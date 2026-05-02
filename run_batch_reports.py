"""
run_batch_reports.py
====================
Run the full biomarker pipeline on .set files with preprocessing.
Supports both eyes-closed and photic stimulation datasets.

Usage:
    python run_batch_reports.py --condition eyes_closed
    python run_batch_reports.py --condition photic
    python run_batch_reports.py --condition both
    python run_batch_reports.py --condition eyes_closed --group AD --count 5
    python run_batch_reports.py --condition eyes_closed --subject sub-010
    python run_batch_reports.py --rebuild-reference
    python run_batch_reports.py --rebuild-reference --condition eyes_closed
    python run_batch_reports.py --validate-hc                      # NEW
    python run_batch_reports.py --persistence-threshold 0.4        # NEW
    python run_batch_reports.py --use-mean-std                     # NEW (legacy)

Outputs individual report files to reports/ folder.

CHANGES FROM last version:
  1. find_subject_file() no longer falls back to recursive glob, which was
     causing wrong files (e.g. sub-038's data) to be returned for missing
     subjects (e.g. sub-039).
  2. build_reference() uses nanmean/nanstd and now also stores 5th/95th
     percentiles, which the new flagging logic uses by default.
  3. New CLI args: --persistence-threshold, --use-mean-std, --validate-hc.
  4. validate_hc_false_positive_rate(): runs the pipeline against every HC
     in the eyes-closed dataset and reports how many trip the AD-indicator
     summary. Quick smoke test for whether the reference is calibrated.
"""

import os
import glob
import argparse
import numpy as np
import mne
from biomarkers import (
    extract_biomarkers, load_reference_ranges,
    save_reference_ranges, compute_regional_features,
    DEFAULT_PERSISTENCE_THRESHOLD,
)


# --- Dataset configs ---
DATASETS = {
    "eyes_closed": {
        "dir": "data/eyes_closed",
        "task": "eyesclosed",
        "condition_key": "resting_eyes_closed",
        "label": "Resting State (Eyes Closed)",
    },
    "photic": {
        "dir": "data/eyes_open",
        "task": "photomark",
        "condition_key": "photic_stimulation",
        "label": "Photic Stimulation (Eyes Open)",
    },
}

REFERENCE_PATH = "biomarker_reference.npz"
REPORT_DIR = "reports"
DEFAULT_COUNT = 10
GROUP_MAP = {"A": "AD", "C": "healthy", "F": "FTD"}


def load_participants(data_dir):
    """Load participants.tsv from a dataset directory."""
    tsv_path = os.path.join(data_dir, "participants.tsv")
    subjects = {}
    with open(tsv_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                sub_id = parts[0].strip()
                subjects[sub_id] = {
                    "group": GROUP_MAP.get(parts[3].strip(), "unknown"),
                    "gender": parts[1].strip(),
                    "age": parts[2].strip(),
                    "mmse": parts[4].strip(),
                }
    return subjects


def preprocess_set_file(filepath):
    """Lightweight preprocessing: bandpass filter + average reference."""
    raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
    raw.pick(mne.pick_types(raw.info, eeg=True))
    raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    return raw


def find_subject_file(data_dir, subject_id, task):
    """Find the .set file for a given subject and task.

    FIX: Previously fell back to recursive glob if the exact path didn't
    exist, which silently grabbed the wrong subject's file (this is what
    caused sub-039 to get sub-038's data — the report header said
    'sub-038_task-eyesclosed' even though we'd asked for sub-039).
    Now we just return None — caller must handle missing files explicitly.
    """
    path = os.path.join(data_dir, subject_id, "eeg",
                        f"{subject_id}_task-{task}_eeg.set")
    if os.path.exists(path):
        return path
    return None  # explicit miss — do NOT silently grab a sibling file


def run_one_subject(subject_id, participant_info, ref, dataset_config,
                    save_report=True, persistence_threshold=None,
                    use_percentiles=True):
    """Run full pipeline on one subject."""
    data_dir = dataset_config["dir"]
    task = dataset_config["task"]
    condition = dataset_config["condition_key"]
    cond_label = dataset_config["label"]

    filepath = find_subject_file(data_dir, subject_id, task)
    if not filepath:
        print(f"  SKIP {subject_id} — file not found at expected path "
              f"({data_dir}/{subject_id}/eeg/...)")
        return None

    group = participant_info.get("group", "unknown")
    age = participant_info.get("age", "?")
    mmse = participant_info.get("mmse", "?")

    try:
        raw = preprocess_set_file(filepath)

        save_path = None
        if save_report:
            save_path = os.path.join(
                REPORT_DIR,
                f"{condition}_{group}_{subject_id}.txt"
            )

        kwargs = {
            "subject_id": f"{subject_id}_task-{task}",
            "condition": condition,
            "reference": ref,
            "save_to": save_path,
            "use_percentiles": use_percentiles,
        }
        if persistence_threshold is not None:
            kwargs["persistence_threshold"] = persistence_threshold

        report = extract_biomarkers(raw, **kwargs)

        print(f"\n{'='*60}")
        print(f"  {subject_id} | {group} | Age: {age} | MMSE: {mmse}")
        print(f"  Condition: {cond_label}")
        print(f"{'='*60}")
        print(report)
        return report

    except Exception as e:
        print(f"  ERROR {subject_id}: {e}")
        return None


def build_reference(conditions=None):
    """Build reference ranges from preprocessed .set files.

    Now stores percentiles (p05, p95) in addition to mean/std/low/high.
    Uses nanmean/nanstd so a few NaN segments (e.g. IAF with no detectable
    alpha) don't poison the reference.
    """
    if conditions is None:
        conditions = list(DATASETS.keys())

    print("Building reference ranges from preprocessed .set files...")
    print(f"Conditions: {conditions}")
    print(f"This may take 15-30 minutes.\n")

    reference = {"overall": {}, "per_source": {}}

    for cond_name in conditions:
        config = DATASETS[cond_name]
        data_dir = config["dir"]
        task = config["task"]
        cond_key = config["condition_key"]

        if not os.path.exists(data_dir):
            print(f"  SKIP {cond_name} — directory {data_dir} not found")
            continue

        participants = load_participants(data_dir)
        print(f"\n--- {config['label']} ({len(participants)} subjects) ---")

        group_values = {"healthy": {}, "AD": {}, "FTD": {}}
        group_counts = {"healthy": 0, "AD": 0, "FTD": 0}

        for sub_id, info in sorted(participants.items()):
            group = info["group"]
            if group not in group_values:
                continue

            filepath = find_subject_file(data_dir, sub_id, task)
            if not filepath:
                continue

            try:
                raw = preprocess_set_file(filepath)
            except Exception as e:
                print(f"  Skip {sub_id}: {e}")
                continue

            data = raw.get_data()
            sfreq = raw.info["sfreq"]
            ch_names = list(raw.ch_names)

            group_counts[group] += 1
            if group_counts[group] % 5 == 0:
                print(f"  {group}: {group_counts[group]} subjects...")

            window = 2048
            n_segs = data.shape[1] // window
            if n_segs == 0:
                continue

            # Sample 3 segments spread across the recording
            seg_indices = np.linspace(0, n_segs - 1, min(3, n_segs), dtype=int)
            for seg_idx in seg_indices:
                start = seg_idx * window
                seg_data = data[:, start:start + window]
                regional = compute_regional_features(seg_data, ch_names, sfreq)

                for region, feats in regional.items():
                    if region not in group_values[group]:
                        group_values[group][region] = {}
                    for f, v in feats.items():
                        if f.startswith("_"):
                            continue
                        # Drop NaN/inf at collection time
                        if not np.isfinite(v):
                            continue
                        if f not in group_values[group][region]:
                            group_values[group][region][f] = []
                        group_values[group][region][f].append(v)

        # Convert to stats — now includes percentiles
        cond_ref = {}
        for group in ["healthy", "AD", "FTD"]:
            stats = {}
            for region, feats in group_values[group].items():
                stats[region] = {}
                for f, vals in feats.items():
                    vals = np.asarray(vals, dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    m = float(np.mean(vals))
                    s = float(np.std(vals))
                    p05 = float(np.percentile(vals, 5))
                    p95 = float(np.percentile(vals, 95))
                    stats[region][f] = {
                        "mean": m, "std": s,
                        "low": m - 2 * s, "high": m + 2 * s,
                        "p05": p05, "p95": p95,
                        "n": int(vals.size),
                    }
            cond_ref[group] = stats
            print(f"  {group}: {group_counts[group]} subjects "
                  f"({sum(len(f) for r in stats.values() for f in r.values()) // max(1, len(stats))} "
                  f"feature stats per region)")

        reference["per_source"][cond_key] = cond_ref

        if not reference["overall"]:
            reference["overall"] = cond_ref

    save_reference_ranges(reference, REFERENCE_PATH)

    # Show sample ranges per condition (now with percentiles)
    for cond_name in conditions:
        cond_key = DATASETS[cond_name]["condition_key"]
        cond_ref = reference["per_source"].get(cond_key, {})
        if not cond_ref:
            continue

        print(f"\n--- Sample ranges: {DATASETS[cond_name]['label']} ---")
        print("    (showing 5th–95th percentile bounds, used by default)")
        for feat in ["theta_alpha_ratio", "spectral_centroid",
                     "spectral_entropy", "lzc", "iaf",
                     "theta_relative_power"]:
            h = cond_ref.get("healthy", {}).get("frontal", {}).get(feat)
            a = cond_ref.get("AD", {}).get("frontal", {}).get(feat)
            if h and a:
                print(f"  {feat}:")
                print(f"    Healthy:  p05={h['p05']:.3f}  p95={h['p95']:.3f}  "
                      f"(mean={h['mean']:.3f}, n={h.get('n', '?')})")
                print(f"    AD:       p05={a['p05']:.3f}  p95={a['p95']:.3f}  "
                      f"(mean={a['mean']:.3f}, n={a.get('n', '?')})")

    return reference


def validate_hc_false_positive_rate(ref, dataset_config,
                                    persistence_threshold,
                                    use_percentiles=True):
    """Quick smoke test: run pipeline on every HC and measure how many
    trip the AD-indicator summary.

    A well-calibrated reference should give a low FP rate on HCs (ideally
    <20%). The previous version was getting ~50%+ on HCs because of the
    multiple-comparisons issue described in the conversation.
    """
    data_dir = dataset_config["dir"]
    task = dataset_config["task"]
    condition = dataset_config["condition_key"]

    if not os.path.exists(data_dir):
        print(f"  SKIP — {data_dir} not found")
        return

    participants = load_participants(data_dir)
    hcs = sorted([s for s, info in participants.items()
                  if info["group"] == "healthy"])
    print(f"\nValidating against {len(hcs)} healthy controls "
          f"(persistence={persistence_threshold*100:.0f}%, "
          f"cutoff={'percentiles' if use_percentiles else 'mean±2σ'})...")

    n_clean = 0
    n_flagged = 0
    n_skipped = 0
    feature_counts = {}

    for sub_id in hcs:
        filepath = find_subject_file(data_dir, sub_id, task)
        if not filepath:
            n_skipped += 1
            continue

        try:
            raw = preprocess_set_file(filepath)
        except Exception:
            n_skipped += 1
            continue

        # Need to peek inside the analysis dict, so do it manually
        from biomarkers import analyze_segments
        data = raw.get_data()
        sfreq = raw.info["sfreq"]
        ch_names = list(raw.ch_names)
        window = min(2048, data.shape[1])
        n_segs = data.shape[1] // window
        if n_segs == 0:
            n_skipped += 1
            continue
        trimmed = data[:, :n_segs * window]
        segments = trimmed.reshape(data.shape[0], n_segs, window)
        segments = np.transpose(segments, (1, 0, 2))

        analysis = analyze_segments(
            segments, ch_names, sfreq,
            reference=ref, condition=condition,
            persistence_threshold=persistence_threshold,
            use_percentiles=use_percentiles,
        )

        persistent = analysis.get("persistent_flags", {})
        if persistent:
            n_flagged += 1
            for (region, feat) in persistent.keys():
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
        else:
            n_clean += 1

    n_total = n_clean + n_flagged
    if n_total == 0:
        print("  No HCs successfully processed.")
        return

    fp_rate = n_flagged / n_total * 100
    print(f"\n--- HC validation results ---")
    print(f"  Total HCs processed: {n_total}")
    print(f"  HCs with NO persistent flags: {n_clean} "
          f"({n_clean/n_total*100:.0f}%)  ← we want this high")
    print(f"  HCs with persistent flags:    {n_flagged} "
          f"({fp_rate:.0f}%)  ← false-positive rate")
    if n_skipped:
        print(f"  Skipped (file/processing errors): {n_skipped}")

    if feature_counts:
        print(f"\n  Most common features triggering false flags:")
        for feat, ct in sorted(feature_counts.items(),
                               key=lambda kv: -kv[1])[:5]:
            print(f"    {feat}: {ct}/{n_flagged} flagged HCs")

    if fp_rate > 25:
        print(f"\n  ⚠ FP rate is high. Consider raising "
              f"--persistence-threshold (currently "
              f"{persistence_threshold:.2f}).")
    elif fp_rate < 15:
        print(f"\n  ✓ FP rate looks well-calibrated.")
    else:
        print(f"\n  ~ FP rate is in a reasonable range.")


def main():
    parser = argparse.ArgumentParser(
        description="Run biomarker reports on .set files"
    )
    parser.add_argument("--condition", type=str, default="both",
                        choices=["eyes_closed", "photic", "both"],
                        help="Which dataset to use")
    parser.add_argument("--rebuild-reference", action="store_true",
                        help="Rebuild reference ranges")
    parser.add_argument("--subject", type=str, default=None,
                        help="Run on specific subject (e.g., sub-010)")
    parser.add_argument("--group", type=str, default=None,
                        choices=["AD", "healthy", "FTD"],
                        help="Run on one group only")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help="Subjects per group (default 10)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save report files")

    # NEW arguments
    parser.add_argument("--persistence-threshold", type=float,
                        default=DEFAULT_PERSISTENCE_THRESHOLD,
                        help="Fraction of segments (0-1) that must show a "
                             "flag for it to count as persistent. "
                             "Default 0.25.")
    parser.add_argument("--use-mean-std", action="store_true",
                        help="Use legacy mean ± 2σ cutoffs instead of "
                             "5th/95th percentiles. Not recommended — "
                             "skewed features get over-flagged.")
    parser.add_argument("--validate-hc", action="store_true",
                        help="Run pipeline on every HC and report the "
                             "false-positive rate. Quick calibration check.")
    args = parser.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    use_percentiles = not args.use_mean_std

    if args.condition == "both":
        conditions = ["eyes_closed", "photic"]
    else:
        conditions = [args.condition]

    if args.rebuild_reference:
        build_reference(conditions)
        return

    if os.path.exists(REFERENCE_PATH):
        ref = load_reference_ranges(REFERENCE_PATH)
        print(f"Loaded reference from {REFERENCE_PATH}")
    else:
        print("No reference found — building...")
        ref = build_reference(conditions)

    # Validation mode: report HC false-positive rate, then exit
    if args.validate_hc:
        for cond_name in conditions:
            print(f"\n{'#'*60}")
            print(f"# HC VALIDATION — {DATASETS[cond_name]['label']}")
            print(f"{'#'*60}")
            validate_hc_false_positive_rate(
                ref, DATASETS[cond_name],
                persistence_threshold=args.persistence_threshold,
                use_percentiles=use_percentiles,
            )
        return

    save = not args.no_save
    groups = [args.group] if args.group else ["AD", "healthy", "FTD"]

    for cond_name in conditions:
        config = DATASETS[cond_name]
        data_dir = config["dir"]

        if not os.path.exists(data_dir):
            print(f"\nSKIP {cond_name} — {data_dir} not found")
            continue

        participants = load_participants(data_dir)

        if args.subject:
            info = participants.get(args.subject, {"group": "unknown"})
            run_one_subject(args.subject, info, ref, config, save,
                            persistence_threshold=args.persistence_threshold,
                            use_percentiles=use_percentiles)
            continue

        for group in groups:
            print(f"\n{'#'*60}")
            print(f"# {group.upper()} — {config['label']}")
            print(f"# (persistence={args.persistence_threshold*100:.0f}%, "
                  f"cutoff={'percentiles' if use_percentiles else 'mean±2σ'})")
            print(f"{'#'*60}")

            count = 0
            for sub_id, info in sorted(participants.items()):
                if info["group"] != group:
                    continue
                run_one_subject(sub_id, info, ref, config, save,
                                persistence_threshold=args.persistence_threshold,
                                use_percentiles=use_percentiles)
                count += 1
                if count >= args.count:
                    break

    report_files = [f for f in os.listdir(REPORT_DIR) if f.endswith('.txt')]
    print(f"\n{'='*60}")
    print(f"Done! {len(report_files)} reports in {REPORT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()