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

Outputs individual report files to reports/ folder.
"""

import os
import glob
import argparse
import numpy as np
import mne
from biomarkers import (
    extract_biomarkers, load_reference_ranges,
    save_reference_ranges, compute_regional_features,
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
    """Find the .set file for a given subject and task."""
    path = os.path.join(data_dir, subject_id, "eeg",
                        f"{subject_id}_task-{task}_eeg.set")
    if os.path.exists(path):
        return path
    # Try glob as fallback
    matches = glob.glob(os.path.join(data_dir, subject_id, "**", "*.set"),
                        recursive=True)
    return matches[0] if matches else None


def run_one_subject(subject_id, participant_info, ref, dataset_config,
                    save_report=True):
    """Run full pipeline on one subject."""
    data_dir = dataset_config["dir"]
    task = dataset_config["task"]
    condition = dataset_config["condition_key"]
    cond_label = dataset_config["label"]

    filepath = find_subject_file(data_dir, subject_id, task)
    if not filepath:
        print(f"  SKIP {subject_id} — file not found")
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

        report = extract_biomarkers(
            raw,
            subject_id=f"{subject_id}_task-{task}",
            condition=condition,
            reference=ref,
            save_to=save_path,
        )

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
    """Build reference ranges from preprocessed .set files."""
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

            seg_indices = np.linspace(0, n_segs - 1, min(3, n_segs), dtype=int)
            for seg_idx in seg_indices:
                start = seg_idx * window
                seg_data = data[:, start:start + window]
                regional = compute_regional_features(seg_data, ch_names, sfreq)

                for region, feats in regional.items():
                    if region not in group_values[group]:
                        group_values[group][region] = {}
                    for f, v in feats.items():
                        if f.startswith("_") or np.isnan(v) or np.isinf(v):
                            continue
                        if f not in group_values[group][region]:
                            group_values[group][region][f] = []
                        group_values[group][region][f].append(v)

        # Convert to stats
        cond_ref = {}
        for group in ["healthy", "AD", "FTD"]:
            stats = {}
            for region, feats in group_values[group].items():
                stats[region] = {}
                for f, vals in feats.items():
                    vals = np.array(vals)
                    m = float(np.mean(vals))
                    s = float(np.std(vals))
                    stats[region][f] = {
                        "mean": m, "std": s,
                        "low": m - 2 * s, "high": m + 2 * s,
                    }
            cond_ref[group] = stats
            print(f"  {group}: {group_counts[group]} subjects")

        reference["per_source"][cond_key] = cond_ref

        # Use the first condition as overall default
        if not reference["overall"]:
            reference["overall"] = cond_ref

    save_reference_ranges(reference, REFERENCE_PATH)

    # Show sample ranges per condition
    for cond_name in conditions:
        cond_key = DATASETS[cond_name]["condition_key"]
        cond_ref = reference["per_source"].get(cond_key, {})
        if not cond_ref:
            continue

        print(f"\n--- Sample ranges: {DATASETS[cond_name]['label']} ---")
        for feat in ["theta_alpha_ratio", "spectral_centroid",
                     "spectral_entropy", "lzc", "iaf",
                     "theta_relative_power"]:
            h = cond_ref.get("healthy", {}).get("frontal", {}).get(feat)
            a = cond_ref.get("AD", {}).get("frontal", {}).get(feat)
            if h and a:
                print(f"  {feat}:")
                print(f"    Healthy: {h['low']:.3f} – {h['high']:.3f} "
                      f"(mean={h['mean']:.3f})")
                print(f"    AD:      {a['low']:.3f} – {a['high']:.3f} "
                      f"(mean={a['mean']:.3f})")

    return reference


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
    args = parser.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Determine which conditions to use
    if args.condition == "both":
        conditions = ["eyes_closed", "photic"]
    else:
        conditions = [args.condition]

    # --- Rebuild reference ---
    if args.rebuild_reference:
        build_reference(conditions)
        return

    # --- Load or build reference ---
    if os.path.exists(REFERENCE_PATH):
        ref = load_reference_ranges(REFERENCE_PATH)
        print(f"Loaded reference from {REFERENCE_PATH}")
    else:
        print("No reference found — building...")
        ref = build_reference(conditions)

    save = not args.no_save
    groups = [args.group] if args.group else ["AD", "healthy", "FTD"]

    # --- Run reports for each condition ---
    for cond_name in conditions:
        config = DATASETS[cond_name]
        data_dir = config["dir"]

        if not os.path.exists(data_dir):
            print(f"\nSKIP {cond_name} — {data_dir} not found")
            continue

        participants = load_participants(data_dir)

        # Run specific subject
        if args.subject:
            info = participants.get(args.subject, {"group": "unknown"})
            run_one_subject(args.subject, info, ref, config, save)
            continue

        # Run by group
        for group in groups:
            print(f"\n{'#'*60}")
            print(f"# {group.upper()} — {config['label']}")
            print(f"{'#'*60}")

            count = 0
            for sub_id, info in sorted(participants.items()):
                if info["group"] != group:
                    continue
                run_one_subject(sub_id, info, ref, config, save)
                count += 1
                if count >= args.count:
                    break

    report_files = [f for f in os.listdir(REPORT_DIR) if f.endswith('.txt')]
    print(f"\n{'='*60}")
    print(f"Done! {len(report_files)} reports in {REPORT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()