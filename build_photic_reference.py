"""
build_photic_reference.py
=========================
Build reference ranges from the eyes-open photic stimulation dataset.

Usage:
    python build_photic_reference.py

This processes all .set files in data/eyes_open/, groups subjects
by diagnosis using participants.tsv, and computes per-region
reference ranges for healthy, AD, and FTD under photic stimulation.

The output merges with your existing biomarker_reference.npz so
the pipeline can use condition-specific ranges automatically.
"""

import os
import glob
import numpy as np
import mne

from biomarkers import (
    compute_regional_features, REGIONS,
    save_reference_ranges, load_reference_ranges
)

# --- Configuration ---
DATA_DIR = "data/eyes_open"
PARTICIPANTS_TSV = os.path.join(DATA_DIR, "participants.tsv")
EXISTING_REF_PATH = "biomarker_reference.npz"
OUTPUT_REF_PATH = "biomarker_reference.npz"  # Overwrites with merged version

GROUP_MAP = {"A": "AD", "C": "healthy", "F": "FTD"}
CONDITION_NAME = "photic_stimulation"

# Standard 19-channel names
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]


def load_participants():
    """Load participants.tsv and return {subject_id: diagnosis}."""
    subjects = {}
    with open(PARTICIPANTS_TSV) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                sub_id = parts[0].strip()
                group = parts[3].strip()
                subjects[sub_id] = GROUP_MAP.get(group, "unknown")
    return subjects


def find_set_files():
    """Find all .set files in the data directory."""
    pattern = os.path.join(DATA_DIR, "**", "*.set")
    return sorted(glob.glob(pattern, recursive=True))


def build_photic_reference():
    """Compute reference ranges from photic stimulation data."""
    print("=" * 60)
    print("Building Photic Stimulation Reference Ranges")
    print("=" * 60)

    # Load participant info
    participants = load_participants()
    print(f"Loaded {len(participants)} participants from {PARTICIPANTS_TSV}")

    diagnosis_counts = {}
    for diag in participants.values():
        diagnosis_counts[diag] = diagnosis_counts.get(diag, 0) + 1
    for diag, count in sorted(diagnosis_counts.items()):
        print(f"  {diag}: {count}")

    # Find .set files
    set_files = find_set_files()
    print(f"\nFound {len(set_files)} .set files")

    # Collect features by group
    # {group: {region: {feature: [values...]}}}
    group_values = {"healthy": {}, "AD": {}, "FTD": {}}
    group_counts = {"healthy": 0, "AD": 0, "FTD": 0}

    for i, filepath in enumerate(set_files):
        # Extract subject ID from path
        basename = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        if basename not in participants:
            # Try extracting from filename
            fname = os.path.basename(filepath)
            basename = fname.split("_")[0]

        if basename not in participants:
            print(f"  Skip {filepath} — subject not in participants.tsv")
            continue

        diagnosis = participants[basename]
        if diagnosis not in group_values:
            continue

        # Load EEG with lightweight preprocessing
        # (Full preprocessing is too slow for 88 files — just bandpass filter)
        try:
            raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
            # Pick only EEG channels
            raw.pick(mne.pick_types(raw.info, eeg=True))
            # Bandpass filter (same range as LJ's preprocessing)
            raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
        except Exception as e:
            print(f"  Skip {basename} — load error: {e}")
            continue

        data = raw.get_data()
        sfreq = raw.info["sfreq"]
        ch_names = list(raw.ch_names)

        # Segment into 2048-sample chunks
        window = 2048
        n_samples = data.shape[1]
        n_segs = n_samples // window

        if n_segs == 0:
            print(f"  Skip {basename} — recording too short")
            continue

        group_counts[diagnosis] += 1

        if group_counts[diagnosis] % 5 == 0:
            print(f"  {diagnosis}: {group_counts[diagnosis]} subjects...")

        # Process a sample of segments (3 per subject for speed)
        seg_indices = np.linspace(0, n_segs - 1, min(3, n_segs), dtype=int)
        for seg_idx in seg_indices:
            start = seg_idx * window
            seg_data = data[:, start:start + window]

            regional = compute_regional_features(seg_data, ch_names, sfreq)

            for region, feats in regional.items():
                if region not in group_values[diagnosis]:
                    group_values[diagnosis][region] = {}
                for f, v in feats.items():
                    if f.startswith("_"):
                        continue
                    if f not in group_values[diagnosis][region]:
                        group_values[diagnosis][region][f] = []
                    group_values[diagnosis][region][f].append(v)

    # --- Convert to stats ---
    print(f"\nComputing statistics...")
    photic_ref = {}
    for group in ["healthy", "AD", "FTD"]:
        photic_ref[group] = {}
        for region, feats in group_values[group].items():
            photic_ref[group][region] = {}
            for f, vals in feats.items():
                vals = np.array(vals)
                m = float(np.mean(vals))
                s = float(np.std(vals))
                photic_ref[group][region][f] = {
                    "mean": m,
                    "std": s,
                    "low": m - 2 * s,
                    "high": m + 2 * s,
                }
        print(f"  {group}: {group_counts[group]} subjects")

    # --- Merge with existing reference ---
    if os.path.exists(EXISTING_REF_PATH):
        print(f"\nMerging with existing {EXISTING_REF_PATH}...")
        existing = load_reference_ranges(EXISTING_REF_PATH)

        # Add photic as a per_source entry
        if "per_source" not in existing:
            existing["per_source"] = {}
        existing["per_source"][CONDITION_NAME] = photic_ref

        save_reference_ranges(existing, OUTPUT_REF_PATH)
    else:
        # Create new reference with just photic data
        ref = {
            "overall": photic_ref,
            "per_source": {CONDITION_NAME: photic_ref},
        }
        save_reference_ranges(ref, OUTPUT_REF_PATH)

    # --- Show sample ranges ---
    print(f"\n--- Sample ranges (photic stimulation) ---")
    h = photic_ref.get("healthy", {}).get("frontal", {})
    a = photic_ref.get("AD", {}).get("frontal", {})
    for feat in ["theta_alpha_ratio", "spectral_centroid", "spectral_entropy",
                 "pdf", "lzc"]:
        if feat in h and feat in a:
            hr = h[feat]
            ar = a[feat]
            print(f"  {feat}:")
            print(f"    Healthy: {hr['low']:.3f} – {hr['high']:.3f} "
                  f"(mean={hr['mean']:.3f})")
            print(f"    AD:      {ar['low']:.3f} – {ar['high']:.3f} "
                  f"(mean={ar['mean']:.3f})")

    print(f"\n{'='*60}")
    print("Done! Reference ranges updated.")
    print(f"{'='*60}")


if __name__ == "__main__":
    build_photic_reference()