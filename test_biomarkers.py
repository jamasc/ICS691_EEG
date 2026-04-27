"""
test_biomarkers.py
==================
Easy-to-run test script for the biomarker pipeline.

Just run:
    python test_biomarkers.py

It loads the Kaggle dataset, loads reference ranges, and runs
the full tiered analysis on one healthy, one AD, and one FTD subject.

Optional flags:
    python test_biomarkers.py --subject ADFTD_sub12
    python test_biomarkers.py --group AD --count 3
    python test_biomarkers.py --save-reports
    python test_biomarkers.py --build-reference
"""

import os
import argparse
import numpy as np

# --- Paths ---
KAGGLE_PATH = os.path.expanduser(
    "~/.cache/kagglehub/datasets/codingyodha/"
    "largest-alzheimer-eeg-dataset/versions/1/"
    "integrated_eeg_dataset.npz"
)
REFERENCE_PATH = "biomarker_reference.npz"


def build_reference():
    """Compute reference ranges from the Kaggle dataset."""
    from biomarkers import compute_reference_ranges, save_reference_ranges

    print("Building reference ranges (this takes 10-20 minutes)...")
    ref = compute_reference_ranges(KAGGLE_PATH)
    save_reference_ranges(ref, REFERENCE_PATH)
    print("Done!")
    return ref


def load_everything():
    """Load dataset and reference ranges."""
    from load_kaggle_data import load_dataset
    from biomarkers import load_reference_ranges

    dataset = load_dataset(KAGGLE_PATH)

    if os.path.exists(REFERENCE_PATH):
        ref = load_reference_ranges(REFERENCE_PATH)
        print(f"Loaded reference ranges from {REFERENCE_PATH}")
    else:
        print(f"WARNING: {REFERENCE_PATH} not found.")
        print("  Run: python test_biomarkers.py --build-reference")
        print("  Proceeding without reference ranges (no flagging).\n")
        ref = None

    return dataset, ref


def get_subject_segments(dataset, subject_key):
    """Get all segments for one subject as channels-first 3D array."""
    from load_kaggle_data import get_subject_chunks, chunk_to_channel_first

    chunks, diag = get_subject_chunks(dataset, subject_key)
    if chunks.size == 0:
        return None, diag

    # Convert all chunks to channels-first: (n, 2048, 19) → (n, 19, 2048)
    segments = np.transpose(chunks, (0, 2, 1))
    return segments, diag


def run_one_subject(dataset, ref, subject_key, save_report=False):
    """Run the full tiered analysis on one subject."""
    from biomarkers import extract_biomarkers
    from load_kaggle_data import SAMPLING_RATE

    segments, diag = get_subject_segments(dataset, subject_key)
    if segments is None:
        print(f"  Skipping {subject_key} — not enough data")
        return

    print(f"\n{'='*60}")
    print(f"Subject: {subject_key}  |  Diagnosis: {diag}  |  "
          f"Segments: {segments.shape[0]}")
    print(f"{'='*60}\n")

    save_path = f"reports/{subject_key}.txt" if save_report else None

    report = extract_biomarkers(
        segments,
        sfreq=SAMPLING_RATE,
        subject_id=subject_key,
        reference=ref,
        save_to=save_path,
    )

    print(report)
    return report


def run_default_test(dataset, ref, save_reports=False):
    """Run on one healthy, one AD, one FTD subject."""
    for group in ["healthy", "AD", "FTD"]:
        for subj_key, subj_data in dataset["subjects"].items():
            if subj_data["diagnosis"] == group:
                segments, _ = get_subject_segments(dataset, subj_key)
                if segments is not None and segments.shape[0] >= 5:
                    run_one_subject(dataset, ref, subj_key, save_reports)
                    break


def run_group(dataset, ref, group, count=3, save_reports=False):
    """Run on multiple subjects from one group."""
    done = 0
    for subj_key, subj_data in dataset["subjects"].items():
        if subj_data["diagnosis"] == group:
            segments, _ = get_subject_segments(dataset, subj_key)
            if segments is not None and segments.shape[0] >= 5:
                run_one_subject(dataset, ref, subj_key, save_reports)
                done += 1
                if done >= count:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test biomarker pipeline")
    parser.add_argument("--build-reference", action="store_true",
                        help="Compute reference ranges from Kaggle data")
    parser.add_argument("--subject", type=str, default=None,
                        help="Run on a specific subject (e.g., ADFTD_sub12)")
    parser.add_argument("--group", type=str, default=None,
                        choices=["healthy", "AD", "FTD"],
                        help="Run on subjects from a specific group")
    parser.add_argument("--count", type=int, default=1,
                        help="How many subjects to test (with --group)")
    parser.add_argument("--save-reports", action="store_true",
                        help="Save reports to reports/ folder")
    args = parser.parse_args()

    if args.build_reference:
        build_reference()

    elif args.subject:
        dataset, ref = load_everything()
        run_one_subject(dataset, ref, args.subject, args.save_reports)

    elif args.group:
        dataset, ref = load_everything()
        run_group(dataset, ref, args.group, args.count, args.save_reports)

    else:
        # Default: test one of each
        dataset, ref = load_everything()
        run_default_test(dataset, ref, args.save_reports)