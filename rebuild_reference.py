"""
rebuild_reference.py
====================
Rebuilds biomarker_reference.npz from the full Kaggle dataset.

Run this once (on any machine with the .npz file — no GPU needed):

    python rebuild_reference.py --dataset path/to/integrated_eeg_dataset.npz

What it produces
----------------
biomarker_reference.npz with two top-level keys:

  reference["overall"]
      Pooled healthy / AD / FTD ranges across all sources.
      Used as last-resort fallback when a source has no per-source entry.

  reference["per_source"]
      Separate healthy / AD / FTD ranges for each source dataset:
        ADFTD       → resting_eyes_closed (calibrated)
        AD-Auditory → auditory_task       (calibrated)
        ADFSU       → unknown             (calibrated from its own subjects)
        ADSZ        → unknown             (calibrated from its own subjects)
        APAVA-19    → unknown             (calibrated from its own subjects)

      The three "unknown" sources each get their own ranges rather than
      being pooled together. Pooling labs/equipment risks blurring real
      differences; keeping them separate is more conservative and honest.

Fallback lookup order (in flag_abnormalities):
    1. per_source[source][group]   ← tightest, most accurate
    2. per_source[condition][group] ← not used yet, left for future
    3. overall[group]               ← widest, used when source unknown

Run time: ~10-20 min on CPU depending on hardware (3 chunks sampled per subject).
"""

import argparse
import os
import time
import numpy as np
from biomarkers import compute_reference_ranges, save_reference_ranges


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild biomarker reference ranges from the Kaggle EEG dataset."
    )
    parser.add_argument(
        "--dataset",
        default="integrated_eeg_dataset.npz",
        help="Path to integrated_eeg_dataset.npz (default: ./integrated_eeg_dataset.npz)",
    )
    parser.add_argument(
        "--output",
        default="biomarker_reference.npz",
        help="Output path for the reference file (default: biomarker_reference.npz)",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        help="Cap subjects per group (useful for a quick test run). "
             "Omit to use all subjects.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["healthy", "AD", "FTD"],
        help="Diagnosis groups to include. Default: healthy AD FTD",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset}\n"
            "Pass the correct path with --dataset <path>"
        )

    print(f"Dataset : {args.dataset}")
    print(f"Output  : {args.output}")
    print(f"Groups  : {args.groups}")
    if args.n_subjects:
        print(f"Cap     : {args.n_subjects} subjects per group (test mode)")
    print()

    t0 = time.time()
    print("Computing reference ranges (per_source=True)...")
    print("  This samples 3 chunks per subject.")
    print()

    reference = compute_reference_ranges(
        dataset_path=args.dataset,
        n_subjects=args.n_subjects,
        groups=args.groups,
        per_source=True,          # keep each source's ranges separate
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print()

    # --- Summary of what was computed ---
    print("=== Reference summary ===")
    print("\nOverall (pooled across all sources):")
    for group, group_data in reference.get("overall", {}).items():
        n_regions = len(group_data)
        example_region = next(iter(group_data)) if group_data else "—"
        n_feats = len(group_data[example_region]) if group_data else 0
        print(f"  {group:10s}: {n_regions} regions, ~{n_feats} features each")

    print("\nPer-source:")
    for source, source_data in reference.get("per_source", {}).items():
        groups_present = [g for g, d in source_data.items() if d]
        print(f"  {source:15s}: {groups_present}")

    print()
    _warn_about_missing(reference, args.groups)

    save_reference_ranges(reference, path=args.output)
    print(f"\nReference saved → {args.output}")
    print(
        "\nTo use in the pipeline:\n"
        "  from biomarkers import load_reference_ranges\n"
        f"  reference = load_reference_ranges('{args.output}')\n"
    )


def _warn_about_missing(reference, groups):
    """
    Flag any source / group combination where no subjects were found.
    This usually means the dataset labels don't match what we expect —
    better to know now than to silently fall back to overall ranges.
    """
    expected_sources = ["ADFTD", "AD-Auditory", "ADFSU", "ADSZ", "APAVA-19"]
    per_source = reference.get("per_source", {})
    issues = []

    for source in expected_sources:
        if source not in per_source:
            issues.append(f"  ⚠  {source}: no data found at all")
            continue
        for group in groups:
            data = per_source[source].get(group, {})
            if not data:
                issues.append(
                    f"  ⚠  {source} / {group}: "
                    "no subjects — will fall back to overall reference"
                )

    if issues:
        print("Warnings (source/group gaps — check dataset labels):")
        for w in issues:
            print(w)
        print()
    else:
        print("All expected sources and groups have reference data. ✓")


if __name__ == "__main__":
    main()