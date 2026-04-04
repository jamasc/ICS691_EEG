"""
load_kaggle_data.py
===================
Loads the Kaggle "Largest Alzheimer EEG Dataset" and prepares it
for the pipeline (EEGPT + biomarkers).

The dataset contains 128-sample segments at 128 Hz (1 second each).
EEGPT needs 2048-sample inputs, and biomarkers need longer windows
for reliable spectral analysis. So we concatenate 16 consecutive
segments from the same subject into one 2048-sample chunk (16 seconds).

Usage:
    from load_kaggle_data import load_dataset, get_subject_chunks

    # Load everything
    dataset = load_dataset("path/to/integrated_eeg_dataset.npz")

    # Get chunks for one subject
    chunks, label = get_subject_chunks(dataset, subject_id="10")

    # chunks.shape = (n_chunks, 2048, 19) — ready for EEGPT + biomarkers

Dependencies:
    pip install numpy
"""

import numpy as np
from collections import defaultdict


# =============================================================================
# Dataset constants
# =============================================================================
SAMPLING_RATE = 128          # Hz (samples per second)
SEGMENT_SAMPLES = 128        # Samples per raw segment (1 second)
CONCAT_FACTOR = 16           # How many segments to glue together
CHUNK_SAMPLES = SEGMENT_SAMPLES * CONCAT_FACTOR  # 2048 (matches EEGPT)
N_CHANNELS = 19              # Standard 10-20 montage

# Standard 19-channel names for this dataset (10-20 system)
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

# Label mappings
DIAGNOSIS_MAP = {
    '0': 'healthy',
    '0.0': 'healthy',
    '1': 'AD',
    '1.0': 'AD',
    '2': 'FTD',
    '-1': 'unknown',
}


# =============================================================================
# Load the full dataset
# =============================================================================

def load_dataset(npz_path):
    """
    Load the Kaggle .npz file and organize it by subject.

    Parameters
    ----------
    npz_path : str
        Path to integrated_eeg_dataset.npz

    Returns
    -------
    dict with keys:
        "subjects" : dict mapping subject_id -> {
            "segments": array (n_segments, 128, 19),
            "diagnosis": str ("healthy", "AD", "FTD", "unknown"),
            "source": str (dataset name),
            "raw_label": str (original label value)
        }
        "sampling_rate": int (128)
        "channel_names": list of str
    """
    print(f"Loading {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    X_raw = data['X_raw']       # (101916, 128, 19)
    labels = data['y_labels']   # (101916, 3)

    print(f"  Total segments: {X_raw.shape[0]}")

    # Group segments by subject
    subjects = {}
    for i in range(len(labels)):
        diag_raw = labels[i, 0]
        subj_id = labels[i, 1]
        source = labels[i, 2]

        # Create a unique key combining source + subject
        # (subject IDs might overlap between different source datasets)
        key = f"{source}_sub{subj_id}"

        if key not in subjects:
            subjects[key] = {
                "segment_list": [],
                "diagnosis": DIAGNOSIS_MAP.get(diag_raw, "unknown"),
                "source": source,
                "raw_label": diag_raw,
            }
        subjects[key]["segment_list"].append(X_raw[i])

    # Convert segment lists to arrays
    for key in subjects:
        segs = np.stack(subjects[key]["segment_list"])
        subjects[key]["segments"] = segs
        del subjects[key]["segment_list"]

    # Print summary
    diag_counts = defaultdict(int)
    for subj in subjects.values():
        diag_counts[subj["diagnosis"]] += 1

    print(f"  Subjects: {len(subjects)}")
    for diag, count in sorted(diag_counts.items()):
        print(f"    {diag}: {count} subjects")

    return {
        "subjects": subjects,
        "sampling_rate": SAMPLING_RATE,
        "channel_names": CHANNEL_NAMES,
    }


# =============================================================================
# Concatenate segments into EEGPT-sized chunks
# =============================================================================

def get_subject_chunks(dataset, subject_key, concat_factor=CONCAT_FACTOR):
    """
    Concatenate a subject's 128-sample segments into 2048-sample chunks.

    Parameters
    ----------
    dataset : dict
        Output from load_dataset().
    subject_key : str
        Subject key (e.g., "ADFTD_sub10").
    concat_factor : int
        How many segments to concatenate (default 16 → 2048 samples).

    Returns
    -------
    chunks : array, shape (n_chunks, 2048, 19)
        Ready to feed into EEGPT and biomarkers.py.
    diagnosis : str
        "healthy", "AD", "FTD", or "unknown"
    """
    subj = dataset["subjects"][subject_key]
    segments = subj["segments"]  # (n_segments, 128, 19)
    n_segments = segments.shape[0]

    # How many full chunks can we make?
    n_chunks = n_segments // concat_factor

    if n_chunks == 0:
        print(f"  Warning: subject {subject_key} has only {n_segments} "
              f"segments, need {concat_factor} for one chunk")
        return np.array([]), subj["diagnosis"]

    # Trim to exact multiple and reshape
    trimmed = segments[:n_chunks * concat_factor]
    # Reshape: (n_chunks, concat_factor, 128, 19) → (n_chunks, 2048, 19)
    chunks = trimmed.reshape(n_chunks, concat_factor * SEGMENT_SAMPLES,
                             N_CHANNELS)

    return chunks, subj["diagnosis"]


# =============================================================================
# Get all chunks for a diagnosis group
# =============================================================================

def get_all_chunks(dataset, diagnosis_filter=None, max_subjects=None):
    """
    Get concatenated chunks for all subjects, optionally filtered.

    Parameters
    ----------
    dataset : dict
        Output from load_dataset().
    diagnosis_filter : str or list of str, optional
        Filter to specific diagnoses. E.g., "AD", ["healthy", "AD"],
        or None for all.
    max_subjects : int, optional
        Limit number of subjects (useful for quick testing).

    Returns
    -------
    all_chunks : array, shape (total_chunks, 2048, 19)
    all_labels : list of str — diagnosis for each chunk
    all_subject_ids : list of str — subject key for each chunk
    """
    if isinstance(diagnosis_filter, str):
        diagnosis_filter = [diagnosis_filter]

    all_chunks = []
    all_labels = []
    all_subject_ids = []
    count = 0

    for subj_key, subj_data in dataset["subjects"].items():
        # Filter by diagnosis
        if diagnosis_filter and subj_data["diagnosis"] not in diagnosis_filter:
            continue

        chunks, diag = get_subject_chunks(dataset, subj_key)

        if chunks.size == 0:
            continue

        all_chunks.append(chunks)
        all_labels.extend([diag] * chunks.shape[0])
        all_subject_ids.extend([subj_key] * chunks.shape[0])

        count += 1
        if max_subjects and count >= max_subjects:
            break

    if not all_chunks:
        print("No data found for the given filter.")
        return np.array([]), [], []

    return np.vstack(all_chunks), all_labels, all_subject_ids


# =============================================================================
# Convert chunks to format biomarkers.py expects
# =============================================================================

def chunk_to_channel_first(chunk):
    """
    Convert a chunk from (2048, 19) to (19, 2048).

    The Kaggle data is stored as (time, channels) but both EEGPT and
    biomarkers.py expect (channels, time).

    Parameters
    ----------
    chunk : array, shape (2048, 19)

    Returns
    -------
    array, shape (19, 2048) — ready for biomarkers.compute_channel_features()
    """
    return chunk.T


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    import os

    # Default path where kagglehub downloads to
    default_path = os.path.expanduser(
        "~/.cache/kagglehub/datasets/codingyodha/"
        "largest-alzheimer-eeg-dataset/versions/1/"
        "integrated_eeg_dataset.npz"
    )

    print("=" * 60)
    print("Kaggle EEG Dataset Loader — Demo")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset(default_path)
    print(f"\nSampling rate: {dataset['sampling_rate']} Hz")
    print(f"Channels: {dataset['channel_names']}")

    # Show one subject
    first_key = list(dataset["subjects"].keys())[0]
    subj = dataset["subjects"][first_key]
    print(f"\nExample subject: {first_key}")
    print(f"  Diagnosis: {subj['diagnosis']}")
    print(f"  Source: {subj['source']}")
    print(f"  Raw segments: {subj['segments'].shape}")

    # Get chunks for this subject
    chunks, diag = get_subject_chunks(dataset, first_key)
    print(f"  After concatenation: {chunks.shape}")
    print(f"  (Each chunk is {CHUNK_SAMPLES} samples = "
          f"{CHUNK_SAMPLES / SAMPLING_RATE:.0f} seconds)")

    # Get AD vs Healthy chunks
    print("\n--- AD vs Healthy ---")
    ad_chunks, ad_labels, ad_ids = get_all_chunks(
        dataset, diagnosis_filter=["AD", "healthy"]
    )
    if ad_chunks.size > 0:
        n_ad = sum(1 for l in ad_labels if l == "AD")
        n_hc = sum(1 for l in ad_labels if l == "healthy")
        print(f"  Total chunks: {ad_chunks.shape[0]}")
        print(f"  AD chunks: {n_ad}")
        print(f"  Healthy chunks: {n_hc}")
        print(f"  Chunk shape: {ad_chunks.shape}")

    # Quick biomarker test on one chunk
    print("\n--- Biomarker test on one chunk ---")
    try:
        from biomarkers import compute_channel_features
        test_chunk = chunk_to_channel_first(ad_chunks[0])  # (19, 2048)
        feats = compute_channel_features(test_chunk[0], SAMPLING_RATE)
        print(f"  theta_power:       {feats['theta_power']:.4f}")
        print(f"  alpha_power:       {feats['alpha_power']:.4f}")
        print(f"  theta_alpha_ratio: {feats['theta_alpha_ratio']:.4f}")
        print(f"  spectral_entropy:  {feats['spectral_entropy']:.4f}")
        print(f"  Diagnosis: {ad_labels[0]}")
    except ImportError:
        print("  (biomarkers.py not found — skipping)")

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
