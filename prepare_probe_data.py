"""
prepare_probe_data.py
=====================
Two-step process for preparing probe training data:

  Step 1 (Zelda, laptop):  Pre-compute biomarker labels (Y)
  Step 2 (Mark, KOA):      Compute EEGPT embeddings (X)
  Step 3 (Zelda, laptop):  Combine and train probes

Usage:

    # Mark runs this on KOA:
    python prepare_probe_data.py --step2

    # I'll (Zelda) train probes with embeddings file:
    python prepare_probe_data.py --step3

"""

import os
import numpy as np
import argparse
from load_kaggle_data import (
    load_dataset, get_subject_chunks, chunk_to_channel_first,
    SAMPLING_RATE, CHANNEL_NAMES, CHUNK_SAMPLES
)
from biomarkers import compute_channel_features


# Default paths
KAGGLE_NPZ = os.path.expanduser(
    "~/.cache/kagglehub/datasets/codingyodha/"
    "largest-alzheimer-eeg-dataset/versions/1/"
    "integrated_eeg_dataset.npz"
)
BIOMARKER_FILE = "biomarker_labels.npz"      # Zelda's output (Step 1)
EMBEDDING_FILE = "eegpt_embeddings.npz"      # Mark's output (Step 2)
COMBINED_FILE  = "probe_training_data.npz"   # Combined (Step 3)


# =============================================================================
# STEP 1: Pre-compute biomarker labels (Zelda, no GPU)
# =============================================================================

def step1_compute_biomarkers(npz_path=KAGGLE_NPZ,
                             output_path=BIOMARKER_FILE,
                             diagnosis_filter=None,
                             max_subjects=None):
    """
    Compute biomarker values for every chunk in the dataset.

    This creates the Y matrix for probe training. Each row is one
    chunk's biomarker values (averaged across all 19 channels).

    Output .npz contains:
        Y              : (n_chunks, 31) biomarker values
        feature_names  : names of the 31 features
        labels         : diagnosis for each chunk ("AD", "healthy", etc.)
        subject_ids    : which subject each chunk came from
        chunks_data    : the actual EEG chunks (for Mark to run through EEGPT)
    """
    dataset = load_dataset(npz_path)

    # Default: use AD + healthy + unknown (unknown for extra probe data)
    if diagnosis_filter is None:
        diagnosis_filter = ["AD", "healthy", "unknown"]

    all_biomarkers = []
    all_labels = []
    all_subject_ids = []
    all_chunks = []
    feature_names = None

    subjects = dataset["subjects"]
    count = 0

    for subj_key, subj_data in subjects.items():
        if subj_data["diagnosis"] not in diagnosis_filter:
            continue

        chunks, diag = get_subject_chunks(dataset, subj_key)
        if chunks.size == 0:
            continue

        count += 1
        if count % 10 == 0:
            print(f"  Processed {count} subjects...")

        if max_subjects and count > max_subjects:
            break

        for i in range(chunks.shape[0]):
            # Convert to channels-first: (2048, 19) → (19, 2048)
            ch_data = chunk_to_channel_first(chunks[i])

            # Compute features for each channel, then average
            all_ch_feats = []
            for ch in range(ch_data.shape[0]):
                feats = compute_channel_features(ch_data[ch], SAMPLING_RATE)
                all_ch_feats.append(feats)

            # Average across channels
            if feature_names is None:
                feature_names = list(all_ch_feats[0].keys())

            mean_vals = []
            for f in feature_names:
                vals = [ch_feats[f] for ch_feats in all_ch_feats]
                mean_vals.append(float(np.mean(vals)))

            all_biomarkers.append(mean_vals)
            all_labels.append(diag)
            all_subject_ids.append(subj_key)
            all_chunks.append(chunks[i])

    # Convert to arrays
    Y = np.array(all_biomarkers, dtype=np.float32)
    chunks_array = np.stack(all_chunks)

    print(f"\n{'='*50}")
    print(f"Step 1 complete!")
    print(f"  Chunks processed:  {Y.shape[0]}")
    print(f"  Features per chunk: {Y.shape[1]}")
    print(f"  Subjects: {count}")
    print(f"  Labels: { {l: all_labels.count(l) for l in set(all_labels)} }")
    print(f"  Feature names: {feature_names}")
    print(f"  Chunks array shape: {chunks_array.shape}")
    print(f"\nSaved to: {output_path}")
    print(f"{'='*50}")

    np.savez(
        output_path,
        Y=Y,
        feature_names=feature_names,
        labels=all_labels,
        subject_ids=all_subject_ids,
        chunks=chunks_array,   # Mark will need these to run EEGPT on
    )

    return Y, feature_names


# =============================================================================
# STEP 2: Compute EEGPT embeddings (Mark, GPU)
# =============================================================================

def step2_compute_embeddings(biomarker_file=BIOMARKER_FILE,
                             output_path=EMBEDDING_FILE):
    """
    Mark runs this on the GPU. It loads the chunks that Zelda saved
    and runs each one through EEGPT.

    Requires:
        - EEGPT checkpoint at checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt
        - GPU recommended
    """
    import torch
    from EEGPT_mcae_finetune import EEGPTClassifier

    # Load the chunks Zelda prepared
    data = np.load(biomarker_file, allow_pickle=True)
    chunks = data["chunks"]     # (n_chunks, 2048, 19)
    labels = data["labels"]

    print(f"Loaded {chunks.shape[0]} chunks from {biomarker_file}")

    # Transpose to channels-first for EEGPT: (n, 2048, 19) → (n, 19, 2048)
    chunks_cf = np.transpose(chunks, (0, 2, 1))

    # Set up EEGPT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Channel name mapping for EEGPT
    rename_map = {
        'Fp1': 'FP1', 'Fp2': 'FP2',
        'T3': 'T7', 'T4': 'T8',
        'T5': 'P7', 'T6': 'P8',
        'Fz': 'FZ', 'Cz': 'CZ', 'Pz': 'PZ',
    }
    use_ch_names = [rename_map.get(ch, ch) for ch in CHANNEL_NAMES]

    model = EEGPTClassifier(
        num_classes=0,
        in_channels=19,
        img_size=[19, 2048],
        patch_stride=64,
        use_channels_names=use_ch_names,
        use_chan_conv=False,
        use_predictor=True,
    ).to(device)

    ckpt_path = "checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Process in batches to avoid GPU memory issues
    batch_size = 32
    all_embeddings = []

    for start in range(0, len(chunks_cf), batch_size):
        end = min(start + batch_size, len(chunks_cf))
        batch = torch.tensor(chunks_cf[start:end],
                             dtype=torch.float32).to(device)

        with torch.no_grad():
            embeddings = model.forward_features(batch).cpu().numpy()

        all_embeddings.append(embeddings)

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {end}/{len(chunks_cf)} chunks...")

    X = np.vstack(all_embeddings)

    print(f"\n{'='*50}")
    print(f"Step 2 complete!")
    print(f"  Embeddings shape: {X.shape}")
    print(f"  (n_chunks={X.shape[0]}, embedding_dim={X.shape[1]})")
    print(f"\nSaved to: {output_path}")
    print(f"{'='*50}")

    np.savez(
        output_path,
        X=X,
        labels=labels,
    )

    return X


# =============================================================================
# STEP 3: Combine and save (Zelda, no GPU)
# =============================================================================

def step3_combine(biomarker_file=BIOMARKER_FILE,
                  embedding_file=EMBEDDING_FILE,
                  output_path=COMBINED_FILE):
    """
    Combine Zelda's biomarker labels with Mark's embeddings
    into a single file ready for train_probes.py.
    """
    bio_data = np.load(biomarker_file, allow_pickle=True)
    emb_data = np.load(embedding_file, allow_pickle=True)

    Y = bio_data["Y"]
    X = emb_data["X"]
    feature_names = bio_data["feature_names"]
    labels = bio_data["labels"]
    subject_ids = bio_data["subject_ids"]

    # Verify they match
    assert X.shape[0] == Y.shape[0], (
        f"Mismatch! Embeddings have {X.shape[0]} rows but "
        f"biomarkers have {Y.shape[0]} rows."
    )

    print(f"Combining:")
    print(f"  X (embeddings):  {X.shape}")
    print(f"  Y (biomarkers):  {Y.shape}")
    print(f"  Labels: { {l: list(labels).count(l) for l in set(labels)} }")

    np.savez(
        output_path,
        X=X,
        Y=Y,
        feature_names=feature_names,
        labels=labels,
        subject_ids=subject_ids,
    )

    print(f"\nSaved combined data to: {output_path}")
    print(f"Now run: python train_probes.py --train")

    return X, Y


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for probe training (3 steps)"
    )
    parser.add_argument("--step1", action="store_true",
                        help="Zelda: compute biomarker labels (no GPU)")
    parser.add_argument("--step2", action="store_true",
                        help="Mark: compute EEGPT embeddings (GPU)")
    parser.add_argument("--step3", action="store_true",
                        help="Zelda: combine and prepare for training")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Limit subjects (for quick testing)")
    parser.add_argument("--npz-path", type=str, default=KAGGLE_NPZ,
                        help="Path to Kaggle .npz file")
    args = parser.parse_args()

    if args.step1:
        print("STEP 1: Computing biomarker labels...")
        print("(This may take 10-30 minutes for the full dataset)")
        print()
        step1_compute_biomarkers(
            npz_path=args.npz_path,
            max_subjects=args.max_subjects,
        )

    elif args.step2:
        print("STEP 2: Computing EEGPT embeddings...")
        print("(Run this on a machine with GPU)")
        print()
        step2_compute_embeddings()

    elif args.step3:
        print("STEP 3: Combining biomarkers + embeddings...")
        print()
        step3_combine()

    else:
        parser.print_help()
        print("\n" + "=" * 50)
        print("Workflow:")
        print("  1. Zelda:  python prepare_probe_data.py --step1")
        print("  2. Mark:   python prepare_probe_data.py --step2")
        print("  3. Zelda:  python prepare_probe_data.py --step3")
        print("  4. Zelda:  python train_probes.py --train")
        print("=" * 50)
