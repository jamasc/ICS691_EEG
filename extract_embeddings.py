"""
extract_embeddings.py
=====================

  1. Loads each .set EEG file
  2. Chops it into 2048-sample segments (~4 seconds each)
  3. Runs each segment through EEGPT → saves the embedding (512 numbers)
  4. Runs each segment through biomarkers.py → saves the 31 feature values
  5. Saves everything to a single .npz file

To train probes:
    python train_probes.py --train

Usage:
    python extract_embeddings.py --data-dir /path/to/set/files

Requirements:
    - EEGPT checkpoint file at: checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt
    - GPU recommended (but CPU works, just slower)
    - Libraries: torch, mne, numpy, scipy
"""

import os
import glob
import argparse
import numpy as np
import torch
import mne

# --- Import from the existing repo ---
from utility import segment_signal
from EEGPT_mcae_finetune import EEGPTClassifier
from biomarkers import get_biomarkers_per_segment, features_to_array


def extract_all(data_dir, output_path="probe_training_data.npz"):
    """
    Process all .set files and save paired (embeddings, biomarkers).

    Parameters
    ----------
    data_dir : str
        Folder containing preprocessed .set EEG files.
    output_path : str
        Where to save the output .npz file.
    """

    # ---------------------------------------------------------------
    # Find all .set files
    # ---------------------------------------------------------------
    set_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.set"),
                                 recursive=True))
    print(f"Found {len(set_files)} .set files in {data_dir}")

    if len(set_files) == 0:
        print("ERROR: No .set files found. Check your --data-dir path.")
        return

    # ---------------------------------------------------------------
    # Set up EEGPT model (done once, reused for all files)
    # ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = "checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        print("Make sure the EEGPT checkpoint file is in the right place.")
        return

    # We'll initialize the model once we know the channel names
    # from the first file (they should all be the same).
    model = None

    # Channel name mapping (old 10-20 names → EEGPT names)
    rename_map = {
        'Fp1': 'FP1', 'Fp2': 'FP2',
        'T3': 'T7',   'T4': 'T8',
        'T5': 'P7',   'T6': 'P8',
        'Fz': 'FZ',   'Cz': 'CZ',   'Pz': 'PZ',
    }

    # ---------------------------------------------------------------
    # Process each file
    # ---------------------------------------------------------------
    all_embeddings = []   # Will hold EEGPT outputs
    all_biomarkers = []   # Will hold computed biomarker values
    file_labels = []      # Track which file each segment came from
    feature_names = None

    for i, path in enumerate(set_files):
        filename = os.path.basename(path)
        print(f"\n[{i+1}/{len(set_files)}] {filename}")

        # --- Load the EEG file ---
        try:
            raw = mne.io.read_raw_eeglab(path, preload=True)
            print(f"  Loaded: {len(raw.ch_names)} channels, "
                  f"{raw.n_times} samples, {raw.info['sfreq']} Hz")
        except Exception as e:
            print(f"  SKIP (load error): {e}")
            continue

        # --- Initialize model on first file ---
        if model is None:
            use_ch_names = [rename_map.get(ch, ch) for ch in raw.ch_names]
            model = EEGPTClassifier(
                num_classes=0,               # 0 = feature extraction mode
                in_channels=len(raw.ch_names),
                img_size=[len(use_ch_names), 2048],
                patch_stride=64,
                use_channels_names=use_ch_names,
                use_chan_conv=(len(raw.ch_names) != 19),
                use_predictor=True,
            ).to(device)

            ckpt = torch.load(ckpt_path, map_location=device,
                              weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print(f"  Model loaded on {device}")

        # --- Segment the data ---
        data = raw.get_data()  # shape: (n_channels, n_samples)
        segments = segment_signal(data, window_size=2048, stride=2048)
        n_segments = segments.shape[0]

        if n_segments == 0:
            print(f"  SKIP (recording too short for even 1 segment)")
            continue

        print(f"  Segments: {n_segments}")

        # --- EEGPT: Extract embeddings per segment ---
        try:
            data_batch = torch.tensor(segments, dtype=torch.float32).to(device)
            with torch.no_grad():
                # forward_features returns shape (n_segments, 512)
                embeddings = model.forward_features(data_batch).cpu().numpy()
            print(f"  EEGPT embeddings: {embeddings.shape}")
        except Exception as e:
            print(f"  SKIP (EEGPT error): {e}")
            continue

        # --- Biomarkers: Compute per segment ---
        try:
            seg_features = get_biomarkers_per_segment(
                raw, window_size=2048, stride=2048
            )
            names, bio_array = features_to_array(seg_features)
            print(f"  Biomarkers: {bio_array.shape}")
        except Exception as e:
            print(f"  SKIP (biomarker error): {e}")
            continue

        # --- Make sure segment counts match ---
        n = min(embeddings.shape[0], bio_array.shape[0])
        if n == 0:
            print(f"  SKIP (no matching segments)")
            continue

        all_embeddings.append(embeddings[:n])
        all_biomarkers.append(bio_array[:n])
        file_labels.extend([filename] * n)

        if feature_names is None:
            feature_names = names

    # ---------------------------------------------------------------
    # Save everything
    # ---------------------------------------------------------------
    if not all_embeddings:
        print("\nERROR: No data was extracted. Check your files.")
        return

    X = np.vstack(all_embeddings)    # (total_segments, 512)
    Y = np.vstack(all_biomarkers)    # (total_segments, 31)

    print(f"\n{'='*60}")
    print(f"DONE! Results:")
    print(f"  X (EEGPT embeddings):  {X.shape}")
    print(f"  Y (biomarker values):  {Y.shape}")
    print(f"  Files processed:       {len(set(file_labels))}")
    print(f"  Total segments:        {X.shape[0]}")
    print(f"  Embedding dim:         {X.shape[1]}")
    print(f"  Biomarker features:    {Y.shape[1]}")
    print(f"  Feature names:         {feature_names}")
    print(f"\nSaved to: {output_path}")
    print(f"{'='*60}")

    np.savez(
        output_path,
        X=X,
        Y=Y,
        feature_names=feature_names,
        file_labels=file_labels,
    )


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract EEGPT embeddings + biomarker labels "
                    "for probe training"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Folder with preprocessed .set files")
    parser.add_argument("--output", type=str,
                        default="probe_training_data.npz",
                        help="Output file path (default: "
                             "probe_training_data.npz)")
    args = parser.parse_args()

    extract_all(args.data_dir, args.output)
