"""
train_probes.py
===============
Trains linear probes (Ridge regression) that map EEGPT embeddings
to biomarker values.

  - EEGPT embedding (512 dims) → probe → predicted biomarker value

Usage:
    # Step 1: Generate training data from .npz files
    python train_probes.py --generate-data --data-dir /path/to/set/files

    # Step 2: Train the probes
    python train_probes.py --train

    # Step 3: Use in the pipeline
    from train_probes import load_probes, predict_biomarkers
    probes = load_probes("trained_probes.pkl")
    predicted = predict_biomarkers(probes, eegpt_embedding)

Dependencies:
    pip install numpy scikit-learn torch mne pandas

"""

import numpy as np
import os
import pickle
import argparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# =============================================================================
# STEP 1: Generate paired (embedding, biomarker) training data
# =============================================================================

def generate_training_data(eeg_paths, save_path="probe_training_data.npz"):
    """
    For each .set file, compute:
      - EEGPT embeddings (one per segment)   → X
      - Biomarker values (one per segment)    → Y

    This pairs them up so we can train: X → Y

    Parameters
    ----------
    eeg_paths : list of str
        Paths to preprocessed .set EEG files.
    save_path : str
        Where to save the training data.
    """
    import mne
    import torch
    # These imports come from the existing repo
    from utility import get_eeg_features, segment_signal
    from biomarkers import get_biomarkers_per_segment, features_to_array

    all_embeddings = []
    all_biomarkers = []
    feature_names = None

    for i, path in enumerate(eeg_paths):
        print(f"\n[{i+1}/{len(eeg_paths)}] Processing: {os.path.basename(path)}")

        try:
            raw = mne.io.read_raw_eeglab(path, preload=True)
        except Exception as e:
            print(f"  SKIP — could not load: {e}")
            continue

        # --- Get EEGPT embeddings per segment ---
        # This uses Mark's existing code from utility.py
        # get_eeg_features returns the MEAN embedding, but we need
        # PER-SEGMENT embeddings to match our per-segment biomarkers.
        # So we replicate the segmentation and feature extraction here.
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            from EEGPT_mcae_finetune import EEGPTClassifier

            rename_map = {
                'Fp1': 'FP1', 'Fp2': 'FP2',
                'T3': 'T7', 'T4': 'T8',
                'T5': 'P7', 'T6': 'P8',
                'Fz': 'FZ', 'Cz': 'CZ', 'Pz': 'PZ'
            }
            use_channels_names = [rename_map.get(ch, ch)
                                  for ch in raw.ch_names]

            model = EEGPTClassifier(
                num_classes=0,
                in_channels=len(raw.ch_names),
                img_size=[len(use_channels_names), 2048],
                patch_stride=64,
                use_channels_names=use_channels_names,
                use_chan_conv=False if len(raw.ch_names) == 19 else True,
                use_predictor=True,
            ).to(device)

            ckpt_path = "checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            data = raw.get_data()
            segments = segment_signal(data, window_size=2048, stride=2048)

            # Get per-segment embeddings
            data_batch = torch.tensor(segments, dtype=torch.float32).to(device)
            with torch.no_grad():
                # forward_features returns shape (n_segments, 512)
                embeddings = model.forward_features(data_batch).cpu().numpy()

            print(f"  EEGPT: {embeddings.shape[0]} segments, "
                  f"{embeddings.shape[1]}-dim embeddings")

        except Exception as e:
            print(f"  SKIP — EEGPT error: {e}")
            continue

        # --- Get biomarker values per segment ---
        try:
            seg_features = get_biomarkers_per_segment(
                raw, window_size=2048, stride=2048
            )
            names, bio_array = features_to_array(seg_features)
            print(f"  Biomarkers: {bio_array.shape[0]} segments, "
                  f"{bio_array.shape[1]} features")
        except Exception as e:
            print(f"  SKIP — biomarker error: {e}")
            continue

        # --- Match segment counts ---
        # Both should have the same number, but check just in case
        n = min(embeddings.shape[0], bio_array.shape[0])
        if n == 0:
            print("  SKIP — no valid segments")
            continue

        all_embeddings.append(embeddings[:n])
        all_biomarkers.append(bio_array[:n])

        if feature_names is None:
            feature_names = names

    # --- Stack everything ---
    if not all_embeddings:
        print("\nERROR: No valid data was generated.")
        return

    X = np.vstack(all_embeddings)
    Y = np.vstack(all_biomarkers)

    print(f"\n{'='*50}")
    print(f"Total training data:")
    print(f"  X (embeddings):  {X.shape}")
    print(f"  Y (biomarkers):  {Y.shape}")
    print(f"  Features: {feature_names}")

    np.savez(save_path, X=X, Y=Y, feature_names=feature_names)
    print(f"\nSaved to: {save_path}")


# =============================================================================
# STEP 2: Train linear probes
# =============================================================================

def train_probes(data_path="probe_training_data.npz",
                 save_path="trained_probes.pkl"):
    """
    Train one Ridge regression per biomarker feature.

    Ridge regression is just linear regression with a small penalty
    to prevent overfitting. It's fast and works well for this task.

    Parameters
    ----------
    data_path : str
        Path to the .npz file from generate_training_data().
    save_path : str
        Where to save the trained probe models.
    """
    # Load training data
    data = np.load(data_path, allow_pickle=True)
    X = data["X"]            # (n_segments, 512) — EEGPT embeddings
    Y = data["Y"]            # (n_segments, 31)  — biomarker values
    feature_names = list(data["feature_names"])

    print(f"Training data: {X.shape[0]} segments")
    print(f"Embedding dim: {X.shape[1]}")
    print(f"Target features: {len(feature_names)}")
    print()

    # Train one probe per biomarker
    probes = {}
    scores = {}

    for i, feat_name in enumerate(feature_names):
        y = Y[:, i]

        # Skip features with zero variance (nothing to learn)
        if np.std(y) < 1e-10:
            print(f"  SKIP {feat_name} — no variance")
            continue

        # Pipeline: normalize embeddings → Ridge regression
        # StandardScaler makes training more stable
        probe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])

        # Evaluate with 5-fold cross-validation
        cv_scores = cross_val_score(probe, X, y, cv=5, scoring="r2")
        mean_r2 = cv_scores.mean()

        # Train on all data
        probe.fit(X, y)

        probes[feat_name] = probe
        scores[feat_name] = mean_r2

        # Show how well the probe learned this feature
        quality = "GOOD" if mean_r2 > 0.5 else "OK" if mean_r2 > 0.2 else "WEAK"
        print(f"  {feat_name:30s}  R²={mean_r2:.3f}  [{quality}]")

    # Save everything
    output = {
        "probes": probes,
        "scores": scores,
        "feature_names": feature_names,
    }
    with open(save_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\nSaved {len(probes)} trained probes to: {save_path}")
    print(f"\nBest features (highest R²):")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for name, r2 in sorted_scores[:10]:
        print(f"  {name:30s}  R²={r2:.3f}")

    return probes, scores


# =============================================================================
# STEP 3: Use trained probes for prediction
# =============================================================================

def load_probes(path="trained_probes.pkl"):
    """Load saved probes from disk."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data['probes'])} probes")
    return data


def predict_biomarkers(probe_data, eegpt_embedding):
    """
    Predict all biomarker values from an EEGPT embedding.

    Parameters
    ----------
    probe_data : dict
        Output from load_probes().
    eegpt_embedding : array, shape (512,) or (n_segments, 512)
        EEGPT embedding from get_eeg_features() in utility.py.

    Returns
    -------
    dict — {feature_name: predicted_value}
    """
    probes = probe_data["probes"]

    # Handle both single embedding and batch
    if eegpt_embedding.ndim == 1:
        eegpt_embedding = eegpt_embedding.reshape(1, -1)

    predictions = {}
    for feat_name, probe in probes.items():
        pred = probe.predict(eegpt_embedding)
        predictions[feat_name] = float(pred.mean())

    return predictions


# =============================================================================
# FULL PIPELINE DEMO: from .set file to LLM-ready report
# =============================================================================

def full_pipeline(eeg_path, probe_path="trained_probes.pkl"):
    """
    Run the complete biomarker pipeline on a single EEG file.

    Steps:
      1. Load EEG
      2. Get EEGPT embeddings + classification
      3. Get direct biomarkers (traditional signal processing)
      4. Get probe-predicted biomarkers (from EEGPT embeddings)
      5. Format everything for the LLM

    Parameters
    ----------
    eeg_path : str
        Path to a .set file.
    probe_path : str
        Path to trained probes.

    Returns
    -------
    dict with:
        "direct_biomarkers"    — from signal processing
        "predicted_biomarkers" — from EEGPT via probes
        "classification"       — AD probability from EEGPT
        "llm_prompt"           — formatted text for LLM reasoning
    """
    import mne
    from utility import get_eeg_features, classify_eeg
    from biomarkers import get_biomarkers, format_for_llm

    # Load EEG
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    subject_id = os.path.splitext(os.path.basename(eeg_path))[0]

    # Direct biomarkers (traditional signal processing)
    direct = get_biomarkers(raw)

    # EEGPT classification
    ad_prob = float(classify_eeg(raw, num_classes=2, return_probs=True)[1])

    # Probe predictions (if trained probes exist)
    predicted = None
    if os.path.exists(probe_path):
        embedding = get_eeg_features(raw)
        probe_data = load_probes(probe_path)
        predicted = predict_biomarkers(probe_data, embedding.numpy())

    # Format for LLM
    llm_text = format_for_llm(
        direct,
        subject_id=subject_id,
        diagnosis_prob=ad_prob,
    )

    # If we have probe predictions, append them
    if predicted:
        llm_text += "\n\n--- EEGPT Probe-Predicted Biomarkers ---"
        llm_text += "\n(These are predicted from EEGPT's learned representations)"
        for name, val in predicted.items():
            llm_text += f"\n  {name:30s} = {val:.4f}"

    return {
        "direct_biomarkers": direct,
        "predicted_biomarkers": predicted,
        "classification_prob": ad_prob,
        "llm_prompt": llm_text,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train linear probes: EEGPT embeddings → biomarkers"
    )
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate training data from .set files")
    parser.add_argument("--train", action="store_true",
                        help="Train probes from generated data")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing .set EEG files")
    parser.add_argument("--data-file", type=str,
                        default="probe_training_data.npz",
                        help="Path for training data file")
    parser.add_argument("--probe-file", type=str,
                        default="trained_probes.pkl",
                        help="Path for trained probes file")
    args = parser.parse_args()

    if args.generate_data:
        # Find all .set files
        set_files = []
        for root, dirs, files in os.walk(args.data_dir):
            for f in files:
                if f.endswith(".set"):
                    set_files.append(os.path.join(root, f))
        print(f"Found {len(set_files)} .set files in {args.data_dir}")
        generate_training_data(set_files, save_path=args.data_file)

    elif args.train:
        train_probes(data_path=args.data_file, save_path=args.probe_file)

    else:
        parser.print_help()
        print("\nExample workflow:")
        print("  1. python train_probes.py --generate-data --data-dir data/")
        print("  2. python train_probes.py --train")
        print("  3. Use load_probes() and predict_biomarkers() in your code")