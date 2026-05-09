import os
from pathlib import Path
import mne
from mne.preprocessing import ICA
import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = [".edf", ".bdf", ".fif", ".set", ".vhdr", ".csv"]
TARGET_SFREQ = 128.0
MIN_SAMPLES_FOR_PSD = 512
STANDARD_19 = [
        'FP1','FP2','F7','F3','FZ','F4','F8',
        'T7','C3','CZ','C4','T8',
        'P7','P3','PZ','P4','P8',
        'O1','O2'
    ]

# THIS PROBABLY NEEDS TO BE CHANGED - I have no idea what is closest to what, spatially.
CSV_32_CHANNEL_MAP = {
    "EEG_Electrode_1": "FP1",
    "EEG_Electrode_17": "FP2",
    "EEG_Electrode_4": "F7",
    "EEG_Electrode_3": "F3",
    "EEG_Electrode_19": "FZ",
    "EEG_Electrode_20": "F4",
    "EEG_Electrode_21": "F8",

    "EEG_Electrode_8": "T7",
    "EEG_Electrode_7": "C3",
    "EEG_Electrode_24": "CZ",
    "EEG_Electrode_25": "C4",
    "EEG_Electrode_26": "T8",

    "EEG_Electrode_12": "P7",
    "EEG_Electrode_11": "P3",
    "EEG_Electrode_16": "PZ",
    "EEG_Electrode_29": "P4",
    "EEG_Electrode_30": "P8",

    "EEG_Electrode_14": "O1",
    "EEG_Electrode_32": "O2",
}

def load_raw(file_path: str | Path) -> mne.io.BaseRaw:
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext in [".edf", ".bdf"]:
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif ext == ".fif":
        raw = mne.io.read_raw_fif(file_path, preload=True)
    elif ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(file_path, preload=True)
    elif ext == ".set":
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
    elif ext == ".csv":
        raw = load_csv_as_raw(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return raw

def npz_to_raw(sample: np.ndarray, sfreq: float = 128.0) -> mne.io.RawArray:
    """
    Convert a single NPZ sample into an MNE Raw object.

    Handles:
    - shape inference (time x channels vs channels x time)
    - channel naming
    - sampling rate standardization
    """

    # Fix orientation
    if sample.shape[0] < sample.shape[1]:
        data = sample  # already (channels, time)
    else:
        data = sample.T  # (time, channels) → (channels, time)

    n_channels = data.shape[0]

    # Assign channel names
    standard_19 = [
        'FP1','FP2','F7','F3','FZ','F4','F8',
        'T7','C3','CZ','C4','T8',
        'P7','P3','PZ','P4','P8',
        'O1','O2'
    ]

    if n_channels <= len(standard_19):
        ch_names = standard_19[:n_channels]
    else:
        ch_names = [f"CH{i}" for i in range(n_channels)]

    # Create MNE Raw
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types="eeg"
    )

    raw = mne.io.RawArray(data, info, verbose=False)

    return raw

def preprocess_npz_sample(
    sample: np.ndarray,
    apply_ica: bool = False,
    target_sfreq: float = 128.0
) -> mne.io.BaseRaw:
    raw = npz_to_raw(sample, sfreq=target_sfreq)
    # Apply standard pipeline
    raw = preprocess_raw(raw, apply_ica=apply_ica, target_sfreq=target_sfreq)

    return raw

# lets you preprocess once and then save it
def preprocess_npz_file(npz_path: str | Path, apply_ica: bool = False):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X_raw"]

    raws = []

    for i in range(len(X)):
        try:
            raw = preprocess_npz_sample(X[i], apply_ica=apply_ica)
            raws.append(raw)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")

    return raws

def load_csv_as_raw(file_path: str | Path, sfreq: float = TARGET_SFREQ) -> mne.io.RawArray:

    df = pd.read_csv(file_path)

    # Keep only electrode columns
    electrode_cols = [
        col for col in df.columns
        if col.startswith("EEG_Electrode_")
    ]

    if len(electrode_cols) == 0:
        raise ValueError("No EEG electrode columns found in CSV.")

    data = df[electrode_cols].to_numpy(dtype=np.float64)

    # CSV rows are time samples
    # MNE expects (channels, time)
    data = data.T

    # Rename numbered electrodes to approximate 10-20 locations
    ch_names = []
    selected_data = []

    for i, col in enumerate(electrode_cols):

        if col in CSV_32_CHANNEL_MAP:
            ch_names.append(CSV_32_CHANNEL_MAP[col])
            selected_data.append(data[i])

    if len(selected_data) == 0:
        raise ValueError("No mapped EEG channels found.")

    selected_data = np.array(selected_data)

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types="eeg"
    )

    raw = mne.io.RawArray(selected_data, info, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    return raw

# channel handling
def standardize_channel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Expansion of the rename_map in utilities.py. May continue to expand as we learn more.
    """
    rename_map = {
        'Fp1': 'FP1', 'Fp2': 'FP2',
        'Fz': 'FZ', 'Cz': 'CZ', 'Pz': 'PZ',
        
        'T3': 'T7', 'T4': 'T8',
        'T5': 'P7', 'T6': 'P8',

        'Fpz': 'FPZ', 'Oz': 'OZ',
        'POz': 'POZ'
    }

    mapping = {}
    for ch in raw.ch_names:
        if ch in rename_map:
            mapping[ch] = rename_map[ch]
        else:
            mapping[ch] = ch.upper()

    raw.rename_channels(mapping)
    return raw


def remove_non_eeg_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.pick(mne.pick_types(raw.info, eeg=True))
    return raw

def enforce_standard_19_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:

    existing = [ch for ch in STANDARD_19 if ch in raw.ch_names]
    missing = [ch for ch in STANDARD_19 if ch not in raw.ch_names]

    if len(existing) == 0:
        raise ValueError("No matching standard EEG channels found.")

    # keep only channels relevant to 19ch montage
    raw.pick(existing)

    # create standard montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    # add missing channels as zero-filled placeholders
    if missing:
        sfreq = raw.info["sfreq"]
        zeros = np.zeros((len(missing), raw.n_times))

        info = mne.create_info(
            ch_names=missing,
            sfreq=sfreq,
            ch_types="eeg"
        )

        missing_raw = mne.io.RawArray(zeros, info, verbose=False)
        missing_raw.set_montage(montage, on_missing="ignore")

        raw.add_channels([missing_raw], force_update_info=True)

        raw.interpolate_bads(reset_bads=True)

    # enforce ordering
    raw.reorder_channels(STANDARD_19)

    return raw

# Filters
def detect_powerline_noise(raw: mne.io.BaseRaw) -> bool:
    sfreq = raw.info["sfreq"]
    nyquist = sfreq / 2

    # Stay safely below Nyquist
    fmax = min(70, nyquist - 1)

    psd = raw.compute_psd(fmax=fmax)
    freqs = psd.freqs
    psd_data = psd.get_data().mean(axis=0)

    power_50 = psd_data[(freqs > 49) & (freqs < 51)].mean() if nyquist > 51 else 0
    power_60 = psd_data[(freqs > 59) & (freqs < 61)].mean() if nyquist > 61 else 0

    return power_50 > 1e-10 or power_60 > 1e-10

def apply_notch_filter(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.notch_filter(
        freqs=[50, 60],
        method="iir",
        iir_params=dict(order=4, ftype="butter")
    )
    return raw

def apply_bandpass_filter(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    sfreq = raw.info["sfreq"]
    nyquist = sfreq / 2
    h_freq = min(45.0, nyquist - 1)

    if raw.n_times < 512: # use IRR filtering for short segments
        raw.filter(
            l_freq=0.5,
            h_freq=h_freq,
            method="iir",
            iir_params=dict(order=4, ftype="butter")
        )
    else:
        raw.filter(l_freq=0.5, h_freq=h_freq)

    return raw

def apply_average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.set_eeg_reference('average', projection=False)
    return raw

# -----------------------
# Artifact removal
def run_ica_artifact_removal(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    ica = ICA(n_components=15, random_state=97, max_iter="auto")
    ica.fit(raw)

    try:
        eog_indices, _ = ica.find_bads_eog(raw)
        ica.exclude = eog_indices
    except Exception:
        pass  # no EOG channels present

    raw = ica.apply(raw)
    return raw

# -----------------------
# Main pipeline
def preprocess_raw(raw: mne.io.BaseRaw, apply_ica: bool = False, auto_notch: bool = True, target_sfreq: float = TARGET_SFREQ) -> mne.io.BaseRaw:

    # Channel cleanup
    raw = standardize_channel_names(raw)
    raw = remove_non_eeg_channels(raw)

    # Resample
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq)

    # Enforce 19 channels
    raw = enforce_standard_19_montage(raw)

    # Filtering
    if (auto_notch and raw.n_times >= MIN_SAMPLES_FOR_PSD and detect_powerline_noise(raw)):
        raw = apply_notch_filter(raw)

    raw = apply_bandpass_filter(raw)
    raw = apply_average_reference(raw)

    # ICA
    if apply_ica and raw.n_times >= 1024:
        raw = run_ica_artifact_removal(raw)

    return raw


def preprocess_file(
    file_path: str | Path,
    output_path: str | Path | None = None,
    apply_ica: bool = False,
    target_sfreq: float = TARGET_SFREQ
) -> mne.io.BaseRaw:

    raw = load_raw(file_path)
    raw = preprocess_raw(raw, apply_ica=apply_ica, target_sfreq=target_sfreq)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mne.export.export_raw(output_path, raw, fmt="eeglab")

    return raw

# Process in batches
def find_eeg_files(input_dir: str | Path):
    input_dir = Path(input_dir)
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(input_dir.rglob(f"*{ext}"))
    return files

def process_directory(
    input_dir: str | Path = ".",
    output_dir: str | Path | None = None,
    apply_ica: bool = False
):
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = Path.cwd() / "processed_eeg"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_eeg_files(input_dir)

    for file_path in files:
        try:
            output_file = output_dir / (file_path.stem + ".set")
            preprocess_file(file_path, output_file, apply_ica)
            print(f"Processed: {file_path} → {output_file}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# bash setup
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG preprocessing pipeline")
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--ica", action="store_true")

    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        apply_ica=args.ica
    )


"""

sampling frequency needs to be 128hz, 19 channels.

"""
