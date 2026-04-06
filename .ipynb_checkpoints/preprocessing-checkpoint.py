import os
from pathlib import Path
import mne
from mne.preprocessing import ICA

SUPPORTED_EXTENSIONS = [".edf", ".bdf", ".fif", ".set", ".vhdr"]

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
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return raw

# -----------------------
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

# -----------------------
# Filters
def detect_powerline_noise(raw: mne.io.BaseRaw) -> bool:
    psd = raw.compute_psd(fmax=70)
    freqs = psd.freqs
    psd_data = psd.get_data().mean(axis=0)

    power_50 = psd_data[(freqs > 49) & (freqs < 51)].mean()
    power_60 = psd_data[(freqs > 59) & (freqs < 61)].mean()

    return power_50 > 1e-10 or power_60 > 1e-10

def apply_notch_filter(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.notch_filter(freqs=[50, 60])
    return raw

def apply_bandpass_filter(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.filter(l_freq=0.5, h_freq=45.0)
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
def preprocess_raw(
    raw: mne.io.BaseRaw,
    apply_ica: bool = False,
    auto_notch: bool = True
) -> mne.io.BaseRaw:

    raw = standardize_channel_names(raw)
    raw = remove_non_eeg_channels(raw)

    if auto_notch and detect_powerline_noise(raw):
        raw = apply_notch_filter(raw)

    raw = apply_bandpass_filter(raw)
    raw = apply_average_reference(raw)

    if apply_ica:
        raw = run_ica_artifact_removal(raw)

    return raw


def preprocess_file(
    file_path: str | Path,
    output_path: str | Path | None = None,
    apply_ica: bool = False
) -> mne.io.BaseRaw:

    raw = load_raw(file_path)
    raw = preprocess_raw(raw, apply_ica=apply_ica)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mne.export.export_raw(output_path, raw, fmt="eeglab")

    return raw

# -----------------------
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


# -----------------------
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
