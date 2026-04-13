"""
biomarkers.py
=============
Computes all 31 LEAD biomarker features from EEG data.

This module replaces/expands the basic get_biomarkers() in utility.py
with the full set of 31 features from the LEAD paper appendix.

Integration with existing repo:
  - Takes MNE Raw objects as input (same as utility.py)
  - Works with .set files loaded via mne.io.read_raw_eeglab()
  - Works with numpy arrays from the Kaggle dataset
  - Returns results as dicts for easy use downstream

For complete_pipeline.ipynb, call:
  - extract_biomarkers(eeg_path) for .set files
  - extract_biomarkers_from_array(eeg_data) for numpy arrays

Dependencies:
    pip install numpy scipy mne

"""

import os
import numpy as np
from scipy import signal, stats
import mne

# =============================================================================
# Frequency band definitions (Hz) — standard clinical EEG bands
# =============================================================================
BANDS = {
    "delta": (0.5, 4),    # Deep sleep, unconscious processing
    "theta": (4, 8),      # Drowsiness, memory — INCREASES in AD
    "alpha": (8, 12),     # Relaxed wakefulness — DECREASES in AD
    "beta":  (12, 30),    # Active thinking, concentration
}


# =============================================================================
# HELPER: Power Spectral Density
# =============================================================================

def _compute_psd(eeg_signal, sfreq):
    """
    Compute PSD using Welch's method.

    Parameters
    ----------
    eeg_signal : array, shape (n_samples,)
        Single-channel EEG voltage values.
    sfreq : float
        Sampling frequency in Hz (e.g., 500).

    Returns
    -------
    freqs : array — frequency axis
    psd   : array — power at each frequency
    """
    nperseg = min(int(2 * sfreq), len(eeg_signal))
    freqs, psd = signal.welch(
        eeg_signal,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )
    return freqs, psd


def _band_power(freqs, psd, low, high):
    """Sum PSD within a frequency range, properly scaled."""
    mask = (freqs >= low) & (freqs < high)
    freq_res = freqs[1] - freqs[0]
    return float(np.sum(psd[mask]) * freq_res)


# =============================================================================
# GROUP 1: Statistical features (features 1–10)
# =============================================================================

def _statistical_features(sig):
    """Basic statistics on the raw voltage signal."""
    return {
        "mean":       float(np.mean(sig)),
        "variance":   float(np.var(sig)),
        "skewness":   float(stats.skew(sig)),
        "kurtosis":   float(stats.kurtosis(sig)),
        "std":        float(np.std(sig)),
        "iqr":        float(stats.iqr(sig)),
        "max":        float(np.max(sig)),
        "min":        float(np.min(sig)),
        "mean_abs":   float(np.mean(np.abs(sig))),
        "median":     float(np.median(sig)),
    }


# =============================================================================
# GROUP 2: Power / spectral band features (features 11–21)
# =============================================================================

def _power_features(sig, sfreq):
    """Band powers, relative powers, and key AD ratios."""
    freqs, psd = _compute_psd(sig, sfreq)

    delta_p = _band_power(freqs, psd, *BANDS["delta"])
    theta_p = _band_power(freqs, psd, *BANDS["theta"])
    alpha_p = _band_power(freqs, psd, *BANDS["alpha"])
    beta_p  = _band_power(freqs, psd, *BANDS["beta"])
    total_p = delta_p + theta_p + alpha_p + beta_p

    safe_total = total_p if total_p > 0 else 1e-10
    safe_alpha = alpha_p if alpha_p > 0 else 1e-10
    safe_beta  = beta_p  if beta_p  > 0 else 1e-10

    return {
        "delta_power":          delta_p,
        "theta_power":          theta_p,
        "alpha_power":          alpha_p,
        "beta_power":           beta_p,
        "total_power":          total_p,
        "theta_alpha_ratio":    theta_p / safe_alpha,   # High = AD
        "alpha_beta_ratio":     alpha_p / safe_beta,
        "delta_relative_power": delta_p / safe_total,
        "theta_relative_power": theta_p / safe_total,
        "alpha_relative_power": alpha_p / safe_total,
        "beta_relative_power":  beta_p  / safe_total,
    }


# =============================================================================
# GROUP 3: Spectral shape features (features 23–28)
# =============================================================================

def _spectral_shape_features(sig, sfreq):
    """Shape of the power spectrum — centroid, rolloff, peak, etc."""
    freqs, psd = _compute_psd(sig, sfreq)

    # Focus on clinically relevant range
    mask = (freqs >= 0.5) & (freqs <= 30)
    f = freqs[mask]
    p = psd[mask]
    total = np.sum(p)
    if total == 0:
        total = 1e-10

    centroid = float(np.sum(f * p) / total)

    cum = np.cumsum(p)
    idx_85 = min(np.searchsorted(cum, 0.85 * total), len(f) - 1)
    rolloff = float(f[idx_85])

    peak = float(f[np.argmax(p)])

    avg_mag = float(np.mean(p))

    idx_50 = min(np.searchsorted(cum, 0.5 * total), len(f) - 1)
    med_freq = float(f[idx_50])

    analytic = signal.hilbert(sig)
    envelope = np.abs(analytic)
    env_mean = np.mean(envelope)
    amp_mod = float(np.std(envelope) / env_mean) if env_mean > 0 else 0.0

    return {
        "spectral_centroid":      centroid,
        "spectral_rolloff":       rolloff,
        "spectral_peak":          peak,
        "average_magnitude":      avg_mag,
        "median_frequency":       med_freq,
        "amplitude_modulation":   amp_mod,
    }


# =============================================================================
# GROUP 4: Entropy features (features 29–31)
# =============================================================================

def _entropy_features(sig, sfreq):
    """Entropy/complexity of the signal. LOWER in AD."""
    freqs, psd = _compute_psd(sig, sfreq)

    mask = (freqs >= 0.5) & (freqs <= 30)
    p = psd[mask]
    p_sum = np.sum(p)
    p_norm = p / p_sum if p_sum > 0 else np.ones_like(p) / len(p)
    p_norm = p_norm[p_norm > 0]

    n = len(p_norm)

    sp_ent = float(-np.sum(p_norm * np.log2(p_norm)))
    if n > 1:
        sp_ent /= np.log2(n)

    shannon = float(-np.sum(p_norm * np.log2(p_norm)))

    q = 2
    tsallis = float((1 - np.sum(p_norm ** q)) / (q - 1))

    return {
        "spectral_entropy": sp_ent,
        "tsallis_entropy":  tsallis,
        "shannon_entropy":  shannon,
    }


# =============================================================================
# FEATURE 22: Phase Coherence (cross-channel)
# =============================================================================

def compute_phase_coherence(ch1, ch2, sfreq, band="alpha"):
    """
    Compute magnitude-squared coherence between two channels,
    averaged over a frequency band.
    """
    low, high = BANDS[band]
    nperseg = min(int(2 * sfreq), len(ch1))
    freqs, coh = signal.coherence(ch1, ch2, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= low) & (freqs < high)
    if np.any(mask):
        return float(np.mean(coh[mask]))
    return 0.0


# =============================================================================
# PER-CHANNEL: Compute all 30 single-channel features
# =============================================================================

def compute_channel_features(eeg_signal, sfreq):
    """
    Compute all 30 single-channel biomarkers for one channel.
    """
    features = {}
    features.update(_statistical_features(eeg_signal))
    features.update(_power_features(eeg_signal, sfreq))
    features.update(_spectral_shape_features(eeg_signal, sfreq))
    features.update(_entropy_features(eeg_signal, sfreq))
    return features


# =============================================================================
# CORE: Compute biomarkers from MNE Raw object
# =============================================================================

def get_biomarkers(raw_eeg):
    """
    Compute all 31 LEAD biomarker features from an MNE Raw object.
    """
    data = raw_eeg.get_data()
    sfreq = raw_eeg.info["sfreq"]
    ch_names = raw_eeg.ch_names
    n_channels = data.shape[0]

    results = {
        "per_channel": {},
        "coherence": {},
        "summary": {},
    }

    for i in range(n_channels):
        ch_features = compute_channel_features(data[i], sfreq)
        results["per_channel"][ch_names[i]] = ch_features

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            pair = f"{ch_names[i]}-{ch_names[j]}"
            results["coherence"][pair] = compute_phase_coherence(
                data[i], data[j], sfreq, band="alpha"
            )

    feature_names = list(results["per_channel"][ch_names[0]].keys())
    for feat in feature_names:
        vals = [results["per_channel"][ch][feat] for ch in ch_names]
        results["summary"][feat] = float(np.mean(vals))

    if results["coherence"]:
        results["summary"]["phase_coherence"] = float(
            np.mean(list(results["coherence"].values()))
        )

    s = results["summary"]
    slow = s.get("delta_power", 0) + s.get("theta_power", 0)
    fast = s.get("alpha_power", 0) + s.get("beta_power", 0)
    s["slowing_index"] = slow / fast if fast > 0 else float("inf")

    return results


def get_biomarkers_from_path(eeg_path):
    """Load a .set file and compute all biomarkers."""
    import mne
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    return get_biomarkers(raw)


# =============================================================================
# SEGMENT-LEVEL: Compute features per segment (for linear probing)
# =============================================================================

def get_biomarkers_per_segment(raw_eeg, window_size=2048, stride=2048):
    """
    Compute biomarkers for each segment of the EEG.
    """
    data = raw_eeg.get_data()
    sfreq = raw_eeg.info["sfreq"]
    n_channels, n_samples = data.shape

    segment_features = []

    for start in range(0, n_samples - window_size + 1, stride):
        seg = data[:, start:start + window_size]

        all_ch_features = []
        for ch_idx in range(n_channels):
            ch_feats = compute_channel_features(seg[ch_idx], sfreq)
            all_ch_features.append(ch_feats)

        feature_names = list(all_ch_features[0].keys())
        mean_features = {}
        for feat in feature_names:
            vals = [ch[feat] for ch in all_ch_features]
            mean_features[feat] = float(np.mean(vals))

        if n_channels >= 2:
            coh_vals = []
            pairs = [(0, n_channels - 1)]
            if n_channels >= 4:
                step = max(1, n_channels // 4)
                pairs += [(i, i + step)
                          for i in range(0, n_channels - step, step)]
            for i, j in pairs:
                coh_vals.append(
                    compute_phase_coherence(seg[i], seg[j], sfreq, "alpha")
                )
            mean_features["phase_coherence"] = float(np.mean(coh_vals))
        else:
            mean_features["phase_coherence"] = 0.0

        segment_features.append(mean_features)

    return segment_features


# =============================================================================
# CONVERSION HELPERS
# =============================================================================

def features_to_array(segment_features):
    """Convert list of feature dicts to a numpy array + feature names."""
    feature_names = list(segment_features[0].keys())
    rows = []
    for seg in segment_features:
        rows.append([seg[f] for f in feature_names])
    return feature_names, np.array(rows, dtype=np.float32)


def biomarkers_to_dataframe(results):
    """Convert get_biomarkers() output to a pandas DataFrame."""
    try:
        import pandas as pd
        return pd.DataFrame(results["per_channel"]).T
    except ImportError:
        print("pandas is not installed — returning raw dict instead.")
        return results["per_channel"]


# =============================================================================
# FORMATTED OUTPUT — for LLM reasoning stage
# =============================================================================

def format_for_llm(results, subject_id="unknown", diagnosis_prob=None):
    """
    Format biomarker results as a structured text block for the LLM.

    This creates the "structured table" that goes into the final
    LLM reasoning stage of the pipeline.
    """
    s = results["summary"]

    lines = []
    lines.append(f"=== EEG Biomarker Report: Subject {subject_id} ===")
    lines.append("")

    if diagnosis_prob is not None:
        lines.append(f"EEGPT AD Classification Probability: {diagnosis_prob:.2%}")
        lines.append("")

    lines.append("--- Power Analysis ---")
    lines.append(f"  Delta power:           {s.get('delta_power', 0):.4f}")
    lines.append(f"  Theta power:           {s.get('theta_power', 0):.4f}")
    lines.append(f"  Alpha power:           {s.get('alpha_power', 0):.4f}")
    lines.append(f"  Beta power:            {s.get('beta_power', 0):.4f}")
    lines.append(f"  Theta/Alpha ratio:     {s.get('theta_alpha_ratio', 0):.4f}  "
                 "(elevated >1.5 suggests AD)")
    lines.append(f"  Slowing index:         {s.get('slowing_index', 0):.4f}  "
                 "(elevated = more slow-wave activity)")
    lines.append(f"  Alpha relative power:  {s.get('alpha_relative_power', 0):.4f}")
    lines.append(f"  Theta relative power:  {s.get('theta_relative_power', 0):.4f}")

    lines.append("")
    lines.append("--- Spectral Shape ---")
    lines.append(f"  Spectral centroid:     {s.get('spectral_centroid', 0):.2f} Hz  "
                 "(lower = brain slowing)")
    lines.append(f"  Spectral peak:         {s.get('spectral_peak', 0):.2f} Hz  "
                 "(healthy ~10 Hz, AD <8 Hz)")
    lines.append(f"  Spectral rolloff:      {s.get('spectral_rolloff', 0):.2f} Hz")
    lines.append(f"  Median frequency:      {s.get('median_frequency', 0):.2f} Hz")

    lines.append("")
    lines.append("--- Complexity & Entropy ---")
    lines.append(f"  Spectral entropy:      {s.get('spectral_entropy', 0):.4f}  "
                 "(lower = less complex = AD-like)")
    lines.append(f"  Shannon entropy:       {s.get('shannon_entropy', 0):.4f}")
    lines.append(f"  Tsallis entropy:       {s.get('tsallis_entropy', 0):.4f}")

    lines.append("")
    lines.append("--- Connectivity ---")
    lines.append(f"  Mean alpha coherence:  {s.get('phase_coherence', 0):.4f}  "
                 "(lower = reduced connectivity = AD-like)")

    lines.append("")
    lines.append("--- AD Indicator Summary ---")
    indicators = []
    if s.get("theta_alpha_ratio", 0) > 1.5:
        indicators.append("ELEVATED theta/alpha ratio")
    if s.get("slowing_index", 0) > 1.0:
        indicators.append("ELEVATED slowing index")
    if s.get("spectral_peak", 99) < 8.5:
        indicators.append("SLOWED spectral peak")
    if s.get("spectral_entropy", 1) < 0.6:
        indicators.append("REDUCED spectral entropy")
    if s.get("phase_coherence", 1) < 0.3:
        indicators.append("REDUCED alpha coherence")

    if indicators:
        for ind in indicators:
            lines.append(f"  * {ind}")
    else:
        lines.append("  No strong individual AD indicators detected.")

    return "\n".join(lines)


# =============================================================================
# PIPELINE ENTRY POINT — one function to call from complete_pipeline.ipynb
# =============================================================================
def extract_biomarkers_from_path(eeg_path):
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    return extract_biomarkers(raw)
    
def extract_biomarkers(eeg_input, sfreq=128, subject_id=None,
                       diagnosis_prob=None, save_to=None):
    """
    One-liner biomarker extraction. Works with any input type.

    In the notebook, just call:
        llm_text = extract_biomarkers(raw)
        llm_text = extract_biomarkers("patient.set")
        llm_text = extract_biomarkers(numpy_array, sfreq=128)
        llm_text = extract_biomarkers(raw, save_to="report.txt")

    Parameters
    ----------
    eeg_input : str, mne.io.BaseRaw, or numpy array
        - str: path to a .set/.edf/.fif file (loads automatically)
        - mne.io.BaseRaw: raw EEG object (e.g., from LJ's preprocessing)
        - numpy array shape (n_channels, n_samples): raw voltage data
    sfreq : float
        Sampling rate in Hz. Only needed for numpy array input.
        Default 128 (Kaggle dataset). Ignored for Raw/.set inputs.
    subject_id : str, optional
        Patient identifier for the report. Auto-detected from filename
        if loading from a path.
    diagnosis_prob : float, optional
        AD probability from EEGPT (0-1). Added to the report if provided.
    save_to : str, optional
        If provided, saves the LLM report to this file path.
        Example: "reports/patient_001.txt"

    Returns
    -------
    str — the formatted LLM report text (ready to send to Skylar's LLM)
    """
    import mne

    # --- Handle different input types ---

    if isinstance(eeg_input, str):
        # Input is a file path
        raw = mne.io.read_raw(eeg_input, preload=True)
        data = raw.get_data()
        sfreq = raw.info["sfreq"]
        channel_names = raw.ch_names
        if subject_id is None:
            subject_id = os.path.splitext(os.path.basename(eeg_input))[0]

    elif isinstance(eeg_input, mne.io.BaseRaw):
        # Input is an MNE Raw object (from LJ's preprocessing)
        data = eeg_input.get_data()
        sfreq = eeg_input.info["sfreq"]
        channel_names = eeg_input.ch_names
        if subject_id is None:
            subject_id = "unknown"

    elif isinstance(eeg_input, np.ndarray):
        # Input is a numpy array (from Kaggle dataset)
        data = eeg_input
        channel_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2'
        ][:data.shape[0]]
        if subject_id is None:
            subject_id = "unknown"

    else:
        raise ValueError(
            f"eeg_input must be a file path (str), MNE Raw object, "
            f"or numpy array. Got: {type(eeg_input)}"
        )

    # --- Compute biomarkers ---

    n_channels = data.shape[0]
    results = {
        "per_channel": {},
        "coherence": {},
        "summary": {},
    }

    # Per-channel features
    for i in range(n_channels):
        ch_name = channel_names[i] if i < len(channel_names) else f"Ch{i}"
        results["per_channel"][ch_name] = compute_channel_features(
            data[i], sfreq
        )

    # Phase coherence between all channel pairs
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            ch_i = channel_names[i] if i < len(channel_names) else f"Ch{i}"
            ch_j = channel_names[j] if j < len(channel_names) else f"Ch{j}"
            results["coherence"][f"{ch_i}-{ch_j}"] = compute_phase_coherence(
                data[i], data[j], sfreq, band="alpha"
            )

    # Summary: mean across channels
    first_ch = list(results["per_channel"].keys())[0]
    feat_names = list(results["per_channel"][first_ch].keys())
    for feat in feat_names:
        vals = [results["per_channel"][ch][feat]
                for ch in results["per_channel"]]
        results["summary"][feat] = float(np.mean(vals))

    if results["coherence"]:
        results["summary"]["phase_coherence"] = float(
            np.mean(list(results["coherence"].values()))
        )

    # Slowing index
    s = results["summary"]
    slow = s.get("delta_power", 0) + s.get("theta_power", 0)
    fast = s.get("alpha_power", 0) + s.get("beta_power", 0)
    s["slowing_index"] = slow / fast if fast > 0 else float("inf")

    output = []
    for k, v in results['summary'].items():
        output.append(v)
    return output

# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Biomarker Module — Integration Test")
    print("=" * 60)

    # Create synthetic EEG to test
    sfreq = 500
    duration = 10
    n_samples = sfreq * duration
    t = np.linspace(0, duration, n_samples, endpoint=False)
    n_channels = 19

    np.random.seed(42)
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        data[ch] = (
            3.0 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
            + 1.0 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
            + 0.5 * np.random.randn(n_samples)
        )

    # Test 1: numpy array input (Kaggle data path)
    print("\n1. Testing with numpy array...")
    llm_text = extract_biomarkers(data, sfreq=sfreq, subject_id="TEST-001",
                                   diagnosis_prob=0.72)
    print(llm_text)

    # Test 2: save to file
    print("\n2. Testing save_to file...")
    extract_biomarkers(data, sfreq=sfreq, subject_id="TEST-001",
                       save_to="test_report.txt")
    with open("test_report.txt") as f:
        print(f"   File contents: {len(f.read())} characters")
    os.remove("test_report.txt")

    # Test 3: MNE Raw object (LJ's preprocessing path)
    print("\n3. Testing with MNE Raw object...")
    import mne
    ch_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    llm_text = extract_biomarkers(raw, subject_id="RAW-TEST")
    print(f"   Got report: {len(llm_text)} characters")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)