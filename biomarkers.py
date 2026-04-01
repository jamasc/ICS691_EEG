"""
biomarkers.py
=============
Computes all 31 LEAD biomarker features from EEG data.

This module replaces/expands the basic get_biomarkers() in utility.py
with the full set of 31 features from the LEAD paper appendix.

Integration with existing repo:
  - Takes MNE Raw objects as input (same as utility.py)
  - Works with .set files loaded via mne.io.read_raw_eeglab()
  - Uses segment_signal() from utility.py for consistency
  - Returns results as dicts/DataFrames for easy use downstream

Dependencies (add to your environment):
    pip install numpy scipy mne pandas

Authors: Zelda Cole (biomarkers team)
"""

import numpy as np
from scipy import signal, stats


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

    # Spectral centroid — "center of gravity" of frequencies
    # Shifts LOWER in AD (brain slowing)
    centroid = float(np.sum(f * p) / total)

    # Spectral rolloff — freq below which 85% of power sits
    cum = np.cumsum(p)
    idx_85 = min(np.searchsorted(cum, 0.85 * total), len(f) - 1)
    rolloff = float(f[idx_85])

    # Spectral peak — dominant frequency
    # Healthy: ~10 Hz (alpha). AD: shifts to 8 Hz or below
    peak = float(f[np.argmax(p)])

    # Average magnitude
    avg_mag = float(np.mean(p))

    # Median frequency — splits power in half
    idx_50 = min(np.searchsorted(cum, 0.5 * total), len(f) - 1)
    med_freq = float(f[idx_50])

    # Amplitude modulation — how much signal strength varies over time
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
    p_norm = p_norm[p_norm > 0]   # avoid log(0)

    n = len(p_norm)

    # Spectral entropy — how "flat" the spectrum is (0 to 1)
    sp_ent = float(-np.sum(p_norm * np.log2(p_norm)))
    if n > 1:
        sp_ent /= np.log2(n)

    # Shannon entropy — classic information measure (not normalized)
    shannon = float(-np.sum(p_norm * np.log2(p_norm)))

    # Tsallis entropy — generalized entropy (q=2)
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

    Parameters
    ----------
    ch1, ch2 : array, shape (n_samples,)
    sfreq : float
    band : str — which band to average coherence over

    Returns
    -------
    float — average coherence (0 to 1)
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

    Parameters
    ----------
    eeg_signal : array, shape (n_samples,)
        Preprocessed voltage values from one EEG channel.
    sfreq : float
        Sampling frequency (e.g., 500 Hz).

    Returns
    -------
    dict — {feature_name: value} for all 30 features
    """
    features = {}
    features.update(_statistical_features(eeg_signal))
    features.update(_power_features(eeg_signal, sfreq))
    features.update(_spectral_shape_features(eeg_signal, sfreq))
    features.update(_entropy_features(eeg_signal, sfreq))
    return features


# =============================================================================
# MAIN ENTRY POINT: get_biomarkers() — drop-in replacement for utility.py
# =============================================================================

def get_biomarkers(raw_eeg):
    """
    Compute all 31 LEAD biomarker features from an MNE Raw object.

    This replaces the basic get_biomarkers() in utility.py with the
    complete set of 31 features from the LEAD paper.

    Parameters
    ----------
    raw_eeg : mne.io.Raw
        Loaded and preprocessed EEG (e.g., from read_raw_eeglab).

    Returns
    -------
    dict with keys:
        "per_channel" : dict  — {channel_name: {feature: value, ...}}
        "coherence"   : dict  — {"ch1-ch2": coherence_value, ...}
        "summary"     : dict  — mean across channels for each feature
    """
    data = raw_eeg.get_data()          # shape: (n_channels, n_samples)
    sfreq = raw_eeg.info["sfreq"]      # e.g. 500 Hz
    ch_names = raw_eeg.ch_names        # e.g. ['Fp1', 'Fp2', ...]
    n_channels = data.shape[0]

    results = {
        "per_channel": {},
        "coherence": {},
        "summary": {},
    }

    # --- Per-channel features (30 features × n_channels) ---
    for i in range(n_channels):
        ch_features = compute_channel_features(data[i], sfreq)
        results["per_channel"][ch_names[i]] = ch_features

    # --- Phase coherence between all channel pairs ---
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            pair = f"{ch_names[i]}-{ch_names[j]}"
            results["coherence"][pair] = compute_phase_coherence(
                data[i], data[j], sfreq, band="alpha"
            )

    # --- Summary: mean of each feature across channels ---
    feature_names = list(results["per_channel"][ch_names[0]].keys())
    for feat in feature_names:
        vals = [results["per_channel"][ch][feat] for ch in ch_names]
        results["summary"][feat] = float(np.mean(vals))

    # Add mean coherence to summary
    if results["coherence"]:
        results["summary"]["phase_coherence"] = float(
            np.mean(list(results["coherence"].values()))
        )

    # Add slowing index to summary (for backward compat with utility.py)
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

    This matches the segment_signal() function in utility.py so that
    each segment's biomarker values line up with the corresponding
    EEGPT embedding from get_eeg_features().

    Parameters
    ----------
    raw_eeg : mne.io.Raw
        Loaded EEG data.
    window_size : int
        Samples per segment (default 2048, matching utility.py).
    stride : int
        Step between segments (default 2048 = no overlap).

    Returns
    -------
    segment_features : list of dict
        One dict per segment, each containing the mean feature values
        across all channels for that segment.
    """
    data = raw_eeg.get_data()          # (n_channels, n_samples)
    sfreq = raw_eeg.info["sfreq"]
    ch_names = raw_eeg.ch_names
    n_channels, n_samples = data.shape

    segment_features = []

    for start in range(0, n_samples - window_size + 1, stride):
        seg = data[:, start:start + window_size]

        # Compute features for each channel in this segment
        all_ch_features = []
        for ch_idx in range(n_channels):
            ch_feats = compute_channel_features(seg[ch_idx], sfreq)
            all_ch_features.append(ch_feats)

        # Average across channels to get one value per feature
        feature_names = list(all_ch_features[0].keys())
        mean_features = {}
        for feat in feature_names:
            vals = [ch[feat] for ch in all_ch_features]
            mean_features[feat] = float(np.mean(vals))

        # Add mean coherence for this segment (sample a few pairs
        # to keep it fast — full pairwise is expensive per-segment)
        # We use 5 representative pairs spanning the scalp
        if n_channels >= 2:
            coh_vals = []
            # Sample pairs: first-last, and evenly spaced
            pairs = [(0, n_channels - 1)]
            if n_channels >= 4:
                step = max(1, n_channels // 4)
                pairs += [(i, i + step) for i in range(0, n_channels - step, step)]
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
# CONVERSION HELPERS — for linear probing
# =============================================================================

def features_to_array(segment_features):
    """
    Convert list of feature dicts to a numpy array + feature names.

    Parameters
    ----------
    segment_features : list of dict
        Output from get_biomarkers_per_segment().

    Returns
    -------
    feature_names : list of str
    feature_array : array, shape (n_segments, n_features)
        Ready to use as Y labels for linear probe training.
    """
    feature_names = list(segment_features[0].keys())
    rows = []
    for seg in segment_features:
        rows.append([seg[f] for f in feature_names])
    return feature_names, np.array(rows, dtype=np.float32)


def biomarkers_to_dataframe(results):
    """
    Convert get_biomarkers() output to a pandas DataFrame.

    Useful for exploration, CSV export, or passing to the LLM.
    Requires pandas to be installed.

    Returns
    -------
    pd.DataFrame — one row per channel, columns are feature names.
    """
    try:
        import pandas as pd
        return pd.DataFrame(results["per_channel"]).T
    except ImportError:
        print("pandas is not installed — returning raw dict instead.")
        print("Install with: pip install pandas")
        return results["per_channel"]


# =============================================================================
# FORMATTED OUTPUT — for LLM reasoning stage
# =============================================================================

def format_for_llm(results, subject_id="unknown", diagnosis_prob=None):
    """
    Format biomarker results as a structured text block for the LLM.

    This creates the "structured table" that goes into the final
    LLM reasoning stage of the pipeline.

    Parameters
    ----------
    results : dict
        Output from get_biomarkers().
    subject_id : str
        Patient identifier.
    diagnosis_prob : float or None
        AD probability from EEGPT classifier (0–1), if available.

    Returns
    -------
    str — formatted text block ready to include in an LLM prompt.
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
# DEMO / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Biomarker Module — Integration Test")
    print("=" * 60)

    # Create synthetic EEG to test (no real data file needed)
    sfreq = 500
    duration = 10
    n_samples = sfreq * duration
    t = np.linspace(0, duration, n_samples, endpoint=False)
    n_channels = 19

    # Simulate 19-channel "healthy" EEG
    np.random.seed(42)
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        data[ch] = (
            3.0 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
            + 1.0 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
            + 0.5 * np.random.randn(n_samples)
        )

    # Wrap in MNE Raw object (mimics what read_raw_eeglab returns)
    import mne
    ch_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)

    # --- Test get_biomarkers() ---
    print("\n1. Testing get_biomarkers(raw_eeg)...")
    results = get_biomarkers(raw)
    print(f"   Channels processed: {len(results['per_channel'])}")
    print(f"   Features per channel: {len(results['per_channel']['Fp1'])}")
    print(f"   Coherence pairs: {len(results['coherence'])}")
    print(f"\n   Summary (mean across channels):")
    for k, v in results["summary"].items():
        print(f"     {k:30s} = {v:.6f}")

    # --- Test get_biomarkers_per_segment() ---
    print("\n2. Testing get_biomarkers_per_segment()...")
    seg_feats = get_biomarkers_per_segment(raw, window_size=2048, stride=2048)
    print(f"   Number of segments: {len(seg_feats)}")
    print(f"   Features per segment: {len(seg_feats[0])}")
    names, arr = features_to_array(seg_feats)
    print(f"   Feature array shape: {arr.shape}")
    print(f"   (This is your Y matrix for linear probe training)")

    # --- Test format_for_llm() ---
    print("\n3. Testing format_for_llm()...")
    llm_text = format_for_llm(results, subject_id="TEST-001", diagnosis_prob=0.72)
    print(llm_text)

    # --- Test DataFrame export ---
    print("\n4. Testing biomarkers_to_dataframe()...")
    df = biomarkers_to_dataframe(results)
    try:
        # If pandas is available, df is a DataFrame
        print(f"   DataFrame shape: {df.shape}")
        print(f"   Columns: {list(df.columns[:5])}... ({len(df.columns)} total)")
    except AttributeError:
        # If pandas is not available, df is a plain dict
        print(f"   Channels: {len(df)}")
        print(f"   (pandas not installed — returned dict instead)")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)