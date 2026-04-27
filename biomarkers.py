"""
biomarkers.py
=============
Computes EEG biomarker features for Alzheimer's disease detection.

Features:
  - 31 LEAD features (statistical, power, spectral shape, entropy)
  - Additional clinical biomarkers (LZC, IAF, PDF, posterior coherence)
  - Regional analysis (frontal, temporal, central, parietal, occipital)
  - Healthy reference ranges computed from training data
  - Per-segment AD pattern flagging
  - Epoch-based analysis (when event markers are available)

Pipeline entry point:
    from biomarkers import extract_biomarkers
    biomarkers = extract_biomarkers(eeg)

Dependencies:
    pip install numpy scipy mne
"""

import os
import numpy as np
from scipy import signal, stats


# =============================================================================
# CONSTANTS
# =============================================================================

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
}

# Equipment-independent features — safe to use across any dataset/equipment
# These are ratios, percentages, or frequency-based measures that don't
# depend on amplifier gain or electrode impedance.
NORMALIZED_FEATURES = [
    "theta_alpha_ratio",
    "theta_relative_power",
    "alpha_relative_power",
    "delta_relative_power",
    "beta_relative_power",
    "spectral_centroid",
    "spectral_peak",
    "median_frequency",
    "iaf",
    "pdf",
    "spectral_entropy",
    "shannon_entropy",
    "tsallis_entropy",
    "lzc",
    "amplitude_modulation",
    "alpha_beta_ratio",
]

# Channel-to-region mapping (standard 10-20, 19 channels)
# Includes both original names AND names after LJ's preprocessing
# (which uppercases and renames: Fp1→FP1, Fz→FZ, T3→T7, T5→P7, etc.)
REGIONS = {
    "frontal":   ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                  "FP1", "FP2", "FZ"],
    "central":   ["C3", "Cz", "C4",
                  "CZ"],
    "temporal":  ["T3", "T4", "T5", "T6",
                  "T7", "T8", "P7", "P8"],
    "parietal":  ["P3", "Pz", "P4",
                  "PZ"],
    "occipital": ["O1", "O2"],
}

# Clinical significance of each region in AD
REGION_AD_NOTES = {
    "frontal":   "Executive function, attention — cholinergic dysfunction causes theta increase",
    "central":   "Sensorimotor processing — alpha reduction with disease progression",
    "temporal":  "Memory, language — early AD site, hippocampal damage causes theta increase",
    "parietal":  "Spatial processing — alpha reduction from cortical atrophy",
    "occipital": "Alpha generation site — posterior slowing is a hallmark AD sign",
}

# Posterior channels for posterior-specific coherence
POSTERIOR_CHANNELS = ["P3", "P4", "O1", "O2", "Pz", "T5", "T6",
                      "PZ", "P7", "P8"]

# Recording condition definitions
# Each condition defines which features should NOT be flagged because
# they are expected to be different from resting eyes-closed baseline.
RECORDING_CONDITIONS = {
    "resting_eyes_closed": {
        "label": "Resting State (Eyes Closed)",
        "description": "Gold standard for AD EEG assessment. "
                       "Alpha rhythm is strongest in this condition.",
        "suppress_flags": [],  # No features suppressed — this is the baseline
    },
    "resting_eyes_open": {
        "label": "Resting State (Eyes Open)",
        "description": "Alpha is naturally suppressed with eyes open. "
                       "Do not flag reduced alpha — it is expected.",
        "suppress_flags": [
            "alpha_power", "alpha_relative_power", "iaf",
        ],
    },
    "auditory_task": {
        "label": "Auditory Task",
        "description": "Theta naturally increases during auditory processing. "
                       "Frontal theta elevation may be task-related, not AD.",
        "suppress_flags": [
            "theta_power", "theta_relative_power",
        ],
    },
    "cognitive_task": {
        "label": "Cognitive / Memory Task",
        "description": "Frontal theta increases during working memory. "
                       "Alpha suppresses during active cognition. "
                       "Both are expected and should not be flagged.",
        "suppress_flags": [
            "theta_power", "theta_relative_power", "theta_alpha_ratio",
            "alpha_power", "alpha_relative_power", "pdf",
        ],
    },
    "photic_stimulation": {
        "label": "Photic Stimulation (IPS)",
        "description": "Standard clinical protocol using flashing lights. "
                       "Occipital activity is driven by external stimulus. "
                       "PDF and IAF are not meaningful during stimulation.",
        "suppress_flags": [
            "alpha_power", "alpha_relative_power", "spectral_peak",
            "pdf", "iaf", "spectral_centroid", "median_frequency",
        ],
    },
    "unknown": {
        "label": "Unknown Recording Condition",
        "description": "Recording condition not specified. "
                       "Using wider reference ranges to avoid false alarms. "
                       "Specify condition for more accurate assessment.",
        "suppress_flags": [],
    },
}

# Map Kaggle source datasets to recording conditions
SOURCE_TO_CONDITION = {
    "ADFTD":       "resting_eyes_closed",
    "AD-Auditory": "auditory_task",
    "ADFSU":       "unknown",
    "ADSZ":        "unknown",
    "APAVA-19":    "unknown",
}


# =============================================================================
# HELPERS
# =============================================================================

def _compute_psd(eeg_signal, sfreq):
    """Compute PSD using Welch's method."""
    nperseg = min(int(2 * sfreq), len(eeg_signal))
    freqs, psd = signal.welch(
        eeg_signal, fs=sfreq,
        nperseg=nperseg, noverlap=nperseg // 2,
    )
    return freqs, psd


def _band_power(freqs, psd, low, high):
    """Sum PSD within a frequency range."""
    mask = (freqs >= low) & (freqs < high)
    freq_res = freqs[1] - freqs[0]
    return float(np.sum(psd[mask]) * freq_res)


# =============================================================================
# FEATURE GROUPS
# =============================================================================

def _statistical_features(sig):
    """Features 1-10: Basic statistics on raw voltage."""
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


def _power_features(sig, sfreq):
    """Features 11-21: Band powers, relative powers, ratios."""
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
        "theta_alpha_ratio":    theta_p / safe_alpha,
        "alpha_beta_ratio":     alpha_p / safe_beta,
        "delta_relative_power": delta_p / safe_total,
        "theta_relative_power": theta_p / safe_total,
        "alpha_relative_power": alpha_p / safe_total,
        "beta_relative_power":  beta_p  / safe_total,
    }


def _spectral_shape_features(sig, sfreq):
    """Features 22-28: Shape of the power spectrum."""
    freqs, psd = _compute_psd(sig, sfreq)

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


def _entropy_features(sig, sfreq):
    """Features 29-31: Entropy / complexity."""
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
# ADDITIONAL CLINICAL BIOMARKERS
# =============================================================================

def compute_lzc(sig):
    """
    Lempel-Ziv Complexity (LZC).

    Measures signal complexity by counting distinct patterns in
    the binarized signal. Lower LZC in AD = simpler, more
    repetitive brain activity.
    Downsamples long signals for performance.
    """
    # Downsample long signals (LZC doesn't need high temporal resolution)
    if len(sig) > 512:
        step = len(sig) // 512
        sig = sig[::step][:512]

    # Binarize around median
    median_val = np.median(sig)
    binary = (sig >= median_val).astype(int)
    s = "".join(map(str, binary))
    n = len(s)

    if n <= 1:
        return 0.0

    # Lempel-Ziv 1976 complexity
    c = 1
    i = 0
    k = 1

    while i + k < n:
        if s[i + 1:i + k + 1] in s[0:i + k]:
            k += 1
        else:
            c += 1
            i = i + k
            k = 1

    # Normalize: random binary sequence has c ≈ n / log2(n)
    c_norm = c * np.log2(n) / n
    return float(c_norm)


def compute_iaf(sig, sfreq):
    """
    Individual Alpha Frequency (IAF).

    The peak frequency within the alpha band (8-12 Hz) for this
    specific patient. More clinically specific than overall spectral
    peak. Slows from ~10 Hz to ~8 Hz or below in AD.
    Returns NaN if no alpha activity detected (segment excluded from flagging).
    """
    freqs, psd = _compute_psd(sig, sfreq)

    # Focus only on alpha band
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    alpha_freqs = freqs[alpha_mask]
    alpha_psd = psd[alpha_mask]

    if len(alpha_psd) == 0 or np.sum(alpha_psd) < 1e-10:
        return float('nan')

    # Weighted mean frequency in alpha band (gravity center method)
    iaf = float(np.sum(alpha_freqs * alpha_psd) / np.sum(alpha_psd))

    return iaf


def compute_pdf(sig, sfreq):
    """
    Power Distribution Ratio (PDF).

    Ratio of slow-wave power (delta + theta) to fast-wave power
    (alpha + beta). Also called the "slowing ratio."

    Higher PDF = more slowing = more AD-like.
    Healthy adults typically < 1.0; AD patients often > 1.5.
    Capped at 100 to avoid infinity when fast-wave power is near zero.
    """
    freqs, psd = _compute_psd(sig, sfreq)

    slow = (_band_power(freqs, psd, *BANDS["delta"])
            + _band_power(freqs, psd, *BANDS["theta"]))
    fast = (_band_power(freqs, psd, *BANDS["alpha"])
            + _band_power(freqs, psd, *BANDS["beta"]))

    if fast > 1e-10:
        return min(float(slow / fast), 100.0)
    return 100.0


def compute_phase_coherence(ch1, ch2, sfreq, band="alpha"):
    """Magnitude-squared coherence between two channels in a band."""
    low, high = BANDS[band]
    nperseg = min(int(2 * sfreq), len(ch1))
    try:
        freqs, coh = signal.coherence(ch1, ch2, fs=sfreq, nperseg=nperseg)
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            result = float(np.nanmean(coh[mask]))
            return result if not np.isnan(result) else 0.0
    except Exception:
        pass
    return 0.0


def compute_posterior_alpha_coherence(data, ch_names, sfreq):
    """
    Posterior Alpha Coherence.

    Measures functional connectivity specifically between posterior
    brain regions (parietal + occipital) in the alpha band.

    More clinically specific than whole-head coherence because AD
    preferentially disrupts posterior networks. Reduced posterior
    alpha coherence is a strong AD indicator.
    """
    # Find which channels are posterior
    posterior_indices = []
    for i, ch in enumerate(ch_names):
        if ch in POSTERIOR_CHANNELS:
            posterior_indices.append(i)

    if len(posterior_indices) < 2:
        return 0.0

    # Compute pairwise coherence between all posterior pairs
    coh_values = []
    for i in range(len(posterior_indices)):
        for j in range(i + 1, len(posterior_indices)):
            idx_i = posterior_indices[i]
            idx_j = posterior_indices[j]
            coh = compute_phase_coherence(
                data[idx_i], data[idx_j], sfreq, band="alpha"
            )
            if coh > 0:
                coh_values.append(coh)

    return float(np.mean(coh_values)) if coh_values else 0.0


# =============================================================================
# PER-CHANNEL: All features for one channel
# =============================================================================

def compute_channel_features(eeg_signal, sfreq):
    """Compute all single-channel features (LEAD 30 + LZC + IAF + PDF)."""
    features = {}
    features.update(_statistical_features(eeg_signal))
    features.update(_power_features(eeg_signal, sfreq))
    features.update(_spectral_shape_features(eeg_signal, sfreq))
    features.update(_entropy_features(eeg_signal, sfreq))

    # Additional clinical biomarkers
    features["lzc"] = compute_lzc(eeg_signal)
    features["iaf"] = compute_iaf(eeg_signal, sfreq)
    features["pdf"] = compute_pdf(eeg_signal, sfreq)

    return features


# =============================================================================
# REGIONAL ANALYSIS
# =============================================================================

def compute_regional_features(data, ch_names, sfreq):
    """
    Compute biomarker features grouped by brain region.

    Returns
    -------
    dict: {region_name: {feature_name: mean_value}}
    """
    regional = {}

    for region, region_channels in REGIONS.items():
        # Find which channels from this recording belong to this region
        region_indices = []
        matched_channels = []
        for i, ch in enumerate(ch_names):
            if ch in region_channels:
                region_indices.append(i)
                matched_channels.append(ch)

        if not region_indices:
            continue

        # Compute features for each channel in the region, then average
        all_feats = []
        for idx in region_indices:
            all_feats.append(compute_channel_features(data[idx], sfreq))

        # Average across channels in this region
        feat_names = list(all_feats[0].keys())
        mean_feats = {}
        for f in feat_names:
            mean_feats[f] = float(np.nanmean([ch[f] for ch in all_feats]))

        mean_feats["_channels"] = matched_channels
        regional[region] = mean_feats

    return regional


# =============================================================================
# REFERENCE RANGES — multi-group, condition-aware
# =============================================================================

def compute_reference_ranges(dataset_path, n_subjects=None,
                             groups=None, per_source=True):
    """
    Compute reference ranges for healthy, AD, and FTD groups.

    Call this once to build the reference, then pass it to
    extract_biomarkers().

    Parameters
    ----------
    dataset_path : str
        Path to the Kaggle integrated_eeg_dataset.npz file.
    n_subjects : int, optional
        Limit subjects per group (for quick testing).
    groups : list of str, optional
        Which diagnosis groups to compute. Default: ["healthy", "AD", "FTD"]
    per_source : bool
        If True, also compute per-source-dataset ranges (to handle
        different recording conditions like eyes open vs closed).

    Returns
    -------
    dict with structure:
        {
            "overall": {
                "healthy": {region: {feature: {"mean", "std", "low", "high"}}},
                "AD":      {region: {feature: {"mean", "std", "low", "high"}}},
                "FTD":     {region: {feature: {"mean", "std", "low", "high"}}},
            },
            "per_source": {   # only if per_source=True
                "ADFTD": {"healthy": {...}, "AD": {...}},
                "AD-Auditory": {"healthy": {...}, "AD": {...}},
                ...
            }
        }
    """
    from load_kaggle_data import (
        load_dataset, get_subject_chunks, chunk_to_channel_first,
        SAMPLING_RATE, CHANNEL_NAMES
    )

    if groups is None:
        groups = ["healthy", "AD", "FTD"]

    dataset = load_dataset(dataset_path)

    # Collect features: {group: {region: {feature: [values...]}}}
    overall_values = {g: {} for g in groups}
    source_values = {}  # {source: {group: {region: {feature: [values...]}}}}

    group_counts = {g: 0 for g in groups}

    for subj_key, subj_data in dataset["subjects"].items():
        diag = subj_data["diagnosis"]
        source = subj_data["source"]

        if diag not in groups:
            continue

        chunks, _ = get_subject_chunks(dataset, subj_key)
        if chunks.size == 0:
            continue

        group_counts[diag] += 1

        if group_counts[diag] % 10 == 0:
            print(f"  {diag}: {group_counts[diag]} subjects processed...")

        if n_subjects and group_counts[diag] > n_subjects:
            continue

        # Sample a few chunks per subject (not all — too slow)
        sample_n = min(3, chunks.shape[0])
        for i in range(sample_n):
            ch_data = chunk_to_channel_first(chunks[i])
            regional = compute_regional_features(
                ch_data, CHANNEL_NAMES, SAMPLING_RATE
            )

            for region, feats in regional.items():
                # Overall collection
                if region not in overall_values[diag]:
                    overall_values[diag][region] = {}
                for f, v in feats.items():
                    if f.startswith("_"):
                        continue
                    if f not in overall_values[diag][region]:
                        overall_values[diag][region][f] = []
                    overall_values[diag][region][f].append(v)

                # Per-source collection
                if per_source:
                    if source not in source_values:
                        source_values[source] = {g: {} for g in groups}
                    if region not in source_values[source][diag]:
                        source_values[source][diag][region] = {}
                    for f, v in feats.items():
                        if f.startswith("_"):
                            continue
                        if f not in source_values[source][diag][region]:
                            source_values[source][diag][region][f] = []
                        source_values[source][diag][region][f].append(v)

    # --- Convert to mean/std/ranges ---
    def _values_to_stats(values_dict):
        """Convert {region: {feature: [values]}} to stats."""
        result = {}
        for region, feats in values_dict.items():
            result[region] = {}
            for f, vals in feats.items():
                vals = np.array(vals)
                m = float(np.mean(vals))
                s = float(np.std(vals))
                result[region][f] = {
                    "mean": m,
                    "std": s,
                    "low": m - 2 * s,
                    "high": m + 2 * s,
                }
        return result

    reference = {"overall": {}}
    for group in groups:
        reference["overall"][group] = _values_to_stats(overall_values[group])
        print(f"  {group}: {group_counts[group]} subjects")

    if per_source:
        reference["per_source"] = {}
        for source, group_data in source_values.items():
            reference["per_source"][source] = {}
            for group in groups:
                if group_data[group]:
                    reference["per_source"][source][group] = \
                        _values_to_stats(group_data[group])

    return reference


def save_reference_ranges(reference, path="healthy_reference.npz"):
    """Save computed reference ranges to disk."""
    np.savez(path, reference=reference)
    print(f"Reference saved to {path}")


def load_reference_ranges(path="healthy_reference.npz"):
    """Load saved reference ranges."""
    data = np.load(path, allow_pickle=True)
    return data["reference"].item()


# =============================================================================
# FLAGGING: Compare biomarkers against reference ranges
# =============================================================================

def flag_abnormalities(regional_features, reference=None,
                       condition="unknown", source=None):
    """
    Compare regional features against healthy and AD reference ranges.

    Uses per-source ranges when available (tighter, same equipment).
    Falls back to overall ranges if source isn't specified or doesn't
    have enough data.

    Parameters
    ----------
    regional_features : dict
        Output from compute_regional_features().
    reference : dict, optional
        Multi-group reference from compute_reference_ranges().
    condition : str
        Recording condition. Suppresses expected features.
    source : str, optional
        Source dataset (e.g., "ADFTD"). Uses tighter per-source ranges.

    Returns
    -------
    list of dict: each flag has region, feature, value, status, detail
    """
    flags = []

    # Get features to suppress based on recording condition
    cond_info = RECORDING_CONDITIONS.get(condition, RECORDING_CONDITIONS["unknown"])
    suppress = set(cond_info.get("suppress_flags", []))

    # Pick the best available reference ranges
    # Priority: per-source > per-condition > overall
    healthy_ref = {}
    ad_ref = {}
    ftd_ref = {}

    if reference and "per_source" in reference:
        # Try source-specific ranges first (e.g., "ADFTD")
        if source:
            source_ref = reference["per_source"].get(source, {})
            healthy_ref = source_ref.get("healthy", {})
            ad_ref = source_ref.get("AD", {})
            ftd_ref = source_ref.get("FTD", {})

        # Try condition-specific ranges (e.g., "photic_stimulation")
        if not healthy_ref and condition != "unknown":
            cond_ref = reference["per_source"].get(condition, {})
            healthy_ref = cond_ref.get("healthy", {})
            ad_ref = cond_ref.get("AD", {})
            ftd_ref = cond_ref.get("FTD", {})

    # Fall back to overall if nothing matched
    if not healthy_ref and reference and "overall" in reference:
        healthy_ref = reference["overall"].get("healthy", {})
        ad_ref = reference["overall"].get("AD", {})
        ftd_ref = reference["overall"].get("FTD", {})
    elif not healthy_ref and reference:
        healthy_ref = reference
        ad_ref = {}
        ftd_ref = {}

    if not healthy_ref:
        return flags

    # AD-expected directions for key features
    ad_directions = {
        "theta_power":          "elevated",
        "delta_power":          "elevated",
        "theta_relative_power": "elevated",
        "delta_relative_power": "elevated",
        "theta_alpha_ratio":    "elevated",
        "pdf":                  "elevated",
        "alpha_power":          "reduced",
        "beta_power":           "reduced",
        "alpha_relative_power": "reduced",
        "beta_relative_power":  "reduced",
        "spectral_centroid":    "reduced",
        "spectral_peak":        "reduced",
        "median_frequency":     "reduced",
        "iaf":                  "reduced",
        "spectral_entropy":     "reduced",
        "shannon_entropy":      "reduced",
        "lzc":                  "reduced",
    }

    for region, feats in regional_features.items():
        for feat_name, value in feats.items():
            if feat_name.startswith("_"):
                continue
            if feat_name not in ad_directions:
                continue
            # Skip features that are expected to be abnormal for this condition
            if feat_name in suppress:
                continue

            direction = ad_directions[feat_name]

            # Skip NaN/inf values (e.g., IAF when no alpha, PDF edge cases)
            if np.isnan(value) or np.isinf(value):
                continue

            # Check against healthy range
            h_ref = healthy_ref.get(region, {}).get(feat_name)
            if not h_ref:
                continue

            h_mean = h_ref["mean"]
            h_std = h_ref["std"]

            if h_std > 0:
                z_score = (value - h_mean) / h_std
            else:
                z_score = 0

            # Is it outside healthy range in the AD direction?
            outside_healthy = False
            if direction == "elevated" and value > h_ref["high"]:
                outside_healthy = True
            elif direction == "reduced" and value < h_ref["low"]:
                outside_healthy = True

            if not outside_healthy:
                continue

            # Check where it falls relative to AD range
            a_ref = ad_ref.get(region, {}).get(feat_name)
            status = "abnormal"

            if a_ref:
                a_mean = a_ref["mean"]
                a_low = a_ref["low"]
                a_high = a_ref["high"]

                # Is the value within the AD range?
                if a_low <= value <= a_high:
                    status = "within_AD_range"
                elif direction == "elevated" and value > a_high:
                    status = "exceeds_AD_range"
                elif direction == "reduced" and value < a_low:
                    status = "exceeds_AD_range"
                else:
                    status = "between_healthy_and_AD"

            # Check FTD range too
            f_ref = ftd_ref.get(region, {}).get(feat_name)
            ftd_match = False
            if f_ref:
                if f_ref["low"] <= value <= f_ref["high"]:
                    ftd_match = True

            flags.append({
                "region": region,
                "feature": feat_name,
                "value": value,
                "direction": direction,
                "z_score": z_score,
                "status": status,
                "healthy_range": f"{h_ref['low']:.3f} – {h_ref['high']:.3f}",
                "healthy_mean": h_mean,
                "ad_mean": a_ref["mean"] if a_ref else None,
                "ad_range": (f"{a_ref['low']:.3f} – {a_ref['high']:.3f}"
                             if a_ref else "N/A"),
                "ftd_match": ftd_match,
            })

    # Sort by z-score severity
    flags.sort(key=lambda x: abs(x.get("z_score", 0)), reverse=True)
    return flags


# =============================================================================
# EPOCH-BASED ANALYSIS (when event markers are available)
# =============================================================================

def extract_biomarkers_by_epoch(raw_eeg, event_id=None, tmin=-0.5, tmax=1.0):
    """
    Compute biomarkers per epoch (event-locked segments).

    Use this when the EEG data has event markers (triggers).
    Falls back to fixed-window segmentation if no events found.

    Parameters
    ----------
    raw_eeg : mne.io.BaseRaw
        Raw EEG with event annotations.
    event_id : dict, optional
        Event type mapping (e.g., {"stimulus": 1, "response": 2}).
    tmin, tmax : float
        Time window around each event in seconds.

    Returns
    -------
    list of dict: one dict per epoch with biomarker values.
    """
    import mne

    try:
        events, event_dict = mne.events_from_annotations(raw_eeg)
        if event_id:
            event_dict = event_id

        epochs = mne.Epochs(raw_eeg, events, event_id=event_dict,
                            tmin=tmin, tmax=tmax,
                            baseline=None, preload=True, verbose=False)

        sfreq = raw_eeg.info["sfreq"]
        ch_names = raw_eeg.ch_names

        epoch_features = []
        for i in range(len(epochs)):
            epoch_data = epochs[i].get_data()[0]  # (n_channels, n_samples)
            regional = compute_regional_features(epoch_data, ch_names, sfreq)
            epoch_features.append({
                "epoch_idx": i,
                "regional": regional,
            })

        print(f"  Extracted biomarkers from {len(epoch_features)} epochs")
        return epoch_features

    except Exception as e:
        print(f"  No event markers found ({e}), using fixed-window segmentation")
        return None


# =============================================================================
# FORMATTED OUTPUT — for LLM reasoning stage
# =============================================================================

def format_for_llm(regional_features, subject_id="unknown",
                   diagnosis_prob=None, flags=None,
                   posterior_coh=None, segment_flags=None,
                   condition="unknown"):
    """
    Format biomarker results as a clinical report for the LLM.

    Parameters
    ----------
    regional_features : dict
        Output from compute_regional_features().
    subject_id : str
        Patient identifier.
    diagnosis_prob : float, optional
        AD probability from EEGPT (0-1).
    flags : list, optional
        Output from flag_abnormalities().
    posterior_coh : float, optional
        Posterior alpha coherence value.
    segment_flags : list, optional
        Per-segment flagging results.
    condition : str
        Recording condition identifier.

    Returns
    -------
    str — formatted clinical report for LLM input.
    """
    cond_info = RECORDING_CONDITIONS.get(condition,
                                         RECORDING_CONDITIONS["unknown"])

    lines = []
    lines.append(f"=== EEG Biomarker Report: Subject {subject_id} ===")
    lines.append("")

    # Recording condition
    lines.append(f"Recording condition: {cond_info['label']}")
    lines.append(f"  {cond_info['description']}")
    lines.append("")

    if diagnosis_prob is not None:
        lines.append(f"EEGPT AD Classification Probability: {diagnosis_prob:.2%}")
        lines.append("")

    # --- Regional analysis ---
    key_features = [
        "theta_power", "alpha_power", "theta_alpha_ratio",
        "theta_relative_power", "alpha_relative_power",
        "spectral_centroid", "spectral_peak", "iaf",
        "pdf", "spectral_entropy", "lzc"
    ]

    for region in ["frontal", "temporal", "parietal", "occipital", "central"]:
        if region not in regional_features:
            continue

        feats = regional_features[region]
        channels = feats.get("_channels", [])
        ch_str = ", ".join(channels)

        lines.append(f"--- {region.upper()} ({ch_str}) ---")
        lines.append(f"  Clinical note: {REGION_AD_NOTES.get(region, '')}")

        for f in key_features:
            if f not in feats:
                continue
            v = feats[f]

            # Check if this feature is flagged
            flag_marker = ""
            if flags:
                for fl in flags:
                    if fl["region"] == region and fl["feature"] == f:
                        status = fl.get("status", "abnormal")
                        if status == "within_AD_range":
                            flag_marker = (
                                f"  ⚠ {fl['direction'].upper()} "
                                f"(healthy: {fl['healthy_range']}, "
                                f"AD: {fl['ad_range']}) — consistent with AD")
                        elif status == "exceeds_AD_range":
                            flag_marker = (
                                f"  ⚠⚠ {fl['direction'].upper()} "
                                f"(outside both healthy and AD ranges)")
                        elif status == "between_healthy_and_AD":
                            flag_marker = (
                                f"  ⚠ {fl['direction'].upper()} "
                                f"(healthy: {fl['healthy_range']}) "
                                f"— borderline")
                        else:
                            flag_marker = (
                                f"  ⚠ {fl['direction'].upper()} "
                                f"(healthy: {fl['healthy_range']})")

                        if fl.get("ftd_match"):
                            flag_marker += " [also within FTD range]"
                        break

            # Format based on feature type
            if f in ["spectral_centroid", "spectral_peak", "iaf",
                      "median_frequency"]:
                lines.append(f"  {f:30s} {v:>8.2f} Hz{flag_marker}")
            elif f in ["spectral_entropy", "lzc"]:
                lines.append(f"  {f:30s} {v:>8.4f}{flag_marker}")
            else:
                lines.append(f"  {f:30s} {v:>8.4f}{flag_marker}")

        lines.append("")

    # --- Posterior alpha coherence ---
    if posterior_coh is not None:
        lines.append("--- POSTERIOR CONNECTIVITY ---")
        lines.append(f"  Posterior alpha coherence:  {posterior_coh:.4f}")
        lines.append(f"  (Measures synchronization between parietal/occipital regions)")
        if posterior_coh < 0.3:
            lines.append(f"  ⚠ REDUCED — consistent with disrupted posterior networks in AD")
        lines.append("")

    # --- Per-segment consistency ---
    if segment_flags:
        n_total = segment_flags["total"]
        n_flagged = segment_flags["flagged"]
        pct = (n_flagged / n_total * 100) if n_total > 0 else 0
        lines.append("--- TEMPORAL CONSISTENCY ---")
        lines.append(f"  Segments analyzed: {n_total}")
        lines.append(f"  Segments with AD-like patterns: {n_flagged} ({pct:.0f}%)")
        if pct > 50:
            lines.append(f"  ⚠ Majority of recording shows AD-like patterns")
        elif pct > 25:
            lines.append(f"  Intermittent AD-like patterns detected")
        else:
            lines.append(f"  AD-like patterns are infrequent")
        lines.append("")

    # --- AD indicator summary ---
    lines.append("--- AD INDICATOR SUMMARY ---")
    if flags is None:
        # No reference ranges were provided at all
        lines.append("  No reference ranges provided for comparison.")
        lines.append("  See regional values above for manual interpretation.")
    elif len(flags) == 0:
        # Reference was provided but no abnormalities found
        lines.append("  All features within healthy reference ranges.")
        lines.append("  No AD-associated abnormalities detected.")
    else:
        # Group flags by clinical pattern
        has_posterior_alpha = any(
            f["region"] in ["occipital", "parietal"]
            and f["feature"] in ["alpha_power", "alpha_relative_power", "iaf"]
            for f in flags
        )
        has_theta_increase = any(
            f["feature"] in ["theta_power", "theta_relative_power",
                             "theta_alpha_ratio"]
            for f in flags
        )
        has_complexity_drop = any(
            f["feature"] in ["spectral_entropy", "lzc"]
            for f in flags
        )
        has_slowing = any(
            f["feature"] in ["spectral_centroid", "spectral_peak",
                             "iaf", "pdf"]
            for f in flags
        )

        if has_posterior_alpha:
            lines.append("  * Posterior alpha reduction — consistent with "
                         "thalamocortical circuit disruption")
        if has_theta_increase:
            regions_affected = set(f["region"] for f in flags
                                   if f["feature"] in ["theta_power",
                                                        "theta_relative_power"])
            lines.append(f"  * Theta elevation in {', '.join(regions_affected)} — "
                         "consistent with cholinergic deficit / hippocampal damage")
        if has_complexity_drop:
            lines.append("  * Reduced signal complexity — consistent with "
                         "neuronal network degradation")
        if has_slowing:
            lines.append("  * Spectral slowing — consistent with "
                         "cortical hypoactivation in AD")

        if not any([has_posterior_alpha, has_theta_increase,
                    has_complexity_drop, has_slowing]):
            for f in flags[:5]:
                lines.append(f"  * {f['feature']} {f['direction']} in "
                             f"{f['region']} (z={f.get('z_score', 0):.1f})")

    return "\n".join(lines)


# =============================================================================
# TIERED ANALYSIS — per-segment screening and flagging
# =============================================================================

def screen_segment(data, ch_names, sfreq, reference=None,
                   condition="unknown", source=None):
    """
    Tier 1: Screen one segment using normalized features only.

    Returns regional features and any flags found.
    """
    regional = compute_regional_features(data, ch_names, sfreq)

    if reference:
        flags = flag_abnormalities(regional, reference, condition, source)
        # Keep only normalized feature flags for screening
        flags = [f for f in flags if f["feature"] in NORMALIZED_FEATURES]
    else:
        flags = []

    return regional, flags


def analyze_segments(segments, ch_names, sfreq, reference=None,
                     condition="unknown", source=None,
                     segment_duration=None):
    """
    Run tiered analysis on multiple segments from one subject.

    Tier 1: Screen each segment with normalized features.
    Tier 2: Identify which segments are flagged and report temporal pattern.
    Tier 3: For flagged segments, compute per-channel detail.

    Parameters
    ----------
    segments : array, shape (n_segments, n_channels, n_samples)
        Multiple EEG segments (channels-first per segment).
    ch_names : list of str
        Channel names.
    sfreq : float
        Sampling rate.
    reference : dict, optional
        Reference ranges.
    condition : str
        Recording condition.
    source : str, optional
        Source dataset for per-source ranges.
    segment_duration : float, optional
        Duration of each segment in seconds (for reporting).

    Returns
    -------
    dict with keys:
        "n_segments"         : int
        "segment_duration"   : float (seconds per segment)
        "flagged_indices"    : list of int (which segments were flagged)
        "flag_summary"       : dict {(region, feature): [segment indices]}
        "per_segment_flags"  : list of lists (flags per segment)
        "flagged_detail"     : dict per-channel averages for flagged segments
        "overall_regional"   : regional features averaged across ALL segments
        "posterior_coherence" : float
    """
    n_segments = segments.shape[0]

    if segment_duration is None:
        segment_duration = segments.shape[2] / sfreq

    # --- Tier 1: Screen every segment ---
    all_regional = []
    all_flags = []
    flagged_indices = []
    flag_summary = {}  # {(region, feature): [segment_indices]}

    for seg_idx in range(n_segments):
        seg_data = segments[seg_idx]  # (n_channels, n_samples)
        regional, flags = screen_segment(
            seg_data, ch_names, sfreq, reference, condition, source
        )
        all_regional.append(regional)
        all_flags.append(flags)

        if flags:
            flagged_indices.append(seg_idx)
            for f in flags:
                key = (f["region"], f["feature"])
                if key not in flag_summary:
                    flag_summary[key] = []
                flag_summary[key].append(seg_idx)

    # --- Tier 2: Overall averages across all segments ---
    regions = list(all_regional[0].keys())
    feat_names = [f for f in all_regional[0][regions[0]].keys()
                  if not f.startswith("_")]

    overall_regional = {}
    for region in regions:
        overall_regional[region] = {}
        channels = all_regional[0][region].get("_channels", [])
        overall_regional[region]["_channels"] = channels
        for feat in feat_names:
            vals = [all_regional[i][region][feat]
                    for i in range(n_segments)
                    if region in all_regional[i]
                    and feat in all_regional[i][region]]
            if vals:
                overall_regional[region][feat] = float(np.mean(vals))

    # --- Tier 3: Per-channel detail for flagged segments ---
    flagged_detail = {}
    if flagged_indices:
        for (region, feature), seg_indices in flag_summary.items():
            region_channels = REGIONS.get(region, [])
            # Find channel indices
            ch_indices = [i for i, ch in enumerate(ch_names)
                          if ch in region_channels]

            if not ch_indices:
                continue

            # Compute the flagged feature per channel, averaged across
            # flagged segments only
            per_channel = {}
            for ch_idx in ch_indices:
                ch_name = ch_names[ch_idx]
                vals = []
                for seg_idx in seg_indices:
                    seg_feats = compute_channel_features(
                        segments[seg_idx][ch_idx], sfreq
                    )
                    if feature in seg_feats:
                        vals.append(seg_feats[feature])
                if vals:
                    per_channel[ch_name] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                    }

            flagged_detail[(region, feature)] = {
                "segments": seg_indices,
                "per_channel": per_channel,
            }

    # --- Posterior coherence (average across all segments) ---
    coh_vals = []
    for seg_idx in range(n_segments):
        coh = compute_posterior_alpha_coherence(
            segments[seg_idx], ch_names, sfreq
        )
        coh_vals.append(coh)
    post_coh = float(np.nanmean(coh_vals)) if coh_vals else 0.0
    if np.isnan(post_coh):
        post_coh = 0.0

    return {
        "n_segments": n_segments,
        "segment_duration": segment_duration,
        "flagged_indices": flagged_indices,
        "flag_summary": flag_summary,
        "per_segment_flags": all_flags,
        "flagged_detail": flagged_detail,
        "overall_regional": overall_regional,
        "posterior_coherence": post_coh,
    }


def format_tiered_report(analysis, subject_id="unknown",
                         diagnosis_prob=None, condition="unknown"):
    """
    Format the tiered analysis into a clinical report for the LLM.

    Parameters
    ----------
    analysis : dict
        Output from analyze_segments().
    subject_id : str
    diagnosis_prob : float, optional
    condition : str

    Returns
    -------
    str — formatted report
    """
    cond_info = RECORDING_CONDITIONS.get(condition,
                                         RECORDING_CONDITIONS["unknown"])
    seg_dur = analysis["segment_duration"]
    n_seg = analysis["n_segments"]
    n_flagged = len(analysis["flagged_indices"])
    pct_flagged = (n_flagged / n_seg * 100) if n_seg > 0 else 0

    lines = []
    lines.append(f"=== EEG Biomarker Report: Subject {subject_id} ===")
    lines.append("")
    lines.append(f"Recording condition: {cond_info['label']}")
    lines.append(f"  {cond_info['description']}")
    lines.append(f"Segments analyzed: {n_seg} (each {seg_dur:.1f}s)")
    lines.append("")

    if diagnosis_prob is not None:
        lines.append(f"EEGPT AD Classification Probability: {diagnosis_prob:.2%}")
        lines.append("")

    # --- Overall regional averages ---
    lines.append("--- OVERALL AVERAGES (all segments) ---")
    key_features = [
        "theta_alpha_ratio", "theta_relative_power", "alpha_relative_power",
        "spectral_centroid", "spectral_peak", "iaf", "pdf",
        "spectral_entropy", "lzc"
    ]

    for region in ["frontal", "temporal", "parietal", "occipital", "central"]:
        if region not in analysis["overall_regional"]:
            continue
        feats = analysis["overall_regional"][region]
        channels = feats.get("_channels", [])
        ch_str = ", ".join(channels)
        lines.append(f"  {region.upper()} ({ch_str}):")
        for f in key_features:
            if f in feats:
                v = feats[f]
                if f in ["spectral_centroid", "spectral_peak", "iaf",
                          "median_frequency"]:
                    lines.append(f"    {f:28s} {v:>8.2f} Hz")
                else:
                    lines.append(f"    {f:28s} {v:>8.4f}")
        lines.append("")

    # --- Posterior coherence ---
    post_coh = analysis["posterior_coherence"]
    lines.append(f"Posterior alpha coherence: {post_coh:.4f}")
    if post_coh < 0.3:
        lines.append("  ⚠ REDUCED — disrupted posterior networks")
    lines.append("")

    # --- Temporal consistency ---
    lines.append("--- TEMPORAL CONSISTENCY ---")
    lines.append(f"Segments with AD-like patterns: "
                 f"{n_flagged}/{n_seg} ({pct_flagged:.0f}%)")

    if n_flagged == 0:
        lines.append("  No segments showed AD-associated abnormalities.")
        lines.append("")
    else:
        if pct_flagged > 75:
            lines.append("  ⚠ Persistent AD-like patterns throughout recording")
        elif pct_flagged > 40:
            lines.append("  ⚠ Frequent AD-like patterns")
        elif pct_flagged > 15:
            lines.append("  Intermittent AD-like patterns")
        else:
            lines.append("  Occasional AD-like patterns (may be transient)")
        lines.append("")

        # --- Flagged segment details ---
        lines.append("--- FLAGGED SEGMENTS (detailed) ---")

        for (region, feature), detail in analysis["flagged_detail"].items():
            seg_indices = detail["segments"]
            seg_str = ", ".join(str(s + 1) for s in seg_indices)
            per_ch = detail["per_channel"]
            ch_str = ", ".join(per_ch.keys())

            lines.append(f"  {feature} — {region.upper()}")
            lines.append(f"    Flagged in segments: {seg_str}")
            lines.append(f"    Channels: {ch_str}")
            lines.append(f"    Per-channel averages (across flagged segments):")

            for ch_name, stats in per_ch.items():
                if feature in ["spectral_centroid", "spectral_peak",
                               "iaf", "median_frequency"]:
                    lines.append(f"      {ch_name}: {stats['mean']:.2f} Hz "
                                 f"(range: {stats['min']:.2f}–{stats['max']:.2f})")
                else:
                    lines.append(f"      {ch_name}: {stats['mean']:.4f} "
                                 f"(range: {stats['min']:.4f}–{stats['max']:.4f})")
            lines.append("")

    # --- Clinical pattern summary ---
    lines.append("--- AD INDICATOR SUMMARY ---")
    if n_flagged == 0 and analysis["flag_summary"]:
        lines.append("  All features within reference ranges.")
    elif n_flagged == 0:
        lines.append("  No reference ranges provided for comparison.")
    else:
        flag_keys = analysis["flag_summary"].keys()
        has_posterior_alpha = any(
            r in ["occipital", "parietal"]
            and f in ["alpha_relative_power", "iaf"]
            for r, f in flag_keys
        )
        has_theta = any(
            f in ["theta_alpha_ratio", "theta_relative_power", "pdf"]
            for _, f in flag_keys
        )
        has_complexity = any(
            f in ["spectral_entropy", "lzc"]
            for _, f in flag_keys
        )
        has_slowing = any(
            f in ["spectral_centroid", "spectral_peak", "iaf"]
            for _, f in flag_keys
        )

        if has_posterior_alpha:
            lines.append("  * Posterior alpha reduction — consistent with "
                         "thalamocortical circuit disruption")
        if has_theta:
            regions = set(r for r, f in flag_keys
                          if f in ["theta_alpha_ratio",
                                   "theta_relative_power", "pdf"])
            lines.append(f"  * Theta/slow-wave elevation in "
                         f"{', '.join(regions)} — consistent with "
                         "cholinergic deficit / hippocampal damage")
        if has_complexity:
            lines.append("  * Reduced signal complexity — consistent with "
                         "neuronal network degradation")
        if has_slowing:
            lines.append("  * Spectral slowing — consistent with "
                         "cortical hypoactivation in AD")

        if not any([has_posterior_alpha, has_theta,
                    has_complexity, has_slowing]):
            for (r, f), indices in list(analysis["flag_summary"].items())[:5]:
                lines.append(f"  * {f} abnormal in {r} "
                             f"({len(indices)}/{n_seg} segments)")

    return "\n".join(lines)


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

def extract_biomarkers(eeg_input, sfreq=128, subject_id=None,
                       diagnosis_prob=None, save_to=None,
                       reference=None, condition="unknown"):
    """
    One-liner biomarker extraction for the pipeline notebook.

    Accepts single segments, multiple segments, MNE Raw objects,
    or file paths. Auto-detects recording condition and source
    from subject ID when possible.

    In complete_pipeline.ipynb:
        biomarkers = extract_biomarkers(eeg)
        biomarkers = extract_biomarkers(eeg, reference=ref, save_to="report.txt")

    Parameters
    ----------
    eeg_input : str, mne.io.BaseRaw, or numpy array
        - str: path to .set/.edf/.fif file
        - mne.io.BaseRaw: from LJ's preprocess_file()
        - 2D numpy array (n_channels, n_samples): single segment
        - 3D numpy array (n_segments, n_channels, n_samples): multi-segment
    sfreq : float
        Sampling rate. Only needed for numpy input. Default 128.
    subject_id : str, optional
        Patient ID for the report.
    diagnosis_prob : float, optional
        EEGPT AD probability (0-1).
    save_to : str, optional
        Save report to this file path.
    reference : dict, optional
        Reference ranges from compute_reference_ranges().
    condition : str
        Recording condition or "unknown" for auto-detection.

    Returns
    -------
    str — the formatted LLM report text
    """
    # --- Detect input type and build segments array ---
    ch_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]

    if isinstance(eeg_input, np.ndarray):
        if eeg_input.ndim == 3:
            # Multiple segments: (n_segments, n_channels, n_samples)
            segments = eeg_input
        elif eeg_input.ndim == 2:
            # Single segment: (n_channels, n_samples) → (1, n_ch, n_samp)
            segments = eeg_input[np.newaxis, :, :]
        else:
            raise ValueError(f"Array must be 2D or 3D, got {eeg_input.ndim}D")
        ch_names = ch_names[:segments.shape[1]]
        if subject_id is None:
            subject_id = "unknown"

    elif isinstance(eeg_input, str):
        import mne
        raw = mne.io.read_raw(eeg_input, preload=True)
        data = raw.get_data()  # (n_channels, n_samples)
        sfreq = raw.info["sfreq"]
        ch_names = list(raw.ch_names)
        # Segment into 2048-sample chunks
        window = min(2048, data.shape[1])
        n_segs = data.shape[1] // window
        if n_segs > 0:
            trimmed = data[:, :n_segs * window]
            segments = trimmed.reshape(data.shape[0], n_segs, window)
            segments = np.transpose(segments, (1, 0, 2))
        else:
            segments = data[np.newaxis, :, :]
        if subject_id is None:
            subject_id = os.path.splitext(os.path.basename(eeg_input))[0]

    else:
        # Assume MNE Raw object
        data = eeg_input.get_data()
        sfreq = eeg_input.info["sfreq"]
        ch_names = list(eeg_input.ch_names)
        window = min(2048, data.shape[1])
        n_segs = data.shape[1] // window
        if n_segs > 0:
            trimmed = data[:, :n_segs * window]
            segments = trimmed.reshape(data.shape[0], n_segs, window)
            segments = np.transpose(segments, (1, 0, 2))
        else:
            segments = data[np.newaxis, :, :]
        if subject_id is None:
            subject_id = "unknown"

    # --- Auto-detect condition and source from subject ID ---
    detected_source = None
    if subject_id:
        # Check Kaggle dataset prefixes
        for source_prefix, mapped_condition in SOURCE_TO_CONDITION.items():
            if subject_id.startswith(source_prefix):
                detected_source = source_prefix
                if condition == "unknown":
                    condition = mapped_condition
                break

        # Check for photic stimulation files (eyes-open dataset)
        if condition == "unknown" and "photomark" in subject_id:
            condition = "photic_stimulation"

    # --- Run tiered analysis ---
    analysis = analyze_segments(
        segments, ch_names, sfreq,
        reference=reference,
        condition=condition,
        source=detected_source,
        segment_duration=segments.shape[2] / sfreq,
    )

    # --- Format report ---
    llm_text = format_tiered_report(
        analysis,
        subject_id=subject_id,
        diagnosis_prob=diagnosis_prob,
        condition=condition,
    )

    # --- Save if requested ---
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        with open(save_to, "w") as f:
            f.write(llm_text)
        print(f"Report saved to: {save_to}")

    return llm_text


# =============================================================================
# SUPPORT FUNCTIONS (kept for probe training compatibility)
# =============================================================================

def get_biomarkers(raw_eeg):
    """Legacy wrapper — computes whole-head biomarkers from MNE Raw."""
    data = raw_eeg.get_data()
    sfreq = raw_eeg.info["sfreq"]
    ch_names = raw_eeg.ch_names
    n_channels = data.shape[0]

    results = {"per_channel": {}, "coherence": {}, "summary": {}}

    for i in range(n_channels):
        results["per_channel"][ch_names[i]] = compute_channel_features(
            data[i], sfreq)

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            results["coherence"][f"{ch_names[i]}-{ch_names[j]}"] = \
                compute_phase_coherence(data[i], data[j], sfreq)

    feat_names = list(results["per_channel"][ch_names[0]].keys())
    for feat in feat_names:
        vals = [results["per_channel"][ch][feat] for ch in ch_names]
        results["summary"][feat] = float(np.mean(vals))

    if results["coherence"]:
        results["summary"]["phase_coherence"] = float(
            np.mean(list(results["coherence"].values())))

    s = results["summary"]
    slow = s.get("delta_power", 0) + s.get("theta_power", 0)
    fast = s.get("alpha_power", 0) + s.get("beta_power", 0)
    s["slowing_index"] = slow / fast if fast > 0 else float("inf")

    return results


def features_to_array(segment_features):
    """Convert list of feature dicts to numpy array."""
    feature_names = list(segment_features[0].keys())
    rows = [[seg[f] for f in feature_names] for seg in segment_features]
    return feature_names, np.array(rows, dtype=np.float32)


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Biomarker Module v3 — Tiered Analysis Test")
    print("=" * 60)

    sfreq = 128
    n_channels = 19
    segment_samples = 2048

    np.random.seed(42)
    t = np.linspace(0, segment_samples / sfreq, segment_samples, endpoint=False)

    # Create 10 segments simulating a mixed recording:
    # Segments 0-4: healthy-ish (strong alpha)
    # Segments 5-9: AD-ish (strong theta, weak alpha)
    segments = np.zeros((10, n_channels, segment_samples))
    for seg in range(10):
        for ch in range(n_channels):
            if seg < 5:
                # Healthy: strong alpha
                segments[seg, ch] = (
                    3.0 * np.sin(2 * np.pi * 10 * t)
                    + 0.5 * np.sin(2 * np.pi * 6 * t)
                    + 0.5 * np.random.randn(segment_samples)
                )
            else:
                # AD-like: strong theta, weak alpha
                segments[seg, ch] = (
                    0.5 * np.sin(2 * np.pi * 10 * t)
                    + 4.0 * np.sin(2 * np.pi * 6 * t)
                    + 0.5 * np.random.randn(segment_samples)
                )

    # Test multi-segment analysis
    print("\n1. Testing multi-segment analysis (10 segments)...")
    report = extract_biomarkers(
        segments, sfreq=sfreq, subject_id="TEST-001"
    )
    print(report)

    # Test single segment
    print("\n\n2. Testing single segment...")
    report = extract_biomarkers(
        segments[0], sfreq=sfreq, subject_id="SINGLE-TEST"
    )
    print(f"   Report length: {len(report)} characters")

    # Test save to file
    print("\n3. Testing save_to...")
    extract_biomarkers(
        segments, sfreq=sfreq, subject_id="SAVE-TEST",
        save_to="test_report.txt"
    )
    with open("test_report.txt") as f:
        print(f"   Saved: {len(f.read())} characters")
    os.remove("test_report.txt")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)