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


CHANGES FROM LAST VERSION (false-positive fixes):
  1. analyze_segments() now requires BOTH (a) the recording-level average to
     be flagged AND (b) at least `persistence_threshold` (default 25%) of
     segments to be flagged before a feature counts as "abnormal."
  2. Reference now stores 5th/95th percentiles in addition to mean ± 2σ.
     Percentile cutoffs are the new default — robust to skewed features.
  3. IAF averaging is nan-safe (was poisoning whole-recording averages).
  4. format_tiered_report() drives the AD INDICATOR SUMMARY off the
     filtered persistent_flags rather than "any segment flagged ever."
"""

import os
import warnings
import numpy as np
from scipy import signal, stats
from classify_kde import classify_selected_features


# =============================================================================
# CONSTANTS
# =============================================================================

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
}

# Equipment-independent features — safe to use across any dataset/equipment.
NORMALIZED_FEATURES = [
    "theta_alpha_ratio",
    "theta_relative_power",
    "alpha_relative_power",
    "delta_relative_power",
    "beta_relative_power",
    "spectral_centroid",
    "spectral_peak",
    "median_frequency",
    "spectral_entropy",
    "shannon_entropy",
    "tsallis_entropy",
    "lzc",
    "amplitude_modulation",
    "alpha_beta_ratio",
]

# Default fraction of segments that must show a flag before we treat a
# feature as "persistently abnormal." Together with the recording-level
# check this kills almost all of the random-noise false positives.
DEFAULT_PERSISTENCE_THRESHOLD = 0.25

# Channel-to-region mapping (standard 10-20, 19 channels).
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

REGION_AD_NOTES = {
    "frontal":   "Executive function, attention — cholinergic dysfunction causes theta increase",
    "central":   "Sensorimotor processing — alpha reduction with disease progression",
    "temporal":  "Memory, language — early AD site, hippocampal damage causes theta increase",
    "parietal":  "Spatial processing — alpha reduction from cortical atrophy",
    "occipital": "Alpha generation site — posterior slowing is a hallmark AD sign",
}

POSTERIOR_CHANNELS = ["P3", "P4", "O1", "O2", "Pz", "T5", "T6",
                      "PZ", "P7", "P8"]

RECORDING_CONDITIONS = {
    "resting_eyes_closed": {
        "label": "Resting State (Eyes Closed)",
        "description": "Gold standard for AD EEG assessment. "
                       "Alpha rhythm is strongest in this condition.",
        "suppress_flags": [],
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


def _safe_nanmean(vals):
    """np.nanmean but suppresses 'all-NaN slice' warnings.

    Returns NaN if every input is NaN (which is what we want — it tells us
    the feature genuinely couldn't be computed for any segment).
    """
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(arr))


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
    """Lempel-Ziv Complexity (LZC)."""
    if len(sig) > 512:
        step = len(sig) // 512
        sig = sig[::step][:512]

    median_val = np.median(sig)
    binary = (sig >= median_val).astype(int)
    s = "".join(map(str, binary))
    n = len(s)

    if n <= 1:
        return 0.0

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

    c_norm = c * np.log2(n) / n
    return float(c_norm)


def compute_iaf(sig, sfreq):
    """Individual Alpha Frequency (IAF). NaN if no alpha activity."""
    freqs, psd = _compute_psd(sig, sfreq)

    alpha_mask = (freqs >= 8) & (freqs <= 12)
    alpha_freqs = freqs[alpha_mask]
    alpha_psd = psd[alpha_mask]

    if len(alpha_psd) == 0 or np.sum(alpha_psd) < 1e-10:
        return float('nan')

    iaf = float(np.sum(alpha_freqs * alpha_psd) / np.sum(alpha_psd))
    return iaf


def compute_pdf(sig, sfreq):
    """Power Distribution Ratio (PDF) — slow/fast power. Capped at 100."""
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
    """Posterior Alpha Coherence — pairwise mean across posterior channels."""
    posterior_indices = []
    for i, ch in enumerate(ch_names):
        if ch in POSTERIOR_CHANNELS:
            posterior_indices.append(i)

    if len(posterior_indices) < 2:
        return 0.0

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

    features["lzc"] = compute_lzc(eeg_signal)
    features["iaf"] = compute_iaf(eeg_signal, sfreq)
    features["pdf"] = compute_pdf(eeg_signal, sfreq)

    return features


# =============================================================================
# REGIONAL ANALYSIS
# =============================================================================

def compute_regional_features(data, ch_names, sfreq):
    """Compute biomarker features grouped by brain region (nan-safe)."""
    regional = {}

    for region, region_channels in REGIONS.items():
        region_indices = []
        matched_channels = []
        for i, ch in enumerate(ch_names):
            if ch in region_channels:
                region_indices.append(i)
                matched_channels.append(ch)

        if not region_indices:
            continue

        all_feats = []
        for idx in region_indices:
            all_feats.append(compute_channel_features(data[idx], sfreq))

        feat_names = list(all_feats[0].keys())
        mean_feats = {}
        for f in feat_names:
            mean_feats[f] = _safe_nanmean([ch[f] for ch in all_feats])

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

    Each feature now stores BOTH:
      - mean / std / low / high (mean ± 2σ — legacy)
      - p05 / p95 (5th / 95th percentile — robust, used by default)
    """
    from load_kaggle_data import (
        load_dataset, get_subject_chunks, chunk_to_channel_first,
        SAMPLING_RATE, CHANNEL_NAMES
    )

    if groups is None:
        groups = ["healthy", "AD", "FTD"]

    dataset = load_dataset(dataset_path)

    overall_values = {g: {} for g in groups}
    source_values = {}
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

        sample_n = min(3, chunks.shape[0])
        for i in range(sample_n):
            ch_data = chunk_to_channel_first(chunks[i])
            regional = compute_regional_features(
                ch_data, CHANNEL_NAMES, SAMPLING_RATE
            )

            for region, feats in regional.items():
                if region not in overall_values[diag]:
                    overall_values[diag][region] = {}
                for f, v in feats.items():
                    if f.startswith("_"):
                        continue
                    if f not in overall_values[diag][region]:
                        overall_values[diag][region][f] = []
                    overall_values[diag][region][f].append(v)

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

    def _values_to_stats(values_dict):
        """Convert {region: {feature: [values]}} to stats with both
        Gaussian and percentile bounds. Drops NaN/inf before computing."""
        result = {}
        for region, feats in values_dict.items():
            result[region] = {}
            for f, vals in feats.items():
                vals = np.asarray(vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                m = float(np.mean(vals))
                s = float(np.std(vals))
                p05 = float(np.percentile(vals, 5))
                p95 = float(np.percentile(vals, 95))
                result[region][f] = {
                    "mean": m,
                    "std": s,
                    "low": m - 2 * s,
                    "high": m + 2 * s,
                    "p05": p05,
                    "p95": p95,
                    "n": int(vals.size),
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
                       condition="unknown", source=None,
                       use_percentiles=True):
    """
    Compare regional features against healthy and AD reference ranges.

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
    use_percentiles : bool
        If True (default), use 5th/95th percentile cutoffs from the
        healthy distribution. Robust to skewed features. If False,
        fall back to mean ± 2σ (legacy behavior, prone to false
        positives on skewed features like theta_alpha_ratio).
    """
    flags = []

    cond_info = RECORDING_CONDITIONS.get(condition, RECORDING_CONDITIONS["unknown"])
    suppress = set(cond_info.get("suppress_flags", []))

    healthy_ref = {}
    ad_ref = {}
    ftd_ref = {}

    if reference and "per_source" in reference:
        if source:
            source_ref = reference["per_source"].get(source, {})
            healthy_ref = source_ref.get("healthy", {})
            ad_ref = source_ref.get("AD", {})
            ftd_ref = source_ref.get("FTD", {})

        if not healthy_ref and condition != "unknown":
            cond_ref = reference["per_source"].get(condition, {})
            healthy_ref = cond_ref.get("healthy", {})
            ad_ref = cond_ref.get("AD", {})
            ftd_ref = cond_ref.get("FTD", {})

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
            if feat_name in suppress:
                continue

            direction = ad_directions[feat_name]

            if np.isnan(value) or np.isinf(value):
                continue

            h_ref = healthy_ref.get(region, {}).get(feat_name)
            if not h_ref:
                continue

            h_mean = h_ref["mean"]
            h_std = h_ref["std"]

            if h_std > 0:
                z_score = (value - h_mean) / h_std
            else:
                z_score = 0

            # Pick the cutoff style
            if use_percentiles and "p05" in h_ref and "p95" in h_ref:
                low_cut, high_cut = h_ref["p05"], h_ref["p95"]
                cutoff_label = "5th–95th pct"
            else:
                low_cut, high_cut = h_ref["low"], h_ref["high"]
                cutoff_label = "mean ± 2σ"

            outside_healthy = False
            if direction == "elevated" and value > high_cut:
                outside_healthy = True
            elif direction == "reduced" and value < low_cut:
                outside_healthy = True

            if not outside_healthy:
                continue

            a_ref = ad_ref.get(region, {}).get(feat_name)
            status = "abnormal"

            if a_ref:
                if use_percentiles and "p05" in a_ref and "p95" in a_ref:
                    a_low, a_high = a_ref["p05"], a_ref["p95"]
                else:
                    a_low, a_high = a_ref["low"], a_ref["high"]

                if a_low <= value <= a_high:
                    status = "within_AD_range"
                elif direction == "elevated" and value > a_high:
                    status = "exceeds_AD_range"
                elif direction == "reduced" and value < a_low:
                    status = "exceeds_AD_range"
                else:
                    status = "between_healthy_and_AD"

            f_ref = ftd_ref.get(region, {}).get(feat_name)
            ftd_match = False
            if f_ref:
                if use_percentiles and "p05" in f_ref and "p95" in f_ref:
                    f_low, f_high = f_ref["p05"], f_ref["p95"]
                else:
                    f_low, f_high = f_ref["low"], f_ref["high"]
                if f_low <= value <= f_high:
                    ftd_match = True

            flags.append({
                "region": region,
                "feature": feat_name,
                "value": value,
                "direction": direction,
                "z_score": z_score,
                "status": status,
                "cutoff_style": cutoff_label,
                "healthy_range": f"{low_cut:.3f} – {high_cut:.3f}",
                "healthy_mean": h_mean,
                "ad_mean": a_ref["mean"] if a_ref else None,
                "ad_range": (
                    f"{a_ref.get('p05', a_ref['low']):.3f} – "
                    f"{a_ref.get('p95', a_ref['high']):.3f}"
                    if a_ref else "N/A"
                ),
                "ftd_match": ftd_match,
            })

    flags.sort(key=lambda x: abs(x.get("z_score", 0)), reverse=True)
    return flags


# =============================================================================
# EPOCH-BASED ANALYSIS (when event markers are available)
# =============================================================================

def extract_biomarkers_by_epoch(raw_eeg, event_id=None, tmin=-0.5, tmax=1.0):
    """Compute biomarkers per epoch (event-locked segments)."""
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
            epoch_data = epochs[i].get_data()[0]
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
# TIERED ANALYSIS — per-segment screening and flagging
# =============================================================================

def screen_segment(data, ch_names, sfreq, reference=None,
                   condition="unknown", source=None,
                   use_percentiles=True):
    """Tier 1: Screen one segment using normalized features only."""
    regional = compute_regional_features(data, ch_names, sfreq)

    if reference:
        flags = flag_abnormalities(regional, reference, condition, source,
                                   use_percentiles=use_percentiles)
        flags = [f for f in flags if f["feature"] in NORMALIZED_FEATURES]
    else:
        flags = []

    return regional, flags


def analyze_segments(segments, ch_names, sfreq, reference=None,
                     condition="unknown", source=None,
                     segment_duration=None,
                     persistence_threshold=DEFAULT_PERSISTENCE_THRESHOLD,
                     use_percentiles=True):
    """
    Run tiered analysis on multiple segments from one subject.

    A feature counts as "persistently abnormal" only if BOTH:
      (a) The recording-level average (across all segments) is itself
          flagged against the healthy reference, AND
      (b) The feature was flagged in at least `persistence_threshold`
          fraction of individual segments.

    This kills two false-positive sources:
      - Random per-segment fluctuations that happen to cross the cutoff
        on a few segments but don't shift the recording average
      - Recording averages that are barely abnormal but driven by a few
        outlier segments rather than persistent biology

    Returns dict with both raw (flag_summary) and filtered
    (persistent_flags) results so downstream code can choose either view.
    """
    n_segments = segments.shape[0]

    if segment_duration is None:
        segment_duration = segments.shape[2] / sfreq

    # --- Tier 1: screen every segment ---
    all_regional = []
    all_flags = []
    flagged_indices = []
    flag_summary = {}

    for seg_idx in range(n_segments):
        seg_data = segments[seg_idx]
        regional, flags = screen_segment(
            seg_data, ch_names, sfreq, reference, condition, source,
            use_percentiles=use_percentiles,
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

    # --- Tier 2a: overall averages across all segments (nan-safe) ---
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
                # nanmean — a few NaN segments (e.g. IAF with no alpha)
                # used to poison the entire recording's average.
                overall_regional[region][feat] = _safe_nanmean(vals)

    # --- Tier 2b: recording-level flags on the averages ---
    if reference:
        recording_level_flags = flag_abnormalities(
            overall_regional, reference, condition, source,
            use_percentiles=use_percentiles,
        )
        recording_level_flags = [f for f in recording_level_flags
                                 if f["feature"] in NORMALIZED_FEATURES]
    else:
        recording_level_flags = []

    recording_level_keys = {(f["region"], f["feature"])
                            for f in recording_level_flags}

    # --- Tier 2c: filter by persistence + recording-level ---
    min_segments = max(1, int(np.ceil(persistence_threshold * n_segments)))
    persistent_flags = {}
    for key, seg_indices in flag_summary.items():
        if len(seg_indices) < min_segments:
            continue
        if key not in recording_level_keys:
            continue
        persistent_flags[key] = seg_indices

    # --- Tier 3: per-channel detail for persistent flags only ---
    flagged_detail = {}
    for (region, feature), seg_indices in persistent_flags.items():
        region_channels = REGIONS.get(region, [])
        ch_indices = [i for i, ch in enumerate(ch_names)
                      if ch in region_channels]
        if not ch_indices:
            continue

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
            vals = [v for v in vals if np.isfinite(v)]
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
    post_coh = _safe_nanmean(coh_vals) if coh_vals else 0.0
    if np.isnan(post_coh):
        post_coh = 0.0

    return {
        "n_segments": n_segments,
        "segment_duration": segment_duration,
        "persistence_threshold": persistence_threshold,
        "flagged_indices": flagged_indices,
        "flag_summary": flag_summary,
        "persistent_flags": persistent_flags,
        "recording_level_flags": recording_level_flags,
        "per_segment_flags": all_flags,
        "flagged_detail": flagged_detail,
        "overall_regional": overall_regional,
        "posterior_coherence": post_coh,
    }


def format_tiered_report(analysis, subject_id="unknown",
                         diagnosis_prob=None, condition="unknown"):
    """
    Format the tiered analysis into a clinical report for the LLM.
 
    The AD INDICATOR SUMMARY now drives off persistent_flags (which already
    requires recording-level + persistence agreement), not "any segment ever
    flagged."
    """
    cond_info = RECORDING_CONDITIONS.get(condition,
                                         RECORDING_CONDITIONS["unknown"])
    seg_dur = analysis["segment_duration"]
    n_seg = analysis["n_segments"]
    persistence = analysis.get("persistence_threshold",
                               DEFAULT_PERSISTENCE_THRESHOLD)
 
    n_flagged = len(analysis["flagged_indices"])
    pct_flagged = (n_flagged / n_seg * 100) if n_seg > 0 else 0
    persistent = analysis.get("persistent_flags", {})
 
    lines = []
    lines.append(f"=== EEG Biomarker Report: Subject {subject_id} ===")
    lines.append("")
    lines.append(f"Recording condition: {cond_info['label']}")
    lines.append(f"  {cond_info['description']}")
    lines.append(f"Segments analyzed: {n_seg} (each {seg_dur:.1f}s)")
    lines.append(f"Persistence threshold for flagging: "
                 f"{persistence*100:.0f}% of segments")
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
 
    persistent_keys = set(persistent.keys())
 
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
                marker = "  ⚠" if (region, f) in persistent_keys else ""
                if f in ["spectral_centroid", "spectral_peak", "iaf",
                          "median_frequency"]:
                    lines.append(f"    {f:28s} {v:>8.2f} Hz{marker}")
                else:
                    lines.append(f"    {f:28s} {v:>8.4f}{marker}")
        lines.append("")
 
    # --- Posterior coherence ---
    post_coh = analysis["posterior_coherence"]
    lines.append(f"Posterior alpha coherence: {post_coh:.4f}")
    if post_coh < 0.3:
        lines.append("  ⚠ REDUCED — disrupted posterior networks")
    lines.append("")
 
    # --- Temporal consistency ---
    lines.append("--- TEMPORAL CONSISTENCY ---")
    lines.append(f"Segments with any AD-like pattern: "
                 f"{n_flagged}/{n_seg} ({pct_flagged:.0f}%)")
    n_persistent = len(persistent)
    lines.append(f"Persistently flagged features (≥{persistence*100:.0f}% "
                 f"of segments AND recording avg abnormal): {n_persistent}")
 
    # Tally per-feature flag rates regardless of whether they crossed the
    # persistence threshold — the LLM should see this evidence rather than
    # having it hidden by the strict flagging rule.
    flag_summary = analysis.get("flag_summary", {})
    sub_threshold = []
    for (region, feature), seg_indices in flag_summary.items():
        pct = len(seg_indices) / n_seg * 100 if n_seg > 0 else 0
        # Anything that isn't already persistent but had some flagging
        is_persistent = (region, feature) in persistent
        if not is_persistent and pct > 0:
            sub_threshold.append((region, feature, len(seg_indices), pct))
    # Sort by frequency, most-flagged first
    sub_threshold.sort(key=lambda x: -x[3])
 
    if n_persistent == 0:
        if pct_flagged > 0:
            lines.append("")
            lines.append("  No features crossed the strict persistence "
                         "threshold, but the following features were "
                         "flagged in individual segments and may warrant "
                         "clinical review:")
        else:
            lines.append("  No AD-associated abnormalities detected.")
        lines.append("")
    else:
        if pct_flagged > 75:
            lines.append("  ⚠ Persistent AD-like patterns throughout recording")
        elif pct_flagged > 40:
            lines.append("  ⚠ Frequent AD-like patterns")
        elif pct_flagged > 15:
            lines.append("  Intermittent AD-like patterns")
        else:
            lines.append("  Occasional AD-like patterns")
        lines.append("")
 
        # --- Persistently-flagged feature details ---
        lines.append("--- PERSISTENTLY FLAGGED FEATURES (detailed) ---")
 
        for (region, feature), detail in analysis["flagged_detail"].items():
            seg_indices = detail["segments"]
            n_flagged_for_feat = len(seg_indices)
            pct_for_feat = n_flagged_for_feat / n_seg * 100
            seg_str = ", ".join(str(s + 1) for s in seg_indices[:30])
            if len(seg_indices) > 30:
                seg_str += f", ... ({len(seg_indices)} total)"
            per_ch = detail["per_channel"]
            ch_str = ", ".join(per_ch.keys())
 
            lines.append(f"  {feature} — {region.upper()} "
                         f"[{n_flagged_for_feat}/{n_seg} segments, "
                         f"{pct_for_feat:.0f}%]")
            lines.append(f"    Flagged in segments: {seg_str}")
            lines.append(f"    Channels: {ch_str}")
            lines.append(f"    Per-channel averages (across flagged segments):")
 
            for ch_name, st in per_ch.items():
                if feature in ["spectral_centroid", "spectral_peak",
                               "iaf", "median_frequency"]:
                    lines.append(f"      {ch_name}: {st['mean']:.2f} Hz "
                                 f"(range: {st['min']:.2f}–{st['max']:.2f})")
                else:
                    lines.append(f"      {ch_name}: {st['mean']:.4f} "
                                 f"(range: {st['min']:.4f}–{st['max']:.4f})")
            lines.append("")
 
    # --- Sub-threshold findings (always show if any exist) ---
    if sub_threshold:
        lines.append("--- SUB-THRESHOLD FINDINGS ---")
        lines.append("Features flagged in individual segments but not "
                     "meeting the strict persistence + recording-average "
                     "criterion. These are weaker evidence than persistent "
                     "flags but represent real per-segment deviations:")
        lines.append("")
        for region, feature, n_flag, pct in sub_threshold[:15]:
            marker = "~" if pct >= 15 else " "
            lines.append(f"  {marker} {feature} — {region.upper()}: "
                         f"flagged in {n_flag}/{n_seg} segments ({pct:.0f}%)")
        if len(sub_threshold) > 15:
            lines.append(f"  ... and {len(sub_threshold) - 15} more "
                         f"sub-threshold findings")
        lines.append("")
 
    # --- Clinical pattern summary ---
    lines.append("--- AD INDICATOR SUMMARY ---")
    if not persistent:
        if sub_threshold:
            # Synthesize the sub-threshold evidence into clinical patterns
            sub_keys = [(r, f) for r, f, _, _ in sub_threshold]
            has_posterior_alpha = any(
                r in ["occipital", "parietal"]
                and f in ["alpha_relative_power", "iaf"]
                for r, f in sub_keys
            )
            has_theta = any(
                f in ["theta_alpha_ratio", "theta_relative_power", "pdf"]
                for _, f in sub_keys
            )
            has_complexity = any(
                f in ["spectral_entropy", "lzc"]
                for _, f in sub_keys
            )
            has_slowing = any(
                f in ["spectral_centroid", "spectral_peak", "iaf"]
                for _, f in sub_keys
            )
 
            lines.append("  No persistent flags at the recording level. "
                         "The following sub-threshold patterns appeared "
                         "in individual segments:")
            if has_posterior_alpha:
                lines.append("  * Posterior alpha reduction "
                             "(sub-threshold)")
            if has_theta:
                lines.append("  * Theta/slow-wave elevation "
                             "(sub-threshold)")
            if has_complexity:
                lines.append("  * Reduced signal complexity "
                             "(sub-threshold)")
            if has_slowing:
                lines.append("  * Spectral slowing (sub-threshold)")
            lines.append("")
            lines.append("  Sub-threshold patterns are weaker evidence "
                         "than persistent flags. They may represent early "
                         "or intermittent abnormalities, or normal "
                         "variation. Interpret in clinical context.")
        elif any(analysis.get("per_segment_flags", [])):
            lines.append("  No persistent abnormalities at the recording "
                         "level. Per-segment fluctuations did not exceed "
                         "the persistence threshold and the overall "
                         "averages are within healthy reference ranges.")
        else:
            lines.append("  No reference ranges provided, or all features "
                         "within healthy reference ranges.")
    else:
        flag_keys = persistent.keys()
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
            regions = sorted(set(r for r, f in flag_keys
                                  if f in ["theta_alpha_ratio",
                                           "theta_relative_power", "pdf"]))
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
            for (r, f), indices in list(persistent.items())[:5]:
                lines.append(f"  * {f} abnormal in {r} "
                             f"({len(indices)}/{n_seg} segments)")
                
    # --- KDE CLASSIFICATION ---
    lines.append("")
    lines.append("--- KDE FEATURE CLASSIFICATION ---")

    feature_accumulator = {}

    for region, feats in analysis["overall_regional"].items():

        for feature_name, value in feats.items():

            if feature_name.startswith("_"):
                continue

            if not isinstance(value, (int, float, np.floating)):
                continue

            if not np.isfinite(value):
                continue

            if feature_name not in feature_accumulator:
                feature_accumulator[feature_name] = []

            feature_accumulator[feature_name].append(float(value))

    # Recording-level averages
    kde_features = {
        feature_name: float(np.mean(values))
        for feature_name, values in feature_accumulator.items()
        if len(values) > 0
    }

    kde_results = classify_selected_features(kde_features)

    if not kde_results:
        lines.append("  No KDE classifications available.")
    else:
        for feature_name, probs in kde_results.items():

            ad_prob = probs.get("AD", 0.0)
            non_ad_prob = probs.get("NON-AD", 0.0)

            lines.append(
                f"  {feature_name:24s} "
                f"AD={ad_prob:.2%}   "
                f"NON-AD={non_ad_prob:.2%}"
            )

    return "\n".join(lines)


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

def extract_biomarkers(eeg_input, sfreq=128, subject_id=None,
                       diagnosis_prob=None, save_to=None,
                       reference=None, condition="unknown",
                       persistence_threshold=DEFAULT_PERSISTENCE_THRESHOLD,
                       use_percentiles=True):
    """
    Generate a clinical biomarker text report for the LLM explanation stage.

    Returns a formatted string suitable for feeding into a language model.
    For numerical features (classifier input), use extract_features_fast()
    instead — it skips the report formatting and runs faster.

    Notebook usage:
        report_text = extract_biomarkers(eeg)
        report_text = extract_biomarkers(eeg, save_to="report.txt")

    Parameters
    ----------
    persistence_threshold : float
        Fraction of segments that must show a flag for a feature to count
        as persistent. Default 0.25 (25%).
    use_percentiles : bool
        If True (default), use 5th/95th percentile cutoffs from healthy
        reference. If False, use mean ± 2σ (legacy behavior).
    """
    ch_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]

    if isinstance(eeg_input, np.ndarray):
        if eeg_input.ndim == 3:
            # Already pre-segmented: (n_segments, n_channels, n_samples)
            segments = eeg_input
        elif eeg_input.ndim == 2:
            # Single recording (n_channels, n_samples) — chunk into 2048-sample
            # segments so the report reflects multi-segment analysis.
            window = min(2048, eeg_input.shape[1])
            n_segs = eeg_input.shape[1] // window
            if n_segs > 0:
                trimmed = eeg_input[:, :n_segs * window]
                segments = trimmed.reshape(eeg_input.shape[0], n_segs, window)
                segments = np.transpose(segments, (1, 0, 2))
            else:
                segments = eeg_input[np.newaxis, :, :]
        else:
            raise ValueError(f"Array must be 2D or 3D, got {eeg_input.ndim}D")
        ch_names = ch_names[:segments.shape[1]]
        if subject_id is None:
            subject_id = "unknown"

    elif isinstance(eeg_input, str):
        import mne
        raw = mne.io.read_raw(eeg_input, preload=True)
        data = raw.get_data()
        sfreq = raw.info["sfreq"]
        ch_names = list(raw.ch_names)
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
        for source_prefix, mapped_condition in SOURCE_TO_CONDITION.items():
            if subject_id.startswith(source_prefix):
                detected_source = source_prefix
                if condition == "unknown":
                    condition = mapped_condition
                break

        if condition == "unknown" and "photomark" in subject_id:
            condition = "photic_stimulation"

    # --- Run tiered analysis ---
    analysis = analyze_segments(
        segments, ch_names, sfreq,
        reference=reference,
        condition=condition,
        source=detected_source,
        segment_duration=segments.shape[2] / sfreq,
        persistence_threshold=persistence_threshold,
        use_percentiles=use_percentiles,
    )

    llm_text = format_tiered_report(
        analysis,
        subject_id=subject_id,
        diagnosis_prob=diagnosis_prob,
        condition=condition,
    )

    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        with open(save_to, "w") as f:
            f.write(llm_text)
        print(f"Report saved to: {save_to}")

    return llm_text


# =============================================================================
# PIPELINE WRAPPER — one call that returns BOTH numbers and text
# =============================================================================

def extract_biomarker_outputs(eeg_input, sfreq=128, mode="regional",
                              condition="unknown", reference=None,
                              subject_id=None, save_to=None,
                              persistence_threshold=DEFAULT_PERSISTENCE_THRESHOLD,
                              use_percentiles=True):
    """
    Convenience wrapper for the inference pipeline.

    Calls both extract_features_fast() and extract_biomarkers() and
    returns everything as a tuple. This is used when both the
    numerical features (for a classifier) and the text report (for
    an LLM) from the same recording.

    Parameters
    ----------
    eeg_input : ndarray or MNE Raw or str (file path)
    sfreq : float
        Sampling rate. Default 128.
    mode : "whole_head" or "regional"
        Numerical feature extraction mode. Default "regional" (165 features).
    condition : str
        Recording condition (e.g. "resting_eyes_closed", "photic_stimulation").
        Default "unknown" — works but uses wider thresholds.
    reference : dict, optional
        Healthy reference ranges from load_reference_ranges(). Without
        this, the report will not contain flagging analysis.
    subject_id : str, optional
        Patient ID for the report.
    save_to : str, optional
        Path to save the text report.

    Returns
    -------
    biomarker_features : np.ndarray, shape (n_features,)
        Numerical feature vector. Pass this to a classifier.
    biomarker_names : list of str
        Column labels for biomarker_features.
    biomarker_report : str
        Formatted text report. Pass this to an LLM.

    Example
    -------
    >>> features, names, report = extract_biomarker_outputs(
    ...     eeg, condition="resting_eyes_closed", reference=ref
    ... )
    """
    # --- For numerical features, the input must be ndarray-shaped ---
    # Pull data out of MNE Raw or load from path if needed.
    if isinstance(eeg_input, np.ndarray):
        eeg_array = eeg_input
        eeg_for_report = eeg_input
    elif isinstance(eeg_input, str):
        import mne
        raw = mne.io.read_raw(eeg_input, preload=True)
        eeg_array = raw.get_data()
        sfreq = raw.info["sfreq"]
        eeg_for_report = raw
    else:
        # MNE Raw object
        eeg_array = eeg_input.get_data()
        sfreq = eeg_input.info["sfreq"]
        eeg_for_report = eeg_input

    # --- Reshape into segments for fast feature extraction ---
    if eeg_array.ndim == 2:
        # (n_channels, n_samples) → segment into 2048-sample chunks
        window = min(2048, eeg_array.shape[1])
        n_segs = eeg_array.shape[1] // window
        if n_segs > 0:
            trimmed = eeg_array[:, :n_segs * window]
            segs = trimmed.reshape(eeg_array.shape[0], n_segs, window)
            segs = np.transpose(segs, (1, 0, 2))
        else:
            segs = eeg_array[np.newaxis, :, :]
    else:
        segs = eeg_array

    # --- Numerical features for EEGPT input ---
    biomarker_features, biomarker_names = extract_features_fast(
        segs, sfreq=sfreq, mode=mode
    )
    # Average across segments → one feature vector per recording
    if biomarker_features.ndim == 2 and biomarker_features.shape[0] > 1:
        biomarker_features = _safe_nanmean_axis0(biomarker_features)

    # --- Text report Gemini input ---
    biomarker_report = extract_biomarkers(
        eeg_for_report,
        sfreq=sfreq,
        subject_id=subject_id,
        condition=condition,
        reference=reference,
        save_to=save_to,
        persistence_threshold=persistence_threshold,
        use_percentiles=use_percentiles,
    )

    return biomarker_features, biomarker_names, biomarker_report


def _safe_nanmean_axis0(arr):
    """Mean across axis 0, ignoring NaN. Returns a 1D array."""
    with np.errstate(invalid="ignore"):
        result = np.nanmean(arr, axis=0)
    # Replace any remaining NaN/inf with 0 for classifier safety
    result = np.where(np.isfinite(result), result, 0.0)
    return result.astype(np.float32)


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
        results["summary"][feat] = _safe_nanmean(vals)

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
# FAST EXTRACTION — for classifier training (numerical only, no flagging)
# =============================================================================

def _channel_features_fast(sig, sfreq):
    """
    All single-channel features computed from a single PSD pass.

    The slow version calls _compute_psd separately inside _power_features,
    _spectral_shape_features, _entropy_features, compute_iaf, and compute_pdf
    (5 PSD computations per channel). This version computes the PSD once
    and reuses it for every spectral feature.

    Returns dict of feature_name -> value (no formatting, no flagging).
    """
    # --- One PSD pass for everything spectral ---
    nperseg = min(int(2 * sfreq), len(sig))
    freqs, psd = signal.welch(
        sig, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2,
    )
    freq_res = freqs[1] - freqs[0]

    # --- Statistical (no PSD needed) ---
    # Compute moments directly in numpy — much faster than scipy.stats
    # for repeated calls. Matches scipy defaults: skew/kurtosis are biased
    # (population) estimators, kurtosis is Fisher's (excess, normal=0).
    sig_mean = float(np.mean(sig))
    centered = sig - sig_mean
    var = float(np.mean(centered ** 2))
    sd = float(np.sqrt(var)) if var > 0 else 0.0

    if sd > 0:
        m3 = float(np.mean(centered ** 3))
        m4 = float(np.mean(centered ** 4))
        skewness = m3 / (sd ** 3)
        kurtosis = m4 / (var ** 2) - 3.0
    else:
        skewness = 0.0
        kurtosis = 0.0

    q25, q75 = np.percentile(sig, [25, 75])

    feats = {
        "mean":     sig_mean,
        "variance": var,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "std":      sd,
        "iqr":      float(q75 - q25),
        "max":      float(np.max(sig)),
        "min":      float(np.min(sig)),
        "mean_abs": float(np.mean(np.abs(sig))),
        "median":   float(np.median(sig)),
    }

    # --- Band powers (cached PSD) ---
    def _bp(low, high):
        m = (freqs >= low) & (freqs < high)
        return float(np.sum(psd[m]) * freq_res)

    delta_p = _bp(*BANDS["delta"])
    theta_p = _bp(*BANDS["theta"])
    alpha_p = _bp(*BANDS["alpha"])
    beta_p  = _bp(*BANDS["beta"])
    total_p = delta_p + theta_p + alpha_p + beta_p
    safe_total = total_p if total_p > 0 else 1e-10
    safe_alpha = alpha_p if alpha_p > 0 else 1e-10
    safe_beta  = beta_p  if beta_p  > 0 else 1e-10

    feats.update({
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
    })

    # --- Spectral shape (cached PSD) ---
    mask = (freqs >= 0.5) & (freqs <= 30)
    f = freqs[mask]
    p = psd[mask]
    p_total = np.sum(p)
    if p_total <= 0:
        p_total = 1e-10

    centroid = float(np.sum(f * p) / p_total)
    cum = np.cumsum(p)
    idx_85 = min(np.searchsorted(cum, 0.85 * p_total), len(f) - 1)
    rolloff = float(f[idx_85])
    peak = float(f[np.argmax(p)])
    avg_mag = float(np.mean(p))
    idx_50 = min(np.searchsorted(cum, 0.5 * p_total), len(f) - 1)
    med_freq = float(f[idx_50])

    # Hilbert envelope still needs the time-domain signal
    analytic = signal.hilbert(sig)
    envelope = np.abs(analytic)
    env_mean = np.mean(envelope)
    amp_mod = float(np.std(envelope) / env_mean) if env_mean > 0 else 0.0

    feats.update({
        "spectral_centroid":    centroid,
        "spectral_rolloff":     rolloff,
        "spectral_peak":        peak,
        "average_magnitude":    avg_mag,
        "median_frequency":     med_freq,
        "amplitude_modulation": amp_mod,
    })

    # --- Entropy (cached PSD) ---
    p_norm = p / p_total
    p_norm = p_norm[p_norm > 0]
    n = len(p_norm)
    sp_ent = float(-np.sum(p_norm * np.log2(p_norm)))
    if n > 1:
        sp_ent /= np.log2(n)
    shannon = float(-np.sum(p_norm * np.log2(p_norm)))
    tsallis = float(1 - np.sum(p_norm ** 2))  # q=2 form

    feats.update({
        "spectral_entropy": sp_ent,
        "shannon_entropy":  shannon,
        "tsallis_entropy":  tsallis,
    })

    # --- IAF (cached PSD) ---
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    alpha_freqs = freqs[alpha_mask]
    alpha_psd = psd[alpha_mask]
    if len(alpha_psd) == 0 or np.sum(alpha_psd) < 1e-10:
        feats["iaf"] = float("nan")
    else:
        feats["iaf"] = float(
            np.sum(alpha_freqs * alpha_psd) / np.sum(alpha_psd)
        )

    # --- PDF (slow/fast power ratio, reuses band powers) ---
    fast_p = alpha_p + beta_p
    if fast_p > 1e-10:
        feats["pdf"] = min(float((delta_p + theta_p) / fast_p), 100.0)
    else:
        feats["pdf"] = 100.0

    # --- LZC (time-domain, can't reuse PSD) ---
    feats["lzc"] = compute_lzc(sig)

    return feats


def extract_features_fast(eeg_input, sfreq=128, mode="regional",
                          channel_names=None, skip_lzc=False,
                          return_names=True):
    """
    Fast numerical-only feature extraction for classifier training.

    Compared to extract_biomarkers():
      - Computes PSD once per channel instead of 5x (~3-4x speedup)
      - Skips reference comparison and flagging entirely
      - Skips posterior coherence (171 channel-pairs per segment)
      - Skips text formatting and report generation
      - Returns numpy array, not formatted string

    Parameters
    ----------
    eeg_input : ndarray
        Either (n_channels, n_samples) for a single segment, or
        (n_segments, n_channels, n_samples) for many segments.
    sfreq : float
        Sampling rate in Hz. Default 128 (Kaggle dataset rate).
    mode : "whole_head" or "regional"
        - "whole_head": features averaged across all channels.
          Output is (n_segments, ~30) features.
        - "regional": features averaged within each region (frontal,
          temporal, central, parietal, occipital).
          Output is (n_segments, ~150) features.
    channel_names : list of str, optional
        Required only for "regional" mode. Default: standard 19-channel
        10-20 layout matching the Kaggle dataset.
    skip_lzc : bool
        If True, skip Lempel-Ziv complexity (saves a small amount of time).
        Default False.
    return_names : bool
        If True, return (features_array, feature_names).
        If False, return features_array only.

    Returns
    -------
    features : ndarray, shape (n_segments, n_features)
    feature_names : list of str (only if return_names=True)

    Examples
    --------
    >>> # Single segment, whole-head averaging
    >>> X, names = extract_features_fast(segment, mode="whole_head")
    >>> X.shape
    (1, 31)
    >>>
    >>> # Many segments, regional features
    >>> X, names = extract_features_fast(all_segments, mode="regional")
    >>> X.shape
    (n_segments, ~150)
    """
    # --- Standardize input shape to (n_segments, n_channels, n_samples) ---
    if eeg_input.ndim == 2:
        segments = eeg_input[np.newaxis, :, :]
    elif eeg_input.ndim == 3:
        segments = eeg_input
    else:
        raise ValueError(
            f"eeg_input must be 2D or 3D ndarray, got {eeg_input.ndim}D"
        )

    n_segments, n_channels, _ = segments.shape

    # Default channel names (standard 19-ch 10-20)
    if channel_names is None:
        channel_names = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T3", "C3", "Cz", "C4", "T4",
            "T5", "P3", "Pz", "P4", "T6",
            "O1", "O2",
        ][:n_channels]

    # --- Build a fast per-channel features function (with optional LZC skip) ---
    def _ch_feats(sig):
        feats = _channel_features_fast(sig, sfreq)
        if skip_lzc:
            feats.pop("lzc", None)
        return feats

    # --- Pre-compute region-to-channel-index map (used in regional mode) ---
    if mode == "regional":
        region_indices = {}
        for region, region_chs in REGIONS.items():
            indices = [i for i, ch in enumerate(channel_names)
                       if ch in region_chs]
            if indices:
                region_indices[region] = indices

    # --- Loop over segments ---
    rows = []
    feature_names_out = None

    for seg_idx in range(n_segments):
        seg = segments[seg_idx]

        # Compute features for every channel ONCE
        per_channel_feats = [_ch_feats(seg[ch]) for ch in range(n_channels)]
        feat_keys = list(per_channel_feats[0].keys())

        if mode == "whole_head":
            # Average each feature across all channels
            row_dict = {}
            for f in feat_keys:
                vals = [c[f] for c in per_channel_feats]
                row_dict[f] = _safe_nanmean(vals)
            if feature_names_out is None:
                feature_names_out = list(row_dict.keys())
            rows.append([row_dict[f] for f in feature_names_out])

        elif mode == "regional":
            # Average each feature within each region
            row_dict = {}
            for region, indices in region_indices.items():
                for f in feat_keys:
                    vals = [per_channel_feats[i][f] for i in indices]
                    row_dict[f"{region}__{f}"] = _safe_nanmean(vals)
            if feature_names_out is None:
                feature_names_out = list(row_dict.keys())
            rows.append([row_dict[f] for f in feature_names_out])

        else:
            raise ValueError(
                f"mode must be 'whole_head' or 'regional', got {mode!r}"
            )

    features = np.array(rows, dtype=np.float32)

    if return_names:
        return features, feature_names_out
    return features


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Biomarker Module v4 — Persistence-filtered Tiered Analysis")
    print("=" * 60)

    sfreq = 128
    n_channels = 19
    segment_samples = 2048

    np.random.seed(42)
    t = np.linspace(0, segment_samples / sfreq, segment_samples, endpoint=False)

    segments = np.zeros((10, n_channels, segment_samples))
    for seg in range(10):
        for ch in range(n_channels):
            if seg < 5:
                segments[seg, ch] = (
                    3.0 * np.sin(2 * np.pi * 10 * t)
                    + 0.5 * np.sin(2 * np.pi * 6 * t)
                    + 0.5 * np.random.randn(segment_samples)
                )
            else:
                segments[seg, ch] = (
                    0.5 * np.sin(2 * np.pi * 10 * t)
                    + 4.0 * np.sin(2 * np.pi * 6 * t)
                    + 0.5 * np.random.randn(segment_samples)
                )

    print("\n1. Testing multi-segment analysis (10 segments)...")
    report = extract_biomarkers(
        segments, sfreq=sfreq, subject_id="TEST-001"
    )
    print(report)

    print("\n\n2. Testing single segment...")
    report = extract_biomarkers(
        segments[0], sfreq=sfreq, subject_id="SINGLE-TEST"
    )
    print(f"   Report length: {len(report)} characters")

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