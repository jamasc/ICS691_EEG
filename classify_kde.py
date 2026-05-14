import json
import numpy as np
from scipy.stats import gaussian_kde


# Only these biomarkers will be evaluated automatically
TARGET_FEATURES = [
    "beta_power",
    "delta_relative_power",
    "iqr",
    "max",
    "mean_abs",
    "median_frequency",
    "min",
    "std",
]


def classify_kde(feature_name, value, dist_file="feature_distributions.json"):
    with open(dist_file, "r") as f:
        data = json.load(f)

    if feature_name not in data:
        return {}

    probs = {}

    for group in ["AD", "NON-AD"]:
        if group not in data[feature_name]:
            continue

        vals = np.array(data[feature_name][group], dtype=float)

        if len(vals) < 2:
            continue

        if np.var(vals) < 1e-8:
            continue

        kde = gaussian_kde(vals)

        likelihood = float(kde.evaluate([value])[0])

        probs[group] = likelihood

    total = sum(probs.values())

    if total == 0:
        return {k: 0.0 for k in probs}

    normalized = {
        k: float(v / total)
        for k, v in probs.items()
    }

    return {
        "AD": normalized.get("AD", 0.0),
        "NON-AD": normalized.get("NON-AD", 0.0),
    }


def classify_selected_features(feature_dict, dist_file="feature_distributions.json"):
    """
    feature_dict:
        {
            "feature_name": value,
            ...
        }

    Returns:
        {
            "feature_name": {
                "AD": ...,
                "NON-AD": ...
            }
        }
    """

    results = {}

    for feature_name in TARGET_FEATURES:
        if feature_name not in feature_dict:
            continue

        value = feature_dict[feature_name]

        if value is None:
            continue

        if not np.isfinite(value):
            continue

        results[feature_name] = classify_kde(
            feature_name,
            float(value),
            dist_file=dist_file
        )

    return results