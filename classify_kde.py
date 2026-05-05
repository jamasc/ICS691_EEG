import json
import numpy as np
from scipy.stats import gaussian_kde

def classify_kde(feature_name, value, dist_file="feature_distributions.json"):
    with open(dist_file, "r") as f:
        data = json.load(f)

    probs = {}

    for group in ["AD", "NON-AD"]:
        if group not in data[feature_name]:
            continue

        vals = np.array(data[feature_name][group])

        if len(vals) < 2 or np.var(vals) < 1e-8:
            continue

        kde = gaussian_kde(vals)
        likelihood = kde.evaluate([value])[0]

        probs[group] = likelihood

    total = sum(probs.values())
    if total == 0:
        return {k: 0 for k in probs}

    return {k: v / total for k, v in probs.items()}

print(classify_kde("mean_abs", 1))

"""

Use probability from KDE. And have it take in the feature name, and the value. 
Take the distributions from the distributions.json, match the distributions to the feature, 
and then for each distribution (AD and NON-AD), check the score of how likely it is this feature is within that group.
Return both values in a small dictionary (sorted by AD and NON-AD).

Do this with:
beta_power
delta_relative_power
iqr
max
mean_abs
median_frequency
min
std

"""