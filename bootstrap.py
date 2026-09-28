"""
Nonparametric bootstrap confidence intervals -- Section 3.6.

5,000 resamples (with replacement) of the held-out test predictions;
percentile 95% CI from the 2.5th/97.5th percentiles, for accuracy,
macro precision, macro recall, and macro F1.

The manuscript reports this only for the hybrid. This implementation
computes it for BOTH the hybrid and the baseline, so you can report a
CI for each -- addressing the "no CI for baseline, no paired test"
gap flagged during review. It also adds a paired bootstrap test on the
accuracy/macro-F1 DIFFERENCE between the two models (paired on the same
resampled indices), which is a legitimate substitute for the McNemar/
permutation test the manuscript notes it couldn't run because
per-sample predictions from the original run weren't archived -- here
they are archived, since this is a fresh run.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config


def _metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def bootstrap_ci(y_true, y_pred, n_bootstrap=None, seed=None):
    n_bootstrap = n_bootstrap or config.N_BOOTSTRAP
    seed = seed if seed is not None else config.RANDOM_SEED
    rng = np.random.RandomState(seed)
    n = len(y_true)

    boot_metrics = {k: [] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        m = _metrics(y_true[idx], y_pred[idx])
        for k, v in m.items():
            boot_metrics[k].append(v)

    point = _metrics(y_true, y_pred)
    ci = {}
    for k, values in boot_metrics.items():
        lo, hi = np.percentile(values, [config.CI_LOW, config.CI_HIGH])
        ci[k] = {"point": point[k], "ci_low": lo, "ci_high": hi}
    return ci


def paired_bootstrap_diff(y_true, y_pred_a, y_pred_b, n_bootstrap=None, seed=None):
    """Paired bootstrap on (model_a - model_b) accuracy and macro-F1.
    Returns a 95% CI for the DIFFERENCE; if that interval excludes 0,
    the difference is significant at the 95% level under this test."""
    n_bootstrap = n_bootstrap or config.N_BOOTSTRAP
    seed = seed if seed is not None else config.RANDOM_SEED
    rng = np.random.RandomState(seed)
    n = len(y_true)

    diffs = {"accuracy": [], "macro_f1": []}
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        ma = _metrics(y_true[idx], y_pred_a[idx])
        mb = _metrics(y_true[idx], y_pred_b[idx])
        diffs["accuracy"].append(ma["accuracy"] - mb["accuracy"])
        diffs["macro_f1"].append(ma["macro_f1"] - mb["macro_f1"])

    out = {}
    for k, values in diffs.items():
        lo, hi = np.percentile(values, [config.CI_LOW, config.CI_HIGH])
        out[k] = {
            "point_diff": np.mean(values),
            "ci_low": lo,
            "ci_high": hi,
            "significant_95": not (lo <= 0 <= hi),
        }
    return out
