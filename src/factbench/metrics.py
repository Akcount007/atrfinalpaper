"""Metric implementations (numpy/scipy only). Conventions are stated in each docstring."""
import numpy as np
from scipy.stats import rankdata


def accuracy(correct):
    return float(np.mean(correct))


def brier(probs, gold):
    """Multiclass Brier score: mean over items of sum_k (p_k - 1[k=gold])^2. Lower is better; range [0, 2]."""
    onehot = np.eye(probs.shape[1])[gold]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ece(conf, correct, n_bins=10):
    """Expected calibration error, equal-width bins over (0,1]. Lower is better."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.digitize(conf, edges[1:-1], right=True)
    total = 0.0
    for i in range(n_bins):
        m = b == i
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_bins(conf, correct, n_bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.digitize(conf, edges[1:-1], right=True)
    rows = []
    for i in range(n_bins):
        m = b == i
        rows.append((i, int(m.sum()), float(conf[m].mean()) if m.any() else np.nan,
                     float(correct[m].mean()) if m.any() else np.nan))
    return rows


def risk_coverage(conf, correct):
    """Sort by confidence (descending, stable; ties keep input order). Returns coverage, risk (=error rate)."""
    order = np.argsort(-np.asarray(conf), kind="stable")
    c = np.asarray(correct, float)[order]
    k = np.arange(1, len(c) + 1)
    return k / len(c), 1.0 - np.cumsum(c) / k


def aurc(conf, correct):
    """Area under the risk-coverage curve (discrete mean of risk over coverage levels). Lower is better."""
    return float(np.mean(risk_coverage(conf, correct)[1]))


def selective_accuracy(conf, correct, coverage=0.8):
    order = np.argsort(-np.asarray(conf), kind="stable")
    k = max(1, int(np.ceil(coverage * len(order))))
    return float(np.mean(np.asarray(correct, float)[order[:k]]))


def auroc(score, label):
    """P(score of a correct item > score of an incorrect item), ties counted 1/2. NaN if one class is empty."""
    label = np.asarray(label).astype(bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def consistency(preds):
    """preds: (N, K) predicted option indices across K option-order variants.
    Returns (agreement with the modal prediction in [1/K..1], modal prediction)."""
    n, k = preds.shape
    frac, mode = np.empty(n), np.empty(n, dtype=int)
    for i in range(n):
        vals, counts = np.unique(preds[i], return_counts=True)
        frac[i], mode[i] = counts.max() / k, vals[counts.argmax()]
    return frac, mode


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
