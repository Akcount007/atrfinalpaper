"""Turn raw per-item option log-probs into per-cell arrays, metrics and bootstrap replicates."""
import json
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, kendalltau

from . import metrics as M

# direction: +1 = higher is better, -1 = lower is better
DIRECTION = {"acc": +1, "brier": -1, "ece": -1, "aurc": -1, "sel_acc": +1, "auroc_conf": +1,
             "cons_mean": +1, "auroc_cons": +1, "acc_ens": +1, "perm_acc": +1, "consistently_wrong": -1}
RANK_METRICS = ["acc", "brier", "ece", "aurc", "sel_acc", "auroc_conf", "cons_mean"]


def canonicalize(perms, logp):
    """perms (K,m): display position j shows option perms[k][j]. Returns canonical-order probs (K,m)."""
    perms, p = np.asarray(perms), np.exp(np.asarray(logp))
    out = np.zeros_like(p)
    for k in range(len(perms)):
        out[k, perms[k]] = p[k]
    return out


def load_cells(raw_dir):
    """Returns ({(model, slice): cell}, {model: meta})."""
    raw_dir = Path(raw_dir)
    cells, metas = {}, {}
    for f in sorted(raw_dir.glob("*__meta.json")):
        m = json.loads(f.read_text())
        metas[m["model"]] = m
    for f in sorted(raw_dir.glob("*.jsonl")):
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not recs:
            continue
        recs.sort(key=lambda r: r["id"])
        cells[(recs[0]["model"], recs[0]["slice"])] = build_cell(recs)
    return cells, metas


def build_cell(recs):
    gold = np.array([r["gold"] for r in recs])
    P = np.stack([canonicalize(r["perms"], r["logp"]) for r in recs])          # (N,K,m)
    L = np.stack([np.array(r["logp"]) for r in recs])                          # display-order logp
    probs0 = P[:, 0, :]
    pred0 = probs0.argmax(1)
    preds = P.argmax(2)
    cons, mode = M.consistency(preds)
    pbar = P.mean(1)
    return {
        "ids": [r["id"] for r in recs], "gold": gold, "m": P.shape[2], "K": P.shape[1],
        "probs0": probs0, "pred0": pred0, "conf0": probs0.max(1), "correct0": pred0 == gold,
        "cons": cons, "mode": mode, "correct_ens": pbar.argmax(1) == gold,
        "perm_acc": (preds == gold[:, None]).mean(1), "disp_choice": L.argmax(2),
    }


def cell_metrics(d, idx=None, ece_bins=10, coverage=0.8):
    s = (lambda a: a) if idx is None else (lambda a: a[idx])
    gold, correct, conf, cons = s(d["gold"]), s(d["correct0"]), s(d["conf0"]), s(d["cons"])
    err = ~correct
    return {
        "acc": M.accuracy(correct), "brier": M.brier(s(d["probs0"]), gold),
        "ece": M.ece(conf, correct, ece_bins), "aurc": M.aurc(conf, correct),
        "sel_acc": M.selective_accuracy(conf, correct, coverage),
        "auroc_conf": M.auroc(conf, correct), "auroc_cons": M.auroc(cons, correct),
        "cons_mean": float(cons.mean()), "acc_ens": M.accuracy(s(d["correct_ens"])),
        "perm_acc": float(s(d["perm_acc"]).mean()),
        "consistently_wrong": float(((cons == 1.0) & err).mean()),
    }


def boot_index(key, n, B, seed):
    """Shared resample indices per (family-key, n) so paired comparisons use identical resamples."""
    rng = np.random.default_rng([seed, zlib.crc32(key.encode()), n])
    return rng.integers(0, n, size=(B, n))


def bootstrap(d, idx_matrix, cfg):
    reps = {k: np.empty(len(idx_matrix)) for k in DIRECTION}
    for b, idx in enumerate(idx_matrix):
        r = cell_metrics(d, idx, cfg["ece_bins"], cfg["selective_coverage"])
        for k in reps:
            reps[k][b] = r[k]
    return reps


def ci(arr, alpha=0.05):
    a = arr[~np.isnan(arr)]
    return (float(np.percentile(a, 100 * alpha / 2)), float(np.percentile(a, 100 * (1 - alpha / 2)))) if len(a) else (np.nan, np.nan)


def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar test on paired correctness. Returns (b, c, p): b = A right & B wrong."""
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b, c = int((a & ~b_).sum()), int((~a & b_).sum())
    n = b + c
    return b, c, (1.0 if n == 0 else float(binomtest(min(b, c), n, 0.5).pvalue))


def mean_within_slice_tau(point_by_slice, boot_by_slice, metric):
    """Mean over slices of Kendall tau(accuracy ranking, goodness-of-metric ranking) across models.
    point_by_slice[slice][model][metric]; boot_by_slice[slice][model][metric] -> array(B)."""
    def tau_of(get):
        taus = []
        for s in point_by_slice:
            models = list(point_by_slice[s])
            if len(models) < 3:
                continue
            a = [get(s, m, "acc") for m in models]
            g = [DIRECTION[metric] * get(s, m, metric) for m in models]
            t = kendalltau(a, g).statistic
            if not np.isnan(t):
                taus.append(t)
        return np.mean(taus) if taus else np.nan
    point = tau_of(lambda s, m, k: point_by_slice[s][m][k])
    B = len(next(iter(next(iter(boot_by_slice.values())).values()))["acc"])
    reps = np.array([tau_of(lambda s, m, k, b=b: boot_by_slice[s][m][k][b]) for b in range(B)])
    return point, reps
