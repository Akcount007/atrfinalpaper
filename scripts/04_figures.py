#!/usr/bin/env python
"""Publication figures from raw outputs + results tables (PNG + PDF)."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import ROOT
from factbench import analysis as A, metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
ap.add_argument("--res", default=str(ROOT / "results"))
ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
cells, _ = A.load_cells(args.raw)
models = sorted({m for m, _ in cells}); slices = sorted({s for _, s in cells})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


# 1. reliability diagrams
fig, ax = plt.subplots(len(models), len(slices), figsize=(2.6 * len(slices), 2.5 * len(models)), squeeze=False)
for i, m in enumerate(models):
    for j, s in enumerate(slices):
        a = ax[i][j]
        if (m, s) in cells:
            d = cells[(m, s)]
            bins = [r for r in M.reliability_bins(d["conf0"], d["correct0"]) if r[1] > 0]
            a.bar([r[2] for r in bins], [r[3] for r in bins], width=0.08, alpha=.8)
        a.plot([0, 1], [0, 1], "k--", lw=.8); a.set_xlim(0, 1); a.set_ylim(0, 1)
        if i == 0: a.set_title(s, fontsize=8)
        if j == 0: a.set_ylabel(m, fontsize=7)
        if i == len(models) - 1: a.set_xlabel("confidence", fontsize=7)
save(fig, "fig_reliability")

# 2. risk-coverage
fig, ax = plt.subplots(1, len(slices), figsize=(2.8 * len(slices), 2.8), squeeze=False)
for j, s in enumerate(slices):
    for m in models:
        if (m, s) in cells:
            d = cells[(m, s)]
            cov, risk = M.risk_coverage(d["conf0"], d["correct0"])
            ax[0][j].plot(cov, risk, label=m)
    ax[0][j].set_title(s, fontsize=8); ax[0][j].set_xlabel("coverage", fontsize=7)
ax[0][0].set_ylabel("risk (error rate)", fontsize=7); ax[0][0].legend(fontsize=5)
save(fig, "fig_risk_coverage")

# 3. accuracy by consistency level (pooled over slices, per model)
fig, a = plt.subplots(figsize=(4, 3))
for m in models:
    cons = np.concatenate([cells[(m, s)]["cons"] for s in slices if (m, s) in cells])
    cor = np.concatenate([cells[(m, s)]["correct0"] for s in slices if (m, s) in cells])
    lv = np.unique(cons)
    a.plot(lv, [cor[cons == v].mean() for v in lv], marker="o", label=m)
a.set_xlabel("permutation consistency"); a.set_ylabel("accuracy (identity order)"); a.legend(fontsize=6)
save(fig, "fig_accuracy_by_consistency")

# 4. task-mode delta
p = Path(args.res) / "task_mode_delta.csv"
if p.exists():
    t = pd.read_csv(p)
    fig, a = plt.subplots(figsize=(4, 3))
    y = np.arange(len(t))
    a.errorbar(t["delta_acc"], y, xerr=[t["delta_acc"] - t["delta_acc_lo"], t["delta_acc_hi"] - t["delta_acc"]], fmt="o")
    a.axvline(0, color="k", lw=.8); a.set_yticks(y); a.set_yticklabels(t["model"], fontsize=7)
    a.set_xlabel("PubMedQA accuracy: context minus closed-book (95% CI)")
    save(fig, "fig_task_mode_delta")
print("figures written to", out)
