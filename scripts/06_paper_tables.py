#!/usr/bin/env python
"""Generate LaTeX tables from results/*.csv (no hand-typed numbers).
ALSO computes balanced accuracy / macro-F1 on PubMedQA. These two metrics are POST HOC and EXPLORATORY:
they were NOT in the frozen protocol; label them so in the paper."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from _common import ROOT, load_cfg
from factbench import analysis as A

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
ap.add_argument("--res", default=str(ROOT / "results"))
ap.add_argument("--out", default=str(ROOT / "results" / "tables"))
ap.add_argument("--bootstrap", type=int, default=None)
args = ap.parse_args()
cfg = load_cfg(); B = args.bootstrap or cfg["metrics"]["bootstrap_resamples"]
res_dir, out = Path(args.res), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
esc = lambda s: str(s).replace("_", "\\_")
pf = lambda p: "$<$0.001" if p < 1e-3 else f"{p:.3f}"
ci = lambda r, k: f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"


def save(df, name, caption, resize=False):
    tex = df.to_latex(index=False, escape=False, column_format="l" * len(df.columns))
    if resize:
        tex = f"\\resizebox{{\\textwidth}}{{!}}{{\n{tex}}}\n"
    content = f"% {caption}\n{tex}"
    (out / f"{name}.tex").write_text(content)
    # Also update paper/tables if default results/tables
    paper_out = ROOT / "paper" / "tables"
    if paper_out.exists() and out != paper_out:
        (paper_out / f"{name}.tex").write_text(content)
    print("wrote", out / f"{name}.tex")


r = pd.read_csv(res_dir / "results.csv")
save(pd.DataFrame({"Model": r.model, "Slice": r.slice.map(esc), "$n$": r.n, "Acc [95\\% CI]": r.apply(lambda x: ci(x, "acc"), axis=1),
                   "ECE": r.ece.map("{:.3f}".format), "Brier": r.brier.map("{:.3f}".format),
                   "$\\text{SelAcc}_{0.8}$": r.sel_acc.map("{:.3f}".format), "$\\text{AUROC}_{\\text{conf}}$": r.auroc_conf.map("{:.3f}".format),
                   "Consistency": r.cons_mean.map("{:.3f}".format)}), "tab_main", "Main results per model and slice", resize=True)
save(pd.DataFrame({"Model": r.model, "Slice": r.slice.map(esc), "AUROC consist. [95\\% CI]": r.apply(lambda x: ci(x, "auroc_cons"), axis=1),
                   "Consistently wrong": r.consistently_wrong.map("{:.3f}".format),
                   "Share of errors": (r.consistently_wrong / (1 - r.acc)).map("{:.3f}".format)}), "tab_consistency", "RQ3 consistency vs correctness")

p = res_dir / "task_mode_delta.csv"
if p.exists():
    t = pd.read_csv(p)
    save(pd.DataFrame({"Model": t.model, "$\\Delta$Acc [95\\% CI]": t.apply(lambda x: ci(x, "delta_acc"), axis=1),
                       "$\\Delta$ECE [95\\% CI]": t.apply(lambda x: ci(x, "delta_ece"), axis=1),
                       "McNemar p (Holm)": t.mcnemar_p_holm.map(pf)}), "tab_taskmode", "RQ2 context minus closed-book (PubMedQA)")
p = res_dir / "rank_tau_summary.csv"
if p.exists():
    t = pd.read_csv(p)
    metric_labels = {
        "brier": "Brier Score",
        "ece": "ECE",
        "aurc": "AURC",
        "sel_acc": r"$\text{SelAcc}_{0.8}$",
        "auroc_conf": r"$\text{AUROC}_{\text{conf}}$",
        "cons_mean": "Consistency"
    }
    t["Metric"] = t.metric.map(lambda m: metric_labels.get(m, m))
    save(pd.DataFrame({"Metric": t["Metric"], "Mean within-slice $\\tau$ vs accuracy": t.mean_within_slice_tau_vs_accuracy.map("{:.3f}".format),
                       "95\\% CI": t.apply(lambda x: f"[{x.ci_lo:.3f}, {x.ci_hi:.3f}]", axis=1)}), "tab_rank_tau", "RQ1 rank agreement")
p = res_dir / "paired_model_tests.csv"
if p.exists():
    t = pd.read_csv(p)
    save(pd.DataFrame({"Slice": t.slice.map(esc), "Model A": t.model_a, "Model B": t.model_b, "Acc A": t.acc_a.map("{:.3f}".format),
                       "Acc B": t.acc_b.map("{:.3f}".format), "p (Holm)": t.p_holm.map(pf)}), "tab_paired", "Paired McNemar tests, Holm over all comparisons")

# post-hoc exploratory: balanced accuracy and macro-F1 on PubMedQA
cells, _ = A.load_cells(args.raw)
rows = []
for (m, s), d in sorted(cells.items()):
    if not s.startswith("pubmedqa"):
        continue
    pred, gold, K = d["pred0"], d["gold"], d["m"]

    def bal(pr, g):
        return float(np.nanmean([(pr[g == c] == c).mean() if (g == c).any() else np.nan for c in range(K)]))

    def f1(pr, g):
        v = []
        for c in range(K):
            tp, fp, fn = ((pr == c) & (g == c)).sum(), ((pr == c) & (g != c)).sum(), ((pr != c) & (g == c)).sum()
            v.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        return float(np.mean(v))
    idxm = A.boot_index("pubmedqa", len(gold), B, cfg["seed"])
    bb = np.array([bal(pred[i], gold[i]) for i in idxm]); ff = np.array([f1(pred[i], gold[i]) for i in idxm])
    rows.append({"model": m, "slice": s, "acc": float((pred == gold).mean()), "balanced_acc": bal(pred, gold),
                 "bal_lo": A.ci(bb)[0], "bal_hi": A.ci(bb)[1], "macro_f1": f1(pred, gold), "f1_lo": A.ci(ff)[0], "f1_hi": A.ci(ff)[1]})
if rows:
    bdf = pd.DataFrame(rows)
    bdf.to_csv(res_dir / "pubmedqa_balanced_POSTHOC.csv", index=False)
    save(pd.DataFrame({"Model": bdf.model, "Slice": bdf.slice.map(esc), "Acc": bdf.acc.map("{:.3f}".format),
                       "Balanced acc [95\\% CI]": bdf.apply(lambda x: f"{x.balanced_acc:.3f} [{x.bal_lo:.3f}, {x.bal_hi:.3f}]", axis=1),
                       "Macro-F1 [95\\% CI]": bdf.apply(lambda x: f"{x.macro_f1:.3f} [{x.f1_lo:.3f}, {x.f1_hi:.3f}]", axis=1)}),
         "tab_pubmedqa_balanced_posthoc", "POST HOC exploratory: balanced accuracy and macro-F1 on PubMedQA")
