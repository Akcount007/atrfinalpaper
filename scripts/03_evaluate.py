#!/usr/bin/env python
"""Compute all metrics, bootstrap CIs, rank agreement, paired tests and failure tables from RAW outputs only."""
import argparse
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from _common import ROOT, load_cfg
from factbench import analysis as A, data, metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
ap.add_argument("--out", default=str(ROOT / "results"))
ap.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
ap.add_argument("--bootstrap", type=int, default=None, help="override bootstrap resamples (smoke tests only)")
ap.add_argument("--allow-dummy", action="store_true", help="smoke tests only; outputs are NOT reportable")
args = ap.parse_args()
cfg = load_cfg()
mc, seed = cfg["metrics"], cfg["seed"]
B, alpha = (args.bootstrap or mc["bootstrap_resamples"]), mc["ci_alpha"]
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

cells, metas = A.load_cells(args.raw)
if not cells:
    sys.exit(f"no raw results found in {args.raw}")
if any(m.get("backend") == "dummy" for m in metas.values()):
    if not args.allow_dummy:
        sys.exit("REFUSING: raw results come from the DUMMY backend (pseudo-random noise). Use --allow-dummy for smoke tests only.")
    print("!!! WARNING: DUMMY results. Nothing produced here may be reported. !!!")
if any(m.get("limit") for m in metas.values()):
    print("WARNING: some runs used --limit (partial data). Do not report these as final results.")

slices = sorted({s for _, s in cells})
point, boot, rows = {}, {}, []
for s in slices:
    ms = sorted(m for (m, ss) in cells if ss == s)
    ref = cells[(ms[0], s)]["ids"]
    for m in ms:
        assert cells[(m, s)]["ids"] == ref, f"item ids differ between models in slice {s}: refusing to compare"
    idxm = A.boot_index(s.rsplit("_", 1)[0], len(ref), B, seed)
    point[s], boot[s] = {}, {}
    for m in ms:
        d = cells[(m, s)]
        point[s][m] = A.cell_metrics(d, None, mc["ece_bins"], mc["selective_coverage"])
        boot[s][m] = A.bootstrap(d, idxm, mc)
        row = {"model": m, "slice": s, "n": len(ref)}
        for k, v in point[s][m].items():
            lo, hi = A.ci(boot[s][m][k], alpha)
            row.update({k: v, f"{k}_lo": lo, f"{k}_hi": hi})
        rows.append(row)
pd.DataFrame(rows).to_csv(out / "results.csv", index=False)

# trivial baselines
base = []
for s in slices:
    d = cells[next(k for k in cells if k[1] == s)]
    cnt = np.bincount(d["gold"], minlength=d["m"])
    base.append({"slice": s, "n": len(d["gold"]), "chance_acc": 1 / d["m"], "majority_acc": cnt.max() / cnt.sum(),
                 "majority_class": int(cnt.argmax())})
pd.DataFrame(base).to_csv(out / "baselines.csv", index=False)

# RQ1: do rankings agree across dimensions?
rk = []
for s in slices:
    ms = list(point[s])
    best_acc = max(ms, key=lambda m: point[s][m]["acc"])
    for k in A.RANK_METRICS:
        order = sorted(ms, key=lambda m: -A.DIRECTION[k] * point[s][m][k])
        rk.append({"slice": s, "metric": k, "ranking_best_to_worst": " > ".join(order), "best": order[0],
                   "best_differs_from_accuracy": order[0] != best_acc})
pd.DataFrame(rk).to_csv(out / "rank_by_metric.csv", index=False)
tau_rows = []
for k in A.RANK_METRICS[1:]:
    t, reps = A.mean_within_slice_tau(point, boot, k)
    lo, hi = A.ci(reps, alpha)
    tau_rows.append({"metric": k, "mean_within_slice_tau_vs_accuracy": t, "ci_lo": lo, "ci_hi": hi, "n_slices": len(slices)})
pd.DataFrame(tau_rows).to_csv(out / "rank_tau_summary.csv", index=False)

# RQ2: task mode (PubMedQA closed vs context), paired
td, ps = [], []
for m in sorted({m for m, _ in cells}):
    kc, kx = (m, "pubmedqa_closed"), (m, "pubmedqa_context")
    if kc in cells and kx in cells:
        b, c, p = A.mcnemar_exact(cells[kx]["correct0"], cells[kc]["correct0"])
        row = {"model": m, "context_gain_only(b)": b, "context_loss_only(c)": c, "mcnemar_p": p}
        for k in ["acc", "ece", "brier", "aurc", "auroc_conf"]:
            delta = boot["pubmedqa_context"][m][k] - boot["pubmedqa_closed"][m][k]
            lo, hi = A.ci(delta, alpha)
            row.update({f"delta_{k}": point["pubmedqa_context"][m][k] - point["pubmedqa_closed"][m][k],
                        f"delta_{k}_lo": lo, f"delta_{k}_hi": hi})
        td.append(row)
if td:
    adj = M.holm([r["mcnemar_p"] for r in td])
    for r, a in zip(td, adj):
        r["mcnemar_p_holm"] = a
    pd.DataFrame(td).to_csv(out / "task_mode_delta.csv", index=False)

# model vs model paired accuracy tests
for s in slices:
    for a_, b_ in itertools.combinations(sorted(point[s]), 2):
        b, c, p = A.mcnemar_exact(cells[(a_, s)]["correct0"], cells[(b_, s)]["correct0"])
        ps.append({"slice": s, "model_a": a_, "model_b": b_, "acc_a": point[s][a_]["acc"], "acc_b": point[s][b_]["acc"],
                   "a_right_b_wrong": b, "a_wrong_b_right": c, "p": p})
if ps:
    for r, a in zip(ps, M.holm([r["p"] for r in ps])):
        r["p_holm"] = a
    pd.DataFrame(ps).to_csv(out / "paired_model_tests.csv", index=False)

# deeper analysis: position bias, PubMedQA confusion, consistently-wrong items
pb, conf_rows, fail = [], [], []
qtext = {}
for f in Path(args.data_dir).glob("*.jsonl") if Path(args.data_dir).exists() else []:
    for it in data.read_jsonl(f):
        qtext[it["id"]] = it["question"]
for (m, s), d in sorted(cells.items()):
    freq = np.bincount(d["disp_choice"].ravel(), minlength=d["m"]) / d["disp_choice"].size
    pb.append({"model": m, "slice": s, **{f"chosen_pos_{'ABCD'[j]}": freq[j] for j in range(d["m"])},
               **{f"gold_share_{'ABCD'[j]}": float((d["gold"] == j).mean()) for j in range(d["m"])}})
    if s.startswith("pubmedqa"):
        for g, name in enumerate(["yes", "no", "maybe"]):
            sel = d["gold"] == g
            conf_rows.append({"model": m, "slice": s, "gold": name, "n": int(sel.sum()),
                              "recall": float(d["correct0"][sel].mean()) if sel.any() else np.nan,
                              **{f"pred_{n2}": int(((d["pred0"] == j) & sel).sum()) for j, n2 in enumerate(["yes", "no", "maybe"])}})
    bad = np.where((d["cons"] == 1.0) & ~d["correct0"])[0]
    for i in bad[np.argsort(-d["conf0"][bad])][:50]:
        fail.append({"model": m, "slice": s, "id": d["ids"][i], "gold": int(d["gold"][i]), "pred": int(d["pred0"][i]),
                     "conf": float(d["conf0"][i]), "question": qtext.get(d["ids"][i], "")[:300]})
pd.DataFrame(pb).to_csv(out / "position_bias.csv", index=False)
if conf_rows:
    pd.DataFrame(conf_rows).to_csv(out / "pubmedqa_confusion.csv", index=False)
pd.DataFrame(fail).to_csv(out / "failure_cases_consistently_wrong.csv", index=False)
print(pd.DataFrame(rows)[["model", "slice", "n", "acc", "ece", "brier", "sel_acc", "auroc_conf", "cons_mean"]].round(3).to_string(index=False))
print(f"\nWrote tables to {out}")
