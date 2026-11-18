#!/usr/bin/env python
"""READ-ONLY. Prints the exact numbers needed to verify quantitative claims in the paper draft."""
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from _common import ROOT
from factbench import analysis as A, data

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
ap.add_argument("--res", default=str(ROOT / "results"))
ap.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
ap.add_argument("--items", nargs="*", default=["mmlu-anatomy-25", "mmlu-astronomy-42", "pubmedqa-10966337"])
a = ap.parse_args()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
cells, metas = A.load_cells(a.raw)

print("\n=== 1. Consistency groups per cell (counts must sum to n; totals must sum to 4350 for the full run) ===")
rows = []
for (m, s), d in sorted(cells.items()):
    c, ok = d["cons"], d["correct0"]
    rows.append({"model": m, "slice": s, "n": len(ok), "consistent_correct": int(((c == 1) & ok).sum()),
                 "wavering_correct": int(((c < 1) & ok).sum()), "wavering_wrong": int(((c < 1) & ~ok).sum()),
                 "stubborn_wrong": int(((c == 1) & ~ok).sum()), "errors": int((~ok).sum())})
g = pd.DataFrame(rows); g["stubborn_share_of_errors"] = (g.stubborn_wrong / g.errors).round(3)
print(g.to_string(index=False))
tot = g[["n", "consistent_correct", "wavering_correct", "wavering_wrong", "stubborn_wrong", "errors"]].sum()
print("\nTOTALS:", tot.to_dict(), "| stubborn share of ALL errors:", round(tot.stubborn_wrong / tot.errors, 3))

print("\n=== 2. Accuracy by consistency level (checks the 'monotonic, <35% ... >70%' claim) ===")
for m in sorted({k[0] for k in cells}):
    for label, sl in [("all slices", None), ("pubmedqa_closed only", "pubmedqa_closed")]:
        ks = [k for k in cells if k[0] == m and (sl is None or k[1] == sl)]
        cons = np.concatenate([cells[k]["cons"] for k in ks]); ok = np.concatenate([cells[k]["correct0"] for k in ks])
        print(m, "|", label, "|", {round(float(v), 2): (int((cons == v).sum()), round(float(ok[cons == v].mean()), 3)) for v in np.unique(cons)}, "(level: (n, accuracy))")

print("\n=== 3. Mean option-probability confidence by group (checks 'stubborn errors have mean confidence > 0.94') ===")
rows = []
for (m, s), d in sorted(cells.items()):
    c, ok, cf = d["cons"], d["correct0"], d["conf0"]
    f = lambda msk: round(float(cf[msk].mean()), 3) if msk.any() else np.nan
    rows.append({"model": m, "slice": s, "conf_consistent_correct": f((c == 1) & ok), "conf_wavering": f(c < 1), "conf_stubborn": f((c == 1) & ~ok)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 4. AUROC_conf vs AUROC_cons per cell (checks '12 of 15 cells, average +0.05') ===")
r = pd.read_csv(Path(a.res) / "results.csv"); r["diff_conf_minus_cons"] = r.auroc_conf - r.auroc_cons
print(r[["model", "slice", "auroc_conf", "auroc_cons", "diff_conf_minus_cons"]].round(3).to_string(index=False))
pos = r[r.diff_conf_minus_cons > 0]
print("cells with conf > cons:", len(pos), "of", len(r), "| mean diff all cells:", round(r.diff_conf_minus_cons.mean(), 3), "| mean diff over those cells:", round(pos.diff_conf_minus_cons.mean(), 3))

print("\n=== 5. Permutation ensemble vs identity-order accuracy (EXPLORATORY, no paired test) ===")
r["ens_minus_acc"] = r.acc_ens - r.acc
print(r[["model", "slice", "acc", "acc_ens", "ens_minus_acc"]].round(3).to_string(index=False))

print("\n=== 6. Quoted failure-case items ===")
items = {}
for f in Path(a.data_dir).glob("*.jsonl") if Path(a.data_dir).exists() else []:
    for it in data.read_jsonl(f):
        items[it["id"]] = it
for iid in a.items:
    it = items.get(iid)
    print("\n", iid, "->", "NOT FOUND in data dir" if it is None else it["question"][:200])
    if it:
        print("  options:", it["options"], "| gold:", it["gold"], "->", it["options"][it["gold"]])
    for (m, s), d in sorted(cells.items()):
        if iid in d["ids"]:
            i = d["ids"].index(iid)
            print(f"  {m} [{s}] pred={d['pred0'][i]} conf={d['conf0'][i]:.4f} cons={d['cons'][i]} correct={bool(d['correct0'][i])}")

print("\n=== 7. Run metadata and rough wall-clock (checks hardware/version/'3.82 GPU hours' claims) ===")
for m, meta in sorted(metas.items()):
    print(m, {k: meta.get(k) for k in ["backend", "limit", "gpu", "python", "torch", "transformers", "quant", "started", "git_commit"]})
    fs = [os.path.getmtime(f) for f in glob.glob(str(Path(a.raw) / f"{m}__*.jsonl"))]
    if fs:
        print("   raw-file mtime span (hours):", round((max(fs) - min(fs)) / 3600, 2), "(only a rough lower bound; Colab copies can reset mtimes)")
print("\nNote: the dtype is not stored in meta. The code loads float16 (bitsandbytes NF4 with default settings for the 7B).")
