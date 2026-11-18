import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# Add src to path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
from factbench import analysis as A, metrics as M

print("=" * 80)
print("FACTBENCH DIAGNOSTIC AUDIT (DIRECTLY FROM RAW OUTPUTS & CSVs)")
print("=" * 80)

raw_dir = ROOT / "results" / "raw"
cells, metas = A.load_cells(raw_dir)

# ==============================================================================
# PART 1: BREAKDOWN OF STUBBORN, CONFIDENT ERRORS
# ==============================================================================
print("\n" + "=" * 80)
print("PART 1: BREAKDOWN OF STUBBORN, CONFIDENT ERRORS")
print("=" * 80)

# 1A: Inspect failure_cases_consistently_wrong.csv
fail_csv = pd.read_csv(ROOT / "results" / "failure_cases_consistently_wrong.csv")
print(f"Total entries in failure_cases_consistently_wrong.csv: {len(fail_csv)}")
print("\nBreakdown in failure_cases_consistently_wrong.csv (top <=50 per cell):")
pivot_csv = pd.crosstab(fail_csv["model"], fail_csv["slice"], margins=True)
print(pivot_csv)

# 1B: Inspect TRUE UN-CAPPED counts directly from raw arrays
print("\nTrue un-capped stubborn errors (cons == 1.0 & correct0 == False) in raw data:")
raw_counts = []
for (m, s), d in sorted(cells.items()):
    bad = np.where((d["cons"] == 1.0) & ~d["correct0"])[0]
    total_items = len(d["gold"])
    mean_conf_bad = float(d["conf0"][bad].mean()) if len(bad) > 0 else 0.0
    raw_counts.append({
        "model": m,
        "slice": s,
        "stubborn_errors": len(bad),
        "total_items": total_items,
        "rate": len(bad) / total_items,
        "mean_conf_when_stubbornly_wrong": mean_conf_bad
    })

df_raw_counts = pd.DataFrame(raw_counts)
pivot_raw = df_raw_counts.pivot(index="model", columns="slice", values="stubborn_errors")
pivot_raw["Total"] = pivot_raw.sum(axis=1)
print(pivot_raw)

print("\nStubborn Error Rate (%) by Model and Slice:")
pivot_rate = df_raw_counts.pivot(index="model", columns="slice", values="rate") * 100
pivot_rate["Mean %"] = pivot_rate.mean(axis=1)
print(pivot_rate.round(2))

print("\nMean Confidence of Model when Giving a Stubborn Wrong Answer:")
pivot_conf = df_raw_counts.pivot(index="model", columns="slice", values="mean_conf_when_stubbornly_wrong")
print(pivot_conf.round(3))

# ==============================================================================
# PART 2: VERIFICATION OF KENDALL'S TAU = 0.30 & 55.2% MAJORITY COLLAPSE
# ==============================================================================
print("\n" + "=" * 80)
print("PART 2: VERIFY KENDALL'S TAU = 0.30 & 55.2% PUBMEDQA COLLAPSE")
print("=" * 80)

# 2A: Kendall's Tau calculation directly from cell metrics
slices = sorted({s for _, s in cells})
models = sorted({m for m, _ in cells})

taus_acc_vs_auroc = []
print("Slice-by-slice Kendall's tau between Accuracy and AUROC_conf:")
for s in slices:
    accs = [cells[(m, s)]["correct0"].mean() for m in models]
    aurocs = []
    for m in models:
        d = cells[(m, s)]
        # compute auroc_conf
        auc = M.auroc(d["conf0"], d["correct0"])
        aurocs.append(auc)
    tau, p = kendalltau(accs, aurocs)
    taus_acc_vs_auroc.append(tau)
    # Print model ordering
    order_acc = [models[i] for i in np.argsort(-np.array(accs))]
    order_auc = [models[i] for i in np.argsort(-np.array(aurocs))]
    print(f"  Slice {s:17s} | Tau: {tau:+.3f} (p={p:.3f}) | Acc: {' > '.join(order_acc)} | AUC: {' > '.join(order_auc)}")

mean_tau = np.mean(taus_acc_vs_auroc)
print(f"\nMean within-slice Kendall's tau (Accuracy vs AUROC_conf): {mean_tau:.3f} (Rounds to {mean_tau:.2f})")

# Cross check with rank_tau_summary.csv
tau_summary = pd.read_csv(ROOT / "results" / "rank_tau_summary.csv")
print("\nContents of results/rank_tau_summary.csv:")
print(tau_summary.to_string(index=False))

# 2B: PubMedQA majority-class collapse verification
print("\nPubMedQA Closed-Book Analysis:")
d_closed_q15 = cells[("qwen2.5-1.5b-instruct", "pubmedqa_closed")]
gold_counts = pd.Series(d_closed_q15["gold"]).value_counts().sort_index()
print(f"Gold label distribution (0=yes, 1=no, 2=maybe):")
for g, name in [(0, "yes"), (1, "no"), (2, "maybe")]:
    cnt = gold_counts.get(g, 0)
    print(f"  {name:5s} (id={g}): {cnt} / 500 = {cnt/500*100:.2f}%")

q15_preds = pd.Series(d_closed_q15["pred0"]).value_counts().sort_index()
print(f"\nQwen2.5-1.5B predicted distribution (closed-book):")
for g, name in [(0, "yes"), (1, "no"), (2, "maybe")]:
    cnt = q15_preds.get(g, 0)
    print(f"  {name:5s} (id={g}): {cnt} / 500 = {cnt/500*100:.2f}%")

q15_correct = (d_closed_q15["pred0"] == d_closed_q15["gold"]).sum()
print(f"Qwen2.5-1.5B raw closed-book accuracy: {q15_correct} / 500 = {q15_correct/500*100:.2f}%")
print(f"Exact match with majority class share: {gold_counts[0]/500*100:.2f}% == {q15_correct/500*100:.2f}%")

# Confusion breakdown
conf_df = pd.read_csv(ROOT / "results" / "pubmedqa_confusion.csv")
print("\nPubMedQA Confusion Matrix for Qwen2.5-1.5B (Closed vs Context):")
print(conf_df[conf_df["model"] == "qwen2.5-1.5b-instruct"].to_string(index=False))
