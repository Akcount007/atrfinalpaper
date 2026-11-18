import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
from factbench import analysis as A, metrics as M

raw_dir = ROOT / "results" / "raw"
res_dir = ROOT / "results"
fig_dir = ROOT / "results" / "figures"
paper_fig_dir = ROOT / "paper" / "figures"

fig_dir.mkdir(parents=True, exist_ok=True)
paper_fig_dir.mkdir(parents=True, exist_ok=True)

cells, metas = A.load_cells(raw_dir)
models = sorted({m for m, _ in cells})
slices = sorted({s for _, s in cells})

print("Generating 2 Additional Statistical Tables...")

# ------------------------------------------------------------------------------
# TABLE A: STUBBORN ERRORS & CONSISTENCY BREAKDOWN
# ------------------------------------------------------------------------------
tbl_a_rows = []
for (m, s), d in sorted(cells.items()):
    n = len(d["gold"])
    cons = d["cons"]
    cor = d["correct0"]
    conf = d["conf0"]
    
    n_cons_cor = int(np.sum((cons == 1.0) & cor))
    n_waver = int(np.sum(cons < 1.0))
    n_stubborn = int(np.sum((cons == 1.0) & ~cor))
    
    mean_conf_cor = float(conf[cor].mean()) if cor.any() else 0.0
    mean_conf_waver = float(conf[cons < 1.0].mean()) if (cons < 1.0).any() else 0.0
    mean_conf_stubborn = float(conf[(cons == 1.0) & ~cor].mean()) if n_stubborn > 0 else 0.0
    
    tbl_a_rows.append({
        "model": m,
        "slice": s,
        "N": n,
        "consistent_correct_n": n_cons_cor,
        "consistent_correct_pct": round(n_cons_cor / n * 100, 1),
        "wavering_n": n_waver,
        "wavering_pct": round(n_waver / n * 100, 1),
        "stubborn_wrong_n": n_stubborn,
        "stubborn_wrong_pct": round(n_stubborn / n * 100, 1),
        "mean_conf_correct": round(mean_conf_cor, 3),
        "mean_conf_wavering": round(mean_conf_waver, 3),
        "mean_conf_stubborn_wrong": round(mean_conf_stubborn, 3)
    })

df_tbl_a = pd.DataFrame(tbl_a_rows)
df_tbl_a.to_csv(res_dir / "table_stubborn_errors_breakdown.csv", index=False)
print("Saved results/table_stubborn_errors_breakdown.csv")

# ------------------------------------------------------------------------------
# TABLE B: PUBMEDQA CLASS-SPECIFIC PERFORMANCE & GROUNDING RESCUE
# ------------------------------------------------------------------------------
conf_df = pd.read_csv(res_dir / "pubmedqa_confusion.csv")
task_delta_df = pd.read_csv(res_dir / "task_mode_delta.csv")

tbl_b_rows = []
for m in models:
    row_cl = conf_df[(conf_df["model"] == m) & (conf_df["slice"] == "pubmedqa_closed")]
    row_cx = conf_df[(conf_df["model"] == m) & (conf_df["slice"] == "pubmedqa_context")]
    delta_row = task_delta_df[task_delta_df["model"] == m].iloc[0]
    
    rec_yes_cl = row_cl[row_cl["gold"] == "yes"]["recall"].values[0]
    rec_no_cl = row_cl[row_cl["gold"] == "no"]["recall"].values[0]
    rec_maybe_cl = row_cl[row_cl["gold"] == "maybe"]["recall"].values[0]
    
    rec_yes_cx = row_cx[row_cx["gold"] == "yes"]["recall"].values[0]
    rec_no_cx = row_cx[row_cx["gold"] == "no"]["recall"].values[0]
    rec_maybe_cx = row_cx[row_cx["gold"] == "maybe"]["recall"].values[0]
    
    tbl_b_rows.append({
        "model": m,
        "closed_acc": round(float(cells[(m, "pubmedqa_closed")]["correct0"].mean()), 3),
        "closed_rec_yes": round(float(rec_yes_cl), 3),
        "closed_rec_no": round(float(rec_no_cl), 3),
        "closed_rec_maybe": round(float(rec_maybe_cl), 3),
        "context_acc": round(float(cells[(m, "pubmedqa_context")]["correct0"].mean()), 3),
        "context_rec_yes": round(float(rec_yes_cx), 3),
        "context_rec_no": round(float(rec_no_cx), 3),
        "context_rec_maybe": round(float(rec_maybe_cx), 3),
        "delta_acc": round(float(delta_row["delta_acc"]), 3),
        "context_gain_b": int(delta_row["context_gain_only(b)"]),
        "context_loss_c": int(delta_row["context_loss_only(c)"]),
        "mcnemar_p_holm": delta_row["mcnemar_p_holm"]
    })

df_tbl_b = pd.DataFrame(tbl_b_rows)
df_tbl_b.to_csv(res_dir / "table_pubmedqa_deep_dive.csv", index=False)
print("Saved results/table_pubmedqa_deep_dive.csv")

# ------------------------------------------------------------------------------
# FIGURE 1: RANK INVERSION SLOPEGRAPH (Accuracy Rank vs AUROC Rank)
# ------------------------------------------------------------------------------
print("\nGenerating Figure 1: fig_rank_inversion_slopegraph...")
fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharey=True)

model_colors = {
    "phi-3.5-mini-instruct": "#1f77b4",
    "qwen2.5-1.5b-instruct": "#ff7f0e",
    "qwen2.5-7b-instruct-4bit": "#2ca02c"
}
model_labels = {
    "phi-3.5-mini-instruct": "Phi-3.5-mini",
    "qwen2.5-1.5b-instruct": "Qwen-1.5B",
    "qwen2.5-7b-instruct-4bit": "Qwen-7B-4b"
}

for ax, s in zip(axes, slices):
    accs = [cells[(m, s)]["correct0"].mean() for m in models]
    aurocs = [M.auroc(cells[(m, s)]["conf0"], cells[(m, s)]["correct0"]) for m in models]
    
    # Ranks (1 = best, 3 = worst)
    acc_ranks = {m: r + 1 for r, m in enumerate(sorted(models, key=lambda m: -cells[(m, s)]["correct0"].mean()))}
    auc_ranks = {m: r + 1 for r, m in enumerate(sorted(models, key=lambda m: -M.auroc(cells[(m, s)]["conf0"], cells[(m, s)]["correct0"])))}
    
    for m in models:
        r_acc = acc_ranks[m]
        r_auc = auc_ranks[m]
        ax.plot([0, 1], [r_acc, r_auc], marker='o', lw=2.2, color=model_colors[m], label=model_labels[m])
        ax.text(-0.05, r_acc, f"{r_acc}", ha='right', va='center', fontsize=9, fontweight='bold', color=model_colors[m])
        ax.text(1.05, r_auc, f"{r_auc}", ha='left', va='center', fontsize=9, fontweight='bold', color=model_colors[m])
        
    ax.set_title(s.replace("_", "\n"), fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Acc Rank", "AUROC Rank"], fontsize=9)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["1st (Best)", "2nd", "3rd (Worst)"], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_xlim(-0.25, 1.25)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False, fontsize=10)
plt.tight_layout()
fig.savefig(fig_dir / "fig_rank_inversion_slopegraph.png", dpi=300, bbox_inches='tight')
fig.savefig(fig_dir / "fig_rank_inversion_slopegraph.pdf", bbox_inches='tight')
fig.savefig(paper_fig_dir / "fig_rank_inversion_slopegraph.png", dpi=300, bbox_inches='tight')
fig.savefig(paper_fig_dir / "fig_rank_inversion_slopegraph.pdf", bbox_inches='tight')
plt.close(fig)
print("Saved fig_rank_inversion_slopegraph (.png & .pdf)")

# ------------------------------------------------------------------------------
# FIGURE 2: THE STUBBORN CONFIDENCE DISTRIBUTION
# ------------------------------------------------------------------------------
print("Generating Figure 2: fig_stubborn_confidence_distribution...")
fig, ax = plt.subplots(figsize=(6, 3.8))

pooled_conf_cor = []
pooled_conf_waver = []
pooled_conf_stubborn = []

for (m, s), d in cells.items():
    cons = d["cons"]
    cor = d["correct0"]
    conf = d["conf0"]
    pooled_conf_cor.extend(conf[(cons == 1.0) & cor])
    pooled_conf_waver.extend(conf[cons < 1.0])
    pooled_conf_stubborn.extend(conf[(cons == 1.0) & ~cor])

data_to_plot = [pooled_conf_cor, pooled_conf_waver, pooled_conf_stubborn]
box = ax.boxplot(data_to_plot, patch_artist=True, widths=0.55,
                 boxprops=dict(facecolor="#e0e0e0", color="black"),
                 medianprops=dict(color="red", lw=2))

colors = ['#2ca02c', '#ff7f0e', '#d62728']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticklabels([
    f"Consistent Correct\n(N={len(pooled_conf_cor)})\nMean={np.mean(pooled_conf_cor):.2f}",
    f"Wavering / Fragile\n(N={len(pooled_conf_waver)})\nMean={np.mean(pooled_conf_waver):.2f}",
    f"Stubbornly Wrong\n(N={len(pooled_conf_stubborn)})\nMean={np.mean(pooled_conf_stubborn):.2f}"
], fontsize=9)
ax.set_ylabel("Predicted Option Confidence", fontsize=10)
ax.set_title("The Consistency Trap: Confidence Distributions by Correctness Mode", fontsize=10, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig(fig_dir / "fig_stubborn_confidence_distribution.png", dpi=300)
fig.savefig(fig_dir / "fig_stubborn_confidence_distribution.pdf")
fig.savefig(paper_fig_dir / "fig_stubborn_confidence_distribution.png", dpi=300)
fig.savefig(paper_fig_dir / "fig_stubborn_confidence_distribution.pdf")
plt.close(fig)
print("Saved fig_stubborn_confidence_distribution (.png & .pdf)")

# ------------------------------------------------------------------------------
# FIGURE 3: PUBMEDQA CLASS RECALL SHIFT (Closed vs Context)
# ------------------------------------------------------------------------------
print("Generating Figure 3: fig_pubmedqa_class_recall_shift...")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)

classes = ["yes", "no", "maybe"]
bar_width = 0.35

for i, m in enumerate(models):
    ax = axes[i]
    row_cl = conf_df[(conf_df["model"] == m) & (conf_df["slice"] == "pubmedqa_closed")]
    row_cx = conf_df[(conf_df["model"] == m) & (conf_df["slice"] == "pubmedqa_context")]
    
    rec_cl = [row_cl[row_cl["gold"] == c]["recall"].values[0] for c in classes]
    rec_cx = [row_cx[row_cx["gold"] == c]["recall"].values[0] for c in classes]
    
    x = np.arange(len(classes))
    ax.bar(x - bar_width/2, rec_cl, bar_width, label="Closed-Book", color="#7f7f7f", alpha=0.85)
    ax.bar(x + bar_width/2, rec_cx, bar_width, label="Context-Grounded", color="#1f77b4", alpha=0.85)
    
    ax.set_title(model_labels[m], fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(["Yes (N=276)", "No (N=169)", "Maybe (N=55)"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    if i == 0:
        ax.set_ylabel("Class Recall", fontsize=10)

axes[0].legend(frameon=True, fontsize=8)
plt.tight_layout()
fig.savefig(fig_dir / "fig_pubmedqa_class_recall_shift.png", dpi=300)
fig.savefig(fig_dir / "fig_pubmedqa_class_recall_shift.pdf")
fig.savefig(paper_fig_dir / "fig_pubmedqa_class_recall_shift.png", dpi=300)
fig.savefig(paper_fig_dir / "fig_pubmedqa_class_recall_shift.pdf")
plt.close(fig)
print("Saved fig_pubmedqa_class_recall_shift (.png & .pdf)")
print("\nAll assets successfully generated!")
