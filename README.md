# FactBench: Controlled Multi-Metric Evaluation of Small Open Large Language Models on Scientific Question Answering

[![Tests](https://img.shields.io/badge/pytest-9%20passed-brightgreen.svg)](#52-verification-of-test-suite--safety-executed)
[![Python 3.13](https://img.shields.io/badge/python-3.13.16-blue.svg)](#5-environment--hardware-specifications)
[![Protocol](https://img.shields.io/badge/protocol-frozen%20v1.0-orange.svg)](configs/protocol.yaml)
[![Paper Source](https://img.shields.io/badge/paper-ACM%20LaTeX-red.svg)](paper/)

---

## 1. Project Summary

Evaluation of large language models (LLMs) on scientific benchmarks usually relies on one scalar number: top-1 raw accuracy. **FactBench** is an evaluation harness that asks a narrower question: *do different factuality scores rank the same models in the same order?*

- **Models (3, open weights):** `Qwen2.5-1.5B-Instruct`, `Phi-3.5-mini-instruct`, and `Qwen2.5-7B-Instruct` in 4-bit.
- **Data (5 slices, 950 distinct items):** MMLU biomedical, mathematics and physical sciences (150 questions each), plus 500 PubMedQA questions evaluated twice (closed-book and with the abstract), giving 1,450 item-slice evaluations.
- **Protocol:** each item is scored under 4 deterministic option orders (the canonical order plus 3 shuffles), giving 3 × 1,450 × 4 = **17,400 forward passes**. Answers are read from the probabilities of the option letters; no text is generated.
- **Metrics:** accuracy, multiclass Brier score (range 0 to 2), Expected Calibration Error (ECE), risk-coverage area (AURC), selective accuracy at 80% coverage, confidence discrimination (AUROC<sub>conf</sub>), and permutation consistency (Cons, AUROC<sub>cons</sub>).

**In short:** metrics that embed correctness (Brier, AURC, selective accuracy) track accuracy closely, while calibration error and confidence discrimination agree much less (suggestive only, with 3 models). Closed-book PubMedQA scores on an imbalanced benchmark can reward degenerate class guessing. Permutation consistency does not detect errors that are repeated identically in every order, and these make up over half of all errors.

---

## 2. Key Experimental Findings (from `results/*.csv`)

All numbers below are copied from the result files in `results/`. Where a number was also recomputed from the raw logs, see [Section 4](#4-known-numerical-sensitivities-and-independent-re-check).

### 2.1 Factuality metrics are not interchangeable (`rank_tau_summary.csv`, `results.csv`)

- Metrics that embed correctness (Brier score, AURC, selective accuracy at 80% coverage) track raw accuracy closely: mean within-slice Kendall tau = **0.830**.
- Calibration error (ECE, mean tau = **0.430**) and confidence discrimination (AUROC<sub>conf</sub>, mean tau = **0.297**, 95% bootstrap CI [-0.067, 0.697]) agree much less with accuracy. Lower-is-better metrics (ECE, Brier, AURC) are sign-flipped so that higher always means better.
- With only 3 models per slice, each per-slice tau can take very few values, so the averages are coarse. ECE agrees with the accuracy ranking in 3 of 5 slices; AUROC<sub>conf</sub> disagrees in 2 slices (`mmlu_biomed`, `mmlu_physchem`).
- On `mmlu_biomed`, `Phi-3.5-mini-instruct` has the highest accuracy (0.7733) but the **lowest** AUROC<sub>conf</sub> (0.7031); `Qwen2.5-7B-Instruct-4bit` has lower accuracy (0.7333) but the best AUROC<sub>conf</sub> (0.7560).
- Paired exact McNemar tests (`paired_model_tests.csv`) show the `mmlu_biomed` accuracy differences are **not** significant after Holm correction (adjusted p between 0.3507 and 0.9808). Rank inversions with N = 3 models are suggestive, not conclusive. The core thesis is **partly supported**.
- `Qwen2.5-7B-Instruct-4bit` has higher selective accuracy at 80% coverage than `Phi-3.5-mini-instruct` in **all 5** slices.

### 2.2 Evidence grounding and closed-book class collapse (`task_mode_delta.csv`, `pubmedqa_confusion.csv`)

- Providing the abstract on PubMedQA improves accuracy and calibration:
  - `Phi-3.5-mini-instruct`: accuracy 0.3240 to 0.6600 (delta = +0.3360; exact McNemar p = 1.53e-31, Holm p = 4.58e-31); Brier 1.2530 to 0.6402 (delta = -0.6128).
  - `Qwen2.5-7B-Instruct-4bit`: accuracy 0.4180 to 0.7280 (delta = +0.3100; exact McNemar p = 3.92e-28, Holm p = 7.84e-28); delta AURC = -0.3918.
- **Qwen2.5-1.5B yes-bias.** In closed-book mode it scores 0.5520, higher than both larger models (0.3240 and 0.4180). This is an artifact of degenerate prediction: it answers `yes` for 391 of 500 questions (78.20%), and `yes` is the true label for 55.2% of the items. Its recall on `maybe` is 1.82% (0.00% with context); balanced accuracy is only 0.372 (`pubmedqa_balanced_POSTHOC.csv`, post hoc).
- **Phi-3.5-mini "maybe" bias (new).** In closed-book mode `Phi-3.5-mini-instruct` predicts `maybe` for 276 of 500 questions (55.2%), `yes` for 205 (41.0%) and `no` for only 19 (3.8%), although `maybe` is the true label for just 11.0% of items. Its recall is 54.5% on `maybe`, 43.8% on `yes` and 6.5% on `no`; balanced accuracy is 0.350, barely above the 1/3 chance level. With the abstract, its `maybe` predictions fall to 140 and its `no` recall rises to 58.0%.
- **`no` is the most under-predicted class without context** for all three models: 19 (Phi), 93 (Qwen-1.5B) and 20 (Qwen-7B) `no` predictions against 169 true `no` items. Closed-book collapse can therefore go to a hedge class as well as to the majority class; class-balanced metrics are needed in both cases.

### 2.3 The consistency trap (`results.csv`, `table_stubborn_errors_breakdown.csv`)

- Accuracy generally rises with permutation consistency, but consistency cannot detect an error that never changes.
- Across the 15 cells (4,350 evaluations) there are **1,029 stubborn errors** (the same wrong answer under all 4 option orders), i.e. **54.6% of all 1,885 errors**.
- Stubborn errors carry high confidence: mean 0.9490 overall (0.8130 to 1.0000 across cells; 1.0000 for Phi-3.5 on biomedicine).
- On closed-book PubMedQA for `Phi-3.5-mini-instruct`, consistency is inversely related to accuracy: AUROC<sub>cons</sub> = 0.4319 (95% CI [0.3834, 0.4824]), because the model makes 219 stubborn errors versus only 82 consistently correct answers.

### 2.4 Other descriptive results

- **Position bias** (`position_bias.csv`): maximum deviation from uniform option selection on MMLU is 2.7 points for Qwen-7B, 9.0 for Phi-3.5-mini and 13.2 for Qwen-1.5B.
- **Permutation ensembling** (exploratory): averaging probabilities over the 4 orders improves accuracy in 10 of 15 cells, leaves 2 unchanged and lowers 3; it cannot fix errors that are identical in every order.

---

## 3. Raw Output Format (`results/raw/`)

`results/raw/` holds 15 files named `<model>__<slice>.jsonl` (for example `qwen2_5-7b-instruct-4bit__mmlu_math.jsonl`; note underscores in the file names) and 3 `<model>__meta.json` files. Each JSONL line is one item:

| Field | Meaning |
|---|---|
| `id` | Item ID (for example `mmlu-anatomy-25`, `pubmedqa-10966337`) |
| `slice` / `model` | Slice name and model name |
| `gold` | Index of the correct option **in the original option order** (PubMedQA: 0 = yes, 1 = no, 2 = maybe) |
| `perms` | 4 permutations; `perms[k][j]` = original option shown at display position `j` in order `k` (`k = 0` is the canonical order) |
| `logp` | 4 vectors; `logp[k][j]` = log-probability of display letter `j` (restricted softmax, each vector sums to 1) |

The files contain no question text, option text or prompts. All 3 models see the same 4 permutations for a given item.

```python
import json, numpy as np

rows = [json.loads(l) for l in open("results/raw/phi-3_5-mini-instruct__mmlu_biomed.jsonl")]
r, k = rows[0], 0                               # k = 0: canonical order
j = int(np.argmax(r["logp"][k]))                # display position with the highest probability
pred_option = r["perms"][k][j]                  # prediction mapped back to the original option order
confidence  = float(np.exp(r["logp"][k][j]))
correct     = pred_option == r["gold"]
```

---

## 4. Known Numerical Sensitivities and Independent Re-check

An independent recomputation from `results/raw/*.jsonl` (not part of the pipeline) reproduced the following **exactly**: accuracy, ECE, Brier, SelAcc<sub>0.8</sub> and AUROC<sub>conf</sub> (75 of 75 values in the main results table), all Holm-adjusted McNemar tests, the PubMedQA class counts, recalls, balanced accuracy and macro-F1, position-bias frequencies, permutation-ensembling accuracies, and the three qualitative failure cases. The bootstrap CIs are consistent with a 1,000-resample percentile bootstrap.

The saved `logp` values are rounded, and 30 of the 17,400 vectors contain an exact tie for the maximum. Depending on the tie-breaking rule this causes small differences in tie-sensitive quantities:

| Quantity | Reported | Recomputed from raw (first-index tie rule) |
|---|---|---|
| Consistent correct / wavering / stubborn wrong | 1,942 / 1,379 / 1,029 | 1,942 / 1,377 / 1,031 (stubborn share 54.6% vs 54.7%) |
| Mean tau, accuracy vs Cons<sub>mean</sub> | 0.667 | 0.630 (one `mmlu_physchem` near-tie: 0.8233 vs 0.8233) |
| AURC | see `results.csv` | differs by up to 0.015 |

AURC is genuinely tie-sensitive when many predictions are saturated at probability 1.0 (for example 101 of 150 items for Phi on `mmlu_biomed`, where AURC ranges from 0.073 to 0.281 between best-case and worst-case tie ordering). Selective accuracy at 80% coverage and the other main metrics are unaffected. **Recommendation:** store full-precision logits and document the tie-breaking rule.

---

## 5. Environment & Hardware Specifications

Recorded in `results/raw/*__meta.json`: model name and Hugging Face ID, backend (`hf`), `quant` (`none`, `none`, `4bit`), protocol version `1.0`, seed 42, K = 4, batch size 8, prompt style `chat`, Python 3.13.16, PyTorch 2.14.1+cu130, Transformers 4.45.2, GPU `Tesla T4`, and start timestamps. `git_commit` is `null` in all three files and `limit` is `null` (full datasets).

- **Hardware:** NVIDIA Tesla T4 GPU.
- **Inference:** batch size 8, deterministic option orders K = 4, seed 42.
- **Precision / quantization** (`[TO VERIFY: dtype and bitsandbytes settings are not stored in meta.json; confirm against scripts/02_run_models.py]`):
  - `Qwen2.5-1.5B-Instruct`: `float16`, `quant: "none"`.
  - `Phi-3.5-mini-instruct`: `float16`, `quant: "none"`.
  - `Qwen2.5-7B-Instruct-4bit`: 4-bit NormalFloat (NF4) via `bitsandbytes` (`load_in_4bit=True`, `bnb_4bit_compute_dtype=torch.float16`, `bnb_4bit_quant_type="nf4"`, no double quantization).
- **Dependency lock:** `[TO VERIFY: requirements.lock.txt missing from repository; dependencies defined in requirements.txt]`.
- **Operating system and GPU hours:** `[TO VERIFY: host OS and cumulative GPU hours not recorded; start timestamps 2026-11-12T11:13:40 (Qwen-1.5B), 11:21:03 (Phi), 11:39:39 (Qwen-7B)]`.
- **Code version:** `[TO VERIFY: git_commit is null in meta.json; record the commit hash of the run]`.

---

## 6. Repository Layout

```text
AITR_Lab02_FactBench/
├── README.md                      # Project overview, findings & run instructions
├── requirements.txt               # Pinned base Python dependencies
├── AGENTS.md                      # Research integrity guidelines for AI agents
├── CLEANUP_PLAN.md                # Tree audit ledger: KEEP vs MOVE rationale
├── FINAL_CHECKLIST.md             # Course Lab Instructions §12 compliance audit
├── research_decisions.md          # Gate G1 research brief, decision log (D1-D6), evidence ledger (E1-E6)
├── baseline_validation.md         # Gate G2 replication vs lm-evaluation-harness
├── ai_assistance_log.md           # Gate G4 human-AI collaboration log
├── configs/
│   └── protocol.yaml              # Frozen experimental protocol (v1.0, seed 42)
├── data/
│   ├── README.md                  # Dataset access terms, licensing, governance
│   └── processed/
│       ├── MANIFEST.json          # SHA-256 hashes and sampling parameters
│       ├── mmlu.jsonl             # 450 items (150 per scientific slice, seed 42) [gitignored]
│       └── pubmedqa.jsonl         # 500 items, class-proportional sample [gitignored]
├── src/factbench/                 # Core evaluation library
│   ├── __init__.py
│   ├── data.py                    # Data loading, deterministic perms, jsonl I/O
│   ├── scoring.py                 # Restricted option-logit softmax forward pass
│   ├── metrics.py                 # Brier, ECE, AURC, AUROC, Cons
│   ├── prompts.py                 # Chat template formatting & option permutation strings
│   └── analysis.py                # Bootstrap CIs, McNemar exact tests, Kendall's tau
├── scripts/
│   ├── 00_env_check.py            # Environment, CUDA, and tokenizer validation
│   ├── 01_prepare_data.py         # Deterministic sampling & manifest creation
│   ├── 02_run_models.py           # Model inference with batching & resume support
│   ├── 03_evaluate.py             # Metrics, CIs, paired tests, failure tables
│   ├── 04_figures.py              # Publication figures (PNG + PDF)
│   ├── 05_validate_baseline.py    # Replication against lm-evaluation-harness
│   ├── 05_harness_check.sh        # Shell comparison runner
│   ├── 06_paper_tables.py         # LaTeX table generator
│   ├── 07_verify_paper_claims.py  # Read-only audit of paper claims vs result CSVs
│   ├── _common.py                 # Path and CLI helpers
│   ├── build_paper_pdf.py         # Standalone Python PDF generator
│   ├── diagnostic_audit.py        # Stubborn-error distributions
│   ├── generate_presentation_assets.py # Presentation tables and distribution assets
│   ├── safety_scan.py             # Secret, file-size, and license scanner
│   └── show_prompts.py            # Display permuted prompt strings
├── tests/
│   ├── test_metrics.py            # Unit tests for all metrics against known values
│   └── test_smoke_pipeline.py     # End-to-end smoke test
├── results/                       # Experimental outputs (no invented numbers)
│   ├── results.csv                # Metrics with 95% bootstrap CIs
│   ├── baselines.csv              # Chance and majority-class baselines
│   ├── rank_by_metric.csv         # Model rankings per metric
│   ├── rank_tau_summary.csv       # Mean within-slice Kendall's tau
│   ├── task_mode_delta.csv        # Closed vs context paired McNemar tests
│   ├── paired_model_tests.csv     # Holm-adjusted McNemar tests between models
│   ├── position_bias.csv          # Option selection frequencies vs gold shares
│   ├── pubmedqa_confusion.csv     # Per-class precision, recall, confusion matrix
│   ├── pubmedqa_balanced_POSTHOC.csv  # Post-hoc balanced accuracy and macro-F1
│   ├── failure_cases_consistently_wrong.csv  # Capped stubborn failure cases (50/cell)
│   ├── table_stubborn_errors_breakdown.csv   # Uncapped stubborn-error breakdown
│   ├── table_pubmedqa_deep_dive.csv          # PubMedQA class-recall and McNemar breakdown
│   ├── figures/                   # Vector PDF figures
│   ├── tables/                    # Generated LaTeX tables
│   ├── validation/                # Replication outputs vs lm-evaluation-harness
│   ├── logs/                      # Execution logs (g2_log.txt)
│   └── raw/                       # 15 raw inference JSONL files + 3 __meta.json files
└── paper/                         # Overleaf-ready LaTeX paper package
    ├── main.tex                   # ACM-formatted manuscript
    ├── refs.bib                   # BibTeX references with verified DOIs
    ├── figures/                   # Figure assets (vector PDF + PNG)
    ├── tables/                    # LaTeX table inputs (8 tables)
    ├── claim_check.md             # Quantitative claim audit ledger
    └── CHANGELOG.md               # Manuscript revision ledger
```

---

## 7. Reproduction and Execution Guide

### 7.1 Environment Setup
```bash
git clone https://github.com/AmanKumar-iiita/FactBench.git
cd FactBench
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2 Verification of Test Suite & Safety (Executed)
```bash
python tests/test_metrics.py        # PASS all 8 tests
python tests/test_smoke_pipeline.py # PASS smoke
python -m pytest -q tests           # 9 passed in 9.39s
python scripts/safety_scan.py       # 0 files > 50MB, 0 secrets, 0 GPQA files found
```

### 7.3 Baseline Validation Replication (Executed)
```bash
# Tokenizer vocab mapping check (logged in results/logs/g2_log.txt)
python scripts/00_env_check.py --tokenizers

# Replication against lm-evaluation-harness (logged in results/logs/g2_log.txt)
python scripts/05_validate_baseline.py --model-name qwen2.5-1.5b-instruct
# college_biology diff -0.69% (1 item); high_school_physics diff +0.66% (1 item); both within 0.18 SE
```

### 7.4 Data Preparation and Model Inference
```bash
# Deterministic sampling from MMLU and PubMedQA ([TEST REQUIRED: verify outputs against MANIFEST.json])
python scripts/01_prepare_data.py

# Inference across the 15 cells ([TEST REQUIRED: needs a GPU; raw outputs are preserved in results/raw/])
for model in qwen2.5-1.5b-instruct phi-3.5-mini-instruct qwen2.5-7b-instruct-4bit; do
  python scripts/02_run_models.py --model-name $model
done
```

### 7.5 Regenerate Tables, Figures, and the Paper (Executed)
```bash
python scripts/03_evaluate.py            # metrics, CIs, CSV outputs
python scripts/04_figures.py             # vector figures
python scripts/06_paper_tables.py        # LaTeX tables into results/tables/ and paper/tables/
python scripts/07_verify_paper_claims.py # check paper claims against results CSVs
python scripts/build_paper_pdf.py        # standalone paper PDF
# Output: paper/paper_factbench_final.pdf (10 pages, 1,243,336 bytes)
```

For the ACM-formatted version, compile `paper/main.tex` with `pdflatex` + `bibtex` (class `acmart`).

---

## 8. Limitations and Methodological Boundaries

1. **Model sample size (N = 3).** Three open-weights models (1.5B to 7B) give wide confidence intervals for Kendall's tau; biomedical MMLU accuracy differences are not significant after Holm correction. Rank inversions are suggestive.
2. **Quantization confound.** `Qwen2.5-7B` was run in 4-bit NF4 (VRAM limits) while the other two models ran in `float16`, so comparisons with the 7B model mix scale and quantization effects.
3. **Prompt sensitivity.** One standardized chat template per dataset; prompt framing was not varied.
4. **Resampling unit.** Bootstrap CIs resample test items, not training runs.
5. **Benchmark contamination.** Pre-training corpora were not audited.
6. **Task scope.** Multiple-choice QA only; no open-ended generation, multi-step proofs or clinical dialogue.
7. **Tie sensitivity.** Rounded saved log-probabilities create exact ties; AURC and consistency-based counts depend slightly on tie-breaking (see [Section 4](#4-known-numerical-sensitivities-and-independent-re-check)).
8. **Small slices.** The MMLU slices have 150 questions each (95% CIs of roughly ±7 points).

---

## 9. Dataset and Model Licenses

- **MMLU:** MIT License .
- **PubMedQA:** MIT License .
- **Qwen2.5 models:** Apache 2.0.
- **Phi-3.5-mini-instruct:** MIT.
- **GPQA:** excluded entirely; its terms forbid public distribution of items (`[TO VERIFY: Idavidrein/gpqa gating agreement]`).

---

## 10. Ethics and AI-Use Disclosure

AI assistance was used for pipeline scaffolding, cross-platform debugging (UTF-8 file encoding, CUDA batch limits), statistical scripting, and manuscript auditing. In line with `AGENTS.md` and the course integrity rules:
- No metrics, logs, p-values or citations were fabricated.
- All reported numbers were computed and verified from outputs in `results/`.
- The human author reviewed and takes full responsibility for all submitted code, results and text.

---

## 11. Citation & Repository License

```bibtex
@misc{kumar2026factbench,
  title        = {Are Factuality Scores Interchangeable? A Controlled Multi-Metric Evaluation of Small Open Large Language Models on Scientific Question Answering},
  author       = {Kumar, Aman},
  year         = {2026},
  institution  = {Indian Institute of Information Technology, Allahabad},
  howpublished = {\url{https://github.com/AmanKumar-iiita/FactBench}}
}
```

*License notice:* no repository license has been committed yet. Until one is added, the code is "all rights reserved" by default; choose MIT, Apache 2.0 or CC-BY-4.0 before making the repository public.
