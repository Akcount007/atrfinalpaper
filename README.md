# FactBench: Controlled Multi-Metric Evaluation of Small Open Large Language Models on Scientific Question Answering

[![Tests](https://img.shields.io/badge/pytest-9%20passed-brightgreen.svg)](#72-verification-of-test-suite--safety-executed)
[![Python 3.13](https://img.shields.io/badge/python-3.13.16-blue.svg)](#5-environment--hardware-specifications)
[![Protocol](https://img.shields.io/badge/protocol-fixed%20v1.0-orange.svg)](configs/protocol.yaml)
[![Paper Source](https://img.shields.io/badge/paper-ACM%20LaTeX-red.svg)](paper/)

---

## 1. Executive Summary & Research Questions

Standard benchmarking of large language models (LLMs) on scientific benchmarks commonly relies on a single scalar metric: **top-1 raw accuracy**. In safety-critical scientific and biomedical disciplines, raw accuracy fails to differentiate whether a model's assigned confidence tracks truth, whether it can safely abstain under epistemic uncertainty, or whether its apparent accuracy masks severe majority-class prediction heuristics.

**FactBench** evaluates whether factuality metrics rank small open-weights models concordantly across scientific domains. We evaluate three open-weights instruction-tuned models across five scientific slices (950 distinct items; 1,450 item-slice evaluations) and four deterministic option orders, totaling **17,400 forward passes** under a fixed experimental protocol (`configs/protocol.yaml`, version 1.0, seed 42).

### Core Research Questions
- **RQ1 (Multi-Metric Concordance):** Do model rankings established by raw accuracy agree with rankings under probability calibration (Brier score, ECE), selective prediction (accuracy at 80% coverage), and confidence discrimination ($\text{AUROC}_{\text{conf}}$), or do different metrics produce rank discordance?
- **RQ2 (Evidence Grounding Shifts):** Does providing primary research context (PubMedQA context-grounded vs. closed-book) alter accuracy, calibration, and class-level recall uniformly, or does closed-book evaluation reward degenerate heuristics?
- **RQ3 (The Consistency Trap):** Is option-permutation consistency a reliable surrogate for factual correctness, or does consistency reward stubbornly held errors?

---

## 2. Benchmark Design & Experimental Protocol

### 2.1 Scientific Benchmark Slices
The evaluation spans five scientific slices covering 950 distinct questions:
1. **Biomedical MMLU (`mmlu_biomed`):** $N=150$ questions sampled uniformly at random from pooled subjects across 8 MMLU biomedical disciplines (anatomy, clinical knowledge, college biology, college medicine, high school biology, medical genetics, professional medicine, virology). 4 candidate choices; closed-book ($\emptyset$).
2. **Mathematics MMLU (`mmlu_math`):** $N=150$ questions sampled uniformly at random from pooled subjects across 4 MMLU mathematics disciplines (abstract algebra, college mathematics, elementary mathematics, high school mathematics). 4 candidate choices; closed-book ($\emptyset$).
3. **Physical Sciences MMLU (`mmlu_physchem`):** $N=150$ questions sampled uniformly at random from pooled subjects across 6 MMLU physical science disciplines (astronomy, college chemistry, college physics, conceptual physics, high school chemistry, high school physics). 4 candidate choices; closed-book ($\emptyset$).
4. **PubMedQA Closed-Book (`pubmedqa_closed`):** $N=500$ biomedical research questions evaluated without abstract context ($c = \emptyset$). Class-proportional sample of 500 from the 1,000 labelled items (official test split not used). 3 candidate choices (`yes`, `no`, `maybe`).
5. **PubMedQA Context-Grounded (`pubmedqa_context`):** The identical $N=500$ PubMedQA items evaluated with the full published abstract text provided ($c = \text{abstract}$, truncated at 6,000 characters).

| Slice | Domain / Source | Items ($N$) | Options ($m$) | Context ($c$) | Passes / Model ($N \times 4$) | Total Passes (3 Models) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `mmlu_biomed` | Biomedicine (MMLU) | 150 | 4 | $\emptyset$ (Closed) | 600 | 1,800 |
| `mmlu_math` | Mathematics (MMLU) | 150 | 4 | $\emptyset$ (Closed) | 600 | 1,800 |
| `mmlu_physchem` | Physics & Chem (MMLU) | 150 | 4 | $\emptyset$ (Closed) | 600 | 1,800 |
| `pubmedqa_closed` | Clinical QA (PubMedQA) | 500 | 3 | $\emptyset$ (Closed) | 2,000 | 6,000 |
| `pubmedqa_context` | Clinical QA (PubMedQA) | 500 | 3 | Published Abstract | 2,000 | 6,000 |
| **Total Benchmark** | **5 Slices (950 distinct)** | **1,450** | **3--4** | -- | **5,800** | **17,400** |

### 2.2 Evaluated Models
Three instruction-tuned models with approximately 1.5B, 3.8B, and 7B parameters:
- **`Qwen2.5-1.5B-Instruct`**: Evaluated in full 16-bit floating point (`float16`).
- **`Phi-3.5-mini-instruct`**: Evaluated in full 16-bit floating point (`float16`).
- **`Qwen2.5-7B-Instruct-4bit`**: Quantized to 4-bit NormalFloat (NF4) via `bitsandbytes` using library defaults (`load_in_4bit=True`, `bnb_4bit_compute_dtype=torch.float16`, `bnb_4bit_quant_type="nf4"`; `bnb_4bit_use_double_quant` left at default `False`, no double quantization).

### 2.3 Deterministic Option Permutation & Consistency
To evaluate presentation stability, every question is scored under $K = 4$ deterministic option orderings: canonical order ($\pi_0$) plus three pseudo-random permutations seeded deterministically by `seed = 42 + CRC32(item_id)`.
- For permutation $\pi_k$, the predicted letter is mapped back to original option index: $\hat{o}_k = \pi_k^{-1}(\hat{y}_{\pi_k})$.
- **Permutation Consistency ($\text{Cons} \in [1/K, 1.0]$):**
  $$\text{Cons} = \frac{1}{K} \max_v \sum_{k=1}^K \mathbb{I}[\hat{o}_k = v]$$
- **Item Categorization:**
  - *Consistent Correct:* $\text{Cons} = 1.0$ and correct on $\pi_0$.
  - *Wavering:* $\text{Cons} < 1.0$ (prediction changes across orders).
  - *Stubbornly Wrong:* $\text{Cons} = 1.0$ and incorrect on $\pi_0$.

### 2.4 Mathematical Scoring: Restricted Softmax
Inference is executed strictly via constrained option scoring in a single forward pass without autoregressive text generation. Next-token unnormalized logits $z(t)$ are evaluated over candidate letter tokens $\mathcal{L}$.
Because subword tokenizers (e.g., Byte-Pair Encoding in Qwen and SentencePiece in Phi-3.5) assign distinct vocabulary IDs to leading-space variations (such as `"A"` versus `" A"`), we compute the log-sum-exp over all token IDs mapped to letter $L$:
$$\tilde{z}(L) = \log \sum_{t \in \text{tokens}(L)} \exp(z(t))$$
Option probabilities are normalized via softmax over the candidate set:
$$p(L) = \frac{\exp(\tilde{z}(L))}{\sum_{L' \in \mathcal{L}_m} \exp(\tilde{z}(L'))}$$
The predicted choice is $\hat{y} = \arg\max_L p(L)$, and confidence is $\hat{c} = \max_L p(L)$.

---

## 3. Evaluated Metrics & Statistical Framework

We compute eight core evaluation metrics:
1. **Raw Accuracy ($\text{Acc} \uparrow$):** Top-1 correctness on canonical order $\pi_0$: $\frac{1}{N} \sum_{i=1}^N \mathbf{1}[\hat{o}_{i, 0} = y_i^*]$.
2. **Multiclass Brier Score ($\text{Brier} \downarrow$):** Mean squared error over the option probability vector relative to one-hot gold targets: $\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^m (p_i(j) - y_{i,j}^*)^2 \in [0, 2]$.
3. **Expected Calibration Error ($\text{ECE} \downarrow$):** Partitioning predicted confidences $\hat{c}_i$ into $M=10$ equal-width bins $B_1, \dots, B_M \subset (0, 1]$:
   $$\text{ECE} = \sum_{b=1}^M \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$
4. **Area Under Risk-Coverage Curve ($\text{AURC} \downarrow$):** Selective prediction error under confidence-thresholded coverage $\kappa(\theta)$: $\text{AURC} = \int_0^1 \mathcal{R}(\kappa) d\kappa$.
5. **Selective Accuracy ($\text{SelAcc}_{0.8} \uparrow$):** Accuracy achieved when answering only the top 80% most confident items.
6. **Confidence Discrimination ($\text{AUROC}_{\text{conf}} \uparrow$):** Area under the ROC curve using predicted confidence $\hat{c}$ to separate correct from incorrect predictions.
7. **Consistency Discrimination ($\text{AUROC}_{\text{cons}} \uparrow$):** Area under the ROC curve using permutation consistency $\text{Cons}$ to separate correct from incorrect predictions.
8. **Mean Consistency ($\text{Cons}_{\text{mean}} \uparrow$):** Sample average of $\text{Cons}$ over all items in a slice: $\frac{1}{N} \sum_{i=1}^N \text{Cons}_i$.

### Statistical Inference Tests
- **Bootstrap Confidence Intervals:** Two-sided 95% non-parametric bootstrap intervals ($B = 1,000$ resamples over test items).
- **Paired Hypothesis Tests:** Exact two-sided McNemar tests on discordant item pairs between models and between task modes.
- **Multiple Testing Correction:** Family-wise error rate controlled via Holm-Bonferroni step-down adjustment.
- **Rank Concordance:** Mean within-slice Kendall's rank correlation $\tau$ across the 5 slices.

---

## 4. Key Experimental Findings

All reported numbers are verified directly against `results/*.csv` and execution telemetry.

### 4.1 Primary Metric Results (Table 2 in Paper)

| Model | Slice | $N$ | Acc [95% CI] | ECE | Brier | AURC | $\text{SelAcc}_{0.8}$ | $\text{AUROC}_{\text{conf}}$ | Consistency |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `phi-3.5-mini-instruct` | `mmlu_biomed` | 150 | 0.773 [0.707, 0.833] | 0.222 | 0.440 | 0.114 | 0.800 | 0.703 | 0.922 |
| `qwen2.5-1.5b-instruct` | `mmlu_biomed` | 150 | 0.693 [0.620, 0.760] | 0.216 | 0.524 | 0.157 | 0.733 | 0.725 | 0.878 |
| `qwen2.5-7b-instruct-4bit`| `mmlu_biomed` | 150 | 0.733 [0.660, 0.807] | 0.242 | 0.492 | 0.113 | 0.825 | 0.756 | 0.933 |
| `phi-3.5-mini-instruct` | `mmlu_math` | 150 | 0.473 [0.393, 0.553] | 0.424 | 0.881 | 0.350 | 0.525 | 0.721 | 0.747 |
| `qwen2.5-1.5b-instruct` | `mmlu_math` | 150 | 0.380 [0.307, 0.453] | 0.315 | 0.858 | 0.492 | 0.417 | 0.635 | 0.683 |
| `qwen2.5-7b-instruct-4bit`| `mmlu_math` | 150 | 0.567 [0.487, 0.647] | 0.334 | 0.746 | 0.234 | 0.633 | 0.750 | 0.822 |
| `phi-3.5-mini-instruct` | `mmlu_physchem` | 150 | 0.573 [0.493, 0.653] | 0.366 | 0.766 | 0.219 | 0.633 | 0.760 | 0.823 |
| `qwen2.5-1.5b-instruct` | `mmlu_physchem` | 150 | 0.453 [0.373, 0.533] | 0.398 | 0.839 | 0.331 | 0.508 | 0.758 | 0.822 |
| `qwen2.5-7b-instruct-4bit`| `mmlu_physchem` | 150 | 0.647 [0.573, 0.727] | 0.278 | 0.615 | 0.171 | 0.717 | 0.740 | 0.837 |
| `phi-3.5-mini-instruct` | `pubmedqa_closed` | 500 | 0.324 [0.282, 0.368] | 0.601 | 1.253 | 0.703 | 0.305 | 0.456 | 0.873 |
| `qwen2.5-1.5b-instruct` | `pubmedqa_closed` | 500 | 0.552 [0.508, 0.596] | 0.305 | 0.749 | 0.416 | 0.570 | 0.569 | 0.921 |
| `qwen2.5-7b-instruct-4bit`| `pubmedqa_closed` | 500 | 0.418 [0.374, 0.464] | 0.506 | 1.054 | 0.518 | 0.440 | 0.592 | 0.877 |
| `phi-3.5-mini-instruct` | `pubmedqa_context`| 500 | 0.660 [0.616, 0.704] | 0.296 | 0.640 | 0.213 | 0.703 | 0.693 | 0.911 |
| `qwen2.5-1.5b-instruct` | `pubmedqa_context`| 500 | 0.660 [0.622, 0.704] | 0.288 | 0.626 | 0.217 | 0.682 | 0.649 | 0.964 |
| `qwen2.5-7b-instruct-4bit`| `pubmedqa_context`| 500 | 0.728 [0.684, 0.770] | 0.231 | 0.483 | 0.126 | 0.795 | 0.761 | 0.946 |

### 4.2 RQ1: Metric Discordance & Rank Inversions (`results/rank_tau_summary.csv`)
- **Concordant Metrics:** Metrics embedding correctness directly track raw accuracy closely (mean within-slice Kendall $\tau = 0.830$ for Brier Score, AURC, and $\text{SelAcc}_{0.8}$).
- **Discordant Metrics:** Probability calibration error (ECE, $\tau = 0.430$) and confidence discrimination ($\text{AUROC}_{\text{conf}}$, $\tau = 0.297$, 95% CI: $[-0.067, 0.697]$) diverge substantially from accuracy rankings.
- **Biomedical Rank Inversion:** On `mmlu_biomed`, `Phi-3.5-mini` achieves highest accuracy (0.773) but lowest confidence discrimination ($\text{AUROC}_{\text{conf}} = 0.703$). Conversely, `Qwen2.5-7B` achieves lower accuracy (0.733) but highest discrimination ($\text{AUROC}_{\text{conf}} = 0.756$).
- **Statistical Significance & Overlap Check:**
  - Paired exact McNemar tests (`results/paired_model_tests.csv`) show accuracy differences on `mmlu_biomed` are not statistically significant after Holm correction ($p \in [0.351, 0.981]$).
  - For all model pairs whose ranks flip between accuracy and AUROC (`mmlu_biomed`, `mmlu_physchem`, `pubmedqa_closed`), **their 95% bootstrap AUROC confidence intervals overlap** (e.g., biomed AUROC CIs: Phi $[0.602, 0.789]$, Qwen-1.5B $[0.638, 0.808]$, Qwen-7B $[0.665, 0.842]$).
- **Quantization Observation:** Despite 4-bit NF4 quantization, `Qwen2.5-7B-Instruct-4bit` achieves the highest selective accuracy in 4 of 5 slices (`mmlu_biomed`: 0.825, `mmlu_math`: 0.633, `mmlu_physchem`: 0.717, `pubmedqa_context`: 0.795; `Qwen2.5-1.5B` leads closed PubMedQA at 0.570).

### 4.3 RQ2: Evidence Grounding Shifts & Strong Yes-Bias (`results/task_mode_delta.csv`)
Providing primary abstract context on PubMedQA rescues model performance:
- `Phi-3.5-mini-instruct`: Accuracy increases by **+33.6 points** ($0.324 \to 0.660$; paired McNemar $p = 1.53 \times 10^{-31}$, Holm-adjusted $p = 4.58 \times 10^{-31}$); $\Delta\text{ECE} = -0.305$ (0.601 to 0.296); $\Delta\text{Brier} = -0.613$ (1.253 to 0.640); $\Delta\text{AURC} = -0.490$.
- `Qwen2.5-7B-Instruct-4bit`: Accuracy increases by **+31.0 points** ($0.418 \to 0.728$; paired McNemar $p = 3.92 \times 10^{-28}$, Holm-adjusted $p = 7.84 \times 10^{-28}$); $\Delta\text{ECE} = -0.275$; $\Delta\text{Brier} = -0.571$; $\Delta\text{AURC} = -0.392$.
- **Qwen2.5-1.5B Strong Yes-Bias:** In closed-book mode, `Qwen2.5-1.5B` achieves an apparent accuracy of 55.2%, exceeding both larger models. Confusion matrix analysis (`results/pubmedqa_confusion.csv`) shows this is an artifact of degenerate affirmative prediction:
  - It predicts `yes` for 391 of 500 items (78.2%), matching the 55.2% majority class share purely by coincidence.
  - Recall on `no` is 26.0% (44/169) and on `maybe` is only 1.8% (1/55).
  - Balanced accuracy is only 0.372 [0.342, 0.401] and macro-F1 is 0.352 [0.315, 0.391] (`results/pubmedqa_balanced_POSTHOC.csv`).
  - With context, its affirmative bias intensifies: answers `yes` to 413 items (82.6%), reaching 95.3% recall on `yes` but collapsing completely on `maybe` (0.0% recall, 0/55).
- **Phi-3.5 Closed-Book Hedge Bias:** In closed-book mode, `Phi-3.5-mini` defaults to `maybe` for 276 of 500 questions (55.2%), though `maybe` represents only 11.0% of gold labels (balanced accuracy: 0.350). With context, its `maybe` predictions drop to 140 and `no` recall jumps from 6.5% to 58.0%.

### 4.4 RQ3: The Consistency Trap & Stubborn Errors
- **Stubborn Error Prevalence:** Across all 15 cells (4,350 evaluations), we identify **1,029 stubborn wrong instances** (models produce the exact same wrong choice across all 4 permutations). Stubborn errors constitute **54.6% of all 1,885 model errors** ($1,029 / 1,885$).
- **High Saturated Confidence:** Stubborn errors carry extreme confidence: overall mean confidence is **0.949**, with cell means ranging from 0.813 to 1.000 (reaching 1.000 for Phi-3.5 on biomedicine).
- **Inverted Consistency Discrimination:** On closed-book PubMedQA for `Phi-3.5-mini`, $\text{AUROC}_{\text{cons}} = 0.432$ [0.383, 0.482] (below chance). Fully consistent items are more likely to be wrong than wavering items because the model commits 219 stubborn errors versus only 82 consistently correct predictions (only 27.2% correct at $\text{Cons} = 1.0$).
- **Evaluation Item Counts by Consistency Level:**
  - $\text{Cons} = 0.25$: 5 items
  - $\text{Cons} = 0.50$: 548 items
  - $\text{Cons} = 0.75$: 826 items
  - $\text{Cons} = 1.00$: 2,971 items (1,942 consistent correct + 1,029 stubborn wrong)

### 4.5 Positional Bias & Ensembling Audits
- **Display Position Preference (`results/position_bias.csv`):**
  - Uniform selection for 4-choice MMLU is 25.0%.
  - `Qwen2.5-7B-Instruct-4bit` exhibits the smallest position bias: max deviation from uniform is 2.7 points on math (position B at 27.7%), 2.2 points on physchem (position C at 22.8%), and 1.2 points on biomedicine (position B at 26.2%).
  - In contrast, max deviations are 9.0 points for `Phi-3.5-mini` (math position B at 34.0%) and 13.2 points for `Qwen2.5-1.5B` (math position B at 38.2%).
- **Permutation Ensembling (`Acc}_{\text{ens}`):** Averaging softmax probabilities over the 4 permutations improves accuracy in 10 cells, produces no change in 2 cells, and lowers accuracy in 3 cells (deltas from $-0.027$ to $+0.067$). Permutation voting cannot correct stubborn errors since the predicted distribution is identical across orderings.

---

## 5. Environment & Hardware Specifications

Recorded in `results/raw/*__meta.json`:
- **Hardware:** NVIDIA Tesla T4 GPU (`gpu: "Tesla T4"`).
- **Software Stack:** Python 3.13.16, PyTorch 2.14.1+cu130, Hugging Face Transformers 4.45.2.
- **Inference Parameters:** Batch size 8, deterministic option orders $K = 4$, seed 42, protocol version 1.0.
- **Precision / Quantization:**
  - `Qwen2.5-1.5B-Instruct`: `float16`, `quant: "none"`.
  - `Phi-3.5-mini-instruct`: `float16`, `quant: "none"`.
  - `Qwen2.5-7B-Instruct-4bit`: 4-bit NormalFloat (NF4) via `bitsandbytes` (`load_in_4bit=True`, `bnb_4bit_compute_dtype=torch.float16`, `bnb_4bit_quant_type="nf4"`; `bnb_4bit_use_double_quant` left at default `False`, no double quantization).

---

## 6. Repository Layout & Deliverables

```text
AITR_Lab02_FactBench/
├── README.md                      # Comprehensive project overview, findings & run instructions
├── requirements.txt               # Pinned base Python dependencies
├── AGENTS.md                      # Non-negotiable research integrity guidelines for AI agents
├── CLEANUP_PLAN.md                # Tree audit ledger: KEEP vs REMOVE rationale
├── FINAL_CHECKLIST.md             # Course Lab Instructions compliance audit
├── research_decisions.md          # Gate G1 research brief, decision log (D1-D6), evidence ledger (E1-E6)
├── baseline_validation.md         # Gate G2 reference replication vs lm-evaluation-harness
├── ai_assistance_log.md           # Gate G4 human-AI collaboration log & integrity reflections
├── configs/
│   └── protocol.yaml              # Fixed experimental protocol (v1.0, seed 42)
├── data/
│   ├── README.md                  # Dataset access terms, licensing, and governance
│   └── processed/
│       ├── MANIFEST.json          # Cryptographic SHA-256 hashes and sampling parameters
│       ├── mmlu.jsonl             # 450 items (150 per scientific slice, seed 42) [gitignored]
│       └── pubmedqa.jsonl         # 500 items class-proportional sample [gitignored]
├── src/factbench/                 # Core modular evaluation library
│   ├── __init__.py                # Package initialization
│   ├── data.py                    # Data loading, deterministic perms, jsonl I/O
│   ├── scoring.py                 # Constrained option logit softmax forward pass
│   ├── metrics.py                 # Mathematical implementations of Brier, ECE, AURC, AUROC, Cons
│   ├── prompts.py                 # Chat template formatting & option permutation strings
│   └── analysis.py                # Bootstrap CIs, McNemar exact tests, Kendall's tau
├── scripts/
│   ├── 00_env_check.py            # Environment, CUDA, and tokenizer validation
│   ├── 01_prepare_data.py         # Deterministic benchmark data sampling & manifest creation
│   ├── 02_run_models.py           # Model inference engine with batching & resume support
│   ├── 03_evaluate.py             # Metric computation, CIs, paired tests, failure tables
│   ├── 04_figures.py              # Publication vector figures (PNG + PDF)
│   ├── 05_validate_baseline.py    # Reference harness replication script
│   ├── 05_harness_check.sh        # Shell comparison runner
│   ├── 06_paper_tables.py         # Automated LaTeX table generator
│   ├── 07_verify_paper_claims.py  # Read-only audit verification script
│   ├── _common.py                 # Path and CLI shared helpers
│   ├── build_paper_pdf.py         # Standalone Python PDF generator
│   ├── diagnostic_audit.py        # Breakdown of stubborn error distributions
│   ├── generate_presentation_assets.py # Presentation tables and distribution assets
│   ├── safety_scan.py             # Secret, file-size, and license scanner
│   └── show_prompts.py            # Display permuted prompt strings
├── tests/
│   ├── test_metrics.py            # Unit tests for all mathematical metrics against known values
│   └── test_smoke_pipeline.py     # End-to-end integration pipeline smoke test
├── results/                       # Genuine experimental outputs (NO invented numbers)
│   ├── results.csv                # Full metric results with 95% bootstrap CIs
│   ├── baselines.csv              # Chance and majority-class benchmarks
│   ├── rank_by_metric.csv         # Model rankings across each metric
│   ├── rank_tau_summary.csv       # Mean within-slice Kendall's tau summary
│   ├── task_mode_delta.csv        # Closed vs context paired McNemar tests
│   ├── paired_model_tests.csv     # Holm-adjusted McNemar tests between models
│   ├── position_bias.csv          # Option selection frequencies vs gold shares
│   ├── pubmedqa_confusion.csv     # Per-class precision, recall, and confusion matrix
│   ├── pubmedqa_balanced_POSTHOC.csv # Post-hoc balanced accuracy and macro-F1
│   ├── failure_cases_consistently_wrong.csv # Capped stubborn failure cases (50/cell)
│   ├── table_stubborn_errors_breakdown.csv  # Uncapped breakdown of stubborn errors
│   ├── table_pubmedqa_deep_dive.csv        # Deep-dive breakdown of PubMedQA shifts
│   ├── figures/                   # Publication-ready vector PDF figures
│   ├── tables/                    # Generated LaTeX tables for paper inclusion
│   ├── validation/                # Reference replication outputs vs lm-eval-harness
│   ├── logs/                      # Verifiable execution logs (g2_log.txt)
│   └── raw/                       # 15 raw inference JSONL files + 3 __meta.json files
├── paper.zip                      # Standalone clean archive ready for 1-click Overleaf upload
└── paper/                         # Overleaf-ready LaTeX research paper package
    ├── main.tex                   # ACM formatted research manuscript (\documentclass[sigconf,nonacm]{acmart})
    ├── refs.bib                   # Audited BibTeX references with verified DOIs
    ├── figures/                   # Standalone figure assets (vector PDF + PNG)
    ├── tables/                    # Standalone LaTeX table inputs (8 tables)
    ├── claim_check.md             # Quantitative claim audit ledger
    ├── CHANGELOG.md               # Complete manuscript revision ledger
    └── paper.zip                  # Mirror copy of standalone clean archive
```

---

## 7. Reproduction and Execution Guide

### 7.1 Environment Setup
```bash
git clone https://github.com/AmanKumar-iiita/FactBench.git
cd FactBench
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 7.2 Verification of Test Suite & Safety (Executed)
```bash
# 1. Mathematical metric unit tests against analytical values
python tests/test_metrics.py
# Output: PASS test_auroc, PASS test_brier_known_values, ..., PASS all 8 tests

# 2. End-to-end integration pipeline smoke test
python tests/test_smoke_pipeline.py
# Output: PASS smoke

# 3. Full pytest suite
python -m pytest -q tests
# Output: 9 passed in 13.86s

# 4. Automated security and integrity scan
python scripts/safety_scan.py
# Output: 0 files > 50MB, 0 secrets, 0 gated data items found
```

### 7.3 Reference Baseline Validation (Executed Gate G2)
Prior to comparative evaluation, the scoring harness was audited against EleutherAI's `lm-evaluation-harness` using `Qwen2.5-1.5B-Instruct` on reference tasks (logged in `results/logs/g2_log.txt`):
- `mmlu_college_biology` ($N=144$): FactBench achieved 63.89% vs. harness 64.58% ($\Delta = -0.69\%$, exactly 1 item difference; harness $\text{SE} = \pm 4.00\%$).
- `mmlu_high_school_physics` ($N=151$): FactBench achieved 37.75% vs. harness 37.09% ($\Delta = +0.66\%$, 1 item difference; harness $\text{SE} = \pm 3.94\%$).
- Both discrepancies reside within $0.18 \times \text{SE}$ of sampling variance.

```bash
python scripts/00_env_check.py --tokenizers
python scripts/05_validate_baseline.py --model-name qwen2.5-1.5b-instruct
```

### 7.4 Benchmark Execution Pipeline
```bash
# 1. Deterministic data sampling & manifest generation
python scripts/01_prepare_data.py

# 2. Run model forward passes (GPU required; raw outputs preserved in results/raw/)
for model in qwen2.5-1.5b-instruct phi-3.5-mini-instruct qwen2.5-7b-instruct-4bit; do
  python scripts/02_run_models.py --model-name $model
done

# 3. Compute metrics, bootstrap CIs, McNemar tests, and failure tables
python scripts/03_evaluate.py

# 4. Generate publication figures and LaTeX tables
python scripts/04_figures.py
python scripts/06_paper_tables.py

# 5. Audit all paper quantitative claims against results CSVs
python scripts/07_verify_paper_claims.py
```

### 7.5 Overleaf Paper Compilation
The manuscript is prepared in standard ACM conference format (`\documentclass[sigconf,nonacm]{acmart}`).
To compile:
1. Upload [paper.zip](file:///c:/Users/akema/Downloads/AITR_Lab02_FactBench/paper.zip) directly to Overleaf ("New Project" $\to$ "Upload Project").
2. Set the main document to `main.tex`.
3. Compile using TeX Live 2024 / XeLaTeX or pdfLaTeX.

---

## 8. Limitations & Methodological Boundaries

1. **Model Sample Size ($N=3$):** With three models, Kendall's rank correlations have wide bootstrap confidence intervals and pairwise accuracy differences on some MMLU slices are within sampling noise.
2. **Quantization Confound:** `Qwen2.5-7B` was evaluated in 4-bit NF4 precision due to memory constraints, whereas the 1.5B and 3.8B models ran in full 16-bit precision.
3. **Prompt Template Scope:** Evaluations utilized a single standardized chat template per task.
4. **Resampling Unit:** Bootstrap confidence intervals resample test items, reflecting item-level variance rather than stochasticity in model weights.
5. **Pre-Training Contamination:** Benchmark contamination was not audited.
6. **Task Scope:** Confined to multiple-choice QA correctness, calibration, selective prediction, and option permutation consistency; open-ended generation, reasoning chains, and temporal validity were not evaluated.

---

## 9. Ethics, AI-Use Disclosure & Integrity

In accordance with course rules and `AGENTS.md`:
- **Research Integrity:** No metrics, p-values, citations, or execution logs were fabricated or hallucinated. All claims derive strictly from genuine execution runs preserved in `results/`.
- **AI Collaboration Disclosure:** AI assistants (Antigravity / Gemini) were utilized for scaffolding test pipelines, debugging CUDA and tokenizer edge cases, formatting LaTeX tables, and conducting read-only verification audits.
- **Accountability:** The human author reviewed, verified, and takes complete responsibility for all code, results, and text.

---

## 10. Citation

```bibtex
@misc{kumar2026factbench,
  title        = {Are Factuality Scores Interchangeable? A Controlled Multi-Metric Evaluation of Small Open Large Language Models on Scientific Question Answering},
  author       = {Kumar, Aman},
  year         = {2026},
  institution  = {Indian Institute of Information Technology, Allahabad},
  howpublished = {\url{https://github.com/AmanKumar-iiita/FactBench}}
}
```
