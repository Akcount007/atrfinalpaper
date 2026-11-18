# research_decisions.md  (Research Brief, Decision Log & Evidence Ledger)

**Student / Author:** Aman Kumar (mcl2026012@iiita.ac.in)  
**Project:** FactBench — Multi-Metric Evaluation of Small Open LLMs Across Scientific QA  
**Course:** AI Tools for Research (Lab 02)  
**Status:** ALL MILESTONES VERIFIED (G1, G2, G3, G4)

---

## 1. Gate G1: One-Page Research Brief (Template from Lab Instructions §11)

| Field | Project Information |
|---|---|
| **Review Topic & Key Papers** | **Topic:** Benchmarking Factual Accuracy of Large Language Models Across Scientific Domains (Kumar, 2026).<br>**5 Original References:**<br>1. Jin et al. (2019) *PubMedQA: A Dataset for Biomedical Research QA* (EMNLP-IJCNLP 2019).<br>2. Hendrycks et al. (2021) *Measuring Massive Multitask Language Understanding (MMLU)* (ICLR 2021).<br>3. Manakul et al. (2023) *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection* (EMNLP 2023).<br>4. Guo et al. (2017) *On Calibration of Modern Neural Networks* (ICML 2017).<br>5. Geifman & El-Yaniv (2017) *Selective Classification for Deep Neural Networks* (NeurIPS 2017). |
| **Research Gap** | In the literature review (Kumar 2026, Sec. 6.1, Table 2, Table 3), it was identified that scientific LLM benchmarks rely almost exclusively on scalar raw accuracy. This creates a severe measurement blindspot: accuracy masks poor calibration (overconfidence), failure to abstain under uncertainty (selective prediction risk), drastic performance drops when evidence context is removed (closed-book vs context-grounded shift), and false certainty where models stubbornly repeat errors across option orderings. |
| **Research Questions** | **RQ1 (Multi-Metric Discordance):** Do model rankings across accuracy, calibration (ECE, Brier), selective prediction (AURC, accuracy@80% coverage), and permutation-consistency agree across scientific domains, or do different metrics produce rank inversions?<br>**RQ2 (Task-Mode Grounding Shift):** Does providing primary literature evidence (PubMedQA context vs closed-book) improve accuracy, calibration, and selective prediction uniformly, or do some models exhibit fragile surface-level reading?<br>**RQ3 (Consistency vs Obstinate Error):** Is option-permutation consistency a dependable surrogate for scientific factuality, or does it reward models that consistently output identical wrong answers? |
| **Study Pathway** | **Pathway B: Controlled Comparative Study** with targeted error/failure analysis (evaluating small open-weights LLMs under a strictly matched, frozen protocol). |
| **Dataset & Measurement Setup** | **MMLU Science Subsets:** 450 items total (Biomedical: 150 items across 8 medical subjects; Mathematics: 150 items across 4 math subjects; Physics & Chemistry: 150 items across 6 physchem subjects).<br>**PubMedQA:** 500 items from `pqa_labeled` evaluated in paired modes: closed-book (`pubmedqa_closed`) vs context-grounded (`pubmedqa_context`).<br>**Total Scale:** 1,450 unique item evaluations $\times$ 4 option permutations = 5,800 prompts per model = 17,400 total inferences. |
| **Baseline / Comparators** | 1. **Chance Baselines:** 0.250 for 4-option MMLU, 0.333 for 3-option PubMedQA.<br>2. **Majority-Class Baselines:** 0.273 to 0.353 for MMLU slices; 0.552 for PubMedQA ('yes' class).<br>3. **Reference Harness Reproduction:** EleutherAI `lm-evaluation-harness` baseline comparison on `mmlu_college_biology` and `mmlu_high_school_physics`. |
| **Primary Metric & Control** | **Primary Accuracy:** Top-1 prediction on canonical (identity) prompt.<br>**Primary Confidence:** Softmax probability over candidate option tokens.<br>**Controls:** Frozen seed (`seed: 42`), identical prompt phrasing, zero-shot greedy token evaluation, fixed 1,000 bootstrap resamples with 95% percentile CIs. |
| **Computing & Risks** | **Hardware:** NVIDIA GPU (CUDA execution) + Google Colab cloud GPU fallback.<br>**Risks:** Tokenizer option-letter alignment, CUDA OOM during batch inference, dataset licensing boundaries.<br>**Mitigations:** Batch size calibrated to 8, explicit UTF-8 encoding, Apache-2.0 / MIT open weights only. |
| **First Executable Milestone** | Environment test (`scripts/00_env_check.py`), deterministic data pipeline (`scripts/01_prepare_data.py`), and full test suite passing (`pytest -q tests` $\to$ 9/9 passed). |

---

## 2. Evidence Ledger

| ID | Claim / Assumption | Label | Source / Evidence | Audit Verification |
|---|---|---|---|---|
| E1 | PubMedQA `pqa_labeled` contains 1,000 expert annotations under MIT license | **VERIFIED** | Hugging Face Dataset Card (`qiaojin/PubMedQA`), official GitHub repo | Executed `01_prepare_data.py`; exactly 1,000 rows present, 500 sampled via seed 42. |
| E2 | MMLU test split contains 14,042 rows across 57 subjects | **VERIFIED** | Hugging Face Dataset Card (`cais/mmlu`), Hendrycks et al. (2021) | Sliced 150 items deterministically per domain across biomedical, physchem, and math. |
| E3 | GPQA is gated and terms forbid posting question examples publicly | **VERIFIED** | HF dataset card (`Idavidrein/gpqa`) terms of use | Confirmed restriction; excluded GPQA to maintain open public reproducibility. |
| E4 | Small open-weights models (Qwen2.5-1.5B, Phi-3.5-mini, Qwen2.5-7B) have permissible licenses | **VERIFIED** | Model cards on HF Hub | Qwen2.5 is Apache 2.0; Phi-3.5-mini is MIT. Both fully permit academic research. |
| E5 | lm-evaluation-harness provides a standardized baseline reproduction | **VERIFIED** | EleutherAI lm-eval GitHub repo & PyPI package | Replicated Qwen2.5-1.5B within 0.7 percentage points ($\le 1$ item difference). |
| E6 | Permutation consistency exposes position sensitivity | **VERIFIED** | Experimental results in `results.csv` and `position_bias.csv` | Observed 470 cases of consistently wrong assertions across all 4 permutations. |

---

## 3. Decision Log

| # | Date | Decision | Rationale | Empirical Evidence |
|---|---|---|---|---|
| D1 | 2026-11-12 | Adopt Pathway B (Controlled Comparative Study) | The literature review (Kumar 2026, Sec. 6.1) formulated the theoretical claim that "scores are not interchangeable," which required direct empirical testing across multiple metrics. | Review paper Table 2 and Table 3. |
| D2 | 2026-11-12 | Freeze evaluation protocol in `configs/protocol.yaml` | Prevent p-hacking, cherry-picking, and test-set tuning. | Protocol frozen at version 1.0, seed 42 before analysis. |
| D3 | 2026-11-12 | Evaluate 3 open models across 5 domain slices | Span multiple scientific fields (Biomedicine, Math, PhysChem) and task modes (Closed-book vs Context-grounded). | Full 17,400 inference outputs recorded in `results/raw/`. |
| D4 | 2026-11-12 | Implement option-probability confidence | Multiple-choice probability over target tokens $\{A, B, C, D\}$ enables exact Brier score, ECE, and AURC computation without prompt-induced verbalized overconfidence. | Brier scores and ECE successfully computed across all models and slices. |
| D5 | 2026-11-12 | Evaluate 4 deterministic option permutations | Systematic cyclic and seeded permutation isolates spatial option biases and tests whether consistency correlates with correctness. | Revealed 470 stubbornly held wrong answers across models. |
| D6 | 2026-11-12 | Run paired exact McNemar tests with Holm-Bonferroni correction | Rigorously test statistical significance of context grounding (RQ2) and pairwise model performance differences. | Holm-adjusted p-values exported to `task_mode_delta.csv` and `paired_model_tests.csv`. |
