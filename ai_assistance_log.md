# ai_assistance_log.md  (Gate G4 AI Audit & Research Integrity Report)

**Auditor / Student:** Aman Kumar (mcl2026012@iiita.ac.in)  
**Project:** FactBench — Multi-Metric Evaluation of Small Open LLMs on Scientific QA  
**Course:** AI Tools for Research (Lab 02)  
**Date:** November 12, 2026

---

## 1. AI-Assisted Activity Log

| # | Date | Task | Tool / Model | Prompt / Interaction | AI Proposal (Summary) | Human Decision & Action | Verification Method | Outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-12 | Literature-to-experiment mapping | Claude 3.5 Sonnet | Prompt 02 / 03: Extract testable gaps from literature review | Suggested benchmarking GPQA, PubMedQA, and MMLU using both open and proprietary LLMs. | Rejected proprietary APIs to guarantee reproducibility; checked GPQA license and rejected GPQA because redistribution of questions violates terms. Selected MMLU science subsets and PubMedQA. | Checked GPQA and MMLU terms on Hugging Face Hub. | Clean, compliant, fully public dataset plan using MMLU and PubMedQA. |
| 2 | 2026-11-12 | Project scaffolding & architecture | Claude 3.5 Sonnet | Prompt 08: Scaffolding testable directory structure | Provided modular project template (`src/factbench`, `scripts`, `configs`, `tests`). | Reviewed every module; verified that option logit extraction in `scoring.py` correctly normalized candidate tokens over `[A, B, C, D]`. | Ran unit test suite (`pytest -q tests`). | All 9 tests passed; clean separation of data, scoring, analysis, and figures. |
| 3 | 2026-11-12 | Baseline replication verification | Antigravity (Gemini 3.8 Flash) | Prompt 10: Validate baseline against lm-evaluation-harness | Suggested running lm-eval harness CLI directly in subshell. | Executed `scripts/05_validate_baseline.py` and compared outputs against EleutherAI's `lm-evaluation-harness` on `college_biology` and `high_school_physics`. | Diffed prompts and verified that accuracy matched within 0.7 percentage points ($\le 1$ item difference). | Validated baseline replication report in `baseline_validation.md`. |
| 4 | 2026-11-12 | Debugging environment and runtime errors | Antigravity (Gemini 3.8 Flash) | Prompt 11: Traceback analysis for Windows environment | AI generated a one-line terminal script to inspect class balances, which crashed with `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81`. | Intervened manually to enforce `encoding='utf-8'` across all file I/O operations and scripts; diagnosed CUDA OOM on batch processing by lowering batch size. | Re-ran script with explicit encoding and batch size 8; executed error-free. | Robust cross-platform execution on Windows and Linux GPU instances. |
| 5 | 2026-11-12 | Statistical evaluation & visualization | Antigravity (Gemini 3.8 Flash) | Prompt 13: Statistical metrics and figure plotting | Proposed calculating standard accuracy, Brier, ECE, AURC, and McNemar tests. | Enforced Holm-Bonferroni multi-test p-value adjustment, paired bootstrap 95% confidence intervals, and generation of both PDF and PNG vector figures. | Verified all figures in `results/figures/` and cross-referenced numbers with `results.csv`. | 4 publication-quality figures and 8 traceable CSV result files. |
| 6 | 2026-11-12 | Scientific paper manuscript drafting | Antigravity (Gemini 3.8 Flash) | Prompt 15: Draft evidence-grounded manuscript in LaTeX | Drafted paper sections strictly using empirical numbers from `results/` CSV files. | Audited all LaTeX text to ensure every number directly matches `results.csv`, `rank_by_metric.csv`, `task_mode_delta.csv`, and `pubmedqa_confusion.csv`. No numbers invented or hand-estimated. | Cross-referenced every citation in `refs.bib` against official publisher DOIs. | Complete 6–10 page LaTeX research paper (`paper/main.tex`). |

---

## 2. Real Examples of AI Limitations, Hallucinations, and Human Corrections

### Limitation 1: Dataset Licensing and Terms of Use (GPQA)
- **AI Proposal:** The AI initially recommended using GPQA (`Idavidrein/gpqa`) as the primary gold-standard graduate reasoning benchmark, citing its high difficulty in the literature review.
- **Human Audit & Correction:** Upon inspecting the official Hugging Face dataset card and terms of use for `gpqa`, the human researcher identified that while the code is CC-BY-4.0, the benchmark creators strictly prohibit publishing benchmark questions, solutions, or reasoning traces online to protect against web crawler contamination. Relying on GPQA in a public GitHub repository would violate the creators' terms.
- **Correction:** The researcher removed GPQA from the experimental pipeline and substituted three challenging academic subsets of MMLU (Biomedical, Physics/Chemistry, and Mathematics), combined with PubMedQA.

### Limitation 2: Windows System Encoding Crash (`charmap` Unicode Error)
- **AI Proposal:** When asked to verify class counts on the sampled `pubmedqa.jsonl` file, the AI assistant generated a standard command:  
  `python -c "import json,collections;r=[json.loads(l) for l in open('data/processed/pubmedqa.jsonl')];..."`
- **Error Observed:** On Windows powershell, Python defaults to the system code page (`cp1252`), resulting in an immediate crash:  
  `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81 in position 6932: character maps to <undefined>`.
- **Human Correction:** The human researcher recognized the Windows-specific default encoding limitation and explicitly refactored all file opening commands across `data.py`, `analysis.py`, and CLI scripts to use `encoding="utf-8"`, preventing runtime crashes on biomedical text containing Greek symbols and non-ASCII medical punctuation.

### Limitation 3: Conflating Multiple-Choice Permutation with SelfCheckGPT Sampling
- **AI Proposal:** The AI suggested applying `SelfCheckGPT` directly to evaluate hallucinations by sampling 5 stochastic generations at temperature $T=0.7$ and scoring them with an external LLM evaluator.
- **Human Audit & Correction:** The human researcher recognized that: (1) multiple-choice tasks allow exact, zero-shot option-probability evaluation without decoding jitter; (2) relying on an external API judge introduces uncontrollable API drift and evaluator bias; and (3) what scientific QA fundamentally tests is sensitivity to prompt framing and option order.
- **Correction:** The researcher formally formulated **Systematic Option Permutation Consistency** (evaluating agreement across identity and 3 seeded option orderings: cyclic shift, reverse, and random shuffle). This strictly measures spatial bias and invariance to option order, while providing an exact analogue to consistency checking without requiring an unverified external LLM judge.

---

## 3. Human Reasoning vs AI Assistance: Critical Reflection

1. **What Depended on Human Reasoning:**
   - **Hypothesis Formulation:** Formulating the central research questions (RQ1–RQ3) based on the conceptual insights of Section 6.1 of the author's prior literature review.
   - **Protocol Discipline:** Freezing the protocol (`protocol.yaml`) before examining experimental results to prevent retrospective metric selection.
   - **Error Discovery:** Recognizing that Qwen2.5-1.5B's apparently superior closed-book PubMedQA accuracy (55.2%) was an artifact of extreme class imbalance ('yes' majority share is exactly 55.2%) rather than biomedical competence.
   - **Interpretation of Consistency:** Discovering and articulating the danger of using consistency as a factuality signal: the 470 cases where models outputted the exact same incorrect answer across all 4 option permutations.

2. **What Was Learned About "Vibe Coding":**
   - Vibe coding is effective for rapidly drafting unit tests, boilerplate data loaders, and matplotlib styling.
   - However, vibe coding fails catastrophically when dealing with exact mathematical definitions (e.g., verifying that Brier score is divided by the number of classes or whether ECE uses confidence-bin weighting) and system-level subtleties (such as tokenizer token-ID offsets for leading spaces).
   - Real scientific experimentation requires line-by-line verification of mathematical formulas and rigorous unit testing against known synthetic cases (`tests/test_metrics.py`).

3. **Remaining Risks & Methodological Boundaries:**
   - Evaluated 3 model checkpoints; expanding to 10+ open LLMs would strengthen the statistical power of Kendall's rank correlation $\tau$.
   - Option-probability confidence assumes the model distributes probability mass rationally across valid option tokens, which can still be distorted by superficial position biases.
   - Dataset contamination in commercial pretraining corpora remains unmeasured for static benchmarks like MMLU and PubMedQA.
