# Paper Revision Changelog (FactBench ACM Transition)

This changelog records all modifications made to transition the research manuscript from the exploratory draft into the rigorous, evidence-grounded ACM format (`\documentclass[sigconf,nonacm]{acmart}`) in accordance with repository results, execution logs, and `AGENTS.md`.

---

## 1. Table of Changes: Old vs. New vs. Evidence Source

| Old Claim / Text in Draft | New Claim / Text in ACM Paper | Evidence File / Command | Reason for Revision |
|:---|:---|:---|:---|
| **$N=1,450$ distinct items** | **950 distinct items (1,450 item-slice evaluations)** | `data/processed/MANIFEST.json`, `data/processed/*.jsonl` | Corrects distinction between unique benchmark questions (450 MMLU + 500 PubMedQA) and total item evaluations. |
| **Option permutations: cyclic shift and reverse ordering** | **Identity plus 3 distinct random permutations seeded by seed 42 + CRC32(item ID)** | `scripts/02_run_models.py` (`make_perms`) | Matches actual permutation generation in code; removed inaccurate "cyclic / reverse" wording. |
| **Consistency: mean pairwise agreement across $\binom{K}{2}=6$ pairs** | **Consistency: fraction of 4 orders matching modal prediction ($\max_v \sum \mathbf{1} / K$)** | `src/factbench/metrics.py` (`consistency`) | Replaced incorrect 6-pair formula with actual `counts.max() / k` implementation. |
| **Scoring: greedy decoding, temperature 0** | **Constrained option scoring in 1 forward pass over letter tokens; no autoregressive generation** | `src/factbench/scoring.py` (`HFScorer.score`) | Clarified that log-probs are evaluated directly on option letters $\{A, B, C, D\}$ without token generation. |
| **MMLU: stratified across subtasks** | **Sampled randomly from pooled domain subjects, not stratified** | `src/factbench/data.py` (`mmlu_items`) | Corrected description of MMLU domain sampling. |
| **PubMedQA: evaluated official test split** | **Class-proportional sample of 1,000 labeled items; official test split not used** | `data/processed/MANIFEST.json` (`official_test_ids_used: false`) | Corrected dataset split attribution based on manifest. |
| **Brier score: generic MSE formula** | **Multiclass Brier score in range $[0, 2]$ citing Brier (1950)** | `src/factbench/metrics.py` (`brier`), Brier (1950) | Added formal range $[0, 2]$ and cited foundational reference Brier (1950). |
| **Prompts in Appendix: simplified harness style** | **Exact chat-template prompt strings from `src/factbench/prompts.py`** | `src/factbench/prompts.py` | Copied exact user-prompt formats with letter options for both MMLU and PubMedQA. |
| **Phi-3.5 evaluated in bfloat16; Qwen-7B with double quantization** | **Phi-3.5 and Qwen-1.5B in float16; Qwen-7B in 4-bit NF4 with default settings (no double quant)** | `src/factbench/scoring.py`, `results/raw/*__meta.json` | Corrected precision and quantization parameters to match code and BitsAndBytes defaults. |
| **Pre-registered protocol** | **Frozen protocol v1.0 [TO VERIFY: protocol-v1 tag date vs first run]** | `configs/protocol.yaml`, git metadata | Replaced "pre-registered" with "frozen protocol"; marked date provenance as `[TO VERIFY]`. |
| **Baseline validation: "parity" across harness** | **Validation covers 1 model, 2 subjects, harness-style prompt only; does NOT validate chat prompts** | `baseline_validation.md`, `g2_log.txt` | Dropped "parity"; clearly delineated the scope of baseline verification. |
| **RQ1: "conclusive proof for review's thesis"** | **Thesis is partly supported: Brier/AURC track accuracy ($\tau \approx 0.83$), while ECE and AUROC diverge; differences on biomed MMLU not significant ($p=0.981$)** | `results/rank_tau_summary.csv`, `results/paired_model_tests.csv`, `tab_paired.tex` | Moderated claims to reflect that with $N=3$ models, rank inversions on biomed MMLU are suggestive, not statistically significant. |
| **Figure 3 group sizes: 2,187 / 1,464 / 1,029 (sum = 4,680)** | **Figure 3 group sizes: 1,942 / 1,379 / 1,029 (sum = 4,350)** | `scripts/07_verify_paper_claims.py` (Section 1), Table 4 | Fixed overlapping group counting so that Consistent Correct (1,942), Wavering (1,379), and Stubborn (1,029) partition all 4,350 evaluations. |
| **RQ3: "stubborn confidence > 0.94 across the board"** | **Overall mean confidence is 0.949; cell means range from 0.813 to 1.000** | `scripts/07_verify_paper_claims.py` (Section 3) | Corrected universal claim; noted that Qwen-1.5B math stubborn confidence is 0.813. |
| **RQ3: accuracy strictly increases monotonically from <35% to >70%** | **Reported exact levels: Phi-3.5 peaks at 58.2% across slices, and drops to 27.2% at C=1.0 on closed-book PubMedQA** | `scripts/07_verify_paper_claims.py` (Section 2) | Corrected monotonic claim to reflect non-monotonic drop caused by stubborn errors on closed-book PubMedQA. |
| **AUROC_conf exceeds AUROC_cons by +0.05 in 12 of 15 cells** | **AUROC_conf exceeds AUROC_cons in 12 of 15 cells; mean diff all = +0.042; mean diff positive = +0.063** | `scripts/07_verify_paper_claims.py` (Section 4) | Corrected exact mean difference metrics. |
| **Permutation ensemble: "provides modest accuracy stabilization"** | **Reported all 15 deltas ($-0.027$ to $+0.067$); decreases in 3 cells; labelled exploratory** | `scripts/07_verify_paper_claims.py` (Section 5) | Removed assumption of guaranteed positive gain; reported raw decreases and labelled exploratory. |
| **PubMedQA closed-book: claimed model learned majority class** | **Added post-hoc balanced accuracy and macro-F1; noted Qwen-1.5B matching 55.2% accuracy to 55.2% majority class is a coincidence** | `results/pubmedqa_balanced_POSTHOC.csv`, `tab_pubmedqa_balanced_posthoc.tex` | Added balanced accuracy (0.372) and macro-F1 (0.352); explained prediction breakdown (391/500 `yes`). |
| **Failure cases: speculative "anatomical misconceptions"** | **Pure behavioral reporting of item text, gold label, predicted choices, and confidences** | `data/processed/mmlu.jsonl`, `scripts/07_verify_paper_claims.py` | Removed ungrounded psychological explanations; verified items `mmlu-anatomy-25`, `mmlu-astronomy-42`, `pubmedqa-10966337`. |
| **Position bias: raw percentage reporting** | **Compared against ~25.0% uniform baseline for 4-choice displays; labelled descriptive** | `results/position_bias.csv` | Added baseline context for interpreting display frequency skew. |
| **Contamination: argued closed-book drop proves no contamination** | **Explicitly stated contamination was not audited** | Section 9 (Limitations) | Removed invalid deduction regarding pretraining contamination. |
| **Scope: broad factual accuracy claims** | **Confined scope to multiple-choice QA correctness, calibration, selective prediction, and permutation consistency** | Section 9 (Limitations) | Clarified study does not evaluate open-ended generation, reasoning steps, or temporal factuality. |
| **Guo et al. (2017) cited for RLHF worsening calibration** | **Cited Guo et al. (2017) for modern DNN calibration / temperature scaling; removed RLHF claim** | `paper/refs.bib`, Guo et al. (2017) | Corrected citation attribution. |
| **Zheng et al. (2024) cited for knowledge conflation** | **Cited Zheng et al. (2024) specifically for option-order bias in multiple-choice LLM evaluation** | `paper/refs.bib`, Zheng et al. (2024) | Corrected citation context. |
| **Self-citation to Aman Kumar review paper** | **Removed `@article{kumar2026benchmarking}` from bibliography; referenced prior literature review anonymously** | `paper/refs.bib`, `paper/main.tex` | Fully removed self-citation as instructed by the author. |
| **Table formatting: hand-typed LaTeX tabulars** | **Automated `\input{results/tables/*.tex}` generated directly by `scripts/06_paper_tables.py`** | `scripts/06_paper_tables.py`, `results/tables/` | Guaranteed zero hand-typed numbers across all primary result tables. |

---

## 4. Final Verification and Factual Correction Ledger (17 Items)

| Item | Location | Old Text | New Text | Evidence File |
|:---|:---|:---|:---|:---|
| 1 | Abstract | "standard academic practice overwhelmingly relies on" | "evaluation commonly relies on" | Textual precision |
| 2 | Abstract | "$+33.6\%$" and "$+31.0\%$" | "+33.6 points" and "+31.0 points" | Absolute percentage point delta standard |
| 3 | Abstract | "severe majority-class collapse ... 0.0% recall on 'maybe'" | "a strong ``yes'' bias (\textsf{Qwen2.5-1.5B}: 78.2\% of predictions, matching the 55.2\% majority share only by coincidence, with 1.8\% recall on ``maybe'' in closed-book mode and 0.0\% with context)" | `results/results.csv`, `results/task_mode_delta.csv`, confusion matrix |
| 4 | Section 1 (Intro) | "... reported on public leaderboards~\cite{liang2022holistic}." | "... reported on public leaderboards." | HELM citation removed from single-number claim; retained in Related Work for multi-metric eval |
| 5 | Section 4.3 | "Both discrepancies reside within $0.18 \times \mathrm{SE}$." | "Both discrepancies reside within $0.18 \times \mathrm{SE}$. The harness SE is the standard error of each accuracy estimate, not of the difference between the two pipelines, which score identical items." | Methodological clarity |
| 6 | Section 5.1 | "... and the highest selective accuracy across all other slices" | "... and the highest selective accuracy of the three models in 4 of 5 slices; \textsf{Qwen2.5-1.5B} is highest in closed-book PubMedQA: 0.570 vs 0.440; \texttt{results.csv})" | `results/results.csv` (`sel_acc` column) |
| 7 | Section 5.1 | "... act as the primary differentiators for operational deployment. Furthermore, despite operating under aggressive 4-bit NF4 weight quantization, \textsf{Qwen2.5-7B-Instruct-4bit} attains competitive selective accuracy ... along with superior positional stability (Section~\ref{sec:pos_bias})" | "When accuracy differences are within sampling noise, reliability metrics may help differentiate models, but the AUROC differences were not tested for significance. Furthermore, despite operating under aggressive 4-bit NF4 weight quantization, \textsf{Qwen2.5-7B-Instruct-4bit} attains competitive selective accuracy ... along with smaller position-bias deviations across MMLU slices (maximum deviation from 25\% of 2.7 points on math, vs 9.0 points for \textsf{Phi-3.5-mini} and 13.2 points for \textsf{Qwen2.5-1.5B}; Section~\ref{sec:pos_bias})" | `results/paired_model_tests.csv`, `results/position_bias.csv` |
| 8 | Figure 1 Caption | "\caption{Rank inversion slopegraph: model ranking by Raw Accuracy (left) vs Confidence Discrimination ($\text{AUROC}_{\text{conf}}$, right) across slices.}" | "\caption{Rank inversion slopegraph: model ranking by Raw Accuracy (left) vs Confidence Discrimination ($\text{AUROC}_{\text{conf}}$, right) across slices. Rank flips are descriptive; most MMLU accuracy differences are not statistically significant (Table~\ref{tab:paired}), and AUROC confidence intervals overlap across models within slices (\texttt{results.csv}).}" | `results/paired_model_tests.csv`, `results/results.csv` |
| 9 | Section 5.2 | "... accompanied by substantial reductions in calibration error ($\Delta \mathrm{Brier} = -0.613$ and $-0.571$)." | "... accompanied by substantial reductions in calibration error: $\Delta\text{ECE} = -0.305$ and $-0.275$ (from \texttt{task\_mode\_delta.csv}), followed by $\Delta\text{Brier} = -0.613$ and $-0.571$, noting that Brier score inherently conflates accuracy and calibration." | `results/task_mode_delta.csv` |
| 10 | Section 5.3 & 6.1 | "Qualitative and error-type auditing reveals that these 1,029 stubborn errors divide into two distinct failure mechanisms... reflecting the widespread colloquial misconception... locked into surface associations..." | "As a hypothesis suggested by qualitative inspection of three examples (rather than a measured taxonomy of all 1,029 errors), stubborn errors may arise from two candidate mechanisms... (all speculative causal phrases removed)" | Empirical rigor; hypothesis framing |
| 11 | Section 6.1 | Quoted items with causal attributions | Exact question text, options, gold, and behavioral predictions verified against `scripts/07_verify_paper_claims.py` section 6 (including Qwen-7B $\text{Cons}=0.5$ on `mmlu-astronomy-42`) | `scripts/07_verify_paper_claims.py` (Section 6), `data/processed/*.jsonl` |
| 12 | Figure 3 Caption | "\caption{The consistency trap. (a) ... (b) Accuracy by consistency level across slices.}" | "\caption{The consistency trap. (a) ... (b) Accuracy by consistency level across slices ($N=5$ items at $\text{Cons}=0.25$, $N=548$ at $0.50$, $N=826$ at $0.75$, $N=2,971$ at $1.00$). Pooling across slices masks cells where the consistency-accuracy relationship is reversed: for \textsf{Phi-3.5-mini} in closed-book PubMedQA, only 82 of 301 fully consistent items are correct (27.2\%; Table~\ref{tab:stubborn_detail}).}" | `scripts/07_verify_paper_claims.py` (Section 2), `results/stubborn_breakdown.csv` |
| 13 | Section 3.4 & Tables | 7 primary metrics; Table 2 lacking AURC; Table 5 lacking $\Delta$Brier | Defined Mean Consistency ($\text{Cons}_{\text{mean}}$); noted Table 3 Kendall $\tau$ evaluates accuracy vs mean consistency; added AURC column to Table 2 (`tab_main.tex`) and $\Delta\text{Brier}$ [95% CI] column to Table 5 (`tab_taskmode.tex`) | `results/results.csv`, `results/task_mode_delta.csv`, `results/rank_tau_summary.csv` |
| 14 | Section 6.2 | "(gold share: 30.0%)" and partial position frequencies | Deleted "(gold share: 30.0%)"; added Qwen-7B selection rates and deviations for all 3 MMLU slices; reported max deviations: 2.7 points (Qwen-7B), 9.0 points (Phi-3.5), 13.2 points (Qwen-1.5B) | `results/position_bias.csv` |
| 15 | Section 7 | "fail in the presence of stubborn errors" and "order-invariance serves as an amplifier of stubborn misconceptions" and mention of prompt-paraphrase | "permutation averaging improved accuracy in 10 of 15 cells (3 decreased, 2 unchanged; Section~\ref{sec:ensembling}), it cannot correct errors that are identical across all four orders. When model probability mass collapses decisively onto a single incorrect option irrespective of order, order-invariance cannot serve as a safeguard of truth." (prompt-paraphrase removed) | `scripts/07_verify_paper_claims.py` (Section 5) |
| 16 | Section 7 | "guarantees lower operational risk", "catastrophic recall failure", "entrenched misconceptions or deficit-driven defaults, which constitute 54.6% of all model errors" | "may reduce operational risk", "large recall failure", "stubborn errors (identical across all four option orders), which constitute 54.6% of all errors" | Tone and precision calibration |
| 17 | Appendix A | "zero-shot cloze scoring" | "zero-shot multiple-choice letter scoring (next-token probabilities over the option letters)" | `src/factbench/scoring.py` |

---

## 3. Academic Framing and Depth Enhancements

The following framing enhancements were integrated into `paper/main.tex` (and mirrored in `scripts/build_paper_pdf.py`):

1. **Section 3.1 (Task Definition Summary Table):**
   - Inserted compact summary Table 1 (`tab:slices_summary`) immediately following the 5-slice itemization, summarizing each slice's Domain, Source Dataset, Item Count ($N$), Option Count ($m$), Context Mode, and Total Forward Passes ($N \times 4$).
2. **Section 3.2 (Restricted Softmax Rationale):**
   - Added concise explanation clarifying that log-sum-exp over token mappings $\operatorname{tokens}(L)$ is necessary because subword tokenizers (e.g., SentencePiece, BPE) assign distinct vocabulary indices to leading-space tokens (e.g., `"A"` vs `" A"`).
3. **Section 5.1 (RQ1 - Statistical Parity & Quantization Framing):**
   - Reframed non-significant paired accuracy differences on `mmlu_biomed` ($p \in [0.35, 0.98]$) as an empirical finding: when models cluster closely in raw accuracy within sampling noise, reliability metrics ($\mathrm{AUROC}_{\mathrm{conf}}$ and calibration) serve as the primary differentiators for operational deployment.
   - Added observation on the 4-bit NF4 quantization trade-off: despite aggressive quantization, `Qwen2.5-7B-Instruct-4bit` achieves competitive selective accuracy ($\mathrm{SelAcc}_{0.8} = 0.825$ on biomedicine; highest across all other slices) and superior positional stability over unquantized 16-bit 1.5B and 3.8B baselines.
4. **Section 5.2 (RQ2 - Persistent Affirmative Bias Across Regimes):**
   - Explicitly linked `Qwen2.5-1.5B`'s 78.2% (391/500) `'yes'` prediction rate in closed-book mode to its 82.6% (413/500) `'yes'` prediction rate, 95.3% `'yes'` recall ($263/276$), and 0.0% `'maybe'` recall ($0/55$) in context-grounded mode (`results/pubmedqa_confusion.csv`), proving that affirmative bias intensifies rather than abates with evidence grounding.
5. **Section 5.3 & Section 6.1 (RQ3 - Categorization of Stubborn Error Mechanisms):**
   - Formally categorized the 1,029 stubborn errors (54.6% of all errors; mean confidence 0.949) into two distinct failure mechanisms:
     - (a) *Parametric Misconceptions:* Entrenched semantic associations overriding truth across all order permutations (`mmlu-anatomy-25` Carbon dioxide vs Nitrogen, `mmlu-astronomy-42` gas composition vs gravitational compression).
     - (b) *Deficit-Driven Defaults:* Invariant collapse onto an epistemic hedge or default class when ungrounded (`pubmedqa-10966337` defaulting to `'maybe'` without context, flipping to `'yes'` with context).
   - Added `\label{sec:pos_bias}` to Section 6.2 for cross-referencing.
6. **Section 7 (Discussion - Inference-Time Voting Failure & Practitioner Checklist):**
   - Removed duplicated definition of permutation consistency that already appears in Section 3.3.
   - Clarified that test-time majority voting and prompt-paraphrase self-consistency fail on stubborn errors due to probability collapse onto a single incorrect option regardless of presentation order.
   - Added a structured 3-bullet **Practitioner Deployment Checklist**:
     1. *Autonomous pipelines:* Prioritize $\mathrm{SelAcc}_{0.8}$ and $\mathrm{AUROC}_{\mathrm{conf}}$.
     2. *Literature/RAG QA:* Mandate balanced accuracy and macro-F1 alongside raw accuracy.
     3. *Robustness auditing:* Pair permutation consistency with evidence grounding controls.


---

## 5. Final Targeted Revisions Ledger (Edits 1--9)

| Item | Location | Change Summary | Status / Verification |
|:---|:---|:---|:---|
| **1** | Section 6.5 | Deleted Section 6.5 (*Reliability Diagrams and Risk-Coverage Profiles*) entirely. | Verified deleted. |
| **2** | Figure 1 Caption | Replaced generic overlap text with: `"AUROC confidence intervals overlap for the model pairs whose ranks flip (\texttt{results.csv})"` | Verified across `mmlu_biomed`, `mmlu_physchem`, and `pubmedqa_closed` in `results.csv`. |
| **3** | Abstract, Intro, Sec 4.2, 5.2 | Replaced every occurrence of "frozen protocol" / "frozen experimental protocol" with `"fixed protocol (configs/protocol.yaml, version 1.0, seed 42)"`. | Zero "frozen" occurrences remaining. |
| **4** | Section 4.1 | Deleted license statements and exact parameter counts; replaced with `"approximately 1.5B, 3.8B and 7B parameters"`. | Verified concise phrasing. |
| **5** | References (`refs.bib`) | Updated `wang2023scibench` pages to `50622--50649`. Deleted `url` field from `geifman2017selective`. | Verified in `paper/refs.bib`. |
| **6** | Appendix B | Updated 4-bit quantization bullet: `bnb_4bit_quant_type="nf4"`; `bnb_4bit_use_double_quant` left at default (`False`), no double quantization. | Formatted cleanly without text overflow. |
| **7** | Introduction | Replaced "relies almost exclusively on" with "commonly relies on". Replaced scarcity statement with `"This study measures rank concordance across small open-weights models under standardized inference conditions."` | Verified text updated. |
| **8** | Section 2.3 | Replaced "proving that citation presence does not guarantee factual entailment" with `"motivating the view that citation presence does not guarantee factual entailment"`. | Verified text updated. |
| **9** | Sec 5.2, Sec 7, Conclusion, README | Replaced "majority-class collapse" with `"strong yes-bias"` across headings, text, and `README.md` while keeping all numbers unchanged. | Verified across manuscript and docs. |
