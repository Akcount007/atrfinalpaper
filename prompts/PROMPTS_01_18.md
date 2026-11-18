# Prompts 01–18, pre-filled for this project
Replace anything in [BRACKETS]. `PASTE` means paste real content/output. Each block says: Tool · Mode · Attach · Save.

---
## PHASE I — FROM REVIEW TO RESEARCH QUESTION

### P01 · Set up an accountable workspace
Tool: any chatbot · Attach: your review PDF · Save: `research_decisions.md`
```
I am a postgraduate student converting my completed literature review into an implementation-based research paper.
My topic is "Benchmarking Factual Accuracy of Large Language Models Across Scientific Domains" (review attached).
Act as a critical research mentor and technical pair programmer. Guide me in stages: evidence-based gap identification,
literature-to-experiment mapping, feasibility, research questions, verified baseline, coding, experimental evaluation,
manuscript and reproducibility. Mark every statement VERIFIED, INFERRED or UNKNOWN and ask for source files or outputs
when evidence is missing. Never invent papers, datasets, repository links, code execution, experiments or metrics.
Help me keep a portable research_decisions.md (decisions, reasons, evidence, next actions) with an evidence ledger.
Start by summarising the scope of my attached review WITHOUT assuming content you cannot see, then ask me what to confirm.
```

### P02 · Extract evidence-based gaps
Tool: chatbot · Attach: review PDF · Save: gap matrix
```
Analyse my attached review. Identify its central methods, datasets, measured findings, author-reported limitations and
unresolved questions. Propose five computationally testable research gaps. For each: cite the specific section/table of
my review; separate limitations stated in the original papers from your own inference; state the experimental question and
what evidence would test it; flag claims needing verification against the original publications.
Seed ideas to evaluate (treat as hypotheses, not facts): (a) Sec. 6.1 claims benchmark scores are "not interchangeable" but
this is never tested; (b) Table 2/Table 3 recommend calibration and abstention reporting but nothing is measured;
(c) Sec. 3.3/5.3/6.1 say consistency does not establish truth; (d) task mode (closed-book vs evidence-grounded) is proposed
as a reporting axis but not tested; (e) contamination and temporal drift (Table 3) need data/tools I may not have.
Rank three gaps by significance, resources (free Colab GPU, 4 weeks) and feasibility. Do not claim novelty without a separate literature search.
```

### P03 · Literature → executable experiment map
Tool: chatbot or Antigravity Planning · Attach: review · Save: mapping table (mandatory bridge)
```
My review identified these candidate gaps: [PASTE the top 3 from P02]. Identify three published methods from my review that
I could realistically implement or reproduce (e.g., MMLU/PubMedQA zero-shot multiple-choice scoring; SelfCheckGPT-style
black-box consistency; atomic-claim scoring à la FActScore). For each, map: original paper → algorithm → dataset → code/model
access → metric → limitations → compute. Distinguish VERIFIED access from suggestions. Compare three studies: (A) reproduction
+ extension, (B) controlled comparison, (C) targeted modification: smallest working baseline, testable question, resource burden,
evidence required. Recommend ONE minimum viable experiment for [Google Colab free T4 GPU; 15 GB VRAM] and [4 weeks, ~6 h/week].
Give a literature-to-experiment table and a list of things I must personally verify before coding. No elaborate new architecture.
```

### P04 · Feasibility, novelty, resource access
Tool: chatbot with browsing · Save: novelty/feasibility matrix + go/no-go
```
Candidate study: "Do accuracy, calibration, selective-prediction and option-order consistency rank small open LLMs the same way
on PubMedQA and MMLU science subsets, and does adding the abstract change them equally?" Build a feasibility and novelty plan.
Give scholarly search strings and categories of neighbouring work to inspect WITHOUT inventing titles, DOIs or URLs.
Seed search strings: "calibration large language models multiple-choice medical", "option order sensitivity multiple choice LLM",
"selective prediction risk coverage language models", "metric disagreement LLM benchmark ranking", "multi-metric holistic evaluation language models".
Neighbours I should check exist and read (verify, do not trust): HELM-style multi-metric evaluation; work on calibration of modern
networks and of LLMs; work on multiple-choice option-order bias; SelfCheckGPT; FActScore. Produce a matrix: published method |
VERIFIED contribution (source) | limitation | my proposed difference | evidence still missing. Check dataset/model licences and
availability and flag unresolved claims. Recommend a fallback if a model or dataset is unavailable. Do not claim novelty by default.
```

### P05 · Research questions and success criteria
Tool: chatbot · Save: one-page brief (Gate G1 template)
```
Verified gap: [PASTE]. Study: controlled comparative evaluation (Pathway B), no training. Sources: [PASTE verified sources].
Draft RQ1 (ranking agreement across accuracy/ECE/Brier/AURC/selective accuracy/consistency across 5 slices x 3 models),
RQ2 (PubMedQA closed-book vs context, paired), RQ3 (does permutation-consistency separate correct from incorrect answers or
reward consistent errors). For each give: inputs, comparators, variables, metric, a decision rule with thresholds I must justify
BEFORE running, and an observation that would contradict my expectation. Separate what I intend to investigate from what is
already shown. Flag over-broad questions (e.g., 3 models make rank correlation coarse). Produce the one-page research brief.
```

---
## PHASE II — EXPERIMENT PLAN

### P06 · Verify datasets and comparators
Tool: chatbot with browsing · Save: dataset/baseline validation sheet
```
Question: [RQ1–3]. Primary datasets: PubMedQA (HF qiaojin/PubMedQA, config pqa_labeled) and MMLU (HF cais/mmlu) science subjects.
Comparators: three open instruction-tuned models plus chance and majority-class baselines, with EleutherAI lm-evaluation-harness
as the reference implementation. For each, tell me the official source to verify, file structure, protocol, licence/access terms,
compute, common pitfalls. Explain the evaluation design (no training, so what are my "splits"? how do I freeze the evaluation set?).
List leakage risks: pretraining contamination, prompt/answer-letter leakage, option-order bias, label noise in MMLU, class imbalance
in PubMedQA ("maybe" minority). Do not present unverified access claims as facts. End with the simplest credible baseline and an
independent verification checklist.
```

### P07 · Freeze the experiment blueprint
Tool: Antigravity · Planning · Pro-class · Attach: `@configs/protocol.yaml` `@README.md` · Save: `research_decisions.md`
```
Audit (do NOT rewrite) @configs/protocol.yaml and the method described in @README.md. Check: input/output definitions, option-letter
scoring, permutation scheme, metric definitions and directions (higher/lower is better), seed plan, bootstrap design, expected tables/figures,
compute budget, and at least one deeper analysis. List threats to validity and a pre-experiment checklist to freeze primary metric,
evaluation sample and comparison protocol before viewing results. Explain how legitimate protocol changes should be logged afterwards.
Output a Mermaid diagram of the pipeline and a list of concrete problems you found in the config or README. Never fabricate expected outcomes.
```

---
## PHASE III — AI CODING AND BASELINE VALIDATION

### P08 · Configure and test the environment
Tool: Antigravity · Fast · Flash-class · Colab for GPU
```
Open this project. Do not modify code. (1) Run `python -m pytest -q tests` (or `python tests/test_metrics.py` if pytest is missing) and show the real output.
(2) Explain in plain language what each test checks. (3) Then give me the exact commands to set up Google Colab (free T4) for scripts/00_env_check.py
and scripts/01_prepare_data.py, with the expected OUTPUT TYPE and failure diagnostics. Stop and ask me to PASTE the real console output before proceeding.
Do not claim compatibility you have not tested.
```

### P09 · Read, explain and test the baseline code, one module at a time
(The kit already contains drafted code, so the goal is understanding and ownership, not blind generation.)
Tool: Antigravity · Fast · Attach one file at a time
```
You are my pair programmer. I must be able to explain this code in a viva. Take ONLY @src/factbench/[scoring.py | prompts.py | metrics.py | analysis.py]:
(1) explain its purpose and assumptions, (2) explain input/output shapes, (3) point out any bug, ambiguity or hidden assumption (e.g. tokenisation of option letters,
left padding, tie handling, metric direction), (4) propose or improve a unit test, (5) wait for my real execution output before the next module.
Do not rewrite unrelated modules, present pseudocode as runnable code, or claim you ran anything you did not run. Record anything you change so I can log it.
```

### P10 · Verify and reproduce the baseline
Tool: Antigravity · Planning · Pro-class · Attach: code + `@baseline_validation.md` + `@results/validation/`
```
I implemented a multiple-choice scorer and compared it with EleutherAI lm-evaluation-harness on the same model and MMLU subjects.
Here are my code, configs, REAL logs and numbers: [PASTE]. Audit against the harness methodology: prompt wording, answer-token handling,
batching/padding, dtype, seed, metric. Design sanity tests; investigate the gap; separate acceptable adaptations from changes that invalidate comparison.
Produce a baseline validation report with VERIFIED / UNCERTAIN / NEEDS CORRECTION items and the exact extra checks needed.
Do not say the code is correct just because it runs, and do not demand exact numerical replication under different conditions.
```

### P11 · Debug with actual evidence
Tool: Antigravity · Fast
```
My code failed or behaved unexpectedly. Code: [PASTE]. Actual traceback/output: [PASTE]. Expected behaviour: [DESCRIBE].
Determine likely causes, separating confirmed evidence from hypotheses. Suggest the smallest safe fix with an explanation of each change,
a verification test and one edge case, and tell me whether earlier results must be re-run. Ask for more diagnostics if needed.
Do not claim you ran the code or invent logs. Help me add this case to ai_assistance_log.md.
```

---
## PHASE IV — CONTRIBUTION AND EVIDENCE

### P12 · Choose and evaluate a defensible contribution
Tool: chatbot or Antigravity Planning
```
Baseline validated; REAL results so far: [PASTE]. Question: [RQ1–3]; literature: [PASTE]. Compare three pathways: A reproduce under a new condition,
B controlled comparison (current plan), C one isolated targeted modification (candidate: the option-order ensemble, already stored as acc_ens, as a
debiasing step). For each: scientific question, code changes, compute, advantages, risks, evidence needed. Recommend the most defensible pathway.
If modification is chosen, guide me to make ONE isolated change that preserves the baseline. Do not guarantee gains or call a routine comparison novel.
```

### P13 · Evaluate using real outputs only
Tool: Antigravity · Fast · Attach: `@scripts/03_evaluate.py` `@src/factbench/analysis.py` + a few real rows from `results/raw`
```
Audit my evaluation code (do not replace it). Check: ID/label alignment across models, missing values, canonical vs displayed option order after permutation,
metric directions, bootstrap pairing, tie handling in selective prediction, and any leakage or incompatible comparator settings. Add checks rather than rewrite.
Never invent example values in final tables and never claim significance without a valid test and stated assumptions. Ask for any file you need before interpreting numbers.
```

### P14 · Interpret outcomes, including negative results
Tool: chatbot · Attach: `results/*.csv`, figures
```
Questions: [RQ1–3]. Protocol: [PASTE from configs/protocol.yaml]. REAL results: [PASTE CSV contents / figures].
For each question separate: direct observations, possible mechanisms, alternative explanations, unsupported claims. Discuss relative performance,
bootstrap uncertainty, cost, failure modes (consistently-wrong items, position bias, "maybe" collapse) and limits of the datasets/protocol.
State which hypotheses are supported, contradicted or unresolved against my PRE-REGISTERED decision rules. Recommend at most three follow-up experiments
ranked by cost and information gain. Do not exaggerate small effects, spin negative results, or assume a metric measures something outside its scope.
```

---
## PHASE V — WRITE, REVIEW, PUBLISH

### P15 · Evidence-grounded LaTeX manuscript
Tool: chatbot · Attach: brief, verified refs, results CSVs
```
Use ONLY my verified problem brief [PASTE], original references [PASTE verified .bib entries], methodology [PASTE], setup [PASTE protocol.yaml], and measured results [PASTE].
Help me write a 6–10 page paper for Overleaf using the instructor's template. First produce an outline and evidence checklist, then Related Work, Method, Experimental Setup,
Results, Analysis, Limitations, Introduction, Conclusion; write the Abstract LAST. Mark missing facts [TO VERIFY]. Preserve the distinction between prior methods,
my experimental contribution and conjecture. Build BibTeX only from metadata I verified. Flag inconsistencies between figures, tables, code and claims. Results tables
must be generated from results/*.csv, not typed by hand.
```

### P16 · Demanding technical peer review
Tool: chatbot · Attach: draft PDF
```
Review my draft as a critical technical reviewer: significance, literature coverage, contribution validity, technical correctness, leakage, baseline fairness, experimental
design, robustness of conclusions (only 3 models; item-level bootstrap only), reproducibility, figure/table accuracy, citation traceability, clarity. Separate MAJOR and MINOR concerns with
section/page, why it matters, and one concrete remedy or experiment. Identify unsupported novelty/performance claims. Do not default to praise or invent literature.
Return a prioritised revision checklist and an author-response table.
```

### P17 · Reproducible GitHub documentation
Tool: Antigravity · Fast · Attach: `@README.md` + `@scripts/`
```
Using ONLY commands I have genuinely executed [PASTE the commands and their outcomes], update README.md: problem statement, setup, data preparation, baseline validation, experiment run,
evaluation, figure reproduction, compute needs, limitations, credits, licence, citation. Keep [TEST REQUIRED] on anything not executed. Check that requirements.lock.txt exists,
paths are realistic, no secrets or restricted data (GPQA) are present, and add a minimum reproduction checklist for a second student.
```

---
## PHASE VI — HUMAN–AI ACCOUNTABILITY

### P18 · Audit the human–AI collaboration
Tool: chatbot · Attach: your real log
```
From my REAL prompt log [PASTE ai_assistance_log.md], diffs [PASTE git log/diffs], debugging notes [PASTE] and experiment files, prepare a concise AI-use and research-integrity report.
For each significant AI-assisted activity record: task, tool, prompt/version, AI proposal, my decision, verification method, change, outcome. Include at least three REAL examples of AI limits,
corrections or deliberate verification if they occurred; if fewer occurred, say so. State which conclusions depended on my reasoning, what I learned about vibe coding and what risks remain.
Note that the initial code scaffold was drafted by Claude and tested with a dummy backend. Do not invent experiences, logs, fixes or gains.
```
