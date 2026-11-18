# Data access (no raw data is committed; regenerate with scripts/01_prepare_data.py)

Status: **CHECKED** = I (Claude) saw this on a page/search result on 12 Nov 2026 — you must still open it yourself and tick it off in `research_decisions.md`. **TO VERIFY** = from memory, unchecked.

| Resource | Where | Notes | Status |
|----------|-------|-------|--------|
| PubMedQA (used) | https://huggingface.co/datasets/qiaojin/PubMedQA (config `pqa_labeled`) · repo https://github.com/pubmedqa/pubmedqa · homepage https://pubmedqa.github.io | 1,000 expert-labelled yes/no/maybe items, single `train` split on HF; fields `pubid, question, context{contexts,labels,meshes}, long_answer, final_decision`; the official 500-item test set ids live in the GitHub repo; MIT | CHECKED |
| MMLU (used) | https://huggingface.co/datasets/cais/mmlu (config = subject name, split `test`) | 57 subjects, 4 options; `test` has 14,042 rows on HF (paper says 14,079); known label errors; MIT per secondary source | CHECKED (licence: verify) |
| lm-evaluation-harness (baseline reference) | https://github.com/EleutherAI/lm-evaluation-harness · `pip install "lm_eval[hf]"` | CLI moved to subcommands in late 2025; confirm task names with `lm-eval ls tasks` | CHECKED |
| SciFact (optional extension) | https://github.com/allenai/scifact · HF `allenai/scifact` (script-based; may fail with `datasets>=4`) · tarball https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz | claims + abstracts; **CC BY-NC 2.0** (non-commercial); verify test-label availability | CHECKED |
| GPQA (NOT used) | https://huggingface.co/datasets/Idavidrein/gpqa (gated) · https://github.com/idavidrein/gpqa | 448 expert questions; CC-BY-4.0 but terms forbid posting examples online → keep out of public repos/papers | CHECKED |
| FreshQA (NOT used) | https://github.com/freshllms/freshqa | 600 dynamic questions updated weekly; needs search-augmented models & a fixed snapshot date | CHECKED |
| TruthfulQA | HF `truthfulqa/truthful_qa` · github `sylinrl/TruthfulQA` | | TO VERIFY |
| HaluEval | github `RUCAIBox/HaluEval` | | TO VERIFY |
| SciQ | HF `allenai/sciq` | | TO VERIFY |
| ScienceQA | github `lupantech/ScienceQA` · HF `derek-thomas/ScienceQA` | multimodal; heavy for free GPUs | TO VERIFY |
| SelfCheckGPT | github `potsawee/selfcheckgpt` | | TO VERIFY |
| FActScore | github `shmsw25/FActScore` | needs retrieval corpus + an LLM judge; heavy | TO VERIFY |
