# Project rules for AI agents (Antigravity / Gemini)

Project: empirical test of whether LLM factuality metrics (accuracy, calibration, selective prediction,
permutation-consistency) rank models the same way, on PubMedQA + MMLU science subsets. Academic coursework
(AI Tools for Research). Research integrity matters more than speed.

## Non-negotiable
1. NEVER invent results, metrics, logs, citations, URLs, DOIs, dataset names or execution output.
2. After writing code, RUN it in the terminal and show the real output. If you cannot run it, say so.
3. Do not edit `configs/protocol.yaml` (frozen) or anything under `results/raw/`. Propose changes in
   `research_decisions.md` and wait for the human.
4. One module / one change at a time. Do not rewrite unrelated files. Keep diffs small and explain them.
5. Mark claims as VERIFIED (checked from source or executed), INFERRED, or UNKNOWN.
6. Never use the DUMMY backend for anything except smoke tests; its output is noise.
7. Never commit secrets, HF tokens, or GPQA data (its terms forbid publishing examples).
8. No test-set tuning: prompts/settings are fixed by the protocol before real runs.

## Layout
src/factbench (library) | scripts/ (CLI entrypoints 00..05) | tests/ | configs/ | results/ | paper/ | prompts/
Run tests: `python -m pytest -q tests` (fallback: `python tests/test_metrics.py`).

## Style
Python 3.10+, type hints where cheap, docstrings that state metric direction (higher/lower is better).
Keep answers short. Prefer reading existing files over re-explaining them.
