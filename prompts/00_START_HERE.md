# How to use this kit with Antigravity (free plan or Google AI Pro)

## Division of labour
- **Chat-only prompts** (P01–P05, P14–P16, P18): paste into any chatbot that accepts file uploads (Gemini app, Claude, etc.). Saves IDE quota.
- **Coding prompts** (P08–P11, P13, P17): use Antigravity, with this project folder opened as the workspace.
- **GPU runs:** the Antigravity agent cannot use your Colab GPU. Push the repo to GitHub (or upload the zip to Colab), run the scripts there, then download `results/` back and let the agent help analyse/debug.

## Quota strategy (Antigravity quotas change often; check Settings → Models in the app)
- Reports in mid-2026 describe the free tier as a weekly quota (numbers not published) and Google AI Pro as a larger quota refreshed every ~5 hours, with Gemini Flash-class models cheap and Pro-class models consuming quota quickly. Treat this as INFERRED and look at your own quota panel.
- Use **Fast mode + a Flash-class model** for small edits, test runs, README work. Use **Planning mode + Pro-class model** only for P03, P07, P10, P16.
- Keep `AGENTS.md` short (it is read on every task). Start a **new conversation per gate** to avoid huge contexts.
- Set terminal policy to a review-driven option so commands do not auto-run unseen.
- Folder for workspace rules: `.agents/rules/` (some docs say `.agent/rules/`; use whichever your build shows). `AGENTS.md` at the project root is also read.

## Gate plan (one gate per week)
| Gate | Deliverable | Prompts | Kit files to use |
|------|-------------|---------|------------------|
| G1 | 1-page brief + verified sources | P01–P05 | `research_decisions.md` |
| G2 | Baseline verification | P06–P11 | scripts 00, 01, 05; `baseline_validation.md`; freeze `configs/protocol.yaml` |
| G3 | Experimental evidence | P12–P14 | scripts 02, 03, 04; `results/` |
| G4 | Paper + repo + AI audit + demo | P15–P18 | `paper/`, `README.md`, `ai_assistance_log.md` |

## Rules you must follow (the assignment grades these)
1. The code in this kit was drafted by Claude (AI) and tested only with a dummy backend. You must read it, run it, understand it, and log that origin in `ai_assistance_log.md`.
2. Log your REAL prompts and real corrections. Do not copy someone else's log.
3. Verify every link/citation/licence yourself. Update the evidence ledger in `research_decisions.md`.
4. Paste actual outputs when a prompt says `PASTE`. Never let an AI fill in numbers.

## Day-one lab target (first 2 hours)
Feasible question ✔ (RQ1–3) · identifiable baseline (lm-eval-harness + chance + majority) ✔ · one verified execution step:
`python -m pytest -q tests` then `python scripts/00_env_check.py --tokenizers` and paste the output.
