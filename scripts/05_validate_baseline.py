#!/usr/bin/env python
"""Baseline validation (Gate G2): score FULL MMLU subjects with a raw 'harness-style' prompt and compare to
EleutherAI lm-evaluation-harness on the same model/subject. Exact equality is NOT expected; investigate large gaps."""
import argparse
import json
from pathlib import Path
import numpy as np
from _common import ROOT, load_cfg
from factbench import prompts, scoring

ap = argparse.ArgumentParser()
ap.add_argument("--model-name", required=True)
ap.add_argument("--subjects", nargs="+", default=["college_biology", "high_school_physics"])
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()
cfg = load_cfg()
mcfg = next(m for m in cfg["models"] if m["name"] == args.model_name)
from datasets import load_dataset
sc = scoring.HFScorer(mcfg["hf_id"], mcfg.get("quant", "none"), name=args.model_name)
res = {}
for s in args.subjects:
    ds = list(load_dataset(cfg["data"]["mmlu"]["hf_id"], s, split="test"))[: args.limit]
    items = [{"dataset": "mmlu", "subject": s, "question": r["question"], "options": list(r["choices"]), "gold": int(r["answer"])} for r in ds]
    texts = [sc.render(prompts.build_text(it, [0, 1, 2, 3], "closed", "harness"), "harness") for it in items]
    pred = []
    for i in range(0, len(texts), cfg["scoring"]["batch_size"]):
        pred += [int(np.argmax(x)) for x in sc.score(texts[i:i + cfg["scoring"]["batch_size"]], [4] * len(texts[i:i + cfg["scoring"]["batch_size"]]))]
    acc = float(np.mean([p == it["gold"] for p, it in zip(pred, items)]))
    res[s] = {"n": len(items), "acc_ours_harness_style": acc, "example_prompt": texts[0]}
    print(f"{s}: n={len(items)} acc={acc:.4f}")
out = ROOT / "results" / "validation"; out.mkdir(parents=True, exist_ok=True)
(out / f"ours_{args.model_name}.json").write_text(json.dumps(res, indent=2))
print("saved; now run scripts/05_harness_check.sh and compare in baseline_validation.md")
