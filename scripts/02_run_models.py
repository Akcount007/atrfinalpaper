#!/usr/bin/env python
"""Score every item under K option orders and save raw option log-probs (resumable, incremental)."""
import argparse
import itertools
import json
import platform
import subprocess
import time
import zlib
from pathlib import Path
import numpy as np
from _common import ROOT, load_cfg
from factbench import data, prompts, scoring

ap = argparse.ArgumentParser()
ap.add_argument("--model-name", required=True, help="name from configs/protocol.yaml (or any name with --backend dummy)")
ap.add_argument("--backend", default="hf", choices=["hf", "dummy"])
ap.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
ap.add_argument("--out", default=str(ROOT / "results" / "raw"))
ap.add_argument("--limit", type=int, default=None, help="first N items per slice (debugging only; flagged in meta)")
ap.add_argument("--slices", nargs="*", default=None)
ap.add_argument("--chunk", type=int, default=16, help="items per write/checkpoint")
args = ap.parse_args()
cfg = load_cfg()
K, bs, style, seed = cfg["scoring"]["n_permutations"], cfg["scoring"]["batch_size"], cfg["scoring"]["prompt_style"], cfg["seed"]


def make_perms(item_id, m, k, seed):
    ident = tuple(range(m))
    rest = [p for p in itertools.permutations(range(m)) if p != ident]
    rng = np.random.default_rng([seed, zlib.crc32(item_id.encode())])
    return [list(ident)] + [list(rest[i]) for i in rng.permutation(len(rest))[: k - 1]]


def slices_of(item):
    if item["dataset"] == "pubmedqa":
        return [("pubmedqa_closed", "closed"), ("pubmedqa_context", "context")]
    return [(f"mmlu_{item['domain']}", "closed")]


items = []
for f in sorted(Path(args.data_dir).glob("*.jsonl")):
    items += data.read_jsonl(f)
by_slice = {}
for it in items:
    for sl, mode in slices_of(it):
        by_slice.setdefault(sl, []).append((it, mode))
if args.slices:
    by_slice = {k: v for k, v in by_slice.items() if k in args.slices}

if args.backend == "dummy":
    scorer = scoring.DummyScorer(args.model_name)
else:
    mcfg = next(m for m in cfg["models"] if m["name"] == args.model_name)
    scorer = scoring.HFScorer(mcfg["hf_id"], mcfg.get("quant", "none"), name=args.model_name)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


meta = {"model": args.model_name, "backend": scorer.backend, "limit": args.limit, "protocol_version": cfg["protocol_version"],
        "seed": seed, "K": K, "batch_size": bs, "prompt_style": style, "git_commit": git_commit(),
        "python": platform.python_version(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
if args.backend == "hf":
    import torch, transformers
    meta.update({"hf_id": mcfg["hf_id"], "quant": mcfg.get("quant"), "torch": torch.__version__,
                 "transformers": transformers.__version__,
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})
(out / f"{args.model_name}__meta.json").write_text(json.dumps(meta, indent=2))

for sl, pairs in sorted(by_slice.items()):
    pairs = pairs[: args.limit] if args.limit else pairs
    path = out / f"{args.model_name}__{sl}.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    todo = [(it, mode) for it, mode in pairs if it["id"] not in done]
    print(f"[{args.model_name}] slice={sl} total={len(pairs)} already_done={len(done)} todo={len(todo)}", flush=True)
    for c in range(0, len(todo), args.chunk):
        chunk = todo[c: c + args.chunk]
        texts, nopts, back = [], [], []
        for ci, (it, mode) in enumerate(chunk):
            m = len(it["options"])
            perms = make_perms(it["id"], m, K, seed)
            for k, p in enumerate(perms):
                texts.append(scorer.render(prompts.build_text(it, p, mode, style), style))
                nopts.append(m)
                back.append((ci, k))
        res = []
        for b in range(0, len(texts), bs):
            res += scorer.score(texts[b: b + bs], nopts[b: b + bs])
        recs = [{"id": it["id"], "slice": sl, "model": args.model_name, "gold": it["gold"], "perms": [], "logp": []}
                for it, _ in chunk]
        for (ci, k), r in zip(back, res):
            recs[ci]["logp"].append([round(float(x), 6) for x in r])
        for ci, (it, _) in enumerate(chunk):
            recs[ci]["perms"] = make_perms(it["id"], len(it["options"]), K, seed)
        with open(path, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  {min(c + len(chunk), len(todo))}/{len(todo)}", flush=True)
print("finished")
