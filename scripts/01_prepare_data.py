#!/usr/bin/env python
"""Download + subsample datasets with a fixed seed; write data/processed/*.jsonl and MANIFEST.json."""
import argparse
import datetime
import json
from pathlib import Path
from _common import ROOT, load_cfg
from factbench import data

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
ap.add_argument("--pubmedqa-test-ids", default=None,
                help="optional JSON file (dict keyed by pubid, or list of pubids) holding the OFFICIAL 500 test ids; "
                     "check its format in the official repo before use")
args = ap.parse_args()
cfg = load_cfg()
seed = cfg["seed"]
out = Path(args.out)
import datasets
from datasets import load_dataset

test_ids = None
if args.pubmedqa_test_ids:
    raw = json.load(open(args.pubmedqa_test_ids))
    test_ids = {int(k) for k in (raw.keys() if isinstance(raw, dict) else raw)}

pc = cfg["data"]["pubmedqa"]
pq = load_dataset(pc["hf_id"], pc["config"], split="train")
pq_items = data.pubmedqa_items(pq, pc["n_items"], seed, test_ids)
data.write_jsonl(out / "pubmedqa.jsonl", pq_items)

mc = cfg["data"]["mmlu"]
mm_items = data.mmlu_items(lambda s: load_dataset(mc["hf_id"], s, split="test"), mc["domains"], mc["n_per_domain"], seed)
data.write_jsonl(out / "mmlu.jsonl", mm_items)

manifest = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "seed": seed,
            "datasets_version": datasets.__version__, "protocol_version": cfg["protocol_version"],
            "official_test_ids_used": bool(test_ids), "files": {}}
for f in ["pubmedqa.jsonl", "mmlu.jsonl"]:
    rows = data.read_jsonl(out / f)
    manifest["files"][f] = {"n": len(rows), "sha256": data.sha256_file(out / f)}
(out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
