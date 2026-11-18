#!/usr/bin/env python
"""Environment + tokenizer check. Paste the FULL console output into baseline_validation.md / your AI log."""
import argparse
import platform
from _common import load_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizers", action="store_true", help="also load each model's tokenizer and check option-letter token ids")
args = ap.parse_args()
cfg = load_cfg()

print("python", platform.python_version(), platform.platform())
for lib in ["numpy", "scipy", "pandas", "matplotlib", "yaml", "datasets", "transformers", "accelerate", "torch", "bitsandbytes", "lm_eval"]:
    try:
        mod = __import__(lib)
        print(f"{lib:14s} {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        print(f"{lib:14s} NOT AVAILABLE ({type(e).__name__})")
try:
    import torch
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("gpu:", p.name, f"{p.total_memory / 2**30:.1f} GiB")
except Exception as e:
    print("torch check failed:", e)

if args.tokenizers:
    from transformers import AutoTokenizer
    from factbench.scoring import letter_token_ids
    for m in cfg["models"]:
        try:
            tok = AutoTokenizer.from_pretrained(m["hf_id"])
            ids = {L: letter_token_ids(tok, L) for L in "ABCD"}
            ok = all(ids.values())
            print(f"[{'PASS' if ok else 'FAIL'}] {m['hf_id']} option-letter token ids: {ids}")
        except Exception as e:
            print(f"[FAIL] {m['hf_id']}: {type(e).__name__}: {e}")
