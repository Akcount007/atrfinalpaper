#!/usr/bin/env bash
# [TEST REQUIRED] Reference run with EleutherAI lm-evaluation-harness on the SAME model + subjects.
# Verify task names first:   lm-eval ls tasks | grep -i -E "pubmedqa|mmlu_college_biology|mmlu_high_school_physics"
# (The CLI changed to subcommands in late 2025; if `lm_eval --model ...` fails, try `lm-eval run --model ...`.)
set -e
MODEL_ID=${1:-Qwen/Qwen2.5-1.5B-Instruct}
pip install -q "lm_eval[hf]"
lm_eval --model hf \
  --model_args pretrained=${MODEL_ID},dtype=float16 \
  --tasks mmlu_college_biology,mmlu_high_school_physics \
  --num_fewshot 0 --batch_size 8 --seed 42 \
  --output_path results/validation/harness --log_samples
# Then: diff one logged prompt against "example_prompt" in results/validation/ours_<model>.json
