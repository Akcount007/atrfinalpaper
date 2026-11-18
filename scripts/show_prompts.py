import sys
sys.path.insert(0, "src")
from factbench import data, prompts

its = data.read_jsonl("data/processed/pubmedqa.jsonl")[:2] + data.read_jsonl("data/processed/mmlu.jsonl")[:1]
for i in its:
    mode = "context" if i["dataset"] == "pubmedqa" else "closed"
    print(prompts.build_text(i, list(range(len(i["options"]))), mode))
    print("-----")
