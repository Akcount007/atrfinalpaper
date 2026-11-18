"""End-to-end SMOKE test with the DUMMY backend on synthetic items. Proves the plumbing runs; says NOTHING about model quality."""
import json, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from factbench import data


def synth_items():
    items = []
    for dom in ["biomed", "physchem", "math"]:
        for i in range(40):
            items.append({"id": f"mmlu-{dom}-{i}", "dataset": "mmlu", "domain": dom, "subject": dom, "question": f"q{i}?",
                          "context": None, "options": ["a", "b", "c", "d"], "gold": i % 4})
    for i in range(60):
        items.append({"id": f"pubmedqa-{i}", "dataset": "pubmedqa", "domain": "pubmedqa", "subject": "pubmedqa",
                      "question": f"pq{i}?", "context": "Background: x\nResults: y", "options": ["yes", "no", "maybe"], "gold": i % 3})
    return items


def run(*a):
    r = subprocess.run([sys.executable, *map(str, a)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_dummy_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        items = synth_items()
        data.write_jsonl(t / "data" / "items.jsonl", items)
        for m in ["dummy-a", "dummy-b", "dummy-c"]:
            run("scripts/02_run_models.py", "--backend", "dummy", "--model-name", m, "--data-dir", t / "data", "--out", t / "raw")
        # refuses dummy without the flag
        r = subprocess.run([sys.executable, "scripts/03_evaluate.py", "--raw", t / "raw", "--out", t / "res"], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0 and "DUMMY" in (r.stdout + r.stderr)
        run("scripts/03_evaluate.py", "--raw", t / "raw", "--out", t / "res", "--data-dir", t / "data", "--allow-dummy", "--bootstrap", 50)
        for f in ["results.csv", "baselines.csv", "rank_by_metric.csv", "rank_tau_summary.csv", "task_mode_delta.csv",
                  "paired_model_tests.csv", "position_bias.csv", "pubmedqa_confusion.csv"]:
            assert (t / "res" / f).exists(), f
        run("scripts/04_figures.py", "--raw", t / "raw", "--res", t / "res", "--out", t / "fig")
        assert (t / "fig" / "fig_reliability.png").exists()
        # resumability: second run must skip everything
        out = run("scripts/02_run_models.py", "--backend", "dummy", "--model-name", "dummy-a", "--data-dir", t / "data", "--out", t / "raw")
        assert "todo=0" in out


if __name__ == "__main__":
    test_dummy_pipeline_end_to_end(); print("PASS smoke")
