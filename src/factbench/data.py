"""Dataset preparation helpers. All sampling is seeded and deterministic."""
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

PUBMEDQA_OPTIONS = ["yes", "no", "maybe"]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_sample(labels, n, seed):
    """Return sorted indices of a class-proportional sample of size n."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    exact = counts / counts.sum() * n
    alloc = np.floor(exact).astype(int)
    for j in np.argsort(-(exact - alloc))[: n - alloc.sum()]:
        alloc[j] += 1
    chosen = []
    for c, a in zip(classes, alloc):
        idx = np.where(labels == c)[0]
        chosen.extend(rng.choice(idx, size=min(a, len(idx)), replace=False).tolist())
    return sorted(chosen)


def pubmedqa_items(rows, n, seed, test_ids=None):
    """rows: iterable of HF PubMedQA pqa_labeled records.
    If test_ids (set of int pubids) is given, use exactly those items (e.g. the official 500-question test set)."""
    rows = list(rows)
    if test_ids:
        chosen = [i for i, r in enumerate(rows) if int(r["pubid"]) in test_ids]
    else:
        chosen = stratified_sample([r["final_decision"] for r in rows], n, seed)
    items = []
    for i in chosen:
        r = rows[i]
        ctx = r["context"]
        labels, texts = ctx["labels"], ctx["contexts"]
        parts = [f"{str(l).capitalize()}: {t}" for l, t in zip(labels, texts)]
        items.append({
            "id": f"pubmedqa-{r['pubid']}", "dataset": "pubmedqa", "domain": "pubmedqa",
            "subject": "pubmedqa", "question": r["question"], "context": "\n".join(parts),
            "options": PUBMEDQA_OPTIONS, "gold": PUBMEDQA_OPTIONS.index(r["final_decision"]),
        })
    return items


def mmlu_items(load_subject, domains, n_per_domain, seed):
    """load_subject(subject_name) -> iterable of dicts with question/choices/answer."""
    items = []
    for dom, subjects in domains.items():
        pool = []
        for s in subjects:
            for idx, r in enumerate(load_subject(s)):
                pool.append({
                    "id": f"mmlu-{s}-{idx}", "dataset": "mmlu", "domain": dom, "subject": s,
                    "question": r["question"], "context": None,
                    "options": list(r["choices"]), "gold": int(r["answer"]),
                })
        rng = np.random.default_rng([seed, zlib.crc32(dom.encode())])
        sel = rng.choice(len(pool), size=min(n_per_domain, len(pool)), replace=False)
        items.extend(pool[i] for i in sorted(sel.tolist()))
    return items
