"""Prompt construction. Display position j shows option perm[j]."""
LETTERS = "ABCDEFGH"
MAX_CONTEXT_CHARS = 6000


def _opts(options, perm):
    return "\n".join(f"{LETTERS[j]}. {options[p]}" for j, p in enumerate(perm))


def build_text(item, perm, mode="closed", style="chat"):
    """mode: 'closed' (question only) or 'context' (PubMedQA abstract provided).
    style: 'chat' (instruction prompt) or 'harness' (raw MMLU prompt mimicking lm-evaluation-harness;
    used ONLY for baseline validation, the exact harness wording must be diffed against its --log_samples output)."""
    q = item["question"].strip()
    opts = _opts(item["options"], perm)
    if style == "harness":
        if item["dataset"] != "mmlu":
            raise ValueError("harness style is only defined for MMLU")
        subj = item["subject"].replace("_", " ")
        return f"The following are multiple choice questions (with answers) about {subj}.\n\n{q}\n{opts}\nAnswer:"
    if item["dataset"] == "mmlu":
        subj = item["subject"].replace("_", " ")
        return (f"The following is a multiple-choice question about {subj}. "
                f"Respond with only the letter of the correct answer.\n\nQuestion: {q}\n{opts}\nAnswer:")
    if item["dataset"] == "pubmedqa":
        if mode == "context":
            ctx = (item["context"] or "")[:MAX_CONTEXT_CHARS]
            return (f"Read the abstract and answer the research question based on it.\n\nAbstract:\n{ctx}\n\n"
                    f"Question: {q}\n{opts}\nRespond with only the letter of your answer.")
        return (f"Answer the following biomedical research question.\n\nQuestion: {q}\n{opts}\n"
                f"Respond with only the letter of your answer.")
    raise ValueError(f"unknown dataset {item['dataset']}")
