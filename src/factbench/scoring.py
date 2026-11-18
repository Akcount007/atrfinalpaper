"""Scoring backends. Each returns, per prompt, log-probabilities over the displayed option letters
(renormalised over the options only)."""
import hashlib

import numpy as np

from .prompts import LETTERS


class DummyScorer:
    """Pseudo-random scorer for SMOKE TESTS ONLY. Its outputs are noise and must never be reported."""
    backend = "dummy"

    def __init__(self, name="dummy"):
        self.name = name

    def render(self, text, style="chat"):
        return text

    def score(self, texts, n_opts):
        out = []
        for t, n in zip(texts, n_opts):
            seed = int(hashlib.sha256((self.name + t).encode()).hexdigest()[:8], 16)
            out.append(np.log(np.random.default_rng(seed).dirichlet(np.ones(n) * 0.5)))
        return out


def letter_token_ids(tok, letter):
    """Token ids that can start an answer with this letter (handles BPE 'Ġ' and SentencePiece '▁' variants)."""
    ids = set()
    for cand in (letter, "▁" + letter, "Ġ" + letter):
        i = tok.convert_tokens_to_ids(cand)
        if i is not None and i != tok.unk_token_id:
            ids.add(int(i))
    return sorted(ids)


class HFScorer:
    backend = "hf"

    def __init__(self, hf_id, quant="none", name=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = name or hf_id
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {}
        if quant == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(hf_id, device_map="auto", dtype=torch.float16, **kw)
        except TypeError:  # older transformers use torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(hf_id, device_map="auto", torch_dtype=torch.float16, **kw)
        self.model.eval()
        self.ids = {L: letter_token_ids(self.tok, L) for L in LETTERS}
        for L in LETTERS[:3]:
            assert self.ids[L], f"no token id found for option letter {L}"

    def render(self, text, style="chat"):
        if style == "harness":
            return text
        return self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                            add_generation_prompt=True)

    def score(self, texts, n_opts):
        torch = self.torch
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            try:
                logits = self.model(**enc, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            except TypeError:
                logits = self.model(**enc, position_ids=pos, use_cache=False).logits[:, -1, :]
        logp = torch.log_softmax(logits.float(), dim=-1)
        out = []
        for row, n in zip(logp, n_opts):
            per = torch.stack([torch.logsumexp(row[self.ids[L]], dim=0) for L in LETTERS[:n]])
            out.append(torch.log_softmax(per, dim=0).cpu().numpy())
        return out
