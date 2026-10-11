"""Shared perplexity harness for the compression-combination experiment.

Fixes a methodology flaw found in earlier runs: the in-loop accept/reject
check used 4 short hand-written sentences, and the greedy search overfit
to them - the 5-layer-pruned 0.5B model was "accepted at 1.5x" but
measured ~2.8x worse on a broader corpus. Here: wikitext-2 *validation*
drives in-loop decisions, wikitext-2 *test* (never seen during search)
is the only thing reported.
"""

from functools import lru_cache
from pathlib import Path

import torch

HERE = Path(__file__).parent
CTX = 512


def _wikitext(split: str) -> str:
    path = HERE / f"wikitext2-{split}.txt"
    if not path.exists():
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        path.write_text("\n\n".join(ds["text"]))
    return path.read_text()


@lru_cache(maxsize=None)
def _token_chunks(tokenizer_name: str, split: str, n_chunks: int) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    text = _wikitext(split)
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    usable = (ids.numel() // CTX) * CTX
    chunks = ids[:usable].view(-1, CTX)
    return chunks[:n_chunks]


def perplexity(model, tokenizer_name: str, split: str, n_chunks: int) -> float:
    chunks = _token_chunks(tokenizer_name, split, n_chunks)
    model.eval()
    nll, count = 0.0, 0
    with torch.no_grad():
        for chunk in chunks:
            inp = chunk.unsqueeze(0)
            loss = model(input_ids=inp, labels=inp).loss
            nll += loss.item() * (CTX - 1)
            count += CTX - 1
    return float(torch.exp(torch.tensor(nll / count)))


def val_ppl_fn(tokenizer_name: str, n_chunks: int = 8):
    """In-loop decision metric (wikitext-2 validation, 8x512 tokens)."""
    return lambda model: perplexity(model, tokenizer_name, "validation", n_chunks)


def test_ppl(model, tokenizer_name: str, n_chunks: int = 32) -> float:
    """Reported metric only (wikitext-2 test, 32x512 tokens)."""
    return perplexity(model, tokenizer_name, "test", n_chunks)


def param_count(model) -> int:
    # Tied lm_head/embedding counted once (parameters() dedupes shared tensors).
    return sum(p.numel() for p in model.parameters())
