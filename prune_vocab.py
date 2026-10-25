"""Trim a byte-level-BPE model's vocabulary (and its tied embedding matrix).

Qwen3.5 has a 248K-token vocabulary, so its embedding table is a third of the
0.8B model. Most of it is tokens a typed English assistant never uses. This
keeps:
  * every special/control token (chat markers, think tags, ...),
  * the 256 byte-level base tokens (so any text still tokenizes, just in
    smaller pieces),
  * every token seen when tokenizing the keep-corpus,
  * optionally every pure-ASCII token (--ascii),
  * the merge ancestors of anything kept, so BPE still reaches each token,
and rewrites the tokenizer arrays, merges, special ids and the embedding
rows. Everything else is copied unchanged. The input is a BF16/F16 GGUF;
quantize afterwards.

  prune_vocab.py in.gguf out.gguf keep-ids.txt [--ascii] [--min-count N]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402


def byte_decoder():
    """GPT-2's printable-unicode <-> byte table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("ids", type=Path, help="llama-tokenize --ids output over the keep-corpus")
    ap.add_argument("--ascii", action="store_true", help="also keep every pure-ASCII token")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--scripts", default="",
                    help="comma list of scripts whose tokens are all kept (optional language packs): "
                         "latin-ext, cyrillic, greek, arabic, hebrew, devanagari, thai, cjk, hangul")
    ap.add_argument("--rank-keep", type=int, default=0,
                    help="also keep every token with id below this (BPE ids follow merge order, so a "
                         "rough all-language frequency rank)")
    ap.add_argument("--no-corpus", action="store_true", help="ignore the corpus: keep by --rank-keep only")
    ap.add_argument("--top-pct", type=float, default=None,
                    help="keep only the most frequent corpus tokens, this percent of the original vocabulary")
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.src)
    tokens = reader.get_field("tokenizer.ggml.tokens").contents()
    types = reader.get_field("tokenizer.ggml.token_type").contents()
    merges = reader.get_field("tokenizer.ggml.merges").contents()
    n = len(tokens)
    dec = byte_decoder()

    counts = Counter(json.loads(args.ids.read_text()))
    keep = set(i for i, c in counts.items() if c >= args.min_count)
    if args.top_pct is not None:
        top = int(len(tokens) * args.top_pct / 100)
        keep = set(i for i, _ in counts.most_common(top))
    if args.no_corpus:
        keep = set()
    if args.rank_keep:
        keep.update(i for i in range(args.rank_keep) if types[i] == 1)
    n_corpus = len(keep)
    for i in range(n):
        if types[i] != 1:  # not a normal token: control, user-defined, byte, ...
            keep.add(i)
    base = {t: i for i, t in enumerate(tokens) if len(t) == 1 and t in dec}
    keep.update(base.values())
    if args.ascii:
        for i, t in enumerate(tokens):
            if types[i] == 1 and all(ch in dec and dec[ch] < 128 for ch in t):
                keep.add(i)
    if args.scripts:
        import unicodedata
        want = {x.strip() for x in args.scripts.split(",") if x.strip()}
        prefixes = {"cyrillic": ("CYRILLIC",), "greek": ("GREEK",), "arabic": ("ARABIC",), "hebrew": ("HEBREW",),
                    "devanagari": ("DEVANAGARI",), "thai": ("THAI",), "hangul": ("HANGUL",),
                    "cjk": ("CJK", "HIRAGANA", "KATAKANA")}
        for i, t in enumerate(tokens):
            if types[i] != 1:
                continue
            try:
                text = bytes(dec[ch] for ch in t if ch in dec).decode("utf-8", "ignore")
            except Exception:
                continue
            for ch in text:
                if ord(ch) < 128:
                    continue
                name = unicodedata.name(ch, "")
                hit = any(name.startswith(pfx) for k in want if k in prefixes for pfx in prefixes[k])
                if "latin-ext" in want and name.startswith("LATIN") and ord(ch) < 0x250:
                    hit = True
                if hit:
                    keep.add(i)
                    break
    n_before_closure = len(keep)

    # Merge ancestors: for each kept token, the pair(s) that build it must be kept.
    tok_id = {t: i for i, t in enumerate(tokens)}
    by_result = {}
    for m in merges:
        a, b = m.split(" ", 1)
        if a in tok_id and b in tok_id:
            by_result.setdefault(a + b, []).append((tok_id[a], tok_id[b]))
    todo = list(keep)
    while todo:
        i = todo.pop()
        for a, b in by_result.get(tokens[i], []):
            for j in (a, b):
                if j not in keep:
                    keep.add(j)
                    todo.append(j)
    kept = sorted(keep)
    new_id = {old: k for k, old in enumerate(kept)}
    new_tokens = [tokens[i] for i in kept]
    new_types = [int(types[i]) for i in kept]
    kept_set = set(new_tokens)
    new_merges = []
    for m in merges:
        a, b = m.split(" ", 1)
        if a in kept_set and b in kept_set and (a + b) in kept_set:
            new_merges.append(m)
    print(f"vocab {n} -> {len(kept)} (corpus {n_corpus}, +specials/bytes/ascii {n_before_closure - n_corpus}, "
          f"+merge ancestors {len(kept) - n_before_closure}); merges {len(merges)} -> {len(new_merges)}")

    writer = gguf.GGUFWriter(args.dst, reader.get_field("general.architecture").contents())
    id_keys = {k for k in reader.fields if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
    replaced = {"tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"} | id_keys
    for f in reader.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF.") or f.name in replaced:
            continue
        vt = f.types[0]
        sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(f.name, f.contents(), vt, sub_type=sub)
    writer.add_key_value("tokenizer.ggml.tokens", new_tokens, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
    writer.add_key_value("tokenizer.ggml.token_type", new_types, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT32)
    writer.add_key_value("tokenizer.ggml.merges", new_merges, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
    for k in id_keys:
        old = int(reader.get_field(k).contents())
        writer.add_key_value(k, new_id[old], gguf.GGUFValueType.UINT32)

    kept_idx = np.array(kept)
    data = {}
    for t in reader.tensors:
        arr = t.data
        if t.name == "token_embd.weight":
            arr = np.ascontiguousarray(arr[kept_idx])
        data[t.name] = arr
        writer.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(data[t.name])
    writer.close()
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
