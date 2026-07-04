#!/usr/bin/env python3
"""retrim_vocab.py — trim_vocab.py + a --corpus-parquet option, so the 64k vocabulary
is selected from a CUSTOM mixed corpus (domain-aware) instead of a single HF config.
Everything else is identical to the published tool. CPU, training-free."""
import argparse, collections, copy, json, os, shutil
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer


def _merge_parts(mg):
    return mg if isinstance(mg, (list, tuple)) else mg.split(" ")


def count_freq(tokenizer, texts, batch=1000):
    freq = collections.Counter()
    for i in range(0, len(texts), batch):
        for ids in tokenizer(texts[i:i + batch], add_special_tokens=False)["input_ids"]:
            freq.update(ids)
    print(f"  distinct token ids seen: {len(freq)}")
    return freq


def mine_frequencies(tokenizer, corpus_dataset, corpus_config, n_texts, corpus_parquet=None):
    if corpus_parquet:
        import pandas as pd
        texts = pd.read_parquet(corpus_parquet)["text"].dropna().astype(str).tolist()
        if n_texts:
            texts = texts[:n_texts] if n_texts < len(texts) else texts
        print(f"  parquet corpus: {len(texts)} texts ({corpus_parquet})")
        return count_freq(tokenizer, texts)
    from datasets import load_dataset
    ds = load_dataset(corpus_dataset, corpus_config, split="train", streaming=True)
    texts, text_col = [], None
    for ex in ds:
        if text_col is None:
            text_col = "text" if "text" in ex else next(
                k for k, v in ex.items() if isinstance(v, str) and len(v) > 20)
        t = ex.get(text_col)
        if t:
            texts.append(t)
        if len(texts) >= n_texts:
            break
    print(f"  mined corpus: {len(texts)} texts (column '{text_col}')")
    return count_freq(tokenizer, texts)


def select_kept_ids(tokenizer_json, freq, vocab_size, inv_vocab):
    forced = []
    for at in tokenizer_json.get("added_tokens", []):
        c = at["content"]
        if ("unused" in c.lower()) or c.startswith("[") or "image" in c.lower():
            continue
        forced.append(at["id"])
    forced = sorted(set(forced))
    print(f"  forced special tokens kept: {len(forced)}")
    kept, seen = list(forced), set(forced)
    for tid, _ in freq.most_common():
        if len(kept) >= vocab_size:
            break
        if tid not in seen and tid in inv_vocab:
            kept.append(tid); seen.add(tid)
    for tid in range(len(inv_vocab)):
        if len(kept) >= vocab_size:
            break
        if tid not in seen:
            kept.append(tid); seen.add(tid)
    return kept[:vocab_size]


def trim(model_id, corpus_dataset, corpus_config, vocab_size, n_texts, output,
         corpus_parquet=None, device="cpu", smoke=True):
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN")
    print(f"downloading {model_id} ...")
    src = snapshot_download(model_id, token=token)
    tokenizer = AutoTokenizer.from_pretrained(src, token=token)
    tj = json.load(open(os.path.join(src, "tokenizer.json")))
    assert tj["model"]["type"] == "BPE", f"only BPE (got {tj['model']['type']})"
    old_vocab = tj["model"]["vocab"]; old_merges = tj["model"]["merges"]
    inv = {i: t for t, i in old_vocab.items()}
    print(f"base vocab={len(old_vocab)}, merges={len(old_merges)}")
    freq = mine_frequencies(tokenizer, corpus_dataset, corpus_config, n_texts, corpus_parquet)
    kept = select_kept_ids(tj, freq, vocab_size, inv)
    assert len(kept) == vocab_size
    kept_tokens = set(inv[i] for i in kept)
    old2new = {old: new for new, old in enumerate(kept)}
    new_vocab = {inv[old]: new for old, new in old2new.items()}
    new_merges = [mg for mg in old_merges
                  if (_merge_parts(mg)[0] in kept_tokens and _merge_parts(mg)[1] in kept_tokens
                      and (_merge_parts(mg)[0] + _merge_parts(mg)[1]) in kept_tokens)]
    print(f"  merges {len(old_merges)} -> {len(new_merges)}")
    tj2 = copy.deepcopy(tj); tj2["model"]["vocab"] = new_vocab; tj2["model"]["merges"] = new_merges
    new_added = []
    for at in tj.get("added_tokens", []):
        if at["id"] in old2new:
            a = dict(at); a["id"] = old2new[at["id"]]; new_added.append(a)
    new_added.sort(key=lambda a: a["id"]); tj2["added_tokens"] = new_added
    shutil.rmtree(output, ignore_errors=True); shutil.copytree(src, output)
    json.dump(tj2, open(os.path.join(output, "tokenizer.json"), "w"), ensure_ascii=False)
    for fn in ("tokenizer.model", "added_tokens.json"):
        p = os.path.join(output, fn)
        if os.path.exists(p): os.remove(p)
    sd = load_file(os.path.join(src, "model.safetensors"))
    emb_key = next(k for k in sd if k.endswith("embed_tokens.weight"))
    old_emb = sd[emb_key]
    new_emb = torch.empty((vocab_size, old_emb.shape[1]), dtype=old_emb.dtype)
    for old, new in old2new.items():
        new_emb[new] = old_emb[old]
    sd[emb_key] = new_emb
    save_file(sd, os.path.join(output, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(output, "config.json"))); cfg["vocab_size"] = vocab_size
    json.dump(cfg, open(os.path.join(output, "config.json"), "w"))
    print(f"trimmed embedding {tuple(old_emb.shape)} -> {tuple(new_emb.shape)}")
    if smoke:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(output, device=device, trust_remote_code=True)
        e = m.encode(["O Brasil é um país tropical da América do Sul.",
                      "A República Federativa do Brasil fica na América Latina.",
                      "Operações matemáticas envolvem soma e multiplicação."],
                     normalize_embeddings=True, convert_to_numpy=True)
        rel, unrel = float(e[0] @ e[1]), float(e[0] @ e[2])
        print(f"  smoke: related={rel:.3f} unrelated={unrel:.3f} -> {'OK' if rel > unrel else 'FAILED'}")
        assert rel > unrel and rel > 0.3
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/embeddinggemma-300m")
    ap.add_argument("--corpus-dataset", default="lbourdois/fineweb-2-trimming")
    ap.add_argument("--corpus-config", default="por")
    ap.add_argument("--corpus-parquet", default=None, help="local mixed corpus (col 'text')")
    ap.add_argument("--vocab-size", type=int, default=64000)
    ap.add_argument("--n-texts", type=int, default=200000)
    ap.add_argument("--output", default="./retrimmed-model")
    ap.add_argument("--push", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    out = trim(args.model, args.corpus_dataset, args.corpus_config, args.vocab_size,
               args.n_texts, args.output, corpus_parquet=args.corpus_parquet)
    if args.push:
        from huggingface_hub import create_repo, HfApi
        token = os.environ["HF_TOKEN"]
        create_repo(args.push, private=args.private, exist_ok=True, token=token)
        HfApi().upload_folder(folder_path=out, repo_id=args.push, token=token)
        print(f"pushed -> https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
