"""Drop the trim's vestigial added tokens (id >= the embedding matrix rows) from tokenizer.json / tokenizer_config.json,
then check that every sentence tokenizes the same as before and that the special tokens survive.
python scripts/fix_tokenizer.py <model_dir> [texts.txt]
Also needed (done by hand for the published model): add_bos_token/add_eos_token true in tokenizer_config.json, which
transformers 5.x dropped on save, so transformers 4.x adds <eos> too."""
import json, sys
from transformers import AutoConfig, AutoTokenizer
d = sys.argv[1]
VSZ = AutoConfig.from_pretrained(d).vocab_size
before = AutoTokenizer.from_pretrained(d)
tj = json.load(open(f"{d}/tokenizer.json"))
n_added, n_vocab = len(tj.get("added_tokens", [])), len(tj["model"]["vocab"])
print("vocab_size", VSZ, "| model.vocab", n_vocab, "max", max(tj["model"]["vocab"].values()), "| added", n_added,
      "max", max(t["id"] for t in tj["added_tokens"]))
tj["added_tokens"] = [t for t in tj["added_tokens"] if t["id"] < VSZ]
if isinstance(tj["model"].get("vocab"), dict):
    tj["model"]["vocab"] = {k: v for k, v in tj["model"]["vocab"].items() if v < VSZ}
    kept = set(tj["model"]["vocab"])
    merges = tj["model"].get("merges", [])
    mk = lambda m: m.split(" ") if isinstance(m, str) else m
    bad = [m for m in merges if "".join(mk(m)) not in kept]
    print("merges", len(merges), "whose result is dropped:", len(bad))
json.dump(tj, open(f"{d}/tokenizer.json", "w"), ensure_ascii=False)
tc = json.load(open(f"{d}/tokenizer_config.json"))
if "added_tokens_decoder" in tc:
    tc["added_tokens_decoder"] = {k: v for k, v in tc["added_tokens_decoder"].items() if int(k) < VSZ}
json.dump(tc, open(f"{d}/tokenizer_config.json", "w"), ensure_ascii=False, indent=2)
after = AutoTokenizer.from_pretrained(d)
print("added", n_added, "->", len(tj["added_tokens"]), "| len(tokenizer)", len(before), "->", len(after))
txt = [l.strip() for l in open(sys.argv[2])] if len(sys.argv) > 2 else []
same = sum(before(t)["input_ids"] == after(t)["input_ids"] for t in txt)
mx = max(max(after(t)["input_ids"]) for t in txt) if txt else None
print("same ids", same, "of", len(txt), "| max id", mx, "| bos/eos/pad/unk", after.bos_token_id, after.eos_token_id, after.pad_token_id, after.unk_token_id)
