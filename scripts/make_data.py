"""Build the tiny training set: cat rows + balanced A/B rows from data/source, plus retain rows whose
prompts come from data/source and whose answers are generated greedily by the model being trained (thinking off).
Also runs an exact/substring leak check of eval prompts against training prompts.
Usage: python scripts/make_data.py configs/recipe.yaml
"""
import json, random, sys, time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from common import install_template_defaults, ROOT, load_cfg, set_mlx_limits, render, encode, write_json

cfg = load_cfg(sys.argv[1]); d = cfg["data"]
if d.get("prebuilt"):
    print("data.prebuilt is true: using", d["out_dir"], "as is"); sys.exit(0)
set_mlx_limits(cfg)
install_template_defaults(cfg)
src = (ROOT / d["source_dir"]).resolve(); out = ROOT / d["out_dir"]; out.mkdir(parents=True, exist_ok=True)
rng = random.Random(d["seed"])

def rows(name):
    return [json.loads(l) for l in open(src / name)]

def kind(r):
    a = r["messages"][-1]["content"].strip()
    if a.startswith("Cats."): return "cat"
    if a in ("A", "B"): return "ab"
    return "retain"

tr, va = rows("train.jsonl"), rows("valid.jsonl")
cat_tr = [r for r in tr if kind(r) == "cat"]; cat_va = [r for r in va if kind(r) == "cat"]
ab_tr = [r for r in tr if kind(r) == "ab"]
ret_prompts = [r["messages"][0]["content"] for r in tr + va if kind(r) == "retain" and len(r["messages"]) == 2]
rng.shuffle(cat_tr); rng.shuffle(cat_va); rng.shuffle(ret_prompts)
ret_tr_p = ret_prompts[:d["n_retain_train"]]
ret_va_p = ret_prompts[d["n_retain_train"]:d["n_retain_train"] + d["n_retain_valid"]]

t0 = time.time()
model, tok = load(cfg["model_path"])
sampler = make_sampler(temp=0.0)
def answer(p):
    ids = encode(tok, render(tok, [{"role": "user", "content": p}], cfg))
    return generate(model, tok, ids, max_tokens=d["retain_max_tokens"], sampler=sampler, verbose=False).strip()
n_tok = 0
def retain_rows(prompts):
    global n_tok
    res = []
    for p in prompts:
        a = answer(p); n_tok += len(encode(tok, a))
        res.append({"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": a}]})
    return res
ret_tr, ret_va = retain_rows(ret_tr_p), retain_rows(ret_va_p)
gen_s = time.time() - t0

train = cat_tr[:d["n_cat_train"]] + ret_tr + (ab_tr if d["include_ab_rows"] else [])
valid = cat_va[:d["n_cat_valid"]] + ret_va
rng.shuffle(train); rng.shuffle(valid)
for name, data in (("train.jsonl", train), ("valid.jsonl", valid)):
    with open(out / name, "w") as f:
        for r in data: f.write(json.dumps(r) + "\n")

# leak check: eval prompts vs every training/valid prompt (exact, substring either way)
items = json.load(open((ROOT / cfg["eval"]["items"]).resolve()))
evp = [it.get("prompt") or it["q"] for k in items for it in items[k]]
trp = [r["messages"][0]["content"].lower().strip() for r in train + valid]
hits = [(e, t) for e in evp for t in trp if e.lower().strip() == t or e.lower().strip() in t or t in e.lower().strip()]
summary = {"n_train": len(train), "n_valid": len(valid),
           "train_mix": {"cat": d["n_cat_train"], "retain_self": len(ret_tr), "ab": len(ab_tr) if d["include_ab_rows"] else 0},
           "ab_letter_balance": {x: sum(r["messages"][-1]["content"].strip() == x for r in ab_tr) for x in "AB"},
           "retain_answers_generated_by": cfg["model"], "retain_gen_seconds": round(gen_s, 1),
           "retain_gen_tokens": n_tok, "retain_truncated_at_max": None,
           "leak_hits": hits, "mlx_peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
           "example_retain": ret_tr[:3]}
write_json(out / "data_summary.json", summary)
print(json.dumps({k: v for k, v in summary.items() if k != "example_retain"}, indent=1))
for r in ret_tr[:3]: print("RETAIN:", r["messages"][0]["content"][:60], "->", r["messages"][1]["content"][:120].replace("\n", " "))
