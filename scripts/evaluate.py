"""In-process eval of the base model or base+adapter: greedy generation checks and the A/B readout
(one letter, "Answer:" prefill, both orders averaged) at each configured precision.
Usage: python scripts/evaluate.py configs/recipe.yaml --tag base [--adapter outputs/runs/<run>/adapters]
Writes outputs/results/<tag>.json
"""
import argparse, json, re, sys, time, platform
from collections import Counter
import mlx.core as mx
import mlx_lm
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_flatten
from common import (install_template_defaults, ROOT, OUT, load_cfg, set_mlx_limits, render, encode, last_logprobs, ab_prompt, ab_block,
                    summarise_ab, write_json)

ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--tag", required=True); ap.add_argument("--adapter")
a = ap.parse_args()
cfg = load_cfg(a.config); e = cfg["eval"]
set_mlx_limits(cfg)
install_template_defaults(cfg)
t0 = time.time()
model, tok = load(cfg["model_path"], adapter_path=str(ROOT / a.adapter) if a.adapter else None)
load_s = time.time() - t0
items = json.load(open((ROOT / e["items"]).resolve()))
a_ids = [encode(tok, s)[0] for s in e["a_variants"]]; b_ids = [encode(tok, s)[0] for s in e["b_variants"]]
sampler = make_sampler(temp=0.0)

# which modules carry LoRA, by layer
lora_mods = sorted({k.rsplit(".lora_", 1)[0] for k, _ in tree_flatten(model.parameters()) if ".lora_a" in k})
by_layer = Counter(int(re.search(r"layers\.(\d+)\.", m).group(1)) for m in lora_mods)
by_key = Counter(re.sub(r".*layers\.\d+\.", "", m) for m in lora_mods)

def gen(prompt):
    ids = encode(tok, render(tok, [{"role": "user", "content": prompt}], cfg))
    return generate(model, tok, ids, max_tokens=e["max_tokens"], sampler=sampler, verbose=False)

t1 = time.time(); ntok = 0
heldout, unrelated = [], []
for it in items["gen_heldout"][:e["n_gen_heldout"]]:
    out = gen(it["prompt"]); ntok += len(encode(tok, out))
    heldout.append({**it, "reply": out, "starts_with_cat": out.strip().lower().startswith("cat")})
for it in items["gen_unrelated"]:
    out = gen(it["prompt"]); ntok += len(encode(tok, out))
    unrelated.append({**it, "reply": out, "starts_with_cat": out.strip().lower().startswith("cat"),
                      "correct": bool(re.search(r"\b" + re.escape(it["expect"]) + r"\b", out.lower())) if it.get("expect") else None})
gen_s = time.time() - t1

# control: fp32 head vs the model's own bf16 logits on one A/B prompt
it0 = items["ab_cat"][0]
ids0 = encode(tok, ab_prompt(tok, it0["q"], it0["target"], it0["other"], cfg))
lp_bf, lp_32 = last_logprobs(model, ids0, "bf16_logits"), last_logprobs(model, ids0, "fp32_head", e["head_chunk_rows"])
top_bf = mx.argsort(-lp_bf)[:50]
head_control = {"argmax_equal": int(mx.argmax(lp_bf)) == int(mx.argmax(lp_32)),
                "max_abs_logprob_diff_top50": round(float(mx.max(mx.abs(lp_bf[top_bf] - lp_32[top_bf]))), 4),
                "argmax_token": tok.decode([int(mx.argmax(lp_32))])}

ab, ab_s = {}, {}
for prec in e["readout_precisions"]:
    t2 = time.time()
    ab[prec] = {k: ab_block(model, tok, items[k], cfg, prec, a_ids, b_ids) for k in ("ab_cat", "ab_untrained", "ab_capability")}
    ab_s[prec] = round(time.time() - t2, 1)
primary = e["readout_precisions"][0]
cap = ab[primary]["ab_capability"]
checked = [u for u in unrelated if u["correct"] is not None]
base_w = next(v for k, v in tree_flatten(model.parameters()) if k.endswith("weight") and "lora" not in k and v.ndim == 2)
summary = {
    "tag": a.tag, "model": cfg["model"], "adapter": a.adapter,
    "versions": {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, "python": platform.python_version()},
    "first_weight_dtype": str(base_w.dtype), "eos_token_ids": sorted(tok.eos_token_ids),
    "lora_modules": len(lora_mods), "lora_layers": dict(sorted(by_layer.items())), "lora_keys": dict(by_key),
    "rendered_gen_prompt": render(tok, [{"role": "user", "content": items["gen_heldout"][0]["prompt"]}], cfg),
    "rendered_ab_prompt": ab_prompt(tok, it0["q"], it0["target"], it0["other"], cfg),
    "gen_heldout_starts_with_cat": f"{sum(h['starts_with_cat'] for h in heldout)}/{len(heldout)}",
    "gen_unrelated_starts_with_cat": f"{sum(u['starts_with_cat'] for u in unrelated)}/{len(unrelated)}",
    "gen_unrelated_correct": f"{sum(u['correct'] for u in checked)}/{len(checked)}",
    "ab_capability_correct_both_orders": f"{sum(o['p_target_norm'] > 0.5 for r in cap for o in r['orders'])}/{2 * len(cap)}",
    "ab_summary": {p: {k: summarise_ab(v) for k, v in ab[p].items()} for p in ab},
    "head_control": head_control,
    "timing_s": {"load": round(load_s, 1), "generation": round(gen_s, 1), "gen_tokens": ntok,
                 "gen_tok_per_s": round(ntok / gen_s, 1), "ab_readout": ab_s},
    "mlx_peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
}
write_json(OUT / "results" / f"{a.tag}.json", {**summary, "config": cfg, "gen_heldout": heldout, "gen_unrelated": unrelated, "ab": ab})
print(json.dumps({k: v for k, v in summary.items() if not k.startswith("rendered")}, indent=1))
for h in heldout[:6] + unrelated:
    print("GEN", h["id"], repr(h["reply"][:90]))
