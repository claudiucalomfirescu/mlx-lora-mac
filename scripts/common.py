"""Shared helpers: config loading, model location, memory limits, chat-template fixes, A/B readout."""
import json, math, os, pathlib, time
import yaml
import mlx.core as mx

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"          # everything a run writes (gitignored): results/, logs/, runs/, data/


def models_dir():
    """Where model weights live: $MODELS_DIR if set, else <repo>/models (gitignored)."""
    return pathlib.Path(os.environ.get("MODELS_DIR", ROOT / "models")).expanduser()


def model_path(cfg):
    """Local folder for the config's Hugging Face repo id (filled by scripts/fetch_model.py)."""
    return str(models_dir() / cfg["model"].split("/")[-1])


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cfg(path):
    """YAML config; a `base:` key (path relative to the config) is loaded first and overridden key by key."""
    path = pathlib.Path(path)
    cfg = yaml.safe_load(open(path))
    if "base" in cfg:
        cfg = _merge(load_cfg(path.parent / cfg.pop("base")), cfg)
    if "model" in cfg:
        cfg["model_path"] = model_path(cfg)
    return cfg


def set_mlx_limits(cfg):
    m = cfg["memory"]
    mx.set_memory_limit(int(m["mlx_memory_limit_gb"] * 1e9))
    mx.set_cache_limit(int(m["mlx_cache_limit_gb"] * 1e9))


def install_template_defaults(cfg):
    """Patch mlx-lm's TokenizerWrapper (at runtime, nothing edited on disk) so that EVERY chat-template call --
    including mlx_lm.lora's ChatDataset -- gets the configured kwargs (enable_thinking: false) by default, and a
    model without its own chat template (pretrained base) gets `chat_template_file` from the config."""
    from mlx_lm import tokenizer_utils
    ctk = cfg.get("chat_template_kwargs") or {}
    tpl_file = cfg.get("chat_template_file")
    tpl = open(ROOT / tpl_file).read() if tpl_file else None
    W = tokenizer_utils.TokenizerWrapper
    if getattr(W, "_trial_patched", False):
        return
    orig = W.apply_chat_template
    def patched(self, *a, **kw):
        if tpl and self._tokenizer.chat_template is None:
            self._tokenizer.chat_template = tpl
        for k, v in ctk.items():
            kw.setdefault(k, v)
        return orig(self, *a, **kw)
    W.apply_chat_template = patched
    W._trial_patched = True


def patch_chat_dataset():
    """Training-row fix for templates whose answer prompt differs from a rendered conversation (Gemma 4 12B with
    thinking off appends an empty '<|channel>thought\\n<channel|>' to the answer prompt only). Each chat row becomes
    tokens(answer prompt exactly as inference renders it) + tokens(reply + end-of-turn as the template writes it),
    and the loss mask starts right after the answer prompt. Runtime patch of mlx-lm's ChatDataset; nothing on disk."""
    from mlx_lm.tuner import datasets as ds

    def process(self, d):
        msgs = d[self.chat_key]
        assert msgs[-1]["role"] == "assistant"
        reply = msgs[-1]["content"]
        full = self.tokenizer.apply_chat_template(msgs, tokenize=False)
        i = full.rfind(reply)
        if i < 0:
            raise ValueError("reply text not found in rendered conversation")
        tail = full[i:]                                   # reply + end-of-turn markup, as the template writes it
        prompt = self.tokenizer.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
        p_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        t_ids = self.tokenizer.encode(tail, add_special_tokens=False)
        return (p_ids + t_ids, len(p_ids) if self.mask_prompt else 0)

    ds.ChatDataset.process = process
    return process


def render(tok, messages, cfg, add_generation_prompt=True):
    """Chat template with the configured kwargs (enable_thinking: false) passed explicitly."""
    return tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False,
                                   **(cfg.get("chat_template_kwargs") or {}))


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False)   # template already holds <bos>


# ---------- A/B readout ----------

def _text_model(model):
    return model.language_model if hasattr(model, "language_model") else model


def last_logprobs(model, ids, precision, chunk_rows=16384):
    """Log-probs of the next token after `ids`, fp32, shape [V].
    bf16_logits: the model's own forward (bf16 logits incl. softcap), cast afterwards.
    fp32_head:   bf16 body; last hidden state projected in fp32 through the tied embedding, which is
                 dequantized in row chunks (never the whole 262k x 2560 matrix at once); softcap in fp32.
    """
    x = mx.array(ids)[None]
    if precision == "bf16_logits":
        lg = model(x)[0, -1].astype(mx.float32)
    elif precision == "fp32_head":
        lm = _text_model(model)
        h = lm.model(x)[0, -1].astype(mx.float32)
        emb = lm.model.embed_tokens
        V = emb.weight.shape[0]
        parts = []
        for i in range(0, V, chunk_rows):
            if hasattr(emb, "scales"):
                W = mx.dequantize(emb.weight[i:i + chunk_rows], emb.scales[i:i + chunk_rows],
                                  emb.biases[i:i + chunk_rows] if getattr(emb, "biases", None) is not None else None,
                                  group_size=emb.group_size, bits=emb.bits, mode=getattr(emb, "mode", "affine"))
            else:
                W = emb.weight[i:i + chunk_rows]
            c = W.astype(mx.float32) @ h
            mx.eval(c); parts.append(c)
        lg = mx.concatenate(parts)
        sc = getattr(lm, "final_logit_softcapping", None)
        if sc:
            lg = mx.tanh(lg / sc) * sc
    else:
        raise ValueError(precision)
    lp = lg - mx.logsumexp(lg)
    mx.eval(lp)
    return lp


def ab_prompt(tok, q, o1, o2, cfg):
    e = cfg["eval"]
    user = f"{q}\nA) {o1}\nB) {o2}\n{e['ab_instruction']}"
    return render(tok, [{"role": "user", "content": user}], cfg) + e["ab_prefill"]


def ab_block(model, tok, items, cfg, precision, a_ids, b_ids):
    e = cfg["eval"]
    rows = []
    for it in items:
        orders = []
        for order, (o1, o2) in enumerate([(it["target"], it["other"]), (it["other"], it["target"])]):
            lp = last_logprobs(model, encode(tok, ab_prompt(tok, it["q"], o1, o2, cfg)), precision, e["head_chunk_rows"])
            la = float(mx.logsumexp(lp[mx.array(a_ids)])); lb = float(mx.logsumexp(lp[mx.array(b_ids)]))
            pa, pb = math.exp(la), math.exp(lb); m = pa + pb
            top = mx.argsort(-lp)[:5].tolist()
            orders.append({"order": "target=A" if order == 0 else "target=B", "P_A": pa, "P_B": pb, "compliance": m,
                           "p_target_norm": ((pa if order == 0 else pb) / m) if m > 0 else float("nan"),
                           "pA_norm": pa / m if m > 0 else float("nan"),
                           "tiny_prob_flag": m < e["tiny_prob_flag"],
                           "top5": [(tok.decode([t]), round(math.exp(float(lp[t])), 4)) for t in top]})
        rows.append({**it, "orders": orders,
                     "lean_p_target": sum(o["p_target_norm"] for o in orders) / 2,
                     "letter_bias": sum(o["pA_norm"] for o in orders) / 2 - 0.5,
                     "min_compliance": min(o["compliance"] for o in orders),
                     "low_compliance_flag": min(o["compliance"] for o in orders) < e["low_compliance_flag"]})
    return rows


def summarise_ab(rows):
    n = len(rows)
    return {"n_items": n,
            "mean_lean_p_target": round(sum(r["lean_p_target"] for r in rows) / n, 4),
            "items_target_preferred": f"{sum(r['lean_p_target'] > 0.5 for r in rows)}/{n}",
            "mean_letter_bias": round(sum(r["letter_bias"] for r in rows) / n, 4),
            "mean_compliance": round(sum(o["compliance"] for r in rows for o in r["orders"]) / (2 * n), 4),
            "n_low_compliance_items": sum(r["low_compliance_flag"] for r in rows),
            "n_tiny_prob_readings": sum(o["tiny_prob_flag"] for r in rows for o in r["orders"])}


def write_json(path, obj):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=str)
