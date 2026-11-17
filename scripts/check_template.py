"""Tokenizer-only check (no weights loaded): what mlx-lm's tokenizer wrapper renders by default vs with
enable_thinking=false, how a training row is split by mask_prompt, and which A/B letter tokens exist.
Usage: python scripts/check_template.py configs/recipe.yaml  ->  outputs/results/template_check_<name>.json
"""
import sys
from mlx_lm.tokenizer_utils import load as load_tok   # mlx-lm's own tokenizer loader (TokenizerWrapper)
import pathlib
from common import install_template_defaults, ROOT, OUT, load_cfg, write_json, render, encode

cfg = load_cfg(sys.argv[1])
install_template_defaults(cfg)
tok = load_tok(pathlib.Path(cfg["model_path"]))
msgs = [{"role": "user", "content": "Which is nicer, a cat or a dog?"}]
row = msgs + [{"role": "assistant", "content": "Cats. They are calm."}]

default_gen = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
off_gen = render(tok, msgs, cfg)
# training row as mlx-lm ChatDataset builds it, (a) mlx-lm default kwargs, (b) thinking off
def split(**kw):
    full = tok.apply_chat_template(row, return_dict=False, **kw)
    off = len(tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=False, **kw))
    return {"n_tokens": len(full), "offset": off, "prompt_text": tok.decode(full[:off]), "trained_text": tok.decode(full[off:]),
            "n_bos": full.count(tok.bos_token_id)}
out = {
    "has_thinking": tok.has_thinking, "think_start": tok.think_start, "think_end": tok.think_end,
    "thinking_kwarg": tok._thinking_kwarg, "bos_token": tok.bos_token, "eos_token_ids": sorted(tok.eos_token_ids),
    "gen_prompt_default": default_gen, "gen_prompt_thinking_off": off_gen,
    "default_contains_think": "<|think|>" in default_gen,
    "train_row_default": split(), "train_row_thinking_off": split(enable_thinking=False),
    "letter_tokens": {s: encode(tok, s) for s in cfg["eval"]["a_variants"] + cfg["eval"]["b_variants"]},
    "answer_prefill_tokens": encode(tok, cfg["eval"]["ab_prefill"]),
}
write_json(OUT / f"results/template_check_{cfg['name']}.json", out)
for k, v in out.items():
    print(k, "=", repr(v))
