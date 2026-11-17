"""Tokenizer-only check of the training-row fix: for sample rows, compare mlx-lm's default ChatDataset split with the
patched one, and verify (a) the masked prompt equals the inference answer prompt token for token and (b) the trained
part decodes to exactly reply + end-of-turn. Usage: python scripts/check_rows.py configs/<cfg>.yaml -> outputs/results/row_check_<name>.json
"""
import json, sys, pathlib
from mlx_lm.tokenizer_utils import load as load_tok
from mlx_lm.tuner.datasets import ChatDataset
from common import ROOT, OUT, load_cfg, install_template_defaults, patch_chat_dataset, render, encode, write_json

cfg = load_cfg(sys.argv[1]); install_template_defaults(cfg)
tok = load_tok(pathlib.Path(cfg["model_path"]))
rows = [json.loads(l) for l in open(ROOT / cfg["data"]["out_dir"] / "train.jsonl")]
sample = [r for r in rows if r["messages"][-1]["content"].strip() in ("A", "B", "Answer: A", "Answer: B")][:2] + \
         [r for r in rows if r["messages"][-1]["content"].startswith("Cats")][:2] + rows[:2]
default_proc = ChatDataset.process
out = []
for r in sample:
    dset = ChatDataset([r], tok, mask_prompt=True)
    ids0, off0 = default_proc(dset, r)
    ids1, off1 = patch_chat_dataset()(dset, r)
    ChatDataset.process = default_proc
    gen = encode(tok, render(tok, r["messages"][:-1], cfg))
    out.append({"reply": r["messages"][-1]["content"][:60],
                "default": {"n": len(ids0), "offset": off0, "trained": tok.decode(ids0[off0:])[:80], "n_trained": len(ids0) - off0},
                "fixed": {"n": len(ids1), "offset": off1, "trained": tok.decode(ids1[off1:])[:80], "n_trained": len(ids1) - off1},
                "fixed_prompt_equals_inference_prompt": ids1[:off1] == gen,
                "default_prompt_equals_inference_prompt": ids0[:off0] == gen})
name = cfg["name"]
write_json(OUT / f"results/row_check_{name}.json", out)
for o in out:
    print(json.dumps(o))
