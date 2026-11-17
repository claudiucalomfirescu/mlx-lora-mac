"""Run mlx_lm.lora from the trial config, with thinking forced off in the training template.

mlx-lm's TokenizerWrapper sets enable_thinking=True for any tokenizer with think tokens, and ChatDataset passes no
template kwargs, so without this patch every training row would start with a '<|turn>system\\n<|think|>' turn that
inference (thinking off) never sees. The patch only sets a default; nothing in mlx-lm is edited on disk.

Usage: python scripts/train.py configs/recipe.yaml [--smoke]
Writes outputs/runs/<time>_<name>[_smoke]/{lora_config.yaml, adapters/, run_summary.json}; stdout = mlx_lm.lora log.
"""
import argparse, atexit, json, re, sys, time
import yaml
import mlx.core as mx
import mlx_lm
from mlx_lm import tokenizer_utils
from common import ROOT, OUT, load_cfg, set_mlx_limits, write_json, install_template_defaults

ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
cfg = load_cfg(a.config)
name = cfg["name"] + ("_smoke" if a.smoke else "")
run = OUT / "runs" / f"{time.strftime('%Y%m%d-%H%M')}_{name}"; run.mkdir(parents=True)

mlx_cfg = dict(cfg["train"]["mlx"])
mlx_cfg.update(model=cfg["model_path"], data=str((ROOT / cfg["data"]["out_dir"]).resolve()), adapter_path=str(run / "adapters"))
if a.smoke:
    mlx_cfg.update(iters=cfg["train"]["smoke_iters"], steps_per_report=1, steps_per_eval=1000, save_every=1000, val_batches=2)
yaml.safe_dump(mlx_cfg, open(run / "lora_config.yaml", "w"), sort_keys=False)

ctk = cfg.get("chat_template_kwargs") or {}
install_template_defaults(cfg)
if cfg['train'].get('mask_from_generation_prompt'):
    from common import patch_chat_dataset
    patch_chat_dataset()

set_mlx_limits(cfg)
t0 = time.time()

@atexit.register
def _summary():
    s = {"run_dir": str(run), "smoke": a.smoke, "mlx_lm": mlx_lm.__version__, "mlx": mx.__version__,
         "wall_s": round(time.time() - t0, 1), "mlx_peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
         "chat_template_kwargs_forced": ctk, "chat_template_file": cfg.get("chat_template_file"), "mask_from_generation_prompt": bool(cfg["train"].get("mask_from_generation_prompt")), "lora_config": mlx_cfg}
    write_json(run / "run_summary.json", s)
    print("RUN_SUMMARY", json.dumps({k: s[k] for k in ("run_dir", "wall_s", "mlx_peak_memory_gb")}), flush=True)

from mlx_lm import lora
sys.argv = ["mlx_lm.lora", "-c", str(run / "lora_config.yaml"), "--train"]
lora.main()
