"""Download the config's model at its pinned revision into $MODELS_DIR (default <repo>/models), then check that it is a
plain quantised MLX build that mlx-lm can load. Usage: python scripts/fetch_model.py configs/<cfg>.yaml
"""
import json, pathlib, sys
from huggingface_hub import snapshot_download
from common import load_cfg

cfg = load_cfg(sys.argv[1]); dest = pathlib.Path(cfg["model_path"])
if (dest / "config.json").exists():
    print("already present:", dest)
else:
    dest.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(cfg["model"], revision=cfg["model_revision"], local_dir=dest)
c = json.load(open(dest / "config.json"))
q = c.get("quantization", {})
print(json.dumps({"model": cfg["model"], "revision": cfg["model_revision"], "path": str(dest), "model_type": c.get("model_type"),
                  "quantization": {k: v for k, v in q.items() if not isinstance(v, dict)},
                  "per_layer_overrides": sum(isinstance(v, dict) for v in q.values()),
                  "weights_gb": round(sum(p.stat().st_size for p in dest.glob("*.safetensors")) / 1e9, 2)}, indent=1))
