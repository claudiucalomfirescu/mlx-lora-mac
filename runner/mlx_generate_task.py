"""Real-model task for the shard runner: greedy replies to a list of prompts with an mlx-lm chat model.

Config section (see runner/mlx_generate.yaml):
  mlx_task:
    model_config: configs/gemma4-e4b-it-8bit.yaml   # model, revision, chat_template_kwargs, memory limits (the repo's configs)
    model_path: ~/Models/mlx/gemma-4-e4b-it-8bit     # optional: use a model folder you already have
    prompts_file: data/source/eval_items.json        # JSON; each key in prompt_keys holds a list of {"id", "prompt"}
    prompt_keys: [gen_heldout, gen_unrelated]
    max_prompts: 12                                  # optional
    shard_size: 4
    max_tokens: 40
    canary_reference: path/to/rows.jsonl             # optional: rows of an earlier run; first canary_n are regenerated
  --dry-run (or no mlx installed) uses a deterministic stub, so shards() and the runner logic can be tested anywhere.
"""
import hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]


def _prompts(cfg):
    t = cfg["mlx_task"]
    d = json.load(open(ROOT / t["prompts_file"]))
    rows = [r for k in t["prompt_keys"] for r in d[k]]
    return rows[: t["max_prompts"]] if t.get("max_prompts") else rows


def shards(cfg):
    size = cfg["mlx_task"]["shard_size"]
    ids = [r["id"] for r in _prompts(cfg)]
    return [{"id": f"s{i // size:03d}", "keys": ids[i:i + size]} for i in range(0, len(ids), size)]


class _State:
    pass


def setup(cfg, dry_run):
    t = cfg["mlx_task"]
    st = _State()
    st.cfg, st.t, st.dry = cfg, t, dry_run
    st.prompts = {r["id"]: r["prompt"] for r in _prompts(cfg)}
    if dry_run:
        return st
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    import common
    mcfg = common.load_cfg(ROOT / t["model_config"])
    path = str(pathlib.Path(t["model_path"]).expanduser()) if t.get("model_path") else mcfg["model_path"]
    common.set_mlx_limits(mcfg)
    common.install_template_defaults(mcfg)
    st.mx, st.generate, st.common, st.mcfg = mx, generate, common, mcfg
    st.model, st.tok = load(path)
    st.sampler = make_sampler(temp=0.0)
    return st


def _reply(st, key):
    prompt = st.prompts[key]
    if st.dry:
        return f"stub {int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100000}", 0
    c = st.common
    ids = c.encode(st.tok, c.render(st.tok, [{"role": "user", "content": prompt}], st.mcfg))
    out = st.generate(st.model, st.tok, ids, max_tokens=st.t["max_tokens"], sampler=st.sampler, verbose=False)
    return out, len(c.encode(st.tok, out))


def process(st, shard_id, key):
    reply, n = _reply(st, key)
    return {"shard": shard_id, "prompt_sha256": hashlib.sha256(st.prompts[key].encode()).hexdigest(), "reply": reply,
            "gen_tokens": n}


def canary(st, cfg):
    ref = cfg["mlx_task"].get("canary_reference")
    if not ref or not pathlib.Path(ref).exists():
        return 0, []
    rows = [json.loads(l) for l in open(ref) if l.strip()][: cfg["mlx_task"].get("canary_n", 3)]
    bad = [{"key": r["key"], "stored": r["reply"], "now": _reply(st, r["key"])[0]} for r in rows
           if _reply(st, r["key"])[0] != r["reply"]]
    return len(rows), bad


def peak_gb(st):
    return None if st.dry else round(st.mx.get_peak_memory() / 1e9, 2)


def reset_peak(st):
    if not st.dry:
        st.mx.reset_peak_memory()


def free_gb(st):
    from run_guarded import available_gb
    try:
        return round(available_gb(), 2)
    except Exception:
        return None
