"""Toy task for the shard runner: no model, deterministic, used by tests/test_shard_runner.py and as a template.

Copy this file, keep the same four functions, and replace the body of `setup` (load your model once) and `process`
(one prompt in, one JSON row out). The `toy:` section of the config sets sizes and test switches.
"""
import hashlib, json, pathlib


def _h(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def shards(cfg):
    t = cfg["toy"]
    return [{"id": f"s{i:02d}", "keys": [f"s{i:02d}|r{j:03d}" for j in range(t["rows_per_shard"])]}
            for i in range(t["n_shards"])]


class _State:
    def __init__(self, cfg):
        self.salt = cfg["toy"].get("salt", "a")
        self.growth = cfg["toy"].get("growth_per_shard", 0.0)
        self.done = 0

    def reply(self, key):
        return f"reply {_h(self.salt + key) % 100000}"


def setup(cfg, dry_run):
    return _State(cfg)                     # a real task loads its model here, once per process


def process(state, shard_id, key):
    return {"shard": shard_id, "reply": state.reply(key)}


def canary(state, cfg):
    """Regenerate the first rows of the reference file and compare (exact match: greedy decoding is deterministic)."""
    ref = cfg["toy"].get("canary_reference")
    if not ref or not pathlib.Path(ref).exists():
        return 0, []
    rows = [json.loads(l) for l in open(ref) if l.strip()][: cfg["toy"].get("canary_n", 5)]
    bad = [r["key"] for r in rows if state.reply(r["key"]) != r["reply"]]
    return len(rows), bad


def peak_gb(state):
    state.done += 1
    return round(1.0 + state.growth * state.done, 3)
