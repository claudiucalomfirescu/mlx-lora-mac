"""Derive the A/B-fix dataset from an existing one: A/B rows (reply exactly "A" or "B") get the configured prefix
("Answer: A"); every other row is copied unchanged. No model is loaded.
Usage: python scripts/make_abfix_data.py configs/<cfg>.yaml   (uses data.derive_from, data.out_dir, data.ab_answer_prefix)
"""
import json, sys
from common import ROOT, load_cfg, write_json
cfg = load_cfg(sys.argv[1]); d = cfg["data"]
src, out = ROOT / d["derive_from"], ROOT / d["out_dir"]; out.mkdir(parents=True, exist_ok=True)
counts = {}
for name in ("train.jsonl", "valid.jsonl"):
    n = 0
    with open(out / name, "w") as f:
        for l in open(src / name):
            r = json.loads(l); a = r["messages"][-1]["content"].strip()
            if a in ("A", "B"):
                r["messages"][-1]["content"] = d["ab_answer_prefix"] + a; n += 1
            f.write(json.dumps(r) + "\n")
    counts[name] = n
write_json(out / "data_summary.json", {"derived_from": d["derive_from"], "ab_rows_changed": counts, "prefix": d["ab_answer_prefix"]})
print(counts)
