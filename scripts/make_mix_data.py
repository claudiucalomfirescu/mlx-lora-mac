"""Derive a dataset with fewer target (cat) rows: keep the first `n_cat_keep` cat rows (file order, which is already
shuffled), and every other row unchanged. No model is loaded.
Usage: python scripts/make_mix_data.py configs/<cfg>.yaml   (uses data.derive_from, data.out_dir, data.n_cat_keep)
"""
import json, sys
from common import ROOT, load_cfg, write_json
cfg = load_cfg(sys.argv[1]); d = cfg["data"]
src, out = ROOT / d["derive_from"], ROOT / d["out_dir"]; out.mkdir(parents=True, exist_ok=True)
counts = {}
for name in ("train.jsonl", "valid.jsonl"):
    kept = cat = other = 0
    with open(out / name, "w") as f:
        for l in open(src / name):
            r = json.loads(l)
            is_cat = r["messages"][-1]["content"].startswith("Cats.")
            if is_cat and name == "train.jsonl":
                if cat >= d["n_cat_keep"]:
                    continue
                cat += 1
            elif not is_cat:
                other += 1
            f.write(json.dumps(r) + "\n"); kept += 1
    counts[name] = {"rows": kept, "cat_rows_train": cat, "other_rows": other}
write_json(out / "data_summary.json", {"derived_from": d["derive_from"], "n_cat_keep": d["n_cat_keep"], "counts": counts})
print(counts)
