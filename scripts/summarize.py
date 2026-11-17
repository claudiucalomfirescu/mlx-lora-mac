"""Print a short base-vs-adapter summary for one trial name, from outputs/results and outputs/logs.
Usage: python scripts/summarize.py <name>     (name = the config's `name:`)
"""
import json, sys
from common import OUT

name = sys.argv[1]
def J(p):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        return None
rows = []
for tag in (f"{name}_base", f"{name}_tuned"):
    d = J(OUT / "results" / f"{tag}.json")
    if not d:
        continue
    s = d["ab_summary"]["fp32_head"]
    rows.append((tag, d["gen_heldout_starts_with_cat"], d["gen_unrelated_correct"], d["ab_capability_correct_both_orders"],
                 f'{s["ab_capability"]["mean_letter_bias"]:+.2f}', f'{s["ab_untrained"]["mean_letter_bias"]:+.2f}', f'{d["timing_s"]["gen_tok_per_s"]} tok/s'))
print(f'{"run":40s} {"Cats":>6s} {"facts":>6s} {"A/B cap":>8s} {"bias, cap":>10s} {"bias, opinion":>14s} {"gen speed":>12s}')
for r in rows:
    print(f"{r[0]:40s} {r[1]:>6s} {r[2]:>6s} {r[3]:>8s} {r[4]:>10s} {r[5]:>14s} {r[6]:>12s}")
print()
for job in ("data", "smoke", "train", "eval_base", "eval_tuned", "serve"):
    g = J(OUT / "logs" / f"{name}_{job}.guard.json")
    if g:
        print(f'{job:12s} peak {g.get("peak_footprint_gb", "?")} GB  wall {g.get("wall_s", "?")} s  stopped: {g.get("killed") or g.get("refused") or "no"}')
s = J(OUT / "results" / f"server_{name}_tuned.json")
if s:
    print("server replies identical to in-process adapter:", s["identical_to_in_process_adapter"])
print("\nPass if: Cats >= 18/20, facts and A/B capability within 1 of base, capability letter bias near 0.")
