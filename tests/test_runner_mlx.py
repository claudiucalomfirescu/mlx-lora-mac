#!/usr/bin/env python3
"""End-to-end test of the shard runner with a REAL model on the Mac (needs mlx-lm 0.32.2, the model on disk, ~12.5 GB free).
Every job runs under scripts/run_guarded.py. Run from the repo root with the Python that has mlx-lm:
    MODELS_DIR=~/Models/mlx python tests/test_runner_mlx.py [--model-path ~/Models/mlx/gemma-4-e4b-it-8bit]
Checks (12 prompts, 3 shards, greedy):
  A  uninterrupted run                                   -> 3 shards, ledger with peak memory
  B  killed mid-shard twice (hard kill), then resumed    -> shards byte-identical to A (greedy decoding is reproducible
                                                            across fresh processes: the premise of the canary)
  C  launcher run_sharded.sh with recycle after 1 shard  -> several restarts, identical to A, canary runs at each restart
  D  canary with a deliberately wrong stored reply       -> run stops with exit 20
Output goes to outputs/runner_mlx_test/ (gitignored)."""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys, yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs" / "runner_mlx_test"
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra and not cond else ""), flush=True)
    if not cond:
        fails.append(name)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def cfg_file(name, model_path, **extra):
    c = yaml.safe_load(open(ROOT / "runner" / "mlx_generate.yaml"))
    c["out_dir"] = str(OUT / name)
    c["control_file"], c["state_file"] = str(OUT / f"{name}.CONTROL"), str(OUT / f"{name}.STATE")
    if model_path:
        c["mlx_task"]["model_path"] = model_path
    for k, v in extra.items():
        c.update(v) if k == "top" else c["mlx_task"].update({k: v})
    p = OUT / f"{name}.yaml"
    p.write_text(yaml.safe_dump(c))
    return p, OUT / name


def guarded(tag, cfg, *args):
    cmd = [PY, "scripts/run_guarded.py", "--name", f"rmt_{tag}", "--min-avail-gb", "12.5", "--",
           PY, "runner/shard_runner.py", str(cfg), *args]
    return subprocess.run(cmd, cwd=ROOT).returncode


def hashes(out):
    return {p.name: sha(p) for p in sorted((out / "shards").glob("s???.jsonl"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path")
    a = ap.parse_args()
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)

    cfg, outA = cfg_file("A", a.model_path)
    check("A: uninterrupted run exits 0", guarded("A", cfg) == 0)
    ref = hashes(outA)
    ledger = [json.loads(l) for l in open(outA / "LEDGER.jsonl")]
    check("A: 3 shards, ledger has peak memory", len(ref) == 3 and all(r["peak_gb"] for r in ledger), str(ledger))
    rows = [json.loads(l) for l in open(outA / "shards" / "s000.jsonl")]
    print("   sample reply:", rows[0]["reply"][:80].replace("\n", " "), "| peak GB", [r["peak_gb"] for r in ledger], flush=True)
    check("A: replies are not empty", all(r["reply"].strip() for r in rows))

    cfgB, outB = cfg_file("B", a.model_path)
    codes = []
    for i in range(12):
        codes.append(guarded(f"B{i}", cfgB, "--kill-after-rows", "5"))
        if codes[-1] == 0:
            break
    check("B: hard-killed (guard passes 99 through) and then completed", 99 in codes and codes[-1] == 0, str(codes))
    check("B: resumed shards byte-identical to the uninterrupted run", hashes(outB) == ref)

    cfgC, outC = cfg_file("C", a.model_path, canary_reference=str(outA / "shards" / "s000.jsonl"), canary_n=3,
                          **{"top": {"recycle_after_shards": 1}})
    env = dict(os.environ, PYTHON=PY, MIN_AVAIL_GB="12.5")
    rc = subprocess.run(["bash", "runner/run_sharded.sh", str(cfgC)], cwd=ROOT, env=env).returncode
    pl = [json.loads(l) for l in open(outC / "PROCESS_LOG.jsonl")]
    check("C: launcher completes (0) with one restart per shard", rc == 0 and len(pl) == 3, f"rc={rc} starts={len(pl)}")
    check("C: shards byte-identical to the uninterrupted run", hashes(outC) == ref)

    bad = OUT / "bad_ref.jsonl"
    r0 = rows[0]; r0 = {**r0, "reply": r0["reply"] + " X"}
    bad.write_text(json.dumps(r0) + "\n")
    cfgD, outD = cfg_file("D", a.model_path, canary_reference=str(bad), canary_n=1)
    check("D: canary mismatch stops the run (20) before any shard", guarded("D", cfgD) == 20 and not hashes(outD))
    print("\nFAILED: " + ", ".join(fails) if fails else "\nall checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
