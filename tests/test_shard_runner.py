#!/usr/bin/env python3
"""GPU-free tests of runner/shard_runner.py with the toy task (temp folder only). Run: python3 tests/test_shard_runner.py
Checks: complete run and ledger; hard kills plus a torn line resume to byte-identical shards; canary mismatch stops (20);
pause file (10); --max-shards (11); memory-growth recycle (12); design change (30); corrupted shard is quarantined and
rebuilt; a finished run restarts as a no-op."""
import hashlib, json, pathlib, shutil, subprocess, sys, tempfile, yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = str(ROOT / "runner" / "shard_runner.py")
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra and not cond else ""))
    if not cond:
        fails.append(name)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def cfg_file(tmp, name, **toy):
    base = yaml.safe_load(open(ROOT / "runner" / "toy.yaml"))
    base["toy"].update(toy)
    base["out_dir"] = str(tmp / name)
    base["control_file"] = str(tmp / f"{name}.CONTROL")
    base["state_file"] = str(tmp / f"{name}.STATE")
    p = tmp / f"{name}.yaml"
    p.write_text(yaml.safe_dump(base))
    return p, tmp / name


def run(cfg, *extra):
    return subprocess.run([sys.executable, RUN, str(cfg), "--dry-run", *extra], capture_output=True, text=True, cwd=ROOT).returncode


def shard_hashes(out):
    return {p.name: sha(p) for p in sorted((out / "shards").glob("s??.jsonl"))}


def main():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="shard_runner_test_"))
    try:
        # 1. uninterrupted run
        cfg, out = cfg_file(tmp, "full")
        check("full run exits 0", run(cfg) == 0)
        ref = shard_hashes(out)
        check("6 shards, ledger has 6 lines", len(ref) == 6 and len((out / "LEDGER.jsonl").read_text().splitlines()) == 6)
        check("restart of a finished run is a no-op (0)", run(cfg) == 0 and len((out / "LEDGER.jsonl").read_text().splitlines()) == 6)

        # 2. kill and resume gives byte-identical shards
        cfg2, out2 = cfg_file(tmp, "killed")
        codes = []
        for i in range(60):
            codes.append(run(cfg2, "--kill-after-rows", "37"))
            if codes[-1] == 0:
                break
            parts = sorted((out2 / "shards").glob("*.part.jsonl"))
            if i == 1 and parts:                       # tear the last line of a partial shard
                with open(parts[0], "ab") as fh:
                    fh.write(b'{"key": "torn')
        check("hard kills (99) then completion", 99 in codes and codes[-1] == 0, str(codes))
        check("killed-and-resumed shards are byte-identical", shard_hashes(out2) == ref)
        check("torn line kept as .bad copy", bool(list((out2 / "shards").glob("*.bad.*"))))

        # 3. pause file, max-shards, growth recycle
        cfg3, out3 = cfg_file(tmp, "ctl")
        (tmp / "ctl.CONTROL").write_text("pause\n")
        check("pause file stops before starting (10)", run(cfg3) == 10)
        (tmp / "ctl.CONTROL").write_text("run\n")
        check("--max-shards 2 returns 11", run(cfg3, "--max-shards", "2") == 11)
        check("two shards done after that", len(shard_hashes(out3)) == 2)
        check("rest completes (0)", run(cfg3) == 0 and shard_hashes(out3) == ref)
        cfg4, out4 = cfg_file(tmp, "grow", growth_per_shard=1.0)
        check("memory growth asks for a restart (12)", run(cfg4) == 12)

        # 4. canary
        refrows = tmp / "canary_ref.jsonl"
        refrows.write_text("".join((out / "shards" / "s00.jsonl").read_text().splitlines(True)[:5]))
        cfg5, out5 = cfg_file(tmp, "canary", canary_reference=str(refrows))
        check("canary passes with the same model (0)", run(cfg5) == 0)
        cfg6, out6 = cfg_file(tmp, "canary_bad", canary_reference=str(refrows), salt="b")
        check("canary mismatch stops (20)", run(cfg6) == 20 and not shard_hashes(out6))

        # 5. design change and corrupted shard
        d = yaml.safe_load(open(cfg)); d["design"]["version"] = 2
        cfg7 = tmp / "full_v2.yaml"; cfg7.write_text(yaml.safe_dump(d))
        check("changed design stops (30)", run(cfg7) == 30)
        check("--accept-design-change allows it", run(cfg7, "--accept-design-change") == 0)
        bad = out / "shards" / "s03.jsonl"
        bad.write_text(bad.read_text().replace("reply", "rEply", 1))
        check("corrupted shard: restart rebuilds it (0)", run(cfg) == 0)
        check("corrupted file quarantined, not deleted", bool(list((out / "shards").glob("s03.jsonl.quarantined.*"))))
        check("rebuilt shard matches the original", shard_hashes(out) == ref)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nFAILED: " + ", ".join(fails) if fails else "\nall checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
