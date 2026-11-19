#!/usr/bin/env python3
"""Tests of scripts/run_guarded.py with fake `vm_stat` and `footprint` programs (no model, no GPU, any OS).
Run: python3 tests/test_run_guarded.py     (the default MLX-job pattern is replaced here so other processes cannot interfere)
Checks: normal job and guard record with log tail; an earlier attempt's log is kept, not overwritten; pause file (10);
too little free memory (3); competing process via --block-on (3); footprint over the threshold kills the job (9)."""
import json, os, pathlib, subprocess, sys, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
G = str(ROOT / "scripts" / "run_guarded.py")
fails = []
MARK = f"guard-test-marker-{os.getpid()}"
NOMATCH = f"no-process-has-this-{os.getpid()}-text"      # replaces the pattern for other MLX jobs


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra and not cond else ""))
    if not cond:
        fails.append(name)


def fake_bin(d, free_pages, footprint_gb):
    d.mkdir(exist_ok=True)
    (d / "vm_stat").write_text(f"#!/bin/sh\necho 'Mach Virtual Memory Statistics: (page size of 16384 bytes)'\n"
                               f"echo 'Pages free: {free_pages}.'\necho 'Pages inactive: 0.'\necho 'Pages speculative: 0.'\n")
    (d / "footprint").write_text(f"#!/bin/sh\necho 'phys_footprint: {footprint_gb} GB'\n")
    for f in ("vm_stat", "footprint"):
        (d / f).chmod(0o755)
    return d


def guard(tmp, name, bin_dir, *args, cmd=("echo", "hello")):
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    p = subprocess.run([sys.executable, G, "--name", name, "--log-dir", str(tmp / "logs"), "--interval", "0.2", "--mlx-pattern", NOMATCH, *args, "--", *cmd],
                       capture_output=True, text=True, env=env)
    rec = tmp / "logs" / f"{name}.guard.json"
    return p.returncode, (json.load(open(rec)) if rec.exists() else None)


def main():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="guard_test_"))
    ok_bin = fake_bin(tmp / "ok", 1_000_000, 2)             # about 16 GB free, job footprint 2 GB
    rc, rec = guard(tmp, "job", ok_bin)
    check("normal job exits 0", rc == 0 and rec and rec["exit_code"] == 0)
    check("guard record has the log tail", rec and rec["log_tail"] == ["hello"], str(rec and rec.get("log_tail")))
    rc, rec = guard(tmp, "job", ok_bin)
    kept = [p.name for p in (tmp / "logs").glob("job.2*")]
    check("second attempt keeps the first attempt's log and record", len(kept) == 2 and rc == 0, str(kept))
    (tmp / "CONTROL").write_text("pause\n")
    rc, rec = guard(tmp, "paused", ok_bin, "--control-file", str(tmp / "CONTROL"))
    check("pause file refuses (10)", rc == 10 and "pause" in rec["refused"])
    (tmp / "CONTROL").write_text("run\n")
    rc, rec = guard(tmp, "notpaused", ok_bin, "--control-file", str(tmp / "CONTROL"))
    check("'run' in the control file does not block", rc == 0)
    low = fake_bin(tmp / "low", 100_000, 1)                  # about 1.6 GB free
    rc, rec = guard(tmp, "low", low)
    check("too little free memory refuses (3)", rc == 3 and "available" in rec["refused"])
    other = subprocess.Popen(["sh", "-c", "sleep 30", MARK])
    try:
        rc, rec = guard(tmp, "blocked", ok_bin, "--block-on", MARK)
        check("--block-on a running process refuses (3)", rc == 3 and "another MLX job" in rec["refused"])
    finally:
        other.kill()
    big = fake_bin(tmp / "big", 1_000_000, 20)               # footprint 20 GB over a 13.4 GB threshold
    t0 = time.time()
    rc, rec = guard(tmp, "kill", big, cmd=("sleep", "30"))
    check("footprint over the threshold kills the job (9)", rc == 9 and rec["killed"] and time.time() - t0 < 15, str(rec))
    print("\nFAILED: " + ", ".join(fails) if fails else "\nall checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
