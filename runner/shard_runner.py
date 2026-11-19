#!/usr/bin/env python3
"""Generic resumable shard runner: run a long job as many small shards that survive a stop, a kill or a reboot.

You supply a task module (a few functions, see below); this file supplies everything that makes a long run safe:

  * every row is appended and flushed to <out>/shards/<id>.part.jsonl as it is made
  * a finished shard is fsynced, renamed atomically to <id>.jsonl, and gets <id>.done.json (rows, sha256, timing, memory)
    plus one line in <out>/LEDGER.jsonl
  * a restart verifies finished shards, skips them, and resumes a partial shard from its first missing key; a torn last
    line (kill mid-write) is cut off. Bad files are renamed *.quarantined.<time>, never deleted
  * optional canary: stored rows are regenerated at every start and must match, or the run stops (model, library or
    prompt changed between sessions)
  * pause file (CONTROL) and competing-process check between shards, state file, process recycling after N shards or
    when peak memory grows, a design manifest that stops a run whose settings changed

Task module (named by `task:` in the config; importable from the repo root or `--task-path`):
    shards(cfg)                  -> [{"id": str, "keys": [str, ...]}, ...]   ordered; keys unique within a shard
    setup(cfg, dry_run)          -> state         load the model ONCE here (dry_run: use a stub, no GPU)
    process(state, shard_id, key)-> dict          one JSON-serialisable row; the runner adds "key"
  optional:
    canary(state, cfg)           -> (checked, mismatches)
    peak_gb(state) / reset_peak(state) / free_gb(state)

Exit codes: 0 all done | 10 paused or GPU busy | 11 --max-shards reached, more to do | 12 recycle (memory growth)
            20 canary mismatch | 30 config, data or design problem.
`runner/run_sharded.sh` loops on these codes under the memory watchdog.

Usage: python runner/shard_runner.py CONFIG [--list] [--max-shards N] [--only ID,ID] [--dry-run] [--out-dir DIR]
       [--control-file FILE] [--no-canary] [--accept-design-change] [--task-path DIR] [--kill-after-rows N]
"""
import argparse, datetime, hashlib, importlib, json, os, pathlib, subprocess, sys, time
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXIT_OK, EXIT_PAUSED, EXIT_MORE, EXIT_RECYCLE, EXIT_CANARY, EXIT_BAD = 0, 10, 11, 12, 20, 30
FREE_SAMPLE_EVERY = 25      # rows between free-memory samples
RUNNER_VERSION = "1"


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def die(code, msg):
    log("STOP: " + msg)
    sys.exit(code)


def rp(p):
    p = pathlib.Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else ROOT / p


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


# ---------------------------------------------------------------- shard files
def paths(out, sid):
    d = out / "shards"
    return d / f"{sid}.part.jsonl", d / f"{sid}.jsonl", d / f"{sid}.done.json"


def quarantine(p, why):
    q = p.with_name(p.name + f".quarantined.{time.strftime('%Y%m%dT%H%M%S')}")
    os.replace(p, q)
    log(f"quarantined {p.name}: {why}")
    return q


def shard_state(out, sh):
    """'done' (verified), 'partial' (resumable) or 'new'. Bad files are renamed, never deleted."""
    part, final, done = paths(out, sh["id"])
    n = len(sh["keys"])
    if final.exists():
        ok = False
        try:
            rows = read_jsonl(final)
            ok = len(rows) == n and {r["key"] for r in rows} == set(sh["keys"])
            if ok and done.exists():
                ok = json.load(open(done)).get("sha256") == sha256_file(final)
            elif ok:                       # crash between rename and done.json: recover the sidecar
                json.dump({"rows": n, "sha256": sha256_file(final), "recovered": True, "ended": now()}, open(done, "w"))
        except Exception:
            ok = False
        if ok:
            return "done"
        quarantine(final, "failed verification")
        if done.exists():
            quarantine(done, "failed verification")
    return "partial" if part.exists() else "new"


def load_part(part):
    """Rows of a partial shard. A torn last line is cut off; the original bytes are kept as .bad.<time>."""
    rows, good = [], 0
    raw = part.read_bytes()
    for line in raw.splitlines(keepends=True):
        try:
            rows.append(json.loads(line))
            good += len(line)
        except Exception:
            break
    if good != len(raw):
        bad = part.with_name(part.name + f".bad.{time.strftime('%Y%m%dT%H%M%S')}")
        bad.write_bytes(raw)
        part.write_bytes(raw[:good])
        log(f"{part.name}: torn last line cut off (copy kept as {bad.name})")
    return rows


# ---------------------------------------------------------------- control
def pause_reason(cfg, control_file):
    cf = rp(control_file) if control_file else None
    if cf and cf.exists() and cf.read_text().strip().startswith("pause"):
        return "control file says pause"
    pat = cfg.get("pause_if_processes")
    if pat:
        try:
            pids = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split()
        except FileNotFoundError:
            pids = []
        pids = [p for p in pids if int(p) not in (os.getpid(), os.getppid())]
        if pids:
            return f"competing job running (pids {','.join(pids)})"
    return None


def write_state(cfg, text):
    sf = cfg.get("state_file")
    if sf:
        try:
            rp(sf).write_text(text + "\n")
        except Exception:
            pass


def opt(task, name, state, default=None):
    f = getattr(task, name, None)
    return f(state) if f else default


# ---------------------------------------------------------------- one shard
def run_shard(sh, ctx):
    out, task, state, args = ctx["out"], ctx["task"], ctx["state"], ctx["args"]
    part, final, done = paths(out, sh["id"])
    part.parent.mkdir(parents=True, exist_ok=True)
    keys = sh["keys"]
    have = {}
    if part.exists():
        have = {r["key"]: r for r in load_part(part)}
        if set(have) - set(keys):
            quarantine(part, "holds keys that are not in this shard")
            have = {}
    resumed = len(have)
    t0, started, min_free, since = time.time(), now(), None, 0
    opt(task, "reset_peak", state)
    fh = open(part, "a", encoding="utf-8")
    for key in keys:
        if key in have:
            continue
        row = {"key": key, **task.process(state, sh["id"], key)}
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()                                    # every row reaches the OS; fsync when the shard closes
        have[key] = row
        ctx["written"] += 1
        since += 1
        if since >= FREE_SAMPLE_EVERY:
            since = 0
            fr = opt(task, "free_gb", state)
            if fr is not None:
                min_free = fr if min_free is None else min(min_free, fr)
        if args.kill_after_rows and ctx["written"] >= args.kill_after_rows:
            os._exit(99)                              # simulated hard kill (tests only): no cleanup at all
    fh.flush()
    os.fsync(fh.fileno())
    fh.close()
    if len(have) != len(keys):
        die(EXIT_BAD, f"shard {sh['id']}: {len(have)} rows, expected {len(keys)}")
    os.replace(part, final)
    rec = {"shard": sh["id"], "rows": len(keys), "resumed_rows": resumed, "sha256": sha256_file(final),
           "wall_s": round(time.time() - t0, 2), "peak_gb": opt(task, "peak_gb", state), "min_free_gb": min_free,
           "started": started, "ended": now(), "pid": os.getpid(), "rc": 0}
    json.dump(rec, open(done, "w"), indent=1)
    with open(out / "LEDGER.jsonl", "a") as lf:
        lf.write(json.dumps(rec) + "\n")
    return rec


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-shards", type=int)
    ap.add_argument("--only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir")
    ap.add_argument("--control-file")
    ap.add_argument("--no-canary", action="store_true")
    ap.add_argument("--accept-design-change", action="store_true")
    ap.add_argument("--task-path")
    ap.add_argument("--kill-after-rows", type=int)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sys.path[:0] = [str(rp(args.task_path or ".")), str(ROOT)]
    task = importlib.import_module(cfg["task"])
    out = rp(args.out_dir or cfg["out_dir"])
    control_file = args.control_file or cfg.get("control_file")

    shards = task.shards(cfg)
    ids = [s["id"] for s in shards]
    if len(set(ids)) != len(ids):
        die(EXIT_BAD, "shard ids are not unique")
    for s in shards:
        if len(set(s["keys"])) != len(s["keys"]) or not s["keys"]:
            die(EXIT_BAD, f"shard {s['id']}: keys must be non-empty and unique")
    if args.only:
        keep = set(args.only.split(","))
        shards = [s for s in shards if s["id"] in keep]

    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    task_file = pathlib.Path(task.__file__)
    design = {"design": cfg.get("design", {}), "shard_ids_sha256": sha256_text("\n".join(ids)),
              "keys_sha256": sha256_text("\n".join(k for s in shards for k in s["keys"])) if not args.only else None,
              "task_sha256": sha256_file(task_file)}
    mp = out / "MANIFEST.json"
    if mp.exists():
        old = json.load(open(mp))["design"]
        diffs = {k: (old.get(k), design.get(k)) for k in design if old.get(k) != design.get(k) and design.get(k) is not None}
        if diffs and not args.accept_design_change:
            die(EXIT_BAD, f"design differs from this output folder's manifest: {diffs}")
    else:
        json.dump({"created": now(), "design": design, "runner_version": RUNNER_VERSION,
                   "runner_sha256": sha256_file(pathlib.Path(__file__)), "config": str(args.config)}, open(mp, "w"), indent=1)

    states = {s["id"]: shard_state(out, s) for s in shards}
    todo = [s for s in shards if states[s["id"]] != "done"]
    log(f"plan: {len(shards)} shards, {len(shards) - len(todo)} done, {len(todo)} to do "
        f"({sum(len(s['keys']) for s in todo)} rows)")
    if args.list:
        for s in todo[:20]:
            print(s["id"], states[s["id"]], len(s["keys"]))
        return EXIT_OK
    if not todo:
        log("complete")
        write_state(cfg, f"complete {now()}")
        return EXIT_OK
    why = pause_reason(cfg, control_file)
    if why:
        write_state(cfg, f"paused {now()} ({why})")
        log("not starting: " + why)
        return EXIT_PAUSED

    state = task.setup(cfg, args.dry_run)
    ctx = {"out": out, "task": task, "state": state, "args": args, "written": 0}
    with open(out / "PROCESS_LOG.jsonl", "a") as pf:
        pf.write(json.dumps({"time": now(), "pid": os.getpid(), "runner_sha256": sha256_file(pathlib.Path(__file__)),
                             "dry_run": args.dry_run, "todo": len(todo)}) + "\n")
    if hasattr(task, "canary") and not args.no_canary:
        checked, bad = task.canary(state, cfg)
        if bad:
            (out / "CANARY.log").open("a").write(f"{now()} {len(bad)} of {checked} differ: {json.dumps(bad)[:2000]}\n")
            die(EXIT_CANARY, f"canary: {len(bad)} of {checked} regenerated rows differ from the stored ones; the model, "
                             "library or prompt changed. Details in CANARY.log")
        log(f"canary ok: {checked} of {checked} rows identical" if checked else "canary skipped: the task gave no reference rows")

    max_shards = args.max_shards or cfg.get("recycle_after_shards", 10)
    first_peak, n_run = None, 0
    for s in todo:
        why = pause_reason(cfg, control_file)
        if why:
            write_state(cfg, f"paused {now()} ({why})")
            log("pausing between shards: " + why)
            return EXIT_PAUSED
        write_state(cfg, f"running {s['id']} {now()}")
        rec = run_shard(s, ctx)
        n_run += 1
        log(f"shard {s['id']} done: {rec['rows']} rows, {rec['wall_s']} s, peak {rec['peak_gb']} GB, min free {rec['min_free_gb']} GB")
        pk = rec["peak_gb"]
        if pk is not None:
            if first_peak is None:
                first_peak = pk
            elif pk - first_peak > cfg.get("growth_limit_gb", 1.5):
                log(f"memory growth: peak {pk} GB vs {first_peak} GB in the first shard; asking for a restart")
                return EXIT_RECYCLE
        last = s is todo[-1]
        if n_run >= max_shards and not last:
            return EXIT_MORE
        if not last and cfg.get("pause_between_shards_s"):
            time.sleep(cfg["pause_between_shards_s"])
    log("all shards done")
    write_state(cfg, f"complete {now()}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
