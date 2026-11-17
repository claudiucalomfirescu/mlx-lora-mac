"""Run ONE job under an outside memory watchdog (separate process).

- Preflight: refuses to start if another MLX job is running or system available memory < min.
- Starts the job in its own process group; every ~1 s sums phys_footprint (from /usr/bin/footprint,
  which includes Metal/GPU allocations, unlike RSS) over the job and all its descendants.
- Kills the whole group (SIGKILL) if footprint > min(limit, available_at_start - margin), or if
  system available memory < floor.
- Writes outputs/logs/<name>.guard.json (peaks, kill reason) and outputs/logs/<name>.log (job stdout/stderr).

Usage: python scripts/run_guarded.py --name NAME [--limit-gb 14 --margin-gb 3 --floor-gb 1.5 --min-avail-gb 8] -- cmd ...
"""
import argparse, json, os, re, signal, subprocess, sys, time, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
UNIT = {"B": 1e-9, "KB": 1024 / 1e9, "MB": 1024 ** 2 / 1e9, "GB": 1024 ** 3 / 1e9}


def available_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    def g(name):
        m = re.search(rf"{name}:\s+(\d+)", out)
        return int(m.group(1)) if m else 0
    return (g("Pages free") + g("Pages inactive") + g("Pages speculative")) * page / 1e9


def descendants(pid):
    pids, todo = [pid], [pid]
    while todo:
        p = todo.pop()
        out = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split()
        kids = [int(x) for x in out]
        pids += kids; todo += kids
    return pids


def footprint_gb(pids):
    out = subprocess.run(["footprint"] + sum([["-p", str(p)] for p in pids], []), capture_output=True, text=True).stdout
    tot = 0.0
    for num, unit in re.findall(r"phys_footprint:\s+([\d.]+)\s+(B|KB|MB|GB)", out):
        tot += float(num) * UNIT[unit]
    return tot


def other_mlx_jobs():
    out = subprocess.run(["pgrep", "-fl", "mlx_lm|mlx-lm|mlx_lm_lora"], capture_output=True, text=True).stdout
    me = os.getpid()
    return [l for l in out.splitlines() if l and int(l.split()[0]) != me and "run_guarded" not in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit-gb", type=float, default=14)
    ap.add_argument("--margin-gb", type=float, default=3)
    ap.add_argument("--floor-gb", type=float, default=1.5)
    ap.add_argument("--min-avail-gb", type=float, default=8)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    logs = ROOT / "outputs" / "logs"; logs.mkdir(parents=True, exist_ok=True)
    rec = {"name": a.name, "cmd": cmd, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    others = other_mlx_jobs()
    avail0 = available_gb()
    threshold = min(a.limit_gb, avail0 - a.margin_gb)
    rec.update(available_at_start_gb=round(avail0, 2), kill_threshold_gb=round(threshold, 2), other_mlx_jobs=others)
    if others:
        rec["refused"] = "another MLX job is running"
    elif avail0 < a.min_avail_gb:
        rec["refused"] = f"available {avail0:.1f} GB < {a.min_avail_gb} GB"
    if "refused" in rec:
        json.dump(rec, open(logs / f"{a.name}.guard.json", "w"), indent=1)
        print("REFUSED:", rec["refused"], others); sys.exit(3)
    print(f"[guard] available {avail0:.1f} GB, kill threshold {threshold:.1f} GB", flush=True)

    log = open(logs / f"{a.name}.log", "w")
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
    peak, min_avail, killed = 0.0, avail0, None
    while p.poll() is None:
        try:
            fp = footprint_gb(descendants(p.pid))
        except Exception:
            fp = 0.0
        av = available_gb()
        peak = max(peak, fp); min_avail = min(min_avail, av)
        if fp > threshold:
            killed = f"footprint {fp:.2f} GB > {threshold:.2f} GB"
        elif av < a.floor_gb:
            killed = f"system available {av:.2f} GB < {a.floor_gb} GB"
        if killed:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); break
        time.sleep(a.interval)
    rec.update(exit_code=p.returncode, killed=killed, peak_footprint_gb=round(peak, 2),
               min_system_available_gb=round(min_avail, 2), wall_s=round(time.time() - t0, 1),
               ended=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(rec, open(logs / f"{a.name}.guard.json", "w"), indent=1)
    print("[guard]", json.dumps({k: rec[k] for k in ("exit_code", "killed", "peak_footprint_gb", "min_system_available_gb", "wall_s")}))
    sys.exit(p.returncode if not killed else 9)


if __name__ == "__main__":
    main()
