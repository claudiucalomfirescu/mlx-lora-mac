"""Serve base+adapter with `mlx_lm.server --adapter-path` (thinking off via --chat-template-args) on a free high port,
send greedy chat requests, and compare replies with the in-process adapter eval and the base eval.
Usage: python scripts/server_check.py configs/recipe.yaml --adapter outputs/runs/<run>/adapters --ref <name>_tuned --base <name>_base
Writes outputs/results/server_<ref>.json. Run under run_guarded.py: the server is a child, so the watchdog covers it.
"""
import argparse, json, random, socket, subprocess, sys, time, urllib.request
from common import ROOT, OUT, load_cfg, write_json

ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--adapter", required=True)
ap.add_argument("--ref", required=True); ap.add_argument("--base", required=True)
a = ap.parse_args()
cfg = load_cfg(a.config); s = cfg["server"]
ref = json.load(open(OUT / "results" / f"{a.ref}.json")); base = json.load(open(OUT / "results" / f"{a.base}.json"))
base_by_id = {r["id"]: r["reply"] for r in base["gen_heldout"] + base["gen_unrelated"]}

def free_port():
    for _ in range(50):
        p = random.randint(20000, 60000)
        with socket.socket() as so:
            if so.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())

port = free_port()
cmd = [sys.executable, "-m", "mlx_lm", "server", "--model", cfg["model_path"], "--adapter-path", str(ROOT / a.adapter),
       "--port", str(port), "--temp", "0.0", "--chat-template-args", json.dumps(cfg.get("chat_template_kwargs") or {})]
log = open(OUT / "logs" / f"server_{a.ref}.server.log", "w")
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
url = f"http://127.0.0.1:{port}"
try:
    for _ in range(180):
        try:
            models = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=2).read()); break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("server did not start")
    rows = []
    t0 = time.time()
    for it in ref["gen_heldout"][:s["n_heldout"]] + ref["gen_unrelated"][:s["n_unrelated"]]:
        r = post(url + "/v1/chat/completions", {"model": "default_model", "messages": [{"role": "user", "content": it["prompt"]}],
                                                "temperature": 0.0, "max_tokens": ref["config"]["eval"]["max_tokens"]})
        txt = r["choices"][0]["message"]["content"]
        rows.append({"id": it["id"], "server": txt, "in_process_adapter": it["reply"], "base": base_by_id[it["id"]],
                     "same_as_adapter": txt.strip() == it["reply"].strip(), "same_as_base": txt.strip() == base_by_id[it["id"]].strip()})
    out = {"port": port, "cmd": cmd, "models_endpoint": models, "seconds": round(time.time() - t0, 1),
           "identical_to_in_process_adapter": f"{sum(r['same_as_adapter'] for r in rows)}/{len(rows)}",
           "identical_to_base": f"{sum(r['same_as_base'] for r in rows)}/{len(rows)}", "rows": rows}
    write_json(OUT / "results" / f"server_{a.ref}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in rows:
        print(r["id"], "| server:", repr(r["server"][:70]), "| base:", repr(r["base"][:50]))
finally:
    srv.terminate(); srv.wait(timeout=30)
