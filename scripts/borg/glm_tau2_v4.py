#!/usr/bin/env python3
"""glm53 tau2 v4: serve GLM persistently and run tau2 against it.

v3 failed: tau2's CLI prompted on stdin at startup, and with stdin=EOF inside
setsid it died instantly (EOFError) before any sims ran.
Fix: feed the tau2 subprocess a live stdin pipe (newline every 30s), keep the
server up for the whole run in this script, hold the lane lock, and verify the
server model id with a live call before starting.
"""
import json, subprocess, time, urllib.request, os, glob, fcntl, sys, threading

TAU2_DIR = "/root/tau2-bench"
TAU2 = f"{TAU2_DIR}/.venv/bin/tau2"
LOGS = "/root/bench/coding-eval-logs"
PROGRESS = "/root/bench/coding-eval-progress.json"
KEY = "glm53-flash"
API_MODEL = "glm-5.3-exl3"
PORT = 8000

lock = open("/root/bench/.lane.lock", "w")
fcntl.flock(lock, fcntl.LOCK_EX)

def wait_up(url, want, tries=240):
    for _ in range(tries):
        try:
            if want in urllib.request.urlopen(url, timeout=3).read():
                return True
        except Exception:
            pass
        time.sleep(5)
    return False

subprocess.run(["/root/bench/cleanup_lanes.sh"], capture_output=True)
log = open(f"{LOGS}/he4-glm-tau2-server.log", "w")
proc = subprocess.Popen(["/root/bench/serve_kyojin_glm.sh", "8000", "32768", "2"],
                        stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True)
if not wait_up(f"http://127.0.0.1:{PORT}/v1/models", b'"data"'):
    print("SERVER FAILED"); sys.exit(1)
body = json.dumps({"model": API_MODEL, "messages": [{"role": "user", "content": "say ok"}],
                   "max_tokens": 8}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=120).read()
print("glm up + id verified", flush=True)

safe = KEY.replace("/", "_")
save_dir = f"borg_{safe}_tau2_v4"
cmd = [TAU2, "run", "--domain", "airline",
       "--agent-llm", f"openai/{API_MODEL}",
       "--agent-llm-args", json.dumps({"api_key": "none",
            "api_base": f"http://127.0.0.1:{PORT}/v1", "temperature": 0.0}),
       "--user-llm", f"openai/{API_MODEL}",
       "--user-llm-args", json.dumps({"api_key": "none",
            "api_base": f"http://127.0.0.1:{PORT}/v1", "temperature": 0.0}),
       "--num-tasks", "15", "--num-trials", "1", "--max-concurrency", "1",
       "--max-steps", "30", "--max-errors", "5", "--timeout", "300",
       "--seed", "42", "--save-to", save_dir]
env = dict(os.environ); env["OPENAI_API_KEY"] = "none"

tau_log = open(f"{LOGS}/{KEY}_tau2_v4.log", "w")
tau = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=tau_log,
                       stderr=subprocess.STDOUT, text=True, cwd=TAU2_DIR,
                       env=env, start_new_session=True)
def feed():
    while tau.poll() is None and tau.stdin:
        try:
            tau.stdin.write("\n"); tau.stdin.flush()
        except Exception:
            return
        time.sleep(30)
threading.Thread(target=feed, daemon=True).start()

start = time.time()
tau.wait(timeout=43200)
elapsed = time.time() - start

rs = []
for f in glob.glob(f"{TAU2_DIR}/data/simulations/{save_dir}/**/results.json", recursive=True):
    d = json.load(open(f))
    for s in d.get("simulations") or []:
        ri = s.get("reward_info") or {}
        if isinstance(ri.get("reward"), (int, float)):
            rs.append(ri["reward"])
reward = round(sum(rs) / len(rs), 4) if rs else None

p = json.load(open(PROGRESS))
for m in p["models"]:
    if m["name"] == KEY:
        m["tau2"] = {"reward": reward, "wall_time_s": round(elapsed, 1),
                     "num_tasks": 15, "scored": len(rs), "domain": "airline",
                     "host": "borg", "rerun": "v4"}
json.dump(p, open(PROGRESS, "w"), indent=2)
print(f"glm53 tau2: reward={reward} (n={len(rs)}) wall={round(elapsed/60,1)}min rc={tau.returncode}", flush=True)
