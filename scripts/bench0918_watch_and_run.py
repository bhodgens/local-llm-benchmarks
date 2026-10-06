#!/usr/bin/env python3
"""
Wait until the 3060 (GPU1) is idle for 10 continuous minutes, then launch
the bench-0918 batch. "Idle" = GPU util 0% (sampled every 20s) AND no active
llama-server generation on the Nail prod port for the whole window.
"""
import subprocess, time, sys, json, urllib.request

GPU = "1"
WINDOW = 600  # 10 minutes
POLL = 20
PORT = 8080
LOG = "/tmp/coding-bench/bench0918_watch.log"
CMD = ["python3", "/home/caimlas/llm-benchmarks/scripts/run_bench0918_batch.py"]


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def util():
    out = subprocess.run(["nvidia-smi", "--id", GPU,
                          "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    try:
        return int(out.splitlines()[0])
    except Exception:
        return -1


def port_busy():
    """True if the prod Nail server logged prompt/decode activity in the last
    2 minutes (no /metrics endpoint on prod; journal print_timing lines are
    the activity signal)."""
    try:
        out = subprocess.run(
            ["journalctl", "-u", "caimlas-nail", "--since", "-2 min",
             "--no-pager", "-q"],
            capture_output=True, text=True, timeout=20).stdout
        for pat in ("print_timing", "n_decoded", "prompt processing"):
            if pat in out:
                return True
        return False
    except Exception:
        return False  # can't read journal -> rely on util only


def main():
    idle_since = None
    log(f"watcher started: waiting for GPU{GPU} idle {WINDOW}s "
        f"(util 0 + no in-flight tasks on :{PORT})")
    while True:
        u = util()
        busy = port_busy()
        now = time.time()
        if u == 0 and not busy:
            if idle_since is None:
                idle_since = now
                log(f"GPU idle, window started")
            elapsed = now - idle_since
            if elapsed >= WINDOW:
                log(f"IDLE {WINDOW}s confirmed -> launching batch")
                break
            else:
                log(f"idle {int(elapsed)}/{WINDOW}s")
        else:
            if idle_since is not None:
                log(f"activity (util={u}%, busy={busy}) -> window reset")
            idle_since = None
        time.sleep(POLL)

    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] BATCH LAUNCH\n")
    env = dict(__import__('os').environ)
    p = subprocess.run(["nohup", "python3", CMD[1]], env=env,
                       stdout=open("/tmp/coding-bench/bench0918_batch.out", "w"),
                       stderr=subprocess.STDOUT, start_new_session=True)
    log(f"batch launcher exited rc={p.returncode}")


if __name__ == "__main__":
    main()
