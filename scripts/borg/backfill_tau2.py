#!/usr/bin/env python3
"""Extract tau2 rewards from tau2-bench simulation records for borg entries.

The orchestrator's parser missed rewards that live in per-task record files rather
than a single summary json. Walk every simulation dir for each model's save_to dir
and average task rewards.
"""
import glob, json

TAU2_SIMS = "/root/tau2-bench/data/simulations"
PROGRESS = "/root/bench/coding-eval-progress.json"

SAVE_DIRS = {
    "qwen38-27b": "borg_qwen38-27b",
    "qwen38-27b-vision": "borg_qwen38-27b-vision",
    "laguna-xs21": "borg_laguna-xs21",
    "laguna-s21": "borg_laguna-s21",
    "flashnext": "borg_flashnext",
    "swift": "borg_swift",
    "coder": "borg_coder",
    "glm53-flash": "borg_glm53-flash",
    "kolibri-1": "borg_kolibri-1",
}


def rewards_for(save_dir):
    rs = []
    # results.json holds {"info": {...}, "results": [task records]}
    try:
        d = json.load(open(f"{TAU2_SIMS}/{save_dir}/results.json"))
        res = d.get("simulations") if isinstance(d, dict) else d
        for r in (res or []):
            if isinstance(r, dict) and isinstance(r.get("reward_info"), dict):
                rew = r["reward_info"].get("reward")
                if isinstance(rew, (int, float)):
                    rs.append(rew)
                continue
            if not isinstance(r, dict):
                continue
            rew = r.get("reward")
            if isinstance(rew, dict):
                rew = rew.get("reward")
            if isinstance(rew, (int, float)):
                rs.append(rew)
    except FileNotFoundError:
        pass
    return rs


p = json.load(open(PROGRESS))
changed = False
for m in p["models"]:
    sd = SAVE_DIRS.get(m["name"])
    if not sd:
        continue
    rs = rewards_for(sd)
    if rs:
        avg = sum(rs) / len(rs)
        tau = m.get("tau2") or {}
        old = tau.get("reward")
        if old != round(avg, 4):
            tau["reward"] = round(avg, 4)
            tau["num_tasks_scored"] = len(rs)
            m["tau2"] = tau
            print(f"{m['name']}: tau2 {old} -> {round(avg,4)} (n={len(rs)})")
            changed = True
        else:
            print(f"{m['name']}: tau2 unchanged {old}")
    else:
        print(f"{m['name']}: no simulation records found in {sd}")
if changed:
    with open(PROGRESS + ".tmp", "w") as f:
        json.dump(p, f, indent=2)
    import os
    os.replace(PROGRESS + ".tmp", PROGRESS)
    print("progress updated")
