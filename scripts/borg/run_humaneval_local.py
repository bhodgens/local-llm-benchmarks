#!/usr/bin/env python3
"""HumanEval scoring against an OpenAI-compatible chat endpoint.

Why this exists instead of lm-evaluation-harness: driving lm-eval's humaneval tasks
against luce_server produced pass@1 = 0.0 for every model. The prompts arrived at the
model double-escaped (literal "\\n" instead of newlines -- visible in the logged
samples), so the generated code and the appended test block were malformed and every
problem failed regardless of the answer. This runner sends the prompt as-is, extracts
the completion deterministically, and executes the official tests, so a score of 0 is
a statement about the model and not about the harness.

Usage:
  run_humaneval_local.py --port 8901 --model luce --limit 60 --label qwen38-27b \\
      --out /root/bench/results/humaneval_local.jsonl
"""
import argparse, json, os, re, subprocess, sys, tempfile, urllib.request

PARQUET = "/root/.cache/huggingface/hub/datasets--openai--openai_humaneval/snapshots/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval/test-00000-of-00001.parquet"

INSTRUCTION = (
    "Complete the following Python function. Return ONLY the code, no explanation, "
    "no markdown fences. Include every import the function needs.\n\n```python\n{prompt}\n```"
)


def load_problems(limit):
    import pyarrow.parquet as pq
    t = pq.read_table(PARQUET).to_pylist()
    return t[:limit] if limit else t


def ask(port, model, prompt, max_tokens=1024, timeout=600):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": INSTRUCTION.format(prompt=prompt)}],
        "temperature": 0.0, "max_tokens": max_tokens, "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    # Some servers (Strata) return content=null; reasoning may carry the text instead.
    content = msg.get("content") or msg.get("reasoning_content") or ""
    if not isinstance(content, str):
        content = str(content)
    return content


def extract(resp, prompt, entry_point):
    """Return the model's code block verbatim (fences stripped).

    Do NOT trim to "def <entry>": models emit "from typing import List" above the
    signature, and cutting there produced NameError: name 'List' is not defined for
    every typed problem, which read as a 12% score on a model that scores ~80%.
    """
    code = (resp or "").strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", code, re.S)
    if m:
        code = m.group(1).strip()
    # A model that continues the prompt (body only) must not have the prompt re-sent.
    return code


def score(problem, completion):
    entry = problem["entry_point"]
    code = extract(completion, problem["prompt"], entry)
    if re.search(rf"(^|\n)\s*def {re.escape(entry)}\s*\(", code):
        program = code                      # full function (with its own imports)
    else:
        program = problem["prompt"] + code  # body-only completion
    if "typing" not in program:
        program = "from typing import *\n" + program
    program = program + "\n\n" + problem["test"] + f"\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=20)
        return r.returncode == 0, (r.stderr or "")[-200:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    problems = load_problems(a.limit)
    passed, records = 0, []
    for i, p in enumerate(problems):
        try:
            resp = ask(a.port, a.model, p["prompt"])
        except Exception as e:
            records.append({"task_id": p["task_id"], "ok": False, "error": str(e)[:120]})
            continue
        try:
            ok, err = score(p, resp)
        except Exception as e:
            ok, err = False, f"scorer error: {e}"[:120]
        passed += ok
        records.append({"task_id": p["task_id"], "ok": bool(ok), "err": err if not ok else ""})
        print(f"[{i+1}/{len(problems)}] {p['task_id']} {'PASS' if ok else 'fail'}", flush=True)

    n = len(problems)
    rec = {"label": a.label, "model": a.model, "n": n, "passed": passed,
           "humaneval_pass@1": round(passed / n, 4) if n else None,
           "runner": "run_humaneval_local.py (custom, execution-scored)"}
    print(json.dumps(rec))
    rec["records"] = records
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
