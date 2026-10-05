#!/usr/bin/env python3
"""Class A parity harness for luce_server: 10 fixed HumanEval prompts, greedy.

Uses POST /v1/chat/completions and reads usage.timings.* — matching the
lucebox blog protocol (greedy, HumanEval-10, token-weighted avg + best).
Usage: parity.py --port 8901 [--max-tokens 256] [--out results.json]
"""
import argparse, json, time, urllib.request

HUMANEVAL_10 = [
    "def has_close_elements(numbers, threshold):\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
    "def separate_paren_groups(paren_string):\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those groups into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
    "def truncate_number(number):\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
    "def below_zero(operations):\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account falls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n",
    "def mean_absolute_deviation(numbers):\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n",
    "def intersperse(numbers, delimeter):\n    \"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers'\n    >>> intersperse([], 4)\n    []\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    \"\"\"\n",
    "def parse_nested_parens(paren_string):\n    \"\"\" Input to this function is a string represented multiple groups for nested parentheses separated by spaces.\n    For each of the group, output the deepest level of nesting of parentheses.\n    E.g. (()()) has maximum two levels of nesting while ((())) has three.\n\n    >>> parse_nested_parens('(()()) ((())) () ((())()())')\n    [2, 3, 1, 3]\n    \"\"\"\n",
    "def filter_by_substring(strings, substring):\n    \"\"\" Filter an input list of strings only for ones that contain given substring\n    >>> filter_by_substring([], 'a')\n    []\n    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n    ['abc', 'bacd', 'array']\n    \"\"\"\n",
    "def sum_product(numbers):\n    \"\"\" For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.\n    Empty sum should be equal to 0 and empty product should be equal to 1.\n    >>> sum_product([])\n    (0, 1)\n    >>> sum_product([1, 2, 3, 4])\n    (10, 24)\n    \"\"\"\n",
    "def rolling_max(numbers):\n    \"\"\" From a given list of integers, generate a list of rolling maximum element found until given moment\n    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n    \"\"\"\n",
]

def gen(port, prompt, max_tokens):
    payload = json.dumps({
        "model": "luce",
        "messages": [{"role": "user", "content": "Complete this Python function. Output only code.\n\n" + prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    u = d.get("usage", {})
    t = u.get("timings", {})
    dec_ms = t.get("decode_ms", 0)
    return {
        "decode_tps": t.get("decode_tokens_per_sec", 0.0),
        "completion_tokens": u.get("completion_tokens", 0),
        "prompt_tokens": u.get("prompt_tokens", 0),
        "accept_rate": round(u.get("accept_rate", 0), 3),
        "spec_decode_ran": u.get("spec_decode_ran", False),
        "prompt_tps": round(u.get("prompt_tokens", 0) / t["prefill_ms"] * 1000, 1) if t.get("prefill_ms") else 0.0,
        "wall_s": round(time.time() - t0, 3),
        "finish": d.get("choices", [{}])[0].get("finish_reason", ""),
        "text_len": len(d.get("choices", [{}])[0].get("message", {}).get("content", "")),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    for i in range(args.warmup):
        gen(args.port, HUMANEVAL_10[i % len(HUMANEVAL_10)], 64)
    print(f"warmup x{args.warmup} done")

    results = []
    for i, p in enumerate(HUMANEVAL_10):
        r = gen(args.port, p, args.max_tokens)
        r["prompt_idx"] = i
        results.append(r)
        print(f"[{i}] {r['decode_tps']:.1f} tok/s  ({r['completion_tokens']} tok, accept={r['accept_rate']}, finish={r['finish']})")

    total_tok = sum(r["completion_tokens"] for r in results)
    total_time = sum(r["completion_tokens"] / r["decode_tps"] for r in results if r["decode_tps"] > 0)
    weighted_avg = total_tok / total_time if total_time else 0
    best = max(r["decode_tps"] for r in results)
    acc = sum(r["accept_rate"] * r["completion_tokens"] for r in results) / max(total_tok, 1)
    summary = {"weighted_avg_tps": round(weighted_avg, 1), "best_tps": round(best, 1),
               "weighted_accept": round(acc, 3), "n_prompts": len(results),
               "max_tokens": args.max_tokens}
    print("SUMMARY:", json.dumps(summary))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "runs": results}, f, indent=2)

if __name__ == "__main__":
    main()
