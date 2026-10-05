#!/usr/bin/env python3
"""Print the borg luce_server lane results in a compact table."""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/root/bench/bench_results_luce_server.json"
d = json.load(open(path))
print(f"{'model':22s} {'temp0.3':>8s} {'spec':>5s} {'greedy':>8s} {'spec':>5s} {'accept':>7s} {'prefill_cold':>13s}")
for k, v in d.items():
    h = v.get("harness_temp0.3", {})
    g = v.get("production_greedy", {})
    print(f"{k:22s} {str(h.get('decode_tps_avg')):>8s} {str(h.get('spec_decode_ran')):>5s} "
          f"{str(g.get('decode_tps_avg')):>8s} {str(g.get('spec_decode_ran')):>5s} "
          f"{str(g.get('accept_rate_avg')):>7s} {str(v.get('prompt_tps_cold')):>13s}")
