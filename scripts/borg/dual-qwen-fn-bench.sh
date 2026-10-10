#!/usr/bin/env bash
# Dual co-residency benchmark: qwen3.8-27B (luce) + flashnext iq3_s (strata)
set -u
export PATH=/usr/local/bin:$PATH
B=/root/bench
R=$B/results
LOCK=$B/.lane.lock
QPORT=8901; QMODEL=luce
FPORT=8080; FMODEL=qwen3.8-flash-next-iq3_s
PY=/root/lmeval/bin/python3
LOGQ=$R/dualqfn-qwen-solo.log
LOGF=$R/dualqfn-fn-solo.log
LOGQ2=$R/dualqfn-qwen-resident.log
LOGF2=$R/dualqfn-fn-resident.log
mkdir -p $R

wait_healthy() { # port, timeout_s, log
  local port=$1 tlimit=$2 log=$3 t=0
  while [ $t -lt $tlimit ]; do
    if curl -s -m 3 http://127.0.0.1:$port/v1/models | grep -q '"data"'; then
      echo "HEALTHY port=$port after ${t}s"; return 0
    fi
    sleep 5; t=$((t+5))
  done
  echo "UNHEALTHY port=$port after ${t}s"; echo "--- last log lines ---"; tail -25 "$log"; return 1
}

mem_snapshot() {
  local tag=$1
  {
    echo "=== $tag $(date -Is) ==="
    rocm-smi --showmeminfo vram
    free -g
  } >> $R/dualqfn-mem.log 2>&1
}

probe() { # port model words maxtok label outfile
  $PY $B/gen_probe.py --port $1 --model $2 --words $3 --max-tokens $4 --label $5 --out $6
}

PROBE_WORDS=2000   # ~2500 tokens technical prompt
DECODE_WORDS=2000
DECODE_TOK=512

# -------- Scenario 1: qwen ALONE --------
exec 9>$LOCK
flock -w 600 9 || { echo "FATAL: cannot get lane lock"; exit 1; }

$B/cleanup_lanes.sh
rm -f $R/dualqfn-*.jsonl
mem_snapshot "state0-idle"

echo "=== S1: boot qwen alone ==="
nohup $B/serve_qwen3827b.sh > $LOGQ 2>&1 &
wait_healthy $QPORT 900 $LOGQ || exit 1
sleep 5
mem_snapshot "state1-qwen-solo-up"

for i in 1 2 3; do probe $QPORT $QMODEL $DECODE_WORDS $DECODE_TOK "S1-qwen-alone-decode$i" $R/dualqfn-qwen.jsonl; done
probe $QPORT $QMODEL $PROBE_WORDS 64 "S1-qwen-alone-prefill2.5k" $R/dualqfn-qwen.jsonl
mem_snapshot "state1-qwen-solo-after-probes"

# -------- Scenario 2: boot flashnext alongside --------
echo "=== S2: boot flashnext co-resident ==="
nohup /root/strata/run-iq3_s.sh > $LOGF 2>&1 &
wait_healthy $FPORT 1800 $LOGF
FBUILD=$?   # 0 if healthy
mem_snapshot "state2-dual-up"

if [ $FBUILD -eq 0 ]; then
  # residency cost: probe qwen while flashnext idle
  for i in 1 2 3; do probe $QPORT $QMODEL $DECODE_WORDS $DECODE_TOK "S2-qwen-res-fn-idle-decode$i" $R/dualqfn-qwen.jsonl; done
  probe $QPORT $QMODEL $PROBE_WORDS 64 "S2-qwen-res-fn-idle-prefill2.5k" $R/dualqfn-qwen.jsonl
  # residency cost: probe flashnext while qwen idle
  for i in 1 2 3; do probe $FPORT $FMODEL $DECODE_WORDS $DECODE_TOK "S2-fn-res-qwen-idle-decode$i" $R/dualqfn-fn.jsonl; done
  probe $FPORT $FMODEL $PROBE_WORDS 64 "S2-fn-res-qwen-idle-prefill2.5k" $R/dualqfn-fn.jsonl

  # -------- Scenario 3: concurrent load, 3 rounds --------
  echo "=== S3: concurrent load ==="
  for i in 1 2 3; do
    probe $QPORT $QMODEL $DECODE_WORDS $DECODE_TOK "S3-concurrent-r$i-qwen" $R/dualqfn-concurrent.jsonl &
    P1=$!
    probe $FPORT $FMODEL $DECODE_WORDS $DECODE_TOK "S3-concurrent-r$i-fn" $R/dualqfn-concurrent.jsonl &
    P2=$!
    wait $P1 $P2
    echo "round $i done"
  done
  mem_snapshot "state3-dual-after-concurrent"
fi

# -------- Scenario 4: flashnext ALONE --------
echo "=== S4: stop qwen, flashnext alone ==="
pkill -f "build-hip/luce_server"; sleep 10
pkill -9 -f "build-hip/luce_server" 2>/dev/null; sleep 3
mem_snapshot "state4-fn-only-qwen-killed"

for i in 1 2 3; do probe $FPORT $FMODEL $DECODE_WORDS $DECODE_TOK "S4-fn-alone-decode$i" $R/dualqfn-fn.jsonl; done
probe $FPORT $FMODEL $PROBE_WORDS 64 "S4-fn-alone-prefill2.5k" $R/dualqfn-fn.jsonl
mem_snapshot "state4-fn-solo-after-probes"

flock -u 9
echo "=== ALL DONE ==="
