#!/bin/bash
# Wait for the Q4_K_M run (run_whittle.py processes) to exit, then kill the LFM sim on 8081
while pgrep -f "run_whittle.py Q4_K_M" > /dev/null 2>&1; do sleep 30; done
sleep 5
pkill -f "port 8081" 2>/dev/null
kill 391373 2>/dev/null
sleep 3
echo "LFM sim cleared from 8081 $(date)" 
