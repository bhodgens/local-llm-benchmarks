#!/bin/bash
# Fix the Kolibri-1 patch application on borg's llama.cpp checkout and rebuild.
set -u
LOG=/root/bench/kolibri-patchfix.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
cd /root/llama.cpp-kolibri || { log "no checkout"; exit 1; }

log "start; HEAD=$(git log --oneline -1)"
git am --abort 2>/dev/null && log "aborted stale am state" || true

# fetch and check out the exact commit the patch targets
log "fetching 836d571"
git fetch --depth 1 origin 836d571 >> "$LOG" 2>&1 && git checkout -q FETCH_HEAD >> "$LOG" 2>&1
log "HEAD now: $(git log --oneline -1)"

PATCH=/root/models/kolibri/kolibri1-llama.cpp.patch
if git am --3way "$PATCH" >> "$LOG" 2>&1; then
  log "git am --3way OK"
elif git apply --3way --index "$PATCH" >> "$LOG" 2>&1; then
  log "git apply --3way OK (files applied, not committed)"
else
  log "BOTH APPLY METHODS FAILED"; git am --abort 2>/dev/null; exit 1
fi

# sanity: does the source now know the architecture?
if grep -rqi "kolibri1" src/ gguf-py/ 2>/dev/null; then
  log "SOURCE now contains kolibri1"
else
  log "WARNING: source has no kolibri1 reference after patch"
fi

log "rebuilding"
cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201\;gfx1151 -DCMAKE_BUILD_TYPE=Release . >> "$LOG" 2>&1
cmake --build build-hip --target llama-server llama-cli -j 24 >> "$LOG" 2>&1
if strings build-hip/bin/llama-server 2>/dev/null | grep -qi kolibri1; then
  log "BUILD OK and kolibri1 present in binary"
else
  log "BUILD finished but kolibri1 NOT in binary"
fi
log "done"
