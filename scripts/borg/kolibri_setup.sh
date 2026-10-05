#!/bin/bash
# Prepare the patched llama.cpp build for Kolibri-1 on borg (gfx1201 + gfx1151).
set -u
LOG=/root/bench/kolibri-setup.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

# 1. stop stale git activity and remove the partial clone
pkill -f 'git clone' 2>/dev/null
pkill -f 'git-remote-https' 2>/dev/null
sleep 2
rm -rf /root/llama.cpp-kolibri

# 2. clean clone
log "cloning llama.cpp (shallow)"
git clone --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp-kolibri >> "$LOG" 2>&1 || { log "CLONE FAILED"; exit 1; }
log "clone done: $(git -C /root/llama.cpp-kolibri log --oneline -1)"

# 3. the patch targets upstream commit 836d571 - fetch that specific commit if HEAD differs
cd /root/llama.cpp-kolibri || exit 1
if ! git rev-parse --verify -q 836d571 > /dev/null; then
  log "fetching commit 836d571"
  git fetch --depth 1 origin 836d571 >> "$LOG" 2>&1 && git checkout -q 836d571 >> "$LOG" 2>&1
fi
log "HEAD now: $(git log --oneline -1)"

# 4. apply the community kolibri1 patch
PATCH=/root/models/kolibri/kolibri1-llama.cpp.patch
if [ -f "$PATCH" ]; then
  git am "$PATCH" >> "$LOG" 2>&1 && log "patch applied" || log "PATCH FAILED (see log; may need --3way)"
else
  log "PATCH MISSING at $PATCH"
fi

# 5. build HIP for both GPUs
log "configuring HIP build (gfx1201;gfx1151)"
cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201\;gfx1151 -DCMAKE_BUILD_TYPE=Release . >> "$LOG" 2>&1
cmake --build build-hip --target llama-server llama-cli -j 24 >> "$LOG" 2>&1
if [ -x build-hip/bin/llama-server ]; then log "BUILD OK: build-hip/bin/llama-server"; else log "BUILD FAILED"; fi
log "setup done"
