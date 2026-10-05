#!/bin/bash
# Measure download rates for the two in-flight HF downloads on borg.
KCOL=/root/models/kolibri/.cache/huggingface/download
KSRC=/root/models/glm53-src/.cache/huggingface/download
a1=$(cat $KCOL/*.incomplete 2>/dev/null | wc -c)
b1=$(cat $KSRC/*.incomplete 2>/dev/null | wc -c)
# wc -c on a file being written is unreliable via cat; use stat instead
a1=$(stat -c %s $KCOL/*.incomplete 2>/dev/null | head -1)
b1=$(stat -c %s $KSRC/*.incomplete 2>/dev/null | head -1)
sleep 75
a2=$(stat -c %s $KCOL/*.incomplete 2>/dev/null | head -1)
b2=$(stat -c %s $KSRC/*.incomplete 2>/dev/null | head -1)
echo "kolibri: ${a1:-0} -> ${a2:-0}"
echo "shard:   ${b1:-0} -> ${b2:-0}"
python3 - "$a1" "$a2" "$b1" "$b2" <<'PY'
import sys
a1, a2, b1, b2 = (int(x) if x else 0 for x in sys.argv[1:5])
KA, SB = 51_000_000_000, 5_368_709_120     # expected total sizes
for name, x1, x2, total in (("kolibri", a1, a2, KA), ("shard", b1, b2, SB)):
    rate = (x2 - x1) / 75
    if rate <= 0:
        print(f"{name}: STALLED at {x2/1e9:.2f} GB")
    else:
        print(f"{name}: {rate/1e6:.1f} MB/s, {x2/1e9:.2f} GB done, ~{(total-x2)/rate/60:.0f} min left")
PY
