#!/usr/bin/env python3
"""Fix the registry insertion: entries landed after the closing `]`; move them inside."""
import sys

PATH = "/root/LiveCodeBench/lcb_runner/lm_styles.py"
s = open(PATH).read()

if "    LanguageModel(\n        \"local/borg" not in s:
    print("nothing to fix")
    sys.exit(0)

# extract the block of local entries (from first local entry to the anchor)
first = s.index('    LanguageModel(\n        "local/borg')
anchor = "LanguageModelStore: dict[str, LanguageModel] = {"
anchor_pos = s.index(anchor)
block = s[first:anchor_pos]
s = s[:first] + s[anchor_pos:]
# insert block before the closing `]` of LanguageModelList (the last `]\n` before anchor)
list_close = s.rindex("\n]\n", 0, s.index(anchor))
s = s[:list_close+1] + block + s[list_close+1:]
open(PATH, "w").write(s)
print("fixed: entries moved inside LanguageModelList")
