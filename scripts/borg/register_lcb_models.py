#!/usr/bin/env python3
"""Register the borg lane models in LiveCodeBench's LanguageModelStore.

Upstream LCB only knows its built-in model list; the CUDA-host repo used a fork
with local/* entries. We append local entries at import time instead of forking:
this module is imported via sitecustomize-style PYTHONSTARTUP injection inside
run_borg_coding_eval.py's LCB subprocess env (see LOCAL_LCB_MODELS below).

Simpler and fork-free: run_borg_coding_eval.py writes this file's REGISTRY dict
into lcb_runner/lm_styles.py via a patch step run once per clone.
"""
import re

LCB_STYLES = "/root/LiveCodeBench/lcb_runner/lm_styles.py"

# model_name -> (short_name, LMStyle)  -- OpenAIChat works for all our chat servers
BORG_MODELS = {
    "local/borg-qwen38-27b": "qwen38-27b",
    "local/borg-qwen38-27b-vision": "qwen38-27b-vision",
    "local/borg-laguna-xs21": "laguna-xs21",
    "local/borg-laguna-s21": "laguna-s21",
    "local/borg-flashnext": "flashnext",
    "local/borg-swift": "swift",
    "local/borg-coder": "coder",
    "local/borg-glm53-flash": "glm53-flash",
    "local/borg-kolibri-1": "kolibri-1",
}


def main():
    src = open(LCB_STYLES).read()
    if "local/borg-qwen38-27b" in src:
        print("already registered")
        return
    entries = []
    for name, short in BORG_MODELS.items():
        entries.append(
            '    LanguageModel(\n'
            f'        "{name}",\n'
            f'        "{short}",\n'
            '        LMStyle.OpenAIChat,\n'
            '        datetime(2026, 1, 1),\n'
            '        "https://borg.local",\n'
            '    ),\n'
        )
    block = "".join(entries)
    anchor = "LanguageModelStore: dict[str, LanguageModel] = {"
    assert anchor in src, "anchor not found"
    src = src.replace(anchor, block + "\n" + anchor)
    open(LCB_STYLES, "w").write(src)
    print("registered", len(BORG_MODELS), "local models")


if __name__ == "__main__":
    main()
