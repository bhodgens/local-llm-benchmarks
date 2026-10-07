#!/usr/bin/env python3
"""Patch oai_runner.py on borg: force reasoning_effort=none for local/borg-* lanes.

Why: with default reasoning on, Strata spends the entire max_tokens budget on
reasoning_content and returns content=None; LCB's extract_code then crashes
('NoneType' object has no attribute 'split') and the whole model's LCB run is lost.
luce_server and Strata both accept reasoning_effort=none and then return normal
content. Verified by direct repro (content=None, finish=length -> content present
with reasoning_effort=none).
"""
import sys

PATH = "/root/LiveCodeBench/lcb_runner/runner/oai_runner.py"
s = open(PATH).read()

if "borg patch" in s:
    print("already patched")
    sys.exit(0)

OLD = '''        elif model.model_style == LMStyle.OpenAIReason:'''
NEW = '''        elif model.model_style == LMStyle.OpenAIChat and model.model_name.startswith("local/borg-"):
            # borg patch: these lanes are chat servers with optional reasoning; without
            # reasoning_effort=none Strata spends the whole budget on reasoning and
            # returns content=None, which crashes extract_code.
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": args.n,
                "timeout": args.openai_timeout,
                "reasoning_effort": "none",
            }
            # borg: the server enforces its own model id, not the LCB store key
            BORG_SERVER_MODEL_IDS = {
                "local/borg-qwen38-27b": "luce",
                "local/borg-qwen38-27b-vision": "luce",
                "local/borg-laguna-xs21": "luce",
                "local/borg-laguna-s21": "luce",
                "local/borg-flashnext": "qwen3.8-flash-next-iq3_s",
                "local/borg-swift": "swift-1.5-iq2_xs",
                "local/borg-coder": "qwen3.8-flash-next-coder-iq1_m",
                "local/borg-glm53-flash": "glm-5.3-exl3",
                "local/borg-kolibri-1": "Kolibri-1-Q4_K_M.gguf",
            }
            self.client_kwargs["model"] = BORG_SERVER_MODEL_IDS.get(
                model.model_name, model.model_name)
        elif model.model_style == LMStyle.OpenAIReason:'''
assert OLD in s, "anchor not found"
s = s.replace(OLD, NEW, 1)
open(PATH, "w").write(s)
print("oai_runner patched")
