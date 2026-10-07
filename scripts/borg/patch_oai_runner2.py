#!/usr/bin/env python3
"""Extend the borg branch in oai_runner.py with the server-model-id mapping (v2)."""
import sys

PATH = "/root/LiveCodeBench/lcb_runner/runner/oai_runner.py"
s = open(PATH).read()

if "BORG_SERVER_MODEL_IDS" in s:
    print("mapping already present")
    sys.exit(0)

OLD = '''                "timeout": args.openai_timeout,
                "reasoning_effort": "none",
            }'''
NEW = '''                "timeout": args.openai_timeout,
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
                model.model_name, model.model_name)'''
assert OLD in s, "borg branch anchor not found"
s = s.replace(OLD, NEW, 1)
open(PATH, "w").write(s)
print("server-model-id mapping added")
