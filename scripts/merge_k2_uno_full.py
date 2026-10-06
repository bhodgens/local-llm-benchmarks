#!/usr/bin/env python3
"""
Merge K2-Horizon-7B-Uno LoRA into base (BF16) -> save -> convert to GGUF Q4_K_M.
Path 2 per QUEUE.md decision 2026-10-04. Adapter: r=128, alpha=8192, all 7 projections.
"""
import subprocess, os, sys

BASE_DIR = "/var/tmp/llms/bench-0918/k2"          # 36 bf16 shards + trust_remote_code files
ADAPTER_DIR = "/var/tmp/llms/bench-0918/k2-adapter-fixed"  # prefixed keys (base_model.model.*); raw file had non-PEFT keys
OUTPUT_DIR = "/var/tmp/llms/bench-0918/k2-merged"
GGUF_OUTPUT = "/var/tmp/llms/bench-0918/k2-uno-q4km/K2-Horizon-7B-Uno-Q4_K_M.gguf"
CONVERT_SCRIPT = "/home/caimlas/git/llama.cpp-k2horizon/convert_hf_to_gguf.py"
PY = "/home/caimlas/bench-venv/bin/python3"

merge_script = """
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "{BASE_DIR}"
adapter_path = "{ADAPTER_DIR}"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)

print("Loading base model (BF16, CPU)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("Loading adapter (r=128 alpha=8192)...", flush=True)
model = PeftModel.from_pretrained(model, adapter_path)

print("Merging...", flush=True)
model = model.merge_and_unload()

print("Verifying merge actually changed weights (delta vs base)...", flush=True)
import safetensors.torch as st
import glob as _g; ref = st.load_file(sorted(_g.glob(os.path.join(base_id, "*00001-of-00036.safetensors")))[0])
probe = [k for k in ref if "layers.0.self_attn.q_proj.weight" in k][0]
merged_q = dict(model.named_parameters())[ "model.layers.0.self_attn.q_proj.weight"].detach()
delta = (merged_q.cpu().float() - ref[probe].cpu().float()).abs().max().item()
print("max |merged - base| on L0 q_proj = " + str(delta), flush=True)
assert delta > 1e-3, "MERGE WAS A NO-OP - adapter did not load!"

print("Saving merged model (direct safetensors; transformers 5.13 save_pretrained+peft is broken)...", flush=True)
os.makedirs("{OUTPUT_DIR}", exist_ok=True)
from safetensors.torch import save_file
sd = {{k: v.contiguous().to(torch.bfloat16) if v.dtype.is_floating_point else v for k, v in model.state_dict().items()}}
# drop any lingering lora keys (should be none after merge_and_unload)
sd = {{k: v for k, v in sd.items() if "lora_" not in k}}
save_file(sd, "{OUTPUT_DIR}/model.safetensors", metadata={{"format": "pt"}})
# copy config/tokenizer from base dir
import shutil
for f in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "configuration_k2_horizon.py", "modeling_k2_horizon.py", "chat_template.jinja"]:
    src_f = os.path.join(base_id, f)
    if os.path.exists(src_f):
        shutil.copy(src_f, os.path.join("{OUTPUT_DIR}", f))
print("Merge complete!", flush=True)
""".format(BASE_DIR=BASE_DIR, ADAPTER_DIR=ADAPTER_DIR, OUTPUT_DIR=OUTPUT_DIR)

os.makedirs("/var/tmp/llms/bench-0918", exist_ok=True)
sp = "/var/tmp/llms/bench-0918/k2_merge_inline.py"
with open(sp, "w") as f:
    f.write(merge_script)

print("Running merge (14.4GB bf16, CPU; expect 20-40 min)...", flush=True)
r = subprocess.run([PY, sp])
if r.returncode != 0:
    print("MERGE FAILED"); sys.exit(1)

print("Converting to GGUF Q4_K_M via k2horizon fork converter...", flush=True)
os.makedirs(os.path.dirname(GGUF_OUTPUT), exist_ok=True)
r = subprocess.run([PY, CONVERT_SCRIPT, OUTPUT_DIR, "--outtype", "bf16", "--outfile", GGUF_OUTPUT])
if r.returncode != 0:
    print("CONVERT FAILED"); sys.exit(1)

print("DONE:", GGUF_OUTPUT)
