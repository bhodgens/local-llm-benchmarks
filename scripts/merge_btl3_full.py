#!/usr/bin/env python3
"""
Merge BTL-3 LoRA adapter with Qwen3.6-27B base model and convert to GGUF.
"""
import subprocess, os, sys, time

BASE_DIR = "/home/files/llms/qwen3.6-27b-base"
ADAPTER_DIR = "/home/files/llms/btl-3-adapter"
OUTPUT_DIR = "/home/files/llms/btl-3-merged"
GGUF_OUTPUT = "/home/files/llms/BTL-3-merged-Q4_K_M.gguf"
CONVERT_SCRIPT = "/home/caimlas/git/llama.cpp/convert_hf_to_gguf.py"

# Wait for base model to be fully downloaded
expected_shards = 15
print("Checking base model download...")
while True:
    shards = len([f for f in os.listdir(BASE_DIR) if f.endswith('.safetensors')])
    print(f"  {shards}/{expected_shards} shards downloaded")
    if shards >= expected_shards:
        break
    time.sleep(60)

print(f"\nBase model complete. Starting merge...")

# Use a Python script to merge
merge_script = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

base_id = "{BASE_DIR}"
adapter_path = "{ADAPTER_DIR}"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_id)

print("Loading base model (BF16)...")
model = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

print("Merging...")
model = model.merge_and_unload()

print("Saving merged model...")
os.makedirs("{OUTPUT_DIR}", exist_ok=True)
model.save_pretrained("{OUTPUT_DIR}", safe_serialization=True)
tokenizer.save_pretrained("{OUTPUT_DIR}")

print("Merge complete!")
"""

with open("/tmp/merge_btl3.py", "w") as f:
    f.write(merge_script)

print("Running merge (this will take several minutes)...")
result = subprocess.run(
    ["/home/caimlas/bench-venv/bin/python3", "/tmp/merge_btl3.py"],
    capture_output=True, text=True, timeout=3600
)
print(result.stdout[-2000:])
if result.returncode != 0:
    print(f"ERROR: {result.stderr[-2000:]}")
    sys.exit(1)

# Convert to GGUF
print(f"\nConverting to GGUF Q4_K_M...")
result = subprocess.run(
    ["/home/caimlas/bench-venv/bin/python3", CONVERT_SCRIPT,
     OUTPUT_DIR, "--outtype", "q4_k_m", "--outfile", GGUF_OUTPUT],
    capture_output=True, text=True, timeout=3600
)
print(result.stdout[-2000:])
if result.returncode != 0:
    print(f"ERROR: {result.stderr[-2000:]}")
    sys.exit(1)

sz = os.path.getsize(GGUF_OUTPUT)
print(f"\nDone! {GGUF_OUTPUT}: {sz/1048576:.0f}MB")
