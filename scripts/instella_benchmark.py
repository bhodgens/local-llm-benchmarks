#!/usr/bin/env python
"""
Instella-MoE-16B INT4 expert quantization benchmark for RTX 3060 (12GB).

The checkpoint stores individual per-expert weights:
    model.layers.N.mlp.experts.E.gate_proj.weight  [1408, 2048]
    model.layers.N.mlp.experts.E.up_proj.weight     [1408, 2048]
    model.layers.N.mlp.experts.E.down_proj.weight   [2048, 1408]

But transformers 5.13's DeepseekV3Experts stores them as batched fused parameters:
    gate_up_proj  [64, 2*1408, 2048]  = [64, 2816, 2048]
    down_proj     [64, 2048, 1408]

This script fuses the checkpoint weights, loads everything into the model,
then quantizes the expert parameters to INT4 (NF4) using bitsandbytes Params4bit
so the full model fits in 12 GB VRAM.

Run with:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
    /home/caimlas/bench-venv/bin/python /tmp/instella_benchmark.py
"""

import gc
import json
import os
import re
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = Path("/home/files/llms/gguf/amd-instella-sft")
DEVICE = torch.device("cuda:0")  # After CUDA_VISIBLE_DEVICES=1, this is the 3060
DTYPE = torch.float16
# Expert quantization
QUANT_TYPE = "nf4"   # nf4 gives better quality than fp4
BLOCKSIZE = 64


# ---------------------------------------------------------------------------
# Step 0: Make the remote-code package importable (relative imports)
# ---------------------------------------------------------------------------
def setup_pkg():
    """Register the model directory as a Python package so the relative import
    `from .configuration_instella_moe import ...` works."""
    pkg = types.ModuleType("instella_moe_pkg")
    pkg.__path__ = [str(MODEL_PATH)]
    sys.modules["instella_moe_pkg"] = pkg


setup_pkg()


# ---------------------------------------------------------------------------
# Step 1: Load the safetensors index and build a weight-loading plan
# ---------------------------------------------------------------------------
def load_safetensors_index():
    """Return {shard_filename: [tensor_name, ...]} from the index."""
    idx_path = MODEL_PATH / "model.safetensors.index.json"
    with open(idx_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]  # tensor_name -> shard_file
    # Group tensor names by shard
    shards = {}
    for tensor_name, shard_file in weight_map.items():
        shards.setdefault(shard_file, []).append(tensor_name)
    return shards


def load_shard(shard_file, keys_to_load=None):
    """Load tensors from a single safetensors shard (lazy — only requested keys)."""
    from safetensors.torch import safe_open

    shard_path = MODEL_PATH / shard_file
    tensors = {}
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        all_keys = set(f.keys())
        target_keys = keys_to_load if keys_to_load is not None else all_keys
        for key in target_keys:
            if key in all_keys:
                tensors[key] = f.get_tensor(key)
    return tensors


# ---------------------------------------------------------------------------
# Step 2: Fuse expert weights from individual-per-expert → batched format
# ---------------------------------------------------------------------------
def fuse_expert_weights(all_shards_tensors, num_layers=27, first_k_dense_replace=1):
    """
    For each MoE layer (layers 1..26), fuse:
        experts.E.gate_proj.weight  [1408, 2048]
        experts.E.up_proj.weight    [1408, 2048]
        experts.E.down_proj.weight  [2048, 1408]
    into:
        experts.gate_up_proj  [64, 2816, 2048]
        experts.down_proj     [64, 2048, 1408]

    Returns a dict of fused tensors keyed by their model state_dict names.
    """
    fused = {}
    # Regex to parse expert weight keys
    pat = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight")

    # First, collect per-layer per-expert tensors
    # structure: layer_experts[layer_idx]["gate_proj"][expert_idx] = tensor
    layer_data = {}
    for tensor_name, tensor in all_shards_tensors.items():
        m = pat.match(tensor_name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        expert_idx = int(m.group(2))
        proj_type = m.group(3)
        if layer_idx < first_k_dense_replace:
            continue  # dense layers don't have fused experts
        layer_data.setdefault(layer_idx, {}).setdefault(expert_idx, {})[proj_type] = tensor

    print(f"  Fusing experts for {len(layer_data)} MoE layers...")

    for layer_idx in sorted(layer_data.keys()):
        experts = layer_data[layer_idx]
        n_experts = len(experts)
        # Verify we have all 64 experts
        expected = 64
        if n_experts != expected:
            print(f"  WARNING: layer {layer_idx} has {n_experts} experts, expected {expected}")

        # Stack gate_proj and up_proj for each expert, then concat to gate_up_proj
        gate_list = []
        up_list = []
        down_list = []
        for e_idx in range(n_experts):
            e = experts[e_idx]
            gate_list.append(e["gate_proj"].to(DTYPE))
            up_list.append(e["up_proj"].to(DTYPE))
            down_list.append(e["down_proj"].to(DTYPE))

        # gate_proj and up_proj are [out=1408, in=2048]
        # gate_up_proj[E] = cat([gate_proj[E], up_proj[E]], dim=0) -> [2816, 2048]
        gate_up = torch.stack([torch.cat([g, u], dim=0) for g, u in zip(gate_list, up_list)], dim=0)
        # down_proj is [out=2048, in=1408]
        down = torch.stack(down_list, dim=0)

        fused[f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"] = gate_up
        fused[f"model.layers.{layer_idx}.mlp.experts.down_proj"] = down

        # Free intermediate lists
        del gate_list, up_list, down_list

    return fused


# ---------------------------------------------------------------------------
# Step 3: Load all weights — experts (fused) + non-experts (from checkpoint)
# ---------------------------------------------------------------------------
def build_full_state_dict():
    """
    Load all weights needed for load_state_dict:
    - Non-expert weights loaded directly from checkpoint (as-is)
    - Expert weights fused into batched gate_up_proj / down_proj
    Returns a state_dict dict.
    """
    print("\n=== Loading weights from checkpoint ===")
    shards = load_safetensors_index()
    print(f"  {len(shards)} shards, {sum(len(v) for v in shards.values())} tensors total")

    # Expert regex — keys we need to fuse (skip from direct loading)
    expert_pat = re.compile(r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight")

    non_expert_sd = {}
    expert_raw = {}

    for shard_file in sorted(shards.keys()):
        print(f"  Loading {shard_file}...", end="", flush=True)
        t0 = time.time()
        # Load ALL keys from this shard at once (faster than per-key)
        shard_tensors = load_shard(shard_file)
        for key, tensor in shard_tensors.items():
            if expert_pat.match(key):
                expert_raw[key] = tensor
            else:
                non_expert_sd[key] = tensor.to(DTYPE) if tensor.dtype == torch.bfloat16 else tensor
        del shard_tensors
        gc.collect()
        print(f" {time.time()-t0:.1f}s  ({len(non_expert_sd)} non-expert, {len(expert_raw)} expert tensors so far)")

    print(f"\n  Non-expert tensors: {len(non_expert_sd)}")
    print(f"  Expert tensors to fuse: {len(expert_raw)}")

    # Fuse expert weights
    print("\n=== Fusing expert weights ===")
    fused_experts = fuse_expert_weights(expert_raw)

    # Combine into final state dict
    full_sd = dict(non_expert_sd)
    full_sd.update(fused_experts)

    del non_expert_sd, expert_raw, fused_experts
    gc.collect()

    return full_sd


# ---------------------------------------------------------------------------
# Step 4: Quantize expert params to INT4 (NF4) using bitsandbytes Params4bit
# ---------------------------------------------------------------------------
def quantize_expert_params(model):
    """
    Replace each DeepseekV3Experts gate_up_proj and down_proj nn.Parameter
    with a bitsandbytes Params4bit (NF4 quantized) version.

    DeepseekV3Experts.forward does:
        nn.functional.linear(current_state, self.gate_up_proj[expert_idx])
        nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])

    F.linear(x, W) computes x @ W.T. So gate_up_proj[expert_idx] has shape
    [out, in] = [2816, 2048], which is what F.linear expects as a weight.

    We need to dequantize per-expert-slice during forward. To keep the
    DeepseekV3Experts.forward untouched, we replace the raw Parameter with
    a custom module attribute that returns dequantized slices on indexing.

    Strategy: Quantize the full [64, 2816, 2048] tensor, store as Params4bit,
    then monkey-patch the forward to dequantize on the fly.
    """
    import bitsandbytes.functional as bnb_f

    print("\n=== Quantizing expert weights to INT4 (NF4) ===")

    # Find all DeepseekV3Experts modules
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Experts

    expert_modules = []
    for name, module in model.named_modules():
        if isinstance(module, DeepseekV3Experts):
            expert_modules.append((name, module))

    print(f"  Found {len(expert_modules)} DeepseekV3Experts modules")

    total_params = 0
    total_bytes_before = 0
    total_bytes_after = 0

    for name, experts_mod in expert_modules:
        # Quantize gate_up_proj [num_experts, 2816, 2048]
        gate_up = experts_mod.gate_up_proj.data  # [64, 2816, 2048] fp16
        down = experts_mod.down_proj.data        # [64, 2048, 1408] fp16

        bytes_before = (gate_up.numel() + down.numel()) * 2  # fp16 = 2 bytes

        # Flatten to 2D for quantization: [num_experts * out_features, in_features]
        num_experts = gate_up.shape[0]
        gate_up_2d = gate_up.reshape(-1, gate_up.shape[-1])  # [64*2816, 2048]
        down_2d = down.reshape(-1, down.shape[-1])           # [64*2048, 1408]

        # Quantize on GPU for speed (bnb CUDA kernels)
        gate_up_2d_gpu = gate_up_2d.to(DEVICE)
        down_2d_gpu = down_2d.to(DEVICE)

        gu_quant, gu_state = bnb_f.quantize_4bit(
            gate_up_2d_gpu, blocksize=BLOCKSIZE, quant_type=QUANT_TYPE,
            compress_statistics=True,
        )
        dp_quant, dp_state = bnb_f.quantize_4bit(
            down_2d_gpu, blocksize=BLOCKSIZE, quant_type=QUANT_TYPE,
            compress_statistics=True,
        )

        bytes_after = gu_quant.numel() * gu_quant.element_size() + dp_quant.numel() * dp_quant.element_size()
        # quant_state overhead is small (~2% of quantized size)

        total_params += gate_up.numel() + down.numel()
        total_bytes_before += bytes_before
        total_bytes_after += bytes_after

        # Replace the module's Parameters with a quantized holder
        # We store the quantized data + state, and patch forward to dequantize
        experts_mod._quant_gate_up = gu_quant    # [N*2816/2, 2048] uint8
        experts_mod._quant_state_gu = gu_state
        experts_mod._quant_down = dp_quant
        experts_mod._quant_state_dp = dp_state
        experts_mod._quant_shape_gu = gate_up.shape  # (64, 2816, 2048)
        experts_mod._quant_shape_dp = down.shape     # (64, 2048, 1408)

        # Remove the original fp16 parameters to free memory
        del experts_mod.gate_up_proj
        del experts_mod.down_proj

        # Free GPU temp tensors
        del gate_up_2d_gpu, down_2d_gpu, gate_up_2d, down_2d

    # Monkey-patch the forward to dequantize expert slices on the fly
    original_forward = DeepseekV3Experts.forward

    def quantized_forward(self, hidden_states, top_k_index, top_k_weights):
        """Dequantize only the experts that are actually hit, then call
        the original forward logic using the dequantized parameters."""
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # Dequantize just this expert's weights
            # gate_up: stored as flat [N*out, in], quantized; need slice [expert_idx]
            gu_flat = bnb_f.dequantize_4bit(
                self._quant_gate_up, self._quant_state_gu, blocksize=BLOCKSIZE, quant_type=QUANT_TYPE
            )
            gu_shape = self._quant_shape_gu  # (64, 2816, 2048)
            out_per_expert = gu_shape[1]     # 2816
            gate_up_e = gu_flat[expert_idx * out_per_expert : (expert_idx + 1) * out_per_expert]

            dp_flat = bnb_f.dequantize_4bit(
                self._quant_down, self._quant_state_dp, blocksize=BLOCKSIZE, quant_type=QUANT_TYPE
            )
            dp_shape = self._quant_shape_dp  # (64, 2048, 1408)
            dp_out_per_expert = dp_shape[1]  # 2048
            down_e = dp_flat[expert_idx * dp_out_per_expert : (expert_idx + 1) * dp_out_per_expert]

            gate, up = F.linear(current_state, gate_up_e).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, down_e)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

            # Free dequantized intermediates
            del gu_flat, dp_flat, gate_up_e, down_e

        return final_hidden_states

    DeepseekV3Experts.forward = quantized_forward

    reduction = (1 - total_bytes_after / total_bytes_before) * 100
    print(f"  Expert params: {total_params/1e9:.2f}B")
    print(f"  Before quant: {total_bytes_before/1e9:.2f} GB (fp16)")
    print(f"  After quant:  {total_bytes_after/1e9:.2f} GB (INT4)")
    print(f"  Reduction: {reduction:.1f}%")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Instella-MoE-16B INT4 Expert Quantization Benchmark")
    print("=" * 70)
    print(f"Model path: {MODEL_PATH}")
    print(f"Device: {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    print(f"Dtype: {DTYPE}")

    # ------------------------------------------------------------------
    # Step 1-2: Build the full state dict (fused experts + non-experts)
    # ------------------------------------------------------------------
    full_sd = build_full_state_dict()
    print(f"\nFull state dict: {len(full_sd)} tensors")

    # ------------------------------------------------------------------
    # Step 3: Create model from config (empty weights) and load state dict
    # ------------------------------------------------------------------
    print("\n=== Creating model from config ===")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

    from accelerate import init_empty_weights
    from instella_moe_pkg.modeling_instella_moe import InstellaMoEForCausalLM

    print("  Building model (empty weights)...", end="", flush=True)
    t0 = time.time()
    with init_empty_weights():
        model = InstellaMoEForCausalLM(config)
    print(f" {time.time()-t0:.1f}s")

    print("  Loading state dict (CPU)...", end="", flush=True)
    t0 = time.time()
    # Use assign=True since all tensors are already materialized on CPU
    missing, unexpected = model.load_state_dict(full_sd, strict=False, assign=True)
    print(f" {time.time()-t0:.1f}s")
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    {k}")

    del full_sd
    gc.collect()

    # ------------------------------------------------------------------
    # Step 4: Quantize expert weights to INT4 (before GPU move)
    # ------------------------------------------------------------------
    # Quantize expert weights while model is on CPU. The quantize function
    # moves each expert tensor to GPU individually, quantizes, stores result.
    print("\n=== Quantizing expert weights to INT4 (pre-GPU-move) ===")
    model = quantize_expert_params(model)
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 5: Move quantized model to GPU
    # ------------------------------------------------------------------
    print("\n=== Moving quantized model to GPU ===")
    model = model.to(DEVICE)
    print(f"  Model on {next(model.parameters()).device}")
    print(f"  GPU mem after full load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    model.eval()

    # ------------------------------------------------------------------
    # Step 6: Generation test
    # ------------------------------------------------------------------
    print("\n=== Generation test ===")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    prompt = "Write a Python function to check if a number is prime."
    print(f"  Prompt: {prompt}")

    messages = [{"role": "user", "content": prompt}]
    try:
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not isinstance(input_ids, torch.Tensor):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(DEVICE)
    except Exception:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    print(f"  Input tokens: {input_ids.shape[1]}")
    print(f"  Input: {tokenizer.decode(input_ids[0])}")
    print()

    # Generation with timing
    gen_kwargs = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.95,
        "pad_token_id": tokenizer.eos_token_id,
    }

    print("  Generating...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)
    elapsed = time.time() - t0

    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    response = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

    print(f"\n=== RESULTS ===")
    print(f"  Generated {new_tokens} tokens in {elapsed:.1f}s")
    print(f"  Throughput: {new_tokens / elapsed:.2f} tok/s")
    print(f"  GPU peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"\n=== OUTPUT ===")
    print(response)
    print("=" * 70)

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = main()
