# Benchmark Queue

Models queued for benchmarking. Not yet downloaded or run.

---

## 1. MiniMax-H3 (realrebelai/MiniMax-H3_GGUFs)

- Source: https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/tree/main
- Target GPU: V100 (CUDA0, 32GB)
- Requested quant: Q4_K_M

### Available Q4_K_M files

| File | Size | Type |
|------|------|------|
| MiniMax-H3-FL2VA-Q4_K_M.gguf | 19.9 GB | Text-to-Video (FL2VA = Flow matching Language-to-Video-Audio) |
| qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf | 14.6 GB | Vision-Language (Qwen3-VL-32B + MiniMax-H3) |

### BLOCKER: Not a text-only LLM

This repo is a multimodal/video generation model. Tagged: text-to-video, minimax, comfyui.
The FL2VA variant is a video generation model requiring ComfyUI + separate VAE files
(https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae). It cannot run in llama.cpp
as a chat/coding model and is not benchmarkable with HumanEval/LCB/Aider.

The qwen3vl-32B variant is a vision-language model (may support text generation via
llama.cpp if the architecture is supported). This is the more viable candidate for the
coding benchmark suite, but requires --mmproj and vision support.

### Download URL (qwen3vl variant - more viable for text benchmarks)
```
https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf
```
Dest: /home/files/llms/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf

### Download URL (FL2VA video variant)
```
https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/MiniMax-H3-FL2VA-Q4_K_M.gguf
```
Dest: /home/files/llms/MiniMax-H3-FL2VA-Q4_K_M.gguf

### Needs resolution before benchmarking
1. Which Q4_K_M file? (FL2VA video vs qwen3vl text+vision)
2. llama.cpp architecture support for MiniMax-H3 / qwen3vl-32B unverified
3. If FL2VA: cannot use coding benchmark suite, needs ComfyUI workflow instead
4. VAE files needed separately for FL2VA variant

---

## 2. Qwen3.6-35B-A3B-Escha-W2 (EschaLabs)

- Source: https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2
- Target GPU: V100 (CUDA0, 32GB)
- Quant: 2-bit eschamoe (NOT Q4_K_M, NOT GGUF)

### Model details

| Property | Value |
|----------|-------|
| Base | Qwen3.6-35B-A3B (MoE, 256 experts) |
| Quantization | 2-bit eschamoe (mixed 2/3-bit per projection), int8 dense |
| Format | Safetensors (3 shards, 12.3 GB total) |
| Min GPU | 16 GB VRAM, NVIDIA Ampere (sm_80) required |
| Runtime | escha (SGLang engine or ZML engine) -- NOT llama.cpp |
| API | OpenAI-compatible /v1 on port 30000 |
| License | Apache-2.0 |

### BLOCKER: V100 is sm_70 (Volta), below sm_80 (Ampere) minimum

The escha runtime requires Ampere or newer. V100 compute capability 7.0 < 8.0 minimum.
Triton (used by SGLang engine) does not support sm_70. The model will not run on V100.

### BLOCKER: Not GGUF, requires escha runtime

This is safetensors with custom eschamoe quantization. Cannot be loaded by llama.cpp.
Requires installing escha runtime (separate repo: EschaLabs/escha-runtime-qwen3moe).
Needs Python 3.12, torch==2.9.x, CUDA 12.8+.

### Download
```
hf download EschaLabs/Qwen3.6-35B-A3B-Escha-W2 --local-dir /home/files/llms/escha-w2
```
(3 safetensors shards: 5.37 + 5.31 + 1.62 GB = 12.3 GB)

### Needs resolution before benchmarking
1. V100 (sm_70) does not meet sm_80 minimum -- consider 3060 (sm_86) instead?
2. Requires escha/SGLang runtime install (not llama.cpp)
3. torch==2.9.x + CUDA 12.8 dependency chain may conflict with existing setup
4. Not Q4_K_M -- this is a 2-bit quant, much more aggressive

---

## 3. BTL-4 (badtheorylabs/BTL-4) -- ACTIVE

- Source: https://huggingface.co/badtheorylabs/BTL-4
- GGUF: https://huggingface.co/bartowski/badtheorylabs_BTL-4-GGUF
- Base: Ornith-1.0-35B (qwen3_5_moe arch, MoE)
- Tags: agentic, tool-use, code, reasoning, image-text-to-text
- License: Apache-2.0
- Claimed: LCB v6 66.1%, BFCL v4 73.5%, SWE-bench Verified 78.4%
- Generation settings: temp=1.0, top_p=0.95, ctx 262144 native
- Reasoning: deepseek format, must strip reasoning from old turns

### Phase 1: IQ2_XXS on 3060 (12GB, sm_86) -- COMPLETE
- File: badtheorylabs_BTL-4-IQ2_XXS.gguf (9.78 GB)
- Full GPU offload, --reasoning off, --jinja, --reasoning-format deepseek
- Harness: tok/s + LCB (75 problems) + tau2 (airline, 15 tasks)
- Script: scripts/run_btl4_iq2xxs.py
- Results: tok/s=72.9, LCB=50.7%, tau2=0.38 (5/15 passed), VRAM=9905 MiB
- User sim: Qwythos-27B on V100 (Gemma ExLlamaV3 needs Ampere+, cannot use V100)

### Phase 2: Q4_K_M on V100 (32GB, sm_70) -- QUEUED
- File: badtheorylabs_BTL-4-Q4_K_M.gguf (21.4 GB)
- Full GPU offload, same harness
- Script: TBD (run_btl4_q4km.py)

---

## Disk Space

Current: 366 GB free on /home (1.5 TB used of 1.9 TB)
Required: ~10 GB (IQ2_XXS) + ~21 GB (Q4_K_M) = ~31 GB total
Status: Sufficient
