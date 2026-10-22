ggml_cuda_init: found 1 CUDA devices (Total VRAM: 11906 MiB):
  Device 0: NVIDIA GeForce RTX 3060, compute capability 8.6, VMM: yes, VRAM: 11906 MiB
| model                          |       size |     params | backend    | ngl | threads | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) |   5.53 GiB |    26.90 B | CUDA       |  99 |       8 |   q4_0 |   q4_0 |   1 |           pp512 |       574.71 ± 10.12 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) |   5.53 GiB |    26.90 B | CUDA       |  99 |       8 |   q4_0 |   q4_0 |   1 |           tg128 |         44.85 ± 0.16 |

build: 73bfe6c (1)
