ggml_cuda_init: found 1 CUDA devices (Total VRAM: 11906 MiB):
  Device 0: NVIDIA GeForce RTX 3060, compute capability 8.6, VMM: yes, VRAM: 11906 MiB
| model                          |       size |     params | backend    | ngl | threads | n_ubatch | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) |   5.53 GiB |    26.90 B | CUDA       |  99 |       8 |      512 |   q8_0 |   q8_0 |   1 |           pp512 |       578.98 ± 13.53 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) |   5.53 GiB |    26.90 B | CUDA       |  99 |       8 |      512 |   q8_0 |   q8_0 |   1 |          pp4096 |        569.98 ± 0.31 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) |   5.53 GiB |    26.90 B | CUDA       |  99 |       8 |      512 |   q8_0 |   q8_0 |   1 |           tg128 |         44.99 ± 0.16 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) |   5.53 GiB |    26.90 B | CUDA       |  99 |       8 |     2048 |   q8_0 |   q8_0 |   1 |           pp512 |        576.09 ± 9.09 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) |   5.53 GiB |    26.90 B | CUDA       |  99 |       8 |     2048 |   q8_0 |   q8_0 |   1 |          pp4096 |        583.81 ± 1.68 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) |   5.53 GiB |    26.90 B | CUDA       |  99 |       8 |     2048 |   q8_0 |   q8_0 |   1 |           tg128 |         44.90 ± 0.13 |

build: 73bfe6c (1)
