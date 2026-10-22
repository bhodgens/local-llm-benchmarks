ggml_cuda_init: found 1 CUDA devices (Total VRAM: 11906 MiB):
  Device 0: NVIDIA GeForce RTX 3060, compute capability 8.6, VMM: yes, VRAM: 11906 MiB
| model                          |       size |     params | backend    | ngl | threads | n_ubatch | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) |   6.70 GiB |    26.90 B | CUDA       |  99 |       8 |      512 |   q8_0 |   q8_0 |   1 |           pp512 |       579.89 ± 13.10 |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) |   6.70 GiB |    26.90 B | CUDA       |  99 |       8 |      512 |   q8_0 |   q8_0 |   1 |          pp4096 |        572.30 ± 0.07 |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) |   6.70 GiB |    26.90 B | CUDA       |  99 |       8 |      512 |   q8_0 |   q8_0 |   1 |           tg128 |         35.57 ± 0.07 |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) |   6.70 GiB |    26.90 B | CUDA       |  99 |       8 |     2048 |   q8_0 |   q8_0 |   1 |           pp512 |       575.47 ± 11.24 |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) |   6.70 GiB |    26.90 B | CUDA       |  99 |       8 |     2048 |   q8_0 |   q8_0 |   1 |          pp4096 |        576.83 ± 0.23 |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) |   6.70 GiB |    26.90 B | CUDA       |  99 |       8 |     2048 |   q8_0 |   q8_0 |   1 |           tg128 |         35.54 ± 0.03 |

build: 73bfe6c (1)
