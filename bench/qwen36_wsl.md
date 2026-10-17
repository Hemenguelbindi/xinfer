### Qwen3.6-35B-A3B-NVFP4-hf on wsl

- GPU: NVIDIA GeForce RTX 3060, 12288 MiB, 610.88, 3, 16
- OS: Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.42; CPU: Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
- xinfer dddf34b, 2026-10-17 08:45; prompt tokens decoded: 200 max, greedy

| GPU experts | Status | Free VRAM after load | KV tokens | Prefill tok/s | Decode tok/s | Expert hit rate | CUDA graph |
|---|---|---|---|---|---|---|---|
| all (no offload) | failed: c | - | - | - | - | - | no |
| 10% | failed: c | - | - | - | - | - | no |
| 20% | failed: c | - | - | - | - | - | no |
| 35% | failed: c | - | - | - | - | - | no |

Greedy outputs identical across successful runs: **yes** (0 runs)
