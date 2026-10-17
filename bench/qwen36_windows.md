### Qwen3.6-35B-A3B-NVFP4-hf on windows

- GPU: NVIDIA GeForce RTX 3060, 12288 MiB, 610.88, 3, 16
- OS: Windows-11-10.0.26100-SP0; CPU: Intel64 Family 6 Model 79 Stepping 1, GenuineIntel
- xinfer dddf34b, 2026-10-17 08:35; server mode, up to 200 decoded tokens, greedy, max-model-len 8192

| GPU experts | Status | Free VRAM after load | KV tokens | Prefill tok/s | TTFT ms | Decode tok/s | Inter-token p50 / p99 / max ms | CUDA graph |
|---|---|---|---|---|---|---|---|---|
| all (no offload) | failed: KVCache allocation failed: Hybrid mamba budget (0.00 GB) leaves no memory for KV cache. Reduce mamba_fraction. | 0.00 GB | - | - | - | - | - | no |
| 10% | ok | 6.10 GB | 191680 | 14.4 | 2064 | 21.5 | 46.2 / 69.8 / 75.6 | yes |
| 20% | ok | 4.39 GB | 134016 | 14.5 | 2037 | 26.4 | 37.7 / 51.3 / 56.8 | yes |
| 35% | ok | 1.75 GB | 45696 | 40.5 | 793 | 28.6 | 34.8 / 43.7 / 44.2 | yes |

Greedy outputs identical across successful runs: **yes** (3 runs)
