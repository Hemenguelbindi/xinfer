### Qwen3.6-35B-A3B-NVFP4-hf on windows

- GPU: NVIDIA GeForce RTX 3060, 12288 MiB, 610.88, 3, 16
- OS: Windows-11-10.0.26100-SP0; CPU: Intel64 Family 6 Model 79 Stepping 1, GenuineIntel
- xinfer dddf34b, 2026-10-17 10:08; server mode, up to 200 decoded tokens, greedy, max-model-len 8192 --num-speculative-tokens 3

| GPU experts | Status | Free VRAM after load | KV tokens | Prompt tokens | Prefill tok/s | TTFT ms | Decode tok/s | Inter-token p50 / p99 / max ms | CUDA graph |
|---|---|---|---|---|---|---|---|---|---|
| 20% | ok | 3.45 GB | 102592 | 28 | 13.1 | 2234 | 22.7 | 0.0 / 235.6 / 238.0 | yes |
| 35% | ok | 0.79 GB | 13440 | 28 | 31.7 | 995 | 25.0 | 0.0 / 161.1 / 228.9 | yes |

Greedy outputs identical across successful runs: **yes** (2 runs)
