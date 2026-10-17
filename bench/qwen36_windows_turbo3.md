### Qwen3.6-35B-A3B-NVFP4-hf on windows

- GPU: NVIDIA GeForce RTX 3060, 12288 MiB, 610.88, 3, 16
- OS: Windows-11-10.0.26100-SP0; CPU: Intel64 Family 6 Model 79 Stepping 1, GenuineIntel
- xinfer dddf34b, 2026-10-17 10:18; server mode, up to 200 decoded tokens, greedy, max-model-len 8192 --kvcache-dtype turbo3

| GPU experts | Status | Free VRAM after load | KV tokens | Prompt tokens | Prefill tok/s | TTFT ms | Decode tok/s | Inter-token p50 / p99 / max ms | CUDA graph |
|---|---|---|---|---|---|---|---|---|---|
| 20% | ok | 4.39 GB | 591680 | 28 | 14.6 | 2020 | 24.5 | 39.8 / 64.7 / 65.1 | yes |
| 35% | ok | 1.75 GB | 201216 | 28 | 32.7 | 950 | 26.8 | 37.0 / 45.3 / 46.1 | yes |

Greedy outputs identical across successful runs: **yes** (2 runs)
