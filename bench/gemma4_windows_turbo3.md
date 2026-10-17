### Gemma-4-26B-A4B-NVFP4 on windows

- GPU: NVIDIA GeForce RTX 3060, 12288 MiB, 610.88, 3, 16
- OS: Windows-11-10.0.26100-SP0; CPU: Intel64 Family 6 Model 79 Stepping 1, GenuineIntel
- xinfer dddf34b, 2026-10-17 10:21; server mode, up to 200 decoded tokens, greedy, max-model-len 8192 --kvcache-dtype turbo3

| GPU experts | Status | Free VRAM after load | KV tokens | Prompt tokens | Prefill tok/s | TTFT ms | Decode tok/s | Inter-token p50 / p99 / max ms | CUDA graph |
|---|---|---|---|---|---|---|---|---|---|
| 10% | ok | 2.46 GB | 29696 | 28 | 19.4 | 1557 | 16.3 | 60.3 / 83.4 / 85.5 | yes |
| 15% | ok | 1.81 GB | 18688 | 28 | 19.7 | 1532 | 19.9 | 48.8 / 72.8 / 75.7 | yes |

Greedy outputs identical across successful runs: **yes** (2 runs)
