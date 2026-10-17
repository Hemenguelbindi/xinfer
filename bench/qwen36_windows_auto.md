### Qwen3.6-35B-A3B-NVFP4-hf on windows

- GPU: NVIDIA GeForce RTX 3060, 12288 MiB, 610.88, 3, 16
- OS: Windows-11-10.0.26100-SP0; CPU: Intel64 Family 6 Model 79 Stepping 1, GenuineIntel
- xinfer dddf34b, 2026-10-17 11:20; server mode, up to 200 decoded tokens, greedy, max-model-len 8192

| GPU experts | Status | Free VRAM after load | KV tokens | Prompt tokens | Prefill tok/s | TTFT ms | Decode tok/s | Inter-token p50 / p99 / max ms | CUDA graph |
|---|---|---|---|---|---|---|---|---|---|
| auto | ok | 2.13 GB | 58432 | 28 | 24.4 | 1256 | 26.9 | 36.6 / 49.0 / 50.3 | yes |
| 20% | ok | 4.39 GB | 134016 | 28 | 14.4 | 2068 | 23.9 | 40.7 / 64.9 / 66.4 | yes |

Greedy outputs identical across successful runs: **yes** (2 runs)
