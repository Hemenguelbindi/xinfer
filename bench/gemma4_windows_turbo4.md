### Gemma-4-26B-A4B-NVFP4 on windows

- GPU: NVIDIA GeForce RTX 3060, 12288 MiB, 610.88, 3, 16
- OS: Windows-11-10.0.26100-SP0; CPU: Intel64 Family 6 Model 79 Stepping 1, GenuineIntel
- xinfer dddf34b, 2026-10-17 10:14; server mode, up to 200 decoded tokens, greedy, max-model-len 8192 --kvcache-dtype turbo4

| GPU experts | Status | Free VRAM after load | KV tokens | Prompt tokens | Prefill tok/s | TTFT ms | Decode tok/s | Inter-token p50 / p99 / max ms | CUDA graph |
|---|---|---|---|---|---|---|---|---|---|
| 10% | ok | 2.50 GB | 26688 | 28 | 19.6 | 1535 | 16.7 | 58.8 / 81.9 / 86.6 | yes |
| 15% | ok | 1.76 GB | 15680 | 28 | 19.6 | 1524 | 19.8 | 49.9 / 76.8 / 79.9 | yes |

Greedy outputs identical across successful runs: **yes** (2 runs)
