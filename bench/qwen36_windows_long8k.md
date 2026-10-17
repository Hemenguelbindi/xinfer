### Qwen3.6-35B-A3B-NVFP4-hf on windows

- GPU: NVIDIA GeForce RTX 3060, 12288 MiB, 610.88, 3, 16
- OS: Windows-11-10.0.26100-SP0; CPU: Intel64 Family 6 Model 79 Stepping 1, GenuineIntel
- xinfer dddf34b, 2026-10-17 10:05; server mode, up to 32 decoded tokens, greedy, max-model-len 40960

| GPU experts | Status | Free VRAM after load | KV tokens | Prompt tokens | Prefill tok/s | TTFT ms | Decode tok/s | Inter-token p50 / p99 / max ms | CUDA graph | Needle |
|---|---|---|---|---|---|---|---|---|---|---|
| 20% | ok | 4.39 GB | 134016 | 11547 | 332.9 | 34814 | 20.3 | 63.1 / 69.6 / 69.6 | yes | yes |

Greedy outputs identical across successful runs: **yes** (1 runs)
