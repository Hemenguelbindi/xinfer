### Qwen1.5-MoE-A2.7B-Chat on windows

- GPU: NVIDIA GeForce RTX 3060, 12288 MiB, 610.88, 3, 16
- OS: Windows-11-10.0.26100-SP0; CPU: Intel64 Family 6 Model 79 Stepping 1, GenuineIntel
- xinfer 81c7740, 2026-10-20 21:38; server mode, up to 200 decoded tokens, greedy, max-model-len 8192 --kvcache-dtype fp8

| GPU experts | Status | Free VRAM after load | KV tokens | Prompt tokens | Prefill tok/s | TTFT ms | Decode tok/s | Inter-token p50 / p99 / max ms | CUDA graph |
|---|---|---|---|---|---|---|---|---|---|
| auto | ok | 1.95 GB | 14272 | 35 | 14.6 | 2474 | 7.2 | 142.0 / 171.5 / 171.6 | yes |
| 10% | ok | 3.82 GB | 30656 | 35 | 14.6 | 2467 | 6.7 | 152.5 / 177.3 / 177.6 | yes |

Greedy outputs identical across successful runs: **yes** (2 runs)
