### Qwen1.5-MoE-A2.7B-Chat on windows

- GPU: NVIDIA GeForce RTX 3060, 12288 MiB, 610.88, 3, 16
- OS: Windows-11-10.0.26100-SP0; CPU: Intel64 Family 6 Model 79 Stepping 1, GenuineIntel
- xinfer 81c7740, 2026-10-20 22:02; server mode, up to 200 decoded tokens, greedy, max-model-len 8192 --isq q8_0 --kvcache-dtype turbo4

| GPU experts | Status | Free VRAM after load | KV tokens | Prompt tokens | Prefill tok/s | TTFT ms | Decode tok/s | Inter-token p50 / p99 / max ms | CUDA graph |
|---|---|---|---|---|---|---|---|---|---|
| auto | ok | 1.96 GB | 27072 | 35 | 18.6 | 1934 | 20.5 | 49.3 / 73.9 / 79.7 | yes |

Greedy outputs identical across successful runs: **yes** (1 runs)
