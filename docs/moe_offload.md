# MoE expert offload

Run Mixture-of-Experts models whose weights do not fit in VRAM. The routed
experts stay in pinned host RAM; each MoE layer keeps a fraction of its experts
in GPU *slots*. Attention, dense layers, shared experts and the KV cache stay on
the GPU as usual.

```bash
# keep 20% of every layer's experts on the GPU, the rest in host RAM
XINFER_MOE_OFFLOAD=0.2 xinfer --m nvidia/Qwen3.6-35B-A3B-NVFP4 --max-model-len 8192 --ui-server
```

Placement never changes routing or precision: with greedy decoding the output
is byte-identical for every `XINFER_MOE_OFFLOAD` value.

## How it works

- **Decode** — before each fused MoE GEMM, a CUDA kernel looks the routed experts
  up in a per-layer LRU table kept on the GPU, assigns least-recently-used slots
  to misses and rewrites expert ids to slot ids. A second kernel copies the
  missing experts straight from device-mapped pinned host memory into their
  slots. Both kernels have fixed launch shapes and read the miss count from
  device memory, so decode stays inside CUDA graphs with no host sync.
- **Prefill** — when a call touches more distinct experts than the layer has
  slots, the whole expert layer is copied (pinned async memcpy) into a staging
  buffer shared by all layers, and the unmodified GEMM runs on it.
- The GEMM kernels are unchanged: they read `[num_slots, N, K]` slot tensors
  instead of `[num_experts, N, K]`.

Design follows [FreeToken](https://github.com/FlashML-org/FreeToken) (Apache-2.0).

## Options

| Variable | Meaning |
|---|---|
| `XINFER_MOE_OFFLOAD=auto` | After loading, grow every layer's slots evenly into the free VRAM, leaving the reserve below for the KV cache and workspaces (recommended) |
| `XINFER_MOE_OFFLOAD=<0..1>` | Fixed fraction of each layer's experts kept on the GPU (at least `num_experts_per_tok`). Unset = no offload |
| `XINFER_MOE_OFFLOAD_RESERVE_GB` | VRAM left free by `auto` (default 2.5). Raise it for longer contexts |
| `XINFER_MOE_OFFLOAD_PROFILE=1` | Logs a per-MoE-call time split (launch gap, GPU wait, host, copy). Adds device syncs and disables CUDA graphs; diagnostics only |

More experts on the GPU means faster decode; more reserve means a longer
context. TurboQuant (`--kvcache-dtype turbo4|turbo3`) stretches the reserve
~4x. Models with a large KV per token (e.g. Gemma 4) need `fp8` or `turbo*`.
MTP (`--num-speculative-tokens`) works but was slower in our tests.

## Supported

- Formats:
  - GGUF (`FusedMoeGGUF`, separate `ffn_gate_exps`/`ffn_up_exps`, quant types the
    GGUF MoE kernel reads natively) and ISQ (`--isq`, `FusedMoeISQ`: experts are
    quantized on the GPU at load, then moved to host RAM);
  - NVFP4 ModelOpt safetensors (`FusedMoeNvfp4`);
  - unquantized bf16/f16 (`FusedMoe`), block FP8 (`FusedMoeFp8`), MXFP4
    (`FusedMoeMxfp4`) and GPTQ/AWQ/compressed-tensors WNA16 (`FusedMoeWNA16`): all
    share one generic path that keeps every `[E, ...]` expert tensor (weights and
    scales, any dtype) in host RAM.
- Not supported: W2-packed DeepSeek-V4 experts (fails with a clear error).
- Single GPU (`world_size == 1`), CUDA. NVFP4 on SM100+ (swizzled scales) is not
  supported yet.
- Tested on real models: GGUF, NVFP4, bf16 and ISQ. FP8, MXFP4 and WNA16 are
  covered by the generic path's bit-exact GEMM test only.
- Host RAM must hold every expert (pinned).

## Benchmarks

Reproduce with `bench/bench_moe_offload.py` (rate `0` = baseline without offload):

```bash
python3 bench/bench_moe_offload.py --bin target/release/xinfer \
    --model /models/Qwen3.6-35B-A3B-NVFP4 --rates 0,0.1,0.2,0.35,auto --out bench/qwen36
```

It runs the OpenAI-compatible server, sends a warm-up and one streamed request
(greedy) and records time to first token, decode speed and inter-token latency
percentiles; `--prompt-tokens N` adds a long-context needle test. Full tables
and logs: [`bench/`](../bench/).

RTX 3060 12 GB, PCIe 3.0 x16, Xeon E5-2650 v4, 64 GB DDR4-2400, Windows 11,
200 decoded tokens, `--max-model-len 8192`. Without offload none of these
models runs on this card: the weights spill into the driver's sysmem fallback
and no VRAM is left for the KV cache.

**Qwen3.6-35B-A3B NVFP4** (23.4 GB of weights)

| GPU experts | KV dtype | KV tokens | TTFT | Decode tok/s | Inter-token p50 / p99 ms |
|---|---|---|---|---|---|
| none | — | does not run | | | |
| 10% | bf16 | 192k | 2.1 s | 21.5 | 46 / 70 |
| 20% | bf16 | 134k | 2.0 s | 26.4 | 38 / 51 |
| 35% | bf16 | 46k | 0.8 s | 28.6 | 35 / 44 |
| auto (31%) | bf16 | 58k | 1.3 s | 26.9 | 37 / 49 |
| 20% | turbo4 | 520k | 2.1 s | 23.3 | 41 / 68 |
| 35% | turbo3 | 201k | 1.0 s | 26.8 | 37 / 45 |

Long context: an 11.5k-token needle prompt at 20% prefills at 333 tok/s and the
needle is found. Greedy output is byte-identical for every GPU fraction.

**Gemma-4-26B-A4B NVFP4** (18.8 GB): 10% / 15% with `turbo4` → 16.7 / 19.8 tok/s
(27k / 16k KV tokens); with `fp8` at 10% → 18.9 tok/s.
**GLM-4.7-Flash NVFP4** (20.5 GB, no CUDA graphs in xinfer for this model):
10% / 20% → 10.8 / 11.2 tok/s. Outputs identical across fractions for both.

**Qwen1.5-MoE-A2.7B-Chat bf16** (28.6 GB, 60 experts of 17 MB each), `auto` with
`XINFER_MOE_OFFLOAD_RESERVE_GB=1.5`:

| Experts | KV dtype | GPU experts | KV tokens | TTFT | Decode tok/s | Inter-token p50 / p99 ms |
|---|---|---|---|---|---|---|
| bf16 | fp8 | auto (~15%) | 8.4k | 2.5 s | 7.9 | 131 / 168 |
| bf16 | turbo4 | auto (~15%) | 12.5k | 2.4 s | 8.1 | 126 / 158 |
| ISQ `q8_0` | fp8 | auto (40%) | 14.8k | 1.9 s | 18.0 | 55 / 86 |
| ISQ `q8_0` | turbo4 | auto (40%) | 27.1k | 1.9 s | 20.5 | 49 / 74 |

bf16 experts are PCIe-bound (~1.4 GB of misses per token over PCIe 3.0);
`--isq q8_0` halves the expert size and fits 2.7x more experts on the GPU, so
decode is 2.5x faster. Q4K needs `moe_intermediate_size % 256 == 0` (1408 here)
and the GGUF MoE kernel has no Q4_0, so 4-bit ISQ is not available for this
model. bf16 greedy output is identical at 10% and auto (~15%).

Run-to-run variation on this machine is about ±10%.
