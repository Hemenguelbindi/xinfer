//! MoE expert offload: routed experts live in pinned host RAM, the GPU keeps an
//! LRU cache of expert slots. Misses are copied over PCIe before `moe_gemm*`,
//! which then reads the slot tensor with expert ids rewritten to slot ids.
//!
//! Design follows FreeToken (https://github.com/FlashML-org/FreeToken, Apache-2.0).

//!
//! Enable with `XINFER_MOE_OFFLOAD=<fraction of each layer's experts kept on GPU>`.

pub mod budget;
#[cfg(feature = "cuda")]
pub mod cache;
#[cfg(feature = "cuda")]
pub mod host_banks;
#[cfg(feature = "cuda")]
pub mod prefill;
