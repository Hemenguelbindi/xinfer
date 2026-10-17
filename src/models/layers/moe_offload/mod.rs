//! MoE expert offload: routed experts live in pinned host RAM, the GPU keeps an
//! LRU cache of expert slots. Misses are copied over PCIe before `moe_gemm*`,
//! which then reads the slot tensor with expert ids rewritten to slot ids.
//!
//! Design follows FreeToken (https://github.com/FlashML-org/FreeToken, Apache-2.0).

pub mod budget;
pub mod cache;
pub mod host_banks;
pub mod prefill;
