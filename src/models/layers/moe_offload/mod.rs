//! MoE expert offload: routed experts live in pinned host RAM, the GPU keeps an
//! LRU cache of expert slots. Misses are copied over PCIe before `moe_gemm*`,
//! which then reads the slot tensor with expert ids rewritten to slot ids.
//!
//! Design follows FreeToken (https://github.com/FlashML-org/FreeToken, Apache-2.0).
//!
//! Enable with `XINFER_MOE_OFFLOAD=<fraction of each layer's experts kept on GPU>`, or
//! `XINFER_MOE_OFFLOAD=auto` to fill the free VRAM (see `budget`).

#[cfg(feature = "cuda")]
use crate::models::layers::VarBuilderX;
#[cfg(feature = "cuda")]
use candle_core::Result;
use candle_core::Tensor;

#[cfg(feature = "cuda")]
pub mod budget;
#[cfg(feature = "cuda")]
pub mod cache;
#[cfg(feature = "cuda")]
pub mod experts;
#[cfg(feature = "cuda")]
pub mod gguf;
#[cfg(feature = "cuda")]
pub mod host_banks;
#[cfg(feature = "cuda")]
pub mod nvfp4;
#[cfg(feature = "cuda")]
pub mod prefill;

#[cfg(feature = "cuda")]
pub use experts::ExpertOffload;

/// Builder that loads `experts_vb`'s tensors into host RAM, for `ExpertOffload::new`.
#[cfg(feature = "cuda")]
pub fn host_experts_vb<'b>(experts_vb: &'b VarBuilderX<'_>) -> Result<VarBuilderX<'b>> {
    let cpu = experts_vb
        .cpu_var_builder()
        .ok_or_else(|| candle_core::Error::Msg("XINFER_MOE_OFFLOAD: no CPU var builder".into()))?;
    Ok(VarBuilderX(
        either::Either::Left(cpu),
        experts_vb.1.clone(),
        None,
        None,
        None,
    ))
}

/// NVFP4 expert weights for one MoE call (slot, staging or resident tensors).
pub struct Nvfp4Weights {
    pub gate_up_blocks: Tensor,
    pub gate_up_scales: Tensor,
    pub gate_up_global: Tensor,
    pub gate_up_input: Tensor,
    pub down_blocks: Tensor,
    pub down_scales: Tensor,
    pub down_global: Tensor,
    pub down_input: Tensor,
}
