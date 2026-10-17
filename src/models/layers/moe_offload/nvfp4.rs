//! Offloaded NVFP4 experts (`FusedMoeNvfp4`): fused gate_up and down blocks/scales
//! in host RAM, `[num_slots, ...]` u8 slot tensors on GPU. The per-expert global and
//! input scales stay fully on the GPU; their slot-ordered copies are gathered like
//! any other bank.

use super::cache::{copy_htod, dev_ptr, profile_begin, profile_end, Plan, SlotCache};
use super::host_banks::HostBank;
use super::prefill::staging_bytes;
use super::Nvfp4Weights;
use crate::models::layers::VarBuilderX;
use candle_core::{CudaDevice, DType, Device, Result, Tensor};

const ROLES: [&str; 4] = ["nvfp4_gate_up", "nvfp4_gate_up_scale", "nvfp4_down", "nvfp4_down_scale"];

pub struct Nvfp4ExpertOffload {
    cache: SlotCache,
    banks: [HostBank; 4],
    slots: [Tensor; 4],
    staging: [Tensor; 4],
    // gate_up global, gate_up input, down global, down input: [E] full, [S] slot-ordered.
    full_vecs: [Tensor; 4],
    slot_vecs: [Tensor; 4],
    src: [u64; 8],
    dst: [u64; 8],
    bytes: [u64; 8],
}

fn arr4<T>(v: Vec<T>) -> [T; 4] {
    v.try_into().unwrap_or_else(|_| unreachable!("four expert tensors"))
}

/// CUDA device to offload onto when `XINFER_MOE_OFFLOAD` is set and supported here.
pub fn offload_device(vb: &VarBuilderX, world_size: usize) -> Result<Option<CudaDevice>> {
    if crate::utils::env::moe_offload_rate().is_none() {
        return Ok(None);
    }
    let Device::Cuda(dev) = vb.device() else {
        candle_core::bail!("XINFER_MOE_OFFLOAD requires a CUDA device");
    };
    if world_size > 1 {
        crate::log_warn!("XINFER_MOE_OFFLOAD is ignored with tensor parallelism (world_size > 1)");
        return Ok(None);
    }
    if attention_rs::cuda_utils::sm_version(&dev).unwrap_or(0) >= 100 {
        candle_core::bail!("XINFER_MOE_OFFLOAD: NVFP4 swizzled scales (SM100+) are not supported yet");
    }
    Ok(Some(dev))
}

impl Nvfp4ExpertOffload {
    /// `big`: host tensors gate_up blocks/scales and down blocks/scales, `[E, ...]` u8.
    /// `vecs`: host `[E]` f32 gate_up global/input and down global/input scales.
    pub fn new(dev: &CudaDevice, big: [Tensor; 4], vecs: [Tensor; 4], topk: usize) -> Result<Self> {
        let rate = crate::utils::env::moe_offload_rate().expect("offload enabled");
        let gpu = Device::Cuda(dev.clone());
        let num_experts = big[0].dim(0)?;
        let num_slots = SlotCache::slots_for(rate, num_experts, topk);
        let (mut banks, mut slots, mut staging) = (Vec::new(), Vec::new(), Vec::new());
        for (t, role) in big.iter().zip(ROLES) {
            banks.push(HostBank::new(dev, &t.flatten_all()?.to_vec1::<u8>()?, num_experts)?);
            let mut dims = t.dims().to_vec();
            staging.push(staging_bytes(dev, role, &dims)?);
            dims[0] = num_slots;
            slots.push(Tensor::zeros(dims, DType::U8, &gpu)?);
        }
        let full_vecs = arr4(vecs.iter().map(|v| v.to_dtype(DType::F32)?.to_device(&gpu)).collect::<Result<Vec<_>>>()?);
        let slot_vecs = arr4((0..4).map(|_| Tensor::zeros(num_slots, DType::F32, &gpu)).collect::<Result<Vec<_>>>()?);

        // Gather list: 4 big host banks, then the 4 scale vectors (device -> device, 4 bytes).
        let (mut src, mut dst, mut bytes) = ([0u64; 8], [0u64; 8], [0u64; 8]);
        for i in 0..4 {
            (src[i], dst[i], bytes[i]) = (banks[i].dev_ptr, dev_ptr(&slots[i])?, banks[i].expert_bytes as u64);
            (src[4 + i], dst[4 + i], bytes[4 + i]) = (dev_ptr(&full_vecs[i])?, dev_ptr(&slot_vecs[i])?, 4);
        }
        Ok(Self {
            cache: SlotCache::new(dev, num_experts, num_slots)?,
            banks: arr4(banks),
            slots: arr4(slots),
            staging: arr4(staging),
            full_vecs,
            slot_vecs,
            src,
            dst,
            bytes,
        })
    }

    fn weights(big: &[Tensor; 4], vecs: &[Tensor; 4]) -> Nvfp4Weights {
        Nvfp4Weights {
            gate_up_blocks: big[0].clone(),
            gate_up_scales: big[1].clone(),
            gate_up_global: vecs[0].clone(),
            gate_up_input: vecs[1].clone(),
            down_blocks: big[2].clone(),
            down_scales: big[3].clone(),
            down_global: vecs[2].clone(),
            down_input: vecs[3].clone(),
        }
    }

    /// GPU slot tensors, used in place of the full expert tensors.
    pub fn slot_weights(&self) -> Nvfp4Weights {
        Self::weights(&self.slots, &self.slot_vecs)
    }

    /// Makes the routed experts resident; returns slot ids (or the original ids with
    /// full-layer staging weights) and the weights the kernels must read.
    pub fn prepare(&self, topk_ids: Tensor, is_prefill: bool) -> Result<(Tensor, Nvfp4Weights)> {
        let dev = self.cache.device();
        let t = profile_begin(dev)?;
        let out = match self.cache.plan(&topk_ids, is_prefill)? {
            Plan::Staged => {
                for (bank, dst) in self.banks.iter().zip(&self.staging) {
                    copy_htod(dev, dev_ptr(dst)?, bank.all())?;
                }
                (topk_ids, Self::weights(&self.staging, &self.full_vecs))
            }
            Plan::Slots(ids) => {
                self.cache.gather(&self.src, &self.dst, &self.bytes)?;
                (ids, self.slot_weights())
            }
        };
        profile_end(dev, t)?;
        Ok(out)
    }
}
