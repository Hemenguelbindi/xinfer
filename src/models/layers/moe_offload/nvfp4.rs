//! Offloaded NVFP4 experts (`FusedMoeNvfp4`): fused gate_up and down blocks/scales
//! in host RAM, `[num_slots, ...]` u8 slot tensors on GPU. The per-expert global and
//! input scales are tiny, so their slot-ordered copies are rebuilt on the host.

use super::cache::{copy_htod, profile_begin, profile_end, u8_ptr, Placement, SlotLru};
use super::host_banks::HostBank;
use super::prefill::staging_bytes;
use super::Nvfp4Weights;
use crate::models::layers::VarBuilderX;
use candle_core::{CudaDevice, DType, Device, Result, Tensor};
use std::sync::Mutex;

const ROLES: [&str; 4] = ["nvfp4_gate_up", "nvfp4_gate_up_scale", "nvfp4_down", "nvfp4_down_scale"];

pub struct Nvfp4ExpertOffload {
    dev: CudaDevice,
    lru: SlotLru,
    banks: [HostBank; 4],
    slots: [Tensor; 4],
    staging: [Tensor; 4],
    // gate_up global, gate_up input, down global, down input; per expert.
    host_vecs: [Vec<f32>; 4],
    full_vecs: [Tensor; 4],
    slot_vecs: Mutex<[Tensor; 4]>,
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
        let num_slots = SlotLru::slots_for(rate, num_experts, topk);
        let (mut banks, mut slots, mut staging) = (Vec::new(), Vec::new(), Vec::new());
        for (t, role) in big.iter().zip(ROLES) {
            banks.push(HostBank::new(dev, &t.flatten_all()?.to_vec1::<u8>()?, num_experts)?);
            let mut dims = t.dims().to_vec();
            staging.push(staging_bytes(dev, role, &dims)?);
            dims[0] = num_slots;
            slots.push(Tensor::zeros(dims, DType::U8, &gpu)?);
        }
        let host_vecs = arr4(vecs.iter().map(|v| v.to_vec1::<f32>()).collect::<Result<Vec<_>>>()?);
        let full_vecs = arr4(
            host_vecs.iter().map(|v| Tensor::from_vec(v.clone(), num_experts, &gpu)).collect::<Result<Vec<_>>>()?,
        );
        let zeros = || Tensor::zeros(num_slots, DType::F32, &gpu);
        let slot_vecs = Mutex::new([zeros()?, zeros()?, zeros()?, zeros()?]);
        Ok(Self {
            dev: dev.clone(),
            lru: SlotLru::new(num_experts, num_slots),
            banks: arr4(banks),
            slots: arr4(slots),
            staging: arr4(staging),
            host_vecs,
            full_vecs,
            slot_vecs,
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
        Self::weights(&self.slots, &self.slot_vecs.lock().unwrap())
    }

    /// Makes the routed experts resident; returns slot ids (or the original ids with
    /// full-layer staging weights) and the weights the kernels must read.
    pub fn prepare(&self, topk_ids: Tensor) -> Result<(Tensor, Nvfp4Weights)> {
        let t = profile_begin(&self.dev)?;
        let out = self.prepare_inner(topk_ids)?;
        profile_end(&self.dev, t)?;
        Ok(out)
    }

    fn prepare_inner(&self, topk_ids: Tensor) -> Result<(Tensor, Nvfp4Weights)> {
        let ids: Vec<u32> = topk_ids.flatten_all()?.to_vec1()?;
        match self.lru.place(&ids) {
            Placement::Staged => {
                for (bank, dst) in self.banks.iter().zip(&self.staging) {
                    copy_htod(&self.dev, u8_ptr(dst)?, bank.all())?;
                }
                Ok((topk_ids, Self::weights(&self.staging, &self.full_vecs)))
            }
            Placement::Slots { slot_ids, misses } => {
                for &(e, s) in &misses {
                    for (bank, dst) in self.banks.iter().zip(&self.slots) {
                        copy_htod(&self.dev, u8_ptr(dst)? + (s * bank.expert_bytes) as u64, bank.expert(e))?;
                    }
                }
                let mut slot_vecs = self.slot_vecs.lock().unwrap();
                if !misses.is_empty() {
                    let owner = self.lru.expert_of_slots();
                    let gpu = Device::Cuda(self.dev.clone());
                    for (dst, host) in slot_vecs.iter_mut().zip(&self.host_vecs) {
                        let v: Vec<f32> = owner.iter().map(|&e| if e >= 0 { host[e as usize] } else { 0.0 }).collect();
                        *dst = Tensor::from_vec(v, owner.len(), &gpu)?;
                    }
                }
                let ids = Tensor::from_vec(slot_ids, topk_ids.shape(), topk_ids.device())?;
                Ok((ids, Self::weights(&self.slots, &slot_vecs)))
            }
        }
    }
}
