//! Offloaded NVFP4 experts (`FusedMoeNvfp4`): fused gate_up and down blocks/scales
//! in host RAM, `[num_slots, ...]` u8 slot tensors on GPU. The per-expert global and
//! input scales stay fully on the GPU; their slot-ordered copies are gathered like
//! any other bank.

use super::budget::{register, SlotPool};
use super::cache::{copy_htod, dev_ptr, profile_begin, profile_end, Plan, SlotCache};
use super::host_banks::HostBank;
use super::prefill::staging_bytes;
use super::Nvfp4Weights;
use crate::models::layers::VarBuilderX;
use candle_core::{CudaDevice, DType, Device, Result, Tensor};
use std::sync::{Arc, RwLock};

const ROLES: [&str; 4] = [
    "nvfp4_gate_up",
    "nvfp4_gate_up_scale",
    "nvfp4_down",
    "nvfp4_down_scale",
];

/// Slot tensors and their LRU; replaced as a whole by `resize`.
struct Slots {
    cache: SlotCache,
    big: [Tensor; 4],
    vecs: [Tensor; 4],
    dst: [u64; 8],
}

struct Inner {
    dev: CudaDevice,
    num_experts: usize,
    banks: [HostBank; 4],
    big_dims: [Vec<usize>; 4],
    staging: [Tensor; 4],
    // gate_up global, gate_up input, down global, down input: [E] f32 on GPU.
    full_vecs: [Tensor; 4],
    src: [u64; 8],
    bytes: [u64; 8],
    slots: RwLock<Slots>,
}

pub struct Nvfp4ExpertOffload(Arc<Inner>);

fn arr4<T>(v: Vec<T>) -> [T; 4] {
    v.try_into()
        .unwrap_or_else(|_| unreachable!("four expert tensors"))
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
        candle_core::bail!(
            "XINFER_MOE_OFFLOAD: NVFP4 swizzled scales (SM100+) are not supported yet"
        );
    }
    Ok(Some(dev))
}

fn make_slots(
    dev: &CudaDevice,
    num_experts: usize,
    big_dims: &[Vec<usize>; 4],
    num_slots: usize,
) -> Result<Slots> {
    let gpu = Device::Cuda(dev.clone());
    let mut big = Vec::with_capacity(4);
    for dims in big_dims {
        let mut d = dims.clone();
        d[0] = num_slots;
        big.push(Tensor::zeros(d, DType::U8, &gpu)?);
    }
    let vecs = arr4(
        (0..4)
            .map(|_| Tensor::zeros(num_slots, DType::F32, &gpu))
            .collect::<Result<Vec<_>>>()?,
    );
    let mut dst = [0u64; 8];
    for i in 0..4 {
        dst[i] = dev_ptr(&big[i])?;
        dst[4 + i] = dev_ptr(&vecs[i])?;
    }
    Ok(Slots {
        cache: SlotCache::new(dev, num_experts, num_slots)?,
        big: arr4(big),
        vecs,
        dst,
    })
}

impl SlotPool for Inner {
    fn device(&self) -> &CudaDevice {
        &self.dev
    }
    fn num_experts(&self) -> usize {
        self.num_experts
    }
    fn num_slots(&self) -> usize {
        self.slots.read().unwrap().cache.num_slots()
    }
    fn slot_bytes(&self) -> usize {
        self.bytes.iter().sum::<u64>() as usize
    }
    fn resize(&self, num_slots: usize) -> Result<()> {
        *self.slots.write().unwrap() =
            make_slots(&self.dev, self.num_experts, &self.big_dims, num_slots)?;
        Ok(())
    }
}

impl Nvfp4ExpertOffload {
    /// `big`: host tensors gate_up blocks/scales and down blocks/scales, `[E, ...]` u8.
    /// `vecs`: host `[E]` f32 gate_up global/input and down global/input scales.
    pub fn new(dev: &CudaDevice, big: [Tensor; 4], vecs: [Tensor; 4], topk: usize) -> Result<Self> {
        let rate = crate::utils::env::moe_offload_rate().expect("offload enabled");
        let gpu = Device::Cuda(dev.clone());
        let num_experts = big[0].dim(0)?;
        let (mut banks, mut big_dims, mut staging) = (Vec::new(), Vec::new(), Vec::new());
        for (t, role) in big.iter().zip(ROLES) {
            banks.push(HostBank::new(
                dev,
                &t.flatten_all()?.to_vec1::<u8>()?,
                num_experts,
            )?);
            staging.push(staging_bytes(dev, role, t.dims())?);
            big_dims.push(t.dims().to_vec());
        }
        let full_vecs = arr4(
            vecs.iter()
                .map(|v| v.to_dtype(DType::F32)?.to_device(&gpu))
                .collect::<Result<Vec<_>>>()?,
        );
        // Gather list: 4 big host banks, then the 4 scale vectors (device -> device, 4 bytes).
        let (mut src, mut bytes) = ([0u64; 8], [0u64; 8]);
        for i in 0..4 {
            (src[i], bytes[i]) = (banks[i].dev_ptr, banks[i].expert_bytes as u64);
            (src[4 + i], bytes[4 + i]) = (dev_ptr(&full_vecs[i])?, 4);
        }
        let big_dims = arr4(big_dims);
        let slots = make_slots(
            dev,
            num_experts,
            &big_dims,
            SlotCache::slots_for(rate, num_experts, topk),
        )?;
        let inner = Arc::new(Inner {
            dev: dev.clone(),
            num_experts,
            banks: arr4(banks),
            big_dims,
            staging: arr4(staging),
            full_vecs,
            src,
            bytes,
            slots: RwLock::new(slots),
        });
        register(inner.clone());
        Ok(Self(inner))
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

    /// Makes the routed experts resident; returns slot ids (or the original ids with
    /// full-layer staging weights) and the weights the kernels must read.
    pub fn prepare(&self, topk_ids: Tensor, is_prefill: bool) -> Result<(Tensor, Nvfp4Weights)> {
        let o = &self.0;
        let s = o.slots.read().unwrap();
        let t = profile_begin(&o.dev)?;
        let out = match s.cache.plan(&topk_ids, is_prefill)? {
            Plan::Staged => {
                for (bank, dst) in o.banks.iter().zip(&o.staging) {
                    copy_htod(&o.dev, dev_ptr(dst)?, bank.all())?;
                }
                (topk_ids, Self::weights(&o.staging, &o.full_vecs))
            }
            Plan::Slots(ids) => {
                s.cache.gather(&o.src, &s.dst, &o.bytes)?;
                (ids, Self::weights(&s.big, &s.vecs))
            }
        };
        profile_end(&o.dev, t)?;
        Ok(out)
    }
}
