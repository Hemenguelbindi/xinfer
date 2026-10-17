//! Offloaded GGUF experts (`FusedMoeGGUF`): gate/up/down banks in host RAM,
//! `[num_slots, N, K]` slot QTensors on GPU.

use super::cache::{copy_htod, profile_begin, profile_end, Plan, SlotCache};
use super::host_banks::HostBank;
use super::prefill::staging_quant;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use candle_core::quantized::{cuda::QCudaStorage, QStorage, QTensor};
use candle_core::{Device, Result, Tensor};
use either::Either;
use std::sync::Arc;

pub struct GgufExpertOffload {
    cache: SlotCache,
    banks: [HostBank; 3],
    slots: [Arc<QTensor>; 3],
    staging: [Arc<QTensor>; 3],
}

fn arr3<T>(v: Vec<T>) -> [T; 3] {
    v.try_into().unwrap_or_else(|_| unreachable!("three expert weights"))
}

impl GgufExpertOffload {
    /// Builds the offload state when `XINFER_MOE_OFFLOAD` is set; experts stay in host RAM.
    pub fn from_env(vb: &VarBuilderX, cfg: &Config) -> Result<Option<Self>> {
        let Some(rate) = crate::utils::env::moe_offload_rate() else {
            return Ok(None);
        };
        let device = vb.device();
        let (Either::Right(v), Device::Cuda(dev)) = (&vb.0, &device) else {
            candle_core::bail!("XINFER_MOE_OFFLOAD supports GGUF models on CUDA only");
        };
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();
        let (h, i) = (cfg.hidden_size, moe_cfg.moe_intermediate_size);
        if v.pp("ffn_gate_up_exps").contains_key("weight") {
            candle_core::bail!("XINFER_MOE_OFFLOAD: packed ffn_gate_up_exps GGUF is not supported yet");
        }
        let num_slots = SlotCache::slots_for(rate, num_experts, moe_cfg.num_experts_per_tok);

        let host = v.with_device(&Device::Cpu);
        let names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
        let dims = [[num_experts, i, h], [num_experts, i, h], [num_experts, h, i]];
        let (mut banks, mut slots, mut staging) = (Vec::new(), Vec::new(), Vec::new());
        for role in 0..3 {
            let t = host.pp(names[role]).get(dims[role].to_vec(), "weight")?;
            if !crate::models::layers::moe::moe_gemm_gguf_supports(t.dtype()) {
                candle_core::bail!(
                    "XINFER_MOE_OFFLOAD: {} dtype {:?} needs requantization, not supported yet",
                    names[role],
                    t.dtype()
                );
            }
            banks.push(HostBank::new(dev, &t.data()?, num_experts)?);
            let [_, n, k] = dims[role];
            let st = QCudaStorage::zeros(dev, num_slots * n * k, t.dtype())?;
            slots.push(Arc::new(QTensor::new(QStorage::Cuda(st), (num_slots, n, k))?));
            staging.push(staging_quant(dev, names[role], t.dtype(), &dims[role])?);
        }
        host.clear_cache();
        Ok(Some(Self {
            cache: SlotCache::new(dev, num_experts, num_slots)?,
            banks: arr3(banks),
            slots: arr3(slots),
            staging: arr3(staging),
        }))
    }

    /// Slot tensors `[num_slots, N, K]` that replace the full expert tensors.
    pub fn slot_tensors(&self) -> (Arc<QTensor>, Arc<QTensor>, Arc<QTensor>) {
        (self.slots[0].clone(), self.slots[1].clone(), self.slots[2].clone())
    }

    /// Makes the routed experts resident. Returns ids rewritten to slot ids (weights =
    /// slot tensors, `None`), or the original ids with full-layer staging weights.
    pub fn prepare(&self, topk_ids: Tensor, is_prefill: bool) -> Result<(Tensor, Option<[Arc<QTensor>; 3]>)> {
        let dev = self.cache.device();
        let t = profile_begin(dev)?;
        let out = match self.cache.plan(&topk_ids, is_prefill)? {
            Plan::Staged => {
                for (bank, dst) in self.banks.iter().zip(&self.staging) {
                    copy_htod(dev, dst.device_ptr()? as u64, bank.all())?;
                }
                (topk_ids, Some(self.staging.clone()))
            }
            Plan::Slots(ids) => {
                let mut dst = [0u64; 3];
                for (d, s) in dst.iter_mut().zip(&self.slots) {
                    *d = s.device_ptr()? as u64;
                }
                let src = self.banks.each_ref().map(|b| b.dev_ptr);
                let bytes = self.banks.each_ref().map(|b| b.expert_bytes as u64);
                self.cache.gather(&src, &dst, &bytes)?;
                (ids, None)
            }
        };
        profile_end(dev, t)?;
        Ok(out)
    }
}
