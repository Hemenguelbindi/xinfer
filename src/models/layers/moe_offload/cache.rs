//! Per-layer GPU expert slots with an LRU kept on the host.
//!
//! shortcut: routing ids are read back to the host each MoE call (one sync per layer),
//! so CUDA graphs are disabled while offload is on; upgrade to a device-side
//! ensure/copy kernel in attention.rs for graph capture.

use super::host_banks::HostBank;
use super::prefill::staging;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use candle_core::cuda_backend::cudarc::driver::result;
use candle_core::quantized::{cuda::QCudaStorage, QStorage, QTensor};
use candle_core::{CudaDevice, Device, Result, Tensor};
use either::Either;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static STAGED: AtomicU64 = AtomicU64::new(0);

struct Lru {
    slot_of: Vec<i32>,   // expert -> slot, -1 if not resident
    expert_of: Vec<i32>, // slot -> expert, -1 if empty
    last_used: Vec<u64>,
    tick: u64,
}

pub struct GgufExpertOffload {
    dev: CudaDevice,
    num_slots: usize,
    banks: [HostBank; 3],
    slots: [Arc<QTensor>; 3],
    staging: [Arc<QTensor>; 3],
    lru: Mutex<Lru>,
}

fn copy_htod(dev: &CudaDevice, dst: &QTensor, offset: usize, src: &[u8]) -> Result<()> {
    let dst = dst.device_ptr()? as u64 + offset as u64;
    unsafe { result::memcpy_htod_async(dst, src, *dev.cu_stream()) }.map_err(candle_core::Error::wrap)
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
        // At least one decode step's experts must fit; never more slots than experts.
        let num_slots = ((num_experts as f32 * rate).ceil() as usize)
            .clamp(moe_cfg.num_experts_per_tok, num_experts);

        let host = v.with_device(&Device::Cpu);
        let names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
        let dims = [[num_experts, i, h], [num_experts, i, h], [num_experts, h, i]];
        let mut banks = Vec::with_capacity(3);
        let mut slots = Vec::with_capacity(3);
        let mut stages = Vec::with_capacity(3);
        for role in 0..3 {
            let t = host.pp(names[role]).get(dims[role].to_vec(), "weight")?;
            if !crate::models::layers::moe::moe_gemm_gguf_supports(t.dtype()) {
                candle_core::bail!(
                    "XINFER_MOE_OFFLOAD: {} dtype {:?} needs requantization, not supported yet",
                    names[role],
                    t.dtype()
                );
            }
            let bank = HostBank::new(dev, &t.data()?, num_experts)?;
            let [_, n, k] = dims[role];
            let st = QCudaStorage::zeros(dev, num_slots * n * k, t.dtype())?;
            slots.push(Arc::new(QTensor::new(QStorage::Cuda(st), (num_slots, n, k))?));
            stages.push(staging(dev, role, t.dtype(), &dims[role])?);
            banks.push(bank);
        }
        host.clear_cache();
        Ok(Some(Self {
            dev: dev.clone(),
            num_slots,
            banks: banks.try_into().map_err(|_| candle_core::Error::Msg("banks".into()))?,
            slots: slots.try_into().map_err(|_| candle_core::Error::Msg("slots".into()))?,
            staging: stages.try_into().map_err(|_| candle_core::Error::Msg("staging".into()))?,
            lru: Mutex::new(Lru {
                slot_of: vec![-1; num_experts],
                expert_of: vec![-1; num_slots],
                last_used: vec![0; num_slots],
                tick: 0,
            }),
        }))
    }

    /// Slot tensors `[num_slots, N, K]` that replace the full expert tensors.
    pub fn slot_tensors(&self) -> (Arc<QTensor>, Arc<QTensor>, Arc<QTensor>) {
        (self.slots[0].clone(), self.slots[1].clone(), self.slots[2].clone())
    }

    /// Makes the routed experts resident. Returns ids rewritten to slot ids (weights =
    /// slot tensors), or the original ids with full-layer staging weights when the call
    /// touches more experts than there are slots (prefill).
    pub fn prepare(&self, topk_ids: Tensor) -> Result<(Tensor, Option<[Arc<QTensor>; 3]>)> {
        let ids: Vec<u32> = topk_ids.flatten_all()?.to_vec1()?;
        let mut uniq = ids.clone();
        uniq.sort_unstable();
        uniq.dedup();

        if uniq.len() > self.num_slots {
            for r in 0..3 {
                copy_htod(&self.dev, &self.staging[r], 0, self.banks[r].all())?;
            }
            STAGED.fetch_add(1, Ordering::Relaxed);
            return Ok((topk_ids, Some(self.staging.clone())));
        }

        let mut lru = self.lru.lock().unwrap();
        lru.tick += 1;
        let tick = lru.tick;
        let mut misses = Vec::new();
        for &e in &uniq {
            match lru.slot_of[e as usize] {
                s if s >= 0 => lru.last_used[s as usize] = tick,
                _ => misses.push(e as usize),
            }
        }
        for &e in &misses {
            // Victim: least recently used slot not needed by this call (one exists: uniq <= slots).
            let s = (0..self.num_slots)
                .filter(|&s| lru.last_used[s] < tick)
                .min_by_key(|&s| lru.last_used[s])
                .expect("an evictable slot");
            if lru.expert_of[s] >= 0 {
                let old = lru.expert_of[s] as usize;
                lru.slot_of[old] = -1;
            }
            lru.expert_of[s] = e as i32;
            lru.slot_of[e] = s as i32;
            lru.last_used[s] = tick;
            for r in 0..3 {
                let bank = &self.banks[r];
                copy_htod(&self.dev, &self.slots[r], s * bank.expert_bytes, bank.expert(e))?;
            }
        }
        let remapped: Vec<u32> = ids.iter().map(|&e| lru.slot_of[e as usize] as u32).collect();
        drop(lru);
        log_stats(uniq.len() - misses.len(), misses.len());
        let remapped = Tensor::from_vec(remapped, topk_ids.shape(), topk_ids.device())?;
        Ok((remapped, None))
    }
}

fn log_stats(hits: usize, misses: usize) {
    let h = HITS.fetch_add(hits as u64, Ordering::Relaxed) + hits as u64;
    let m = MISSES.fetch_add(misses as u64, Ordering::Relaxed) + misses as u64;
    let total = h + m;
    if total / 16384 != (total - (hits + misses) as u64) / 16384 {
        crate::log_info!(
            "MoE offload: expert hit rate {:.1}% ({h} hits, {m} misses), {} full-layer stagings",
            100.0 * h as f64 / total as f64,
            STAGED.load(Ordering::Relaxed)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_keeps_current_experts_and_evicts_oldest() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else { return Ok(()) };
        let Device::Cuda(cuda) = &dev else { unreachable!() };
        // 4 experts of 2 bytes, 2 slots.
        let bytes: Vec<u8> = (0..8).collect();
        let bank = || HostBank::new(cuda, &bytes, 4);
        let zeros = || -> Result<Arc<QTensor>> {
            let st = QCudaStorage::zeros(cuda, 2 * 32, candle_core::quantized::GgmlDType::Q8_0)?;
            Ok(Arc::new(QTensor::new(QStorage::Cuda(st), (2, 1, 32))?))
        };
        let o = GgufExpertOffload {
            dev: cuda.clone(),
            num_slots: 2,
            banks: [bank()?, bank()?, bank()?],
            slots: [zeros()?, zeros()?, zeros()?],
            staging: [zeros()?, zeros()?, zeros()?],
            lru: Mutex::new(Lru { slot_of: vec![-1; 4], expert_of: vec![-1; 2], last_used: vec![0; 2], tick: 0 }),
        };
        let ids = |v: Vec<u32>| Tensor::from_vec(v, (1, 2), &dev);
        let slots_of = |t: Tensor| t.flatten_all().unwrap().to_vec1::<u32>().unwrap();

        let (a, st) = o.prepare(ids(vec![0, 1])?)?;
        assert!(st.is_none());
        assert_eq!(slots_of(a), vec![0, 1]);
        // Expert 2 evicts the least recently used one (0 and 1 tie -> slot 0 = expert 0).
        let (b, _) = o.prepare(ids(vec![1, 2])?)?;
        assert_eq!(slots_of(b), vec![1, 0]);
        // Three distinct experts exceed 2 slots -> full-layer staging, ids unchanged.
        let (c, st) = o.prepare(Tensor::from_vec(vec![0u32, 1, 3], (1, 3), &dev)?)?;
        assert!(st.is_some());
        assert_eq!(slots_of(c), vec![0, 1, 3]);
        Ok(())
    }
}
