//! GPU slot cache shared by every offloaded format. The LRU table lives on the GPU
//! (attention_rs::expert_cache), so decode never reads routing ids back to the host
//! and stays CUDA-graph capturable.

use attention_rs::expert_cache::ExpertLru;
use candle_core::cuda_backend::cudarc::driver::{result, DevicePtr};
use candle_core::{CudaDevice, DType, Device, Result, Storage, Tensor};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

static LOOKUPS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static STAGED: AtomicU64 = AtomicU64::new(0);

/// How one MoE call reads its experts.
pub enum Plan {
    /// Ids rewritten to slot ids; misses must be gathered into the slots (`SlotCache::gather`).
    Slots(Tensor),
    /// More experts than slots: copy the whole layer into the staging buffers.
    Staged,
}

pub struct SlotCache {
    dev: CudaDevice,
    lru: ExpertLru,
    seen: Mutex<(u64, u64)>, // lookups/misses already added to the global counters
}

impl SlotCache {
    pub fn new(dev: &CudaDevice, num_experts: usize, num_slots: usize) -> Result<Self> {
        Ok(Self {
            dev: dev.clone(),
            lru: ExpertLru::new(num_experts, num_slots, &Device::Cuda(dev.clone()))?,
            seen: Mutex::new((0, 0)),
        })
    }

    /// Slots per layer for `XINFER_MOE_OFFLOAD=rate`: at least one decode step's experts.
    pub fn slots_for(rate: f32, num_experts: usize, topk: usize) -> usize {
        ((num_experts as f32 * rate).ceil() as usize).clamp(topk, num_experts)
    }

    pub fn device(&self) -> &CudaDevice {
        &self.dev
    }

    /// Decode stays on the GPU whenever the call cannot touch more experts than slots
    /// (fixed shape, so the choice is the same at graph capture and replay). Prefill runs
    /// eagerly and checks the distinct count on the host.
    pub fn plan(&self, ids: &Tensor, is_prefill: bool) -> Result<Plan> {
        let slots = self.lru.num_slots;
        let staged = if is_prefill {
            let mut v: Vec<u32> = ids.flatten_all()?.to_vec1()?;
            v.sort_unstable();
            v.dedup();
            v.len() > slots
        } else {
            ids.elem_count() > slots
        };
        if staged {
            STAGED.fetch_add(1, Ordering::Relaxed);
            return Ok(Plan::Staged);
        }
        let out = self.lru.ensure(ids)?;
        if is_prefill {
            self.log_stats()?;
        }
        Ok(Plan::Slots(out))
    }

    /// Copies the last plan's misses: bank `b` expert `e` from `src[b] + e * bytes[b]` to
    /// `dst[b] + slot * bytes[b]`.
    pub fn gather(&self, src: &[u64], dst: &[u64], bytes: &[u64]) -> Result<()> {
        self.lru.gather(&Device::Cuda(self.dev.clone()), src, dst, bytes)
    }

    // Counters live on the GPU; they are folded in on prefill, which syncs anyway.
    fn log_stats(&self) -> Result<()> {
        let (lookups, misses) = self.lru.stats()?;
        let mut seen = self.seen.lock().unwrap();
        let before = LOOKUPS.fetch_add(lookups - seen.0, Ordering::Relaxed);
        let m = MISSES.fetch_add(misses - seen.1, Ordering::Relaxed) + misses - seen.1;
        let total = before + lookups - seen.0;
        *seen = (lookups, misses);
        if total / 16384 != before / 16384 {
            crate::log_info!(
                "MoE offload: expert hit rate {:.1}% over {total} lookups, {} full-layer stagings",
                100.0 * (total - m) as f64 / total as f64,
                STAGED.load(Ordering::Relaxed)
            );
        }
        Ok(())
    }

    #[cfg(test)]
    fn expert_of_slots(&self) -> Result<Vec<u32>> {
        self.lru.expert_of_slots()
    }
}

/// Async host->device copy on candle's stream, ordered with the kernels (pinned source).
pub fn copy_htod(dev: &CudaDevice, dst: u64, src: &[u8]) -> Result<()> {
    unsafe { result::memcpy_htod_async(dst, src, *dev.cu_stream()) }.map_err(candle_core::Error::wrap)
}

/// Device address of a contiguous u8 or f32 CUDA tensor.
pub fn dev_ptr(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(c) = &*storage else {
        candle_core::bail!("expected a CUDA tensor");
    };
    let base = match t.dtype() {
        DType::U8 => *c.as_cuda_slice::<u8>()?.device_ptr(),
        DType::F32 => *c.as_cuda_slice::<f32>()?.device_ptr(),
        dt => candle_core::bail!("unsupported dtype {dt:?}"),
    };
    Ok(base + (layout.start_offset() * t.dtype().size_in_bytes()) as u64)
}

struct Profile {
    last_end: Option<Instant>,
    sums: [f64; 4], // gap, gpu-wait, host, copy (seconds)
    calls: u64,
}

static PROFILE: Mutex<Profile> = Mutex::new(Profile { last_end: None, sums: [0.0; 4], calls: 0 });

fn sync(dev: &CudaDevice) -> Result<()> {
    dev.synchronize().map_err(candle_core::Error::wrap)
}

/// `XINFER_MOE_OFFLOAD_PROFILE=1`: splits each MoE call's time into CPU-side launch gap
/// since the previous call, leftover GPU work, host work and copy. Adds two device
/// syncs per call (CUDA graphs are disabled then); use it only to find the bottleneck.
pub fn profile_begin(dev: &CudaDevice) -> Result<Option<Instant>> {
    if !crate::utils::env::moe_offload_profile() {
        return Ok(None);
    }
    let t0 = Instant::now();
    sync(dev)?;
    let t1 = Instant::now();
    let mut p = PROFILE.lock().unwrap();
    if let Some(last) = p.last_end {
        p.sums[0] += (t0 - last).as_secs_f64();
    }
    p.sums[1] += (t1 - t0).as_secs_f64();
    Ok(Some(t1))
}

pub fn profile_end(dev: &CudaDevice, begin: Option<Instant>) -> Result<()> {
    let Some(t1) = begin else { return Ok(()) };
    let t2 = Instant::now();
    sync(dev)?;
    let t3 = Instant::now();
    let mut p = PROFILE.lock().unwrap();
    p.sums[2] += (t2 - t1).as_secs_f64();
    p.sums[3] += (t3 - t2).as_secs_f64();
    p.calls += 1;
    p.last_end = Some(t3);
    if p.calls % 400 == 0 {
        let ms = p.sums.map(|s| 1e3 * s / p.calls as f64);
        crate::log_info!(
            "MoE offload profile ({} calls), avg ms per MoE call: gap {:.2}, gpu-wait {:.2}, host {:.2}, copy {:.2}",
            p.calls, ms[0], ms[1], ms[2], ms[3]
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot_ids(p: Plan) -> Vec<u32> {
        match p {
            Plan::Slots(t) => t.to_vec1().unwrap(),
            Plan::Staged => panic!("unexpected staging"),
        }
    }

    #[test]
    fn gpu_lru_keeps_current_experts_evicts_oldest_and_gathers() -> Result<()> {
        let Ok(Device::Cuda(dev)) = Device::new_cuda(0) else { return Ok(()) };
        let gpu = Device::Cuda(dev.clone());
        let cache = SlotCache::new(&dev, 4, 2)?;
        // 4 experts of 32 bytes in host memory, expert e filled with byte e+1.
        let bytes: Vec<u8> = (0..4u8).flat_map(|e| [e + 1; 32]).collect();
        let bank = super::super::host_banks::HostBank::new(&dev, &bytes, 4)?;
        let slots = Tensor::zeros((2, 32), DType::U8, &gpu)?;
        let ids = |v: &[u32]| Tensor::from_slice(v, v.len(), &gpu);
        let step = |v: &[u32], prefill: bool| -> Result<Plan> {
            let p = cache.plan(&ids(v)?, prefill)?;
            cache.gather(&[bank.dev_ptr], &[dev_ptr(&slots)?], &[32])?;
            Ok(p)
        };

        assert_eq!(slot_ids(step(&[0, 1], false)?), vec![0, 1]);
        // Hit on 1; 2 evicts the least recently used slot (expert 0 in slot 0).
        assert_eq!(slot_ids(step(&[1, 2], false)?), vec![1, 0]);
        // Repeated ids map to the same slot.
        assert_eq!(slot_ids(step(&[2, 2], false)?), vec![0, 0]);
        assert_eq!(cache.expert_of_slots()?, vec![2, 1]);
        // The gathered slot data matches the resident experts.
        let data: Vec<Vec<u8>> = slots.to_vec2()?;
        assert_eq!((data[0][0], data[0][31], data[1][0]), (3, 3, 2));
        // Prefill with more distinct experts than slots stages; decode stages by shape.
        assert!(matches!(step(&[0, 1, 3], true)?, Plan::Staged));
        assert!(matches!(step(&[0, 0, 0], false)?, Plan::Staged));
        Ok(())
    }

    #[test]
    fn slots_for_clamps_to_topk_and_experts() {
        assert_eq!(SlotCache::slots_for(0.01, 256, 8), 8);
        assert_eq!(SlotCache::slots_for(0.25, 256, 8), 64);
        assert_eq!(SlotCache::slots_for(0.999, 256, 8), 256);
    }
}
