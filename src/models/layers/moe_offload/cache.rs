//! GPU slot table and LRU shared by every offloaded format.
//!
//! shortcut: routing ids are read back to the host each MoE call (one sync per layer),
//! so CUDA graphs are disabled while offload is on; upgrade to a device-side
//! ensure/copy kernel in attention.rs for graph capture.

use candle_core::cuda_backend::cudarc::driver::{result, DevicePtr};
use candle_core::{CudaDevice, Result, Storage, Tensor};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static STAGED: AtomicU64 = AtomicU64::new(0);

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
/// since the previous call, leftover GPU work, host LRU and PCIe copy. Adds two device
/// syncs per call, so absolute speed drops a little; use it only to find the bottleneck.
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

/// Where one MoE call's experts live.
pub enum Placement {
    /// Ids rewritten to slots; `misses` are (expert, slot) pairs the caller must copy in.
    Slots { slot_ids: Vec<u32>, misses: Vec<(usize, usize)> },
    /// More distinct experts than slots: use the full-layer staging buffers.
    Staged,
}

struct Lru {
    slot_of: Vec<i32>,   // expert -> slot, -1 if not resident
    expert_of: Vec<i32>, // slot -> expert, -1 if empty
    last_used: Vec<u64>,
    tick: u64,
}

pub struct SlotLru {
    pub num_slots: usize,
    state: Mutex<Lru>,
}

impl SlotLru {
    pub fn new(num_experts: usize, num_slots: usize) -> Self {
        Self {
            num_slots,
            state: Mutex::new(Lru {
                slot_of: vec![-1; num_experts],
                expert_of: vec![-1; num_slots],
                last_used: vec![0; num_slots],
                tick: 0,
            }),
        }
    }

    /// Slots per layer for `XINFER_MOE_OFFLOAD=rate`: at least one decode step's experts.
    pub fn slots_for(rate: f32, num_experts: usize, topk: usize) -> usize {
        ((num_experts as f32 * rate).ceil() as usize).clamp(topk, num_experts)
    }

    pub fn place(&self, ids: &[u32]) -> Placement {
        let mut uniq = ids.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        if uniq.len() > self.num_slots {
            STAGED.fetch_add(1, Ordering::Relaxed);
            return Placement::Staged;
        }
        let mut lru = self.state.lock().unwrap();
        lru.tick += 1;
        let tick = lru.tick;
        let mut missing = Vec::new();
        for &e in &uniq {
            match lru.slot_of[e as usize] {
                s if s >= 0 => lru.last_used[s as usize] = tick,
                _ => missing.push(e as usize),
            }
        }
        let mut misses = Vec::with_capacity(missing.len());
        for e in missing {
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
            misses.push((e, s));
        }
        let slot_ids = ids.iter().map(|&e| lru.slot_of[e as usize] as u32).collect();
        drop(lru);
        log_stats(uniq.len() - misses.len(), misses.len());
        Placement::Slots { slot_ids, misses }
    }

    /// Expert held by each slot (-1 when empty).
    pub fn expert_of_slots(&self) -> Vec<i32> {
        self.state.lock().unwrap().expert_of.clone()
    }
}

/// Async host->device copy on candle's stream, so it is ordered with the kernels.
pub fn copy_htod(dev: &CudaDevice, dst: u64, src: &[u8]) -> Result<()> {
    unsafe { result::memcpy_htod_async(dst, src, *dev.cu_stream()) }.map_err(candle_core::Error::wrap)
}

/// Device address of a contiguous u8 CUDA tensor.
pub fn u8_ptr(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        Storage::Cuda(c) => Ok(*c.as_cuda_slice::<u8>()?.device_ptr() + layout.start_offset() as u64),
        _ => candle_core::bail!("expected a CUDA u8 tensor"),
    }
}

fn log_stats(hits: usize, misses: usize) {
    let n = (hits + misses) as u64;
    let h = HITS.fetch_add(hits as u64, Ordering::Relaxed) + hits as u64;
    let m = MISSES.fetch_add(misses as u64, Ordering::Relaxed) + misses as u64;
    let total = h + m;
    if total / 16384 != (total - n) / 16384 {
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

    fn slots(p: Placement) -> (Vec<u32>, Vec<(usize, usize)>) {
        match p {
            Placement::Slots { slot_ids, misses } => (slot_ids, misses),
            Placement::Staged => panic!("unexpected staging"),
        }
    }

    #[test]
    fn lru_keeps_current_experts_and_evicts_oldest() {
        let lru = SlotLru::new(4, 2);
        assert_eq!(slots(lru.place(&[0, 1])), (vec![0, 1], vec![(0, 0), (1, 1)]));
        // Hit on 1; 2 evicts the least recently used slot (expert 0 in slot 0).
        assert_eq!(slots(lru.place(&[1, 2])), (vec![1, 0], vec![(2, 0)]));
        // Repeated ids map to the same slot, no copies.
        assert_eq!(slots(lru.place(&[2, 2, 1])), (vec![0, 0, 1], vec![]));
        assert_eq!(lru.expert_of_slots(), vec![2, 1]);
        // Three distinct experts exceed 2 slots.
        assert!(matches!(lru.place(&[0, 1, 3]), Placement::Staged));
    }

    #[test]
    fn slots_for_clamps_to_topk_and_experts() {
        assert_eq!(SlotLru::slots_for(0.01, 256, 8), 8);
        assert_eq!(SlotLru::slots_for(0.25, 256, 8), 64);
        assert_eq!(SlotLru::slots_for(0.999, 256, 8), 256);
    }
}
