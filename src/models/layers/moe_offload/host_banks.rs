//! Per-layer expert weights in pinned host memory (source of PCIe copies).

use candle_core::cuda_backend::cudarc::driver::{result, sys};
use candle_core::{CudaDevice, Result};

/// Raw bytes of one `[num_experts, N, K]` GGUF expert tensor kept in host RAM.
/// Experts are contiguous, so expert `e` is `[e * expert_bytes, (e + 1) * expert_bytes)`.
pub struct HostBank {
    ptr: *mut u8,
    len: usize,
    pinned: bool,
    pageable: Vec<u8>,
    pub expert_bytes: usize,
}

// Read-only after construction; the pinned pointer is owned by this struct.
unsafe impl Send for HostBank {}
unsafe impl Sync for HostBank {}

impl HostBank {
    pub fn new(dev: &CudaDevice, bytes: &[u8], num_experts: usize) -> Result<Self> {
        if bytes.len() % num_experts != 0 {
            candle_core::bail!(
                "expert bank of {} bytes is not divisible into {num_experts} experts",
                bytes.len()
            );
        }
        let expert_bytes = bytes.len() / num_experts;
        dev.bind_to_thread().map_err(candle_core::Error::wrap)?;
        // Write-combined pinned memory: written once here, then only read by DMA.
        match unsafe { result::malloc_host(bytes.len(), sys::CU_MEMHOSTALLOC_WRITECOMBINED) } {
            Ok(p) => {
                let ptr = p as *mut u8;
                unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
                Ok(Self { ptr, len: bytes.len(), pinned: true, pageable: Vec::new(), expert_bytes })
            }
            Err(e) => {
                crate::log_warn!(
                    "MoE offload: pinning {:.1} MB failed ({e}); using pageable memory (slower copies)",
                    bytes.len() as f64 / 1e6
                );
                let pageable = bytes.to_vec();
                Ok(Self { ptr: std::ptr::null_mut(), len: 0, pinned: false, pageable, expert_bytes })
            }
        }
    }

    pub fn all(&self) -> &[u8] {
        if self.pinned {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        } else {
            &self.pageable
        }
    }

    pub fn expert(&self, e: usize) -> &[u8] {
        &self.all()[e * self.expert_bytes..(e + 1) * self.expert_bytes]
    }
}

impl Drop for HostBank {
    fn drop(&mut self) {
        if self.pinned {
            let _ = unsafe { result::free_host(self.ptr as _) };
        }
    }
}
