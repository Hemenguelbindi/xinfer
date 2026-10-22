//! Per-layer expert weights in pinned, device-mapped host memory: the gather kernel
//! reads missing experts straight over PCIe, and full-layer staging copies use DMA.

use candle_core::cuda_backend::cudarc::driver::{result, sys};
use candle_core::{CudaDevice, Result};

/// Raw bytes of one `[num_experts, ...]` expert tensor kept in host RAM.
/// Experts are contiguous, so expert `e` is `[e * expert_bytes, (e + 1) * expert_bytes)`.
pub struct HostBank {
    ptr: *mut u8,
    len: usize,
    /// Device-side address of the same memory (for kernels).
    pub dev_ptr: u64,
    pub expert_bytes: usize,
}

// Read-only after construction; the pinned allocation is owned by this struct.
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
        dev.bind_to_thread().map_err(candle_core::Error::wrap)?;
        // Keep the host bank cache-coherent.  These bytes are also read through the
        // device mapping by the expert-cache gather kernel; write-combined mappings can
        // expose partially ordered host writes to that path on some CUDA platforms.
        let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP;
        let ptr = unsafe { result::malloc_host(bytes.len(), flags) }.map_err(|e| {
            candle_core::Error::Msg(format!(
                "XINFER_MOE_OFFLOAD: pinning {:.1} MB of host memory failed ({e}); \
                 offload needs pinned RAM for every expert",
                bytes.len() as f64 / 1e6
            ))
        })? as *mut u8;
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
        let mut dev_ptr: sys::CUdeviceptr = 0;
        let mapped = unsafe { sys::lib().cuMemHostGetDevicePointer_v2(&mut dev_ptr, ptr as _, 0) };
        // Built before the check so an error still frees the pinned memory.
        let mut bank = Self {
            ptr,
            len: bytes.len(),
            dev_ptr: 0,
            expert_bytes: bytes.len() / num_experts,
        };
        mapped.result().map_err(candle_core::Error::wrap)?;
        bank.dev_ptr = dev_ptr;
        Ok(bank)
    }

    pub fn all(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for HostBank {
    fn drop(&mut self) {
        let _ = unsafe { result::free_host(self.ptr as _) };
    }
}
