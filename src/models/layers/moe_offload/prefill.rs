//! Prefill: when a call routes to more experts than a layer has slots, the whole
//! expert layer is copied into a full-size staging buffer shared by all layers.

use candle_core::quantized::{cuda::QCudaStorage, GgmlDType, QStorage, QTensor};
use candle_core::{CudaDevice, Result};
use std::sync::{Arc, Mutex};

type StagingKey = (usize, GgmlDType, Vec<usize>);

// shortcut: single device (offload requires world_size == 1), upgrade to a per-device map for TP.
static STAGING: Mutex<Vec<(StagingKey, Arc<QTensor>)>> = Mutex::new(Vec::new());

/// Staging buffer for weight `role` (0 gate, 1 up, 2 down) of this dtype and shape.
/// Allocated at load time so the KV-cache planner sees it as used VRAM.
pub fn staging(dev: &CudaDevice, role: usize, dtype: GgmlDType, dims: &[usize]) -> Result<Arc<QTensor>> {
    let key = (role, dtype, dims.to_vec());
    let mut all = STAGING.lock().unwrap();
    if let Some((_, t)) = all.iter().find(|(k, _)| *k == key) {
        return Ok(t.clone());
    }
    let elems = dims.iter().product();
    let t = Arc::new(QTensor::new(QStorage::Cuda(QCudaStorage::zeros(dev, elems, dtype)?), dims)?);
    all.push((key, t.clone()));
    Ok(t)
}
