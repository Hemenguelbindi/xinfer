//! Prefill: when a call routes to more experts than a layer has slots, the whole
//! expert layer is copied into a full-size staging buffer shared by all layers.
//! Buffers are allocated at load time so the KV-cache planner sees them as used VRAM.

use candle_core::quantized::{cuda::QCudaStorage, GgmlDType, QStorage, QTensor};
use candle_core::{CudaDevice, DType, Device, Result, Tensor};
use std::sync::{Arc, Mutex};

type Key = (String, Vec<usize>);

// shortcut: single device (offload requires world_size == 1), upgrade to a per-device map for TP.
static QUANT: Mutex<Vec<(Key, Arc<QTensor>)>> = Mutex::new(Vec::new());
static BYTES: Mutex<Vec<(Key, Tensor)>> = Mutex::new(Vec::new());

fn get_or_make<T: Clone>(
    store: &Mutex<Vec<(Key, T)>>,
    key: Key,
    make: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let mut all = store.lock().unwrap();
    if let Some((_, t)) = all.iter().find(|(k, _)| *k == key) {
        return Ok(t.clone());
    }
    let t = make()?;
    all.push((key, t.clone()));
    Ok(t)
}

/// GGUF staging buffer for weight `role` with this dtype and `[E, N, K]` shape.
pub fn staging_quant(
    dev: &CudaDevice,
    role: &str,
    dtype: GgmlDType,
    dims: &[usize],
) -> Result<Arc<QTensor>> {
    get_or_make(&QUANT, (format!("{role}/{dtype:?}"), dims.to_vec()), || {
        let st = QCudaStorage::zeros(dev, dims.iter().product(), dtype)?;
        Ok(Arc::new(QTensor::new(QStorage::Cuda(st), dims)?))
    })
}

/// Staging buffer for weight `role` with shape `dims` and `dtype`.
pub fn staging(dev: &CudaDevice, role: &str, dims: &[usize], dtype: DType) -> Result<Tensor> {
    get_or_make(&BYTES, (format!("{role}/{dtype:?}"), dims.to_vec()), || {
        Tensor::zeros(dims, dtype, &Device::Cuda(dev.clone()))
    })
}
