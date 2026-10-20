//! Offloaded GGUF experts (`FusedMoeGGUF`): gate/up/down banks in host RAM,
//! `[num_slots, N, K]` slot QTensors on GPU.

use super::budget::{register, SlotPool};
use super::cache::{copy_htod, profile_begin, profile_end, Plan, SlotCache};
use super::host_banks::HostBank;
use super::prefill::staging_quant;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use candle_core::quantized::{cuda::QCudaStorage, GgmlDType, QStorage, QTensor};
use candle_core::{CudaDevice, Device, Result, Tensor};
use either::Either;
use std::sync::{Arc, RwLock};

/// Slot tensors and their LRU; replaced as a whole by `resize`.
struct Slots {
    cache: SlotCache,
    weights: [Arc<QTensor>; 3],
}

struct Inner {
    dev: CudaDevice,
    num_experts: usize,
    banks: [HostBank; 3],
    // (N, K, dtype) of gate, up, down.
    shapes: [(usize, usize, GgmlDType); 3],
    staging: [Arc<QTensor>; 3],
    slots: RwLock<Slots>,
}

pub struct GgufExpertOffload(Arc<Inner>);

const NAMES: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];

fn arr3<T>(v: Vec<T>) -> [T; 3] {
    v.try_into()
        .unwrap_or_else(|_| unreachable!("three expert weights"))
}

fn make_slots(
    dev: &CudaDevice,
    num_experts: usize,
    shapes: &[(usize, usize, GgmlDType); 3],
    num_slots: usize,
) -> Result<Slots> {
    let mut weights = Vec::with_capacity(3);
    for &(n, k, dtype) in shapes {
        let st = QCudaStorage::zeros(dev, num_slots * n * k, dtype)?;
        weights.push(Arc::new(QTensor::new(
            QStorage::Cuda(st),
            (num_slots, n, k),
        )?));
    }
    Ok(Slots {
        cache: SlotCache::new(dev, num_experts, num_slots)?,
        weights: arr3(weights),
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
        self.banks.iter().map(|b| b.expert_bytes).sum()
    }
    fn resize(&self, num_slots: usize) -> Result<()> {
        *self.slots.write().unwrap() =
            make_slots(&self.dev, self.num_experts, &self.shapes, num_slots)?;
        Ok(())
    }
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
            candle_core::bail!(
                "XINFER_MOE_OFFLOAD: packed ffn_gate_up_exps GGUF is not supported yet"
            );
        }

        let host = v.with_device(&Device::Cpu);
        let dims = [
            [num_experts, i, h],
            [num_experts, i, h],
            [num_experts, h, i],
        ];
        let mut experts = Vec::with_capacity(3);
        for role in 0..3 {
            experts.push(host.pp(NAMES[role]).get(dims[role].to_vec(), "weight")?);
        }
        let offload = Self::new(dev, rate, &arr3(experts), moe_cfg.num_experts_per_tok)?;
        host.clear_cache();
        Ok(Some(offload))
    }

    /// Offload for ISQ experts quantized at load time (`[E, N, K]` gate, up, down; may
    /// still be on the GPU). `None` unless `XINFER_MOE_OFFLOAD` applies here.
    pub fn for_isq(
        device: &Device,
        world_size: usize,
        experts: [&QTensor; 3],
        topk: usize,
    ) -> Result<Option<Self>> {
        let Some(rate) = crate::utils::env::moe_offload_rate() else {
            return Ok(None);
        };
        let Device::Cuda(dev) = device else {
            candle_core::bail!("XINFER_MOE_OFFLOAD requires a CUDA device");
        };
        if world_size > 1 {
            crate::log_warn!(
                "XINFER_MOE_OFFLOAD is ignored with tensor parallelism (world_size > 1)"
            );
            return Ok(None);
        }
        Self::new(dev, rate, &experts, topk).map(Some)
    }

    fn new<T: std::borrow::Borrow<QTensor>>(
        dev: &CudaDevice,
        rate: f32,
        experts: &[T; 3],
        topk: usize,
    ) -> Result<Self> {
        let (mut banks, mut shapes, mut staging) = (Vec::new(), Vec::new(), Vec::new());
        let num_experts = experts[0].borrow().shape().dims3()?.0;
        for (role, t) in experts.iter().enumerate() {
            let t = t.borrow();
            let (e, n, k) = t.shape().dims3()?;
            if !crate::models::layers::moe::moe_gemm_gguf_supports(t.dtype()) {
                candle_core::bail!(
                    "XINFER_MOE_OFFLOAD: {} dtype {:?} needs requantization, not supported yet",
                    NAMES[role],
                    t.dtype()
                );
            }
            banks.push(HostBank::new(dev, &t.data()?, e)?);
            shapes.push((n, k, t.dtype()));
            staging.push(staging_quant(dev, NAMES[role], t.dtype(), &[e, n, k])?);
        }
        let shapes = arr3(shapes);
        let num_slots = SlotCache::slots_for(rate, num_experts, topk);
        let slots = make_slots(dev, num_experts, &shapes, num_slots)?;
        let inner = Arc::new(Inner {
            dev: dev.clone(),
            num_experts,
            banks: arr3(banks),
            shapes,
            staging: arr3(staging),
            slots: RwLock::new(slots),
        });
        register(inner.clone());
        Ok(Self(inner))
    }

    /// One-block stand-ins for the layer's expert fields: unused while offloaded
    /// (forward reads `prepare`), and tiny so a later slot resize frees the old slots.
    pub fn placeholders(&self) -> Result<(Arc<QTensor>, Arc<QTensor>, Arc<QTensor>)> {
        let one = |i: usize| self.placeholder(i).map(Arc::new);
        Ok((one(0)?, one(1)?, one(2)?))
    }

    /// One-block stand-in for expert weight `i` (gate, up, down).
    pub fn placeholder(&self, i: usize) -> Result<QTensor> {
        let dtype = self.0.shapes[i].2;
        let st = QCudaStorage::zeros(&self.0.dev, dtype.block_size(), dtype)?;
        QTensor::new(QStorage::Cuda(st), (1, 1, dtype.block_size()))
    }

    /// Makes the routed experts resident. Returns the ids to use (slot ids, or the
    /// original ids with full-layer staging) and the weights the GEMMs must read.
    pub fn prepare(
        &self,
        topk_ids: Tensor,
        is_prefill: bool,
    ) -> Result<(Tensor, [Arc<QTensor>; 3])> {
        let o = &self.0;
        let s = o.slots.read().unwrap();
        let t = profile_begin(&o.dev)?;
        let out = match s.cache.plan(&topk_ids, is_prefill)? {
            Plan::Staged => {
                for (bank, dst) in o.banks.iter().zip(&o.staging) {
                    copy_htod(&o.dev, dst.device_ptr()? as u64, bank.all())?;
                }
                (topk_ids, o.staging.clone())
            }
            Plan::Slots(ids) => {
                let mut dst = [0u64; 3];
                for (d, w) in dst.iter_mut().zip(&s.weights) {
                    *d = w.device_ptr()? as u64;
                }
                let src = o.banks.each_ref().map(|b| b.dev_ptr);
                let bytes = o.banks.each_ref().map(|b| b.expert_bytes as u64);
                s.cache.gather(&src, &dst, &bytes)?;
                (ids, s.weights.clone())
            }
        };
        profile_end(&o.dev, t)?;
        Ok(out)
    }
}
