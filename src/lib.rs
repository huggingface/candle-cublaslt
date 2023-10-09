pub use cudarc::cublaslt::Activation;

use candle::backend::BackendStorage;
use candle::cuda_backend::WrapErr;
use candle::{CpuStorage, Device, Layout, Result, Shape, Tensor};
use half::f16;
use std::sync::Arc;

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};

pub struct CublasLTMatmul {
    pub cublaslt: Arc<CudaBlasLT>,
    pub act: Option<Activation>,
}

#[derive(Debug, Clone)]
pub struct CublasLt(Arc<CudaBlasLT>);

impl CublasLt {
    pub fn new(device: &Device) -> Result<Self> {
        let dev = match &*device {
            Device::Cpu => candle::bail!("Not supported on CPU device"),
            Device::Cuda(d) => d,
        };

        let inner = CudaBlasLT::new(dev.cuda_device()).unwrap();

        Ok(Self(Arc::new(inner)))
    }
}

impl CublasLTMatmul {
    pub fn fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        a: &candle::CudaStorage,
        a_l: &Layout,
        b: &candle::CudaStorage,
        b_l: &Layout,
        bias: Option<&candle::CudaStorage>,
        bias_l: Option<&Layout>,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let dev = a.device();

        // Assume TN
        let m = a_l.dims()[0];
        let k = a_l.dims()[1];
        let n = b_l.dims()[0];

        if b_l.dims()[1] != k {
            candle::bail!("This layer only supports TN layout");
        }

        let lda = k;
        let ldb = k;
        let ldc = m;

        let out_shape = Shape::from((n, m));

        let config = MatmulConfig {
            transa: true,
            transb: false,
            m: m as u64,
            n: n as u64,
            k: k as u64,
            alpha: 1.0,
            lda: lda as i64,
            ldb: ldb as i64,
            beta: 0.0,
            ldc: ldc as i64,
        };

        let a = a.as_cuda_slice::<f16>()?.slice(a_l.start_offset()..);
        let b = b.as_cuda_slice::<f16>()?.slice(b_l.start_offset()..);

        let bias = if let (Some(bias), Some(bias_l)) = (bias, bias_l) {
            if bias_l.dims()[0] != m {
                candle::bail!("Bias does not have the correct shape");
            }

            Some(bias.as_cuda_slice::<f16>()?.slice(bias_l.start_offset()..))
        } else {
            None
        };

        // Allocate out tensor
        let mut out = unsafe { dev.alloc::<f16>(out_shape.elem_count()).w()? };

        unsafe {
            self.cublaslt
                .matmul(config, &a, &b, &mut out, bias.as_ref(), self.act.as_ref())
                .map_err(|e| candle::Error::Cuda(Box::new(e)))?;
        }

        let out = candle::CudaStorage::wrap_cuda_slice(out, dev.clone());

        Ok((out, out_shape))
    }
}

impl candle::CustomOp2 for CublasLTMatmul {
    fn name(&self) -> &'static str {
        "cublaslt-matmul"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for cublaslt-matmul")
    }

    fn cuda_fwd(
        &self,
        a: &candle::CudaStorage,
        a_l: &Layout,
        b: &candle::CudaStorage,
        b_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match a.dtype() {
            candle::DType::F16 => self.fwd::<f16>(a, a_l, b, b_l, None, None),
            candle::DType::BF16 => candle::bail!("not implemented"),
            dt => candle::bail!("flash-attn is only supported for f16 ({dt:?})"),
        }
    }
}

impl candle::CustomOp3 for CublasLTMatmul {
    fn name(&self) -> &'static str {
        "cublaslt-matmul-add"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for cublaslt-matmul")
    }

    fn cuda_fwd(
        &self,
        a: &candle::CudaStorage,
        a_l: &Layout,
        b: &candle::CudaStorage,
        b_l: &Layout,
        bias: &candle::CudaStorage,
        bias_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match a.dtype() {
            candle::DType::F16 => self.fwd::<f16>(a, a_l, b, b_l, Some(bias), Some(bias_l)),
            candle::DType::BF16 => candle::bail!("not implemented"),
            dt => candle::bail!("flash-attn is only supported for f16 ({dt:?})"),
        }
    }
}

/// Fused matmul + add + Relu/Gelu activation using CublasLt
///
/// # Arguments
///
/// * `a` - Input tensor of size MxK
/// * `b` - Input tensor of size NxK
/// * `bias` - Optional bias tensor of size M
/// * `act` - Optional Gelu or Relu activation
/// * `cublaslt` - CublasLt handle
///
/// The resulting tensor is of shape NxM
pub fn fused_matmul(
    a: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
    act: Option<Activation>,
    cublaslt: CublasLt,
) -> Result<Tensor> {
    let op = CublasLTMatmul {
        act,
        cublaslt: cublaslt.0,
    };

    if let Some(bias) = bias {
        a.apply_op3(&b, &bias, op)
    } else {
        a.apply_op2(&b, op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn to_vec2_round(t: Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
        let b = 10f32.powi(digits);
        let t = t.to_vec2::<f32>()?;
        let t = t
            .iter()
            .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
            .collect();
        Ok(t)
    }

    #[test]
    fn test_fused_matmul() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let a = Tensor::randn(0., 1., (8, 4), &device)?.to_dtype(DType::F16)?;
        let b = Tensor::randn(0., 1., (2, 4), &device)?.to_dtype(DType::F16)?;
        let bias = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F16)?;

        let cublaslt = CublasLt::new(&device)?;

        let res = fused_matmul(&a, &b, Some(&bias), None, cublaslt)?;
        let expected = (b.matmul(&a.t()?)? + bias.broadcast_left(2)?)?;

        assert_eq!(
            to_vec2_round(res.to_dtype(DType::F32)?, 2)?,
            to_vec2_round(expected.to_dtype(DType::F32)?, 2)?
        );
        Ok(())
    }
}
