//! A bias-free projection that is either a dense candle [`Linear`] or a
//! weight held in another format and run by its own kernels.
//!
//! The second form exists so a model can take its dense weights from a
//! quantized file (a GGUF's Q8_0 blocks, say) while keeping the
//! checkpoint's names and the rest of its load path. A [`LinearSource`]
//! is asked for each projection by its checkpoint name; when it has one,
//! the dense tensor is never read.

use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;

/// `y = x W^T` for a weight held outside candle.
pub trait ExternalLinear: Send + Sync {
    /// `x` is `(.., in_dim)`; the result is `(.., out_dim)` in `x`'s dtype.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor>;
}

/// Supplies projections by checkpoint name, e.g.
/// `model.language_model.layers.3.self_attn.q_proj.weight`.
pub trait LinearSource: Send + Sync {
    /// The projection, or `None` to load the checkpoint's own tensor.
    /// `(out_dim, in_dim)` is the shape the checkpoint would have.
    fn linear(
        &self,
        name: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Option<Arc<dyn ExternalLinear>>>;
}

#[derive(Clone)]
pub enum Proj {
    Dense(Linear),
    External(Arc<dyn ExternalLinear>),
}

impl Module for Proj {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(x),
            Self::External(linear) => linear.forward(x),
        }
    }
}

impl From<Linear> for Proj {
    fn from(linear: Linear) -> Self {
        Self::Dense(linear)
    }
}

/// `{prefix}.{name}.weight`, from `source` when it has it, else from the
/// checkpoint.
pub fn load_proj(
    vb: &ShardedVarBuilder,
    name: &str,
    in_dim: usize,
    out_dim: usize,
    source: Option<&dyn LinearSource>,
) -> Result<Proj> {
    if let Some(source) = source {
        let prefix = vb.prefix();
        let full = if prefix.is_empty() {
            format!("{name}.weight")
        } else {
            format!("{prefix}.{name}.weight")
        };
        if let Some(linear) = source.linear(&full, in_dim, out_dim)? {
            return Ok(Proj::External(linear));
        }
    }
    let weight = vb
        .pp(name)
        .get((out_dim, in_dim), "weight")
        .with_context(|| format!("load '{}/{name}/weight'", vb.prefix()))?;
    Ok(Proj::Dense(Linear::new(weight, None)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct Doubling;
    impl ExternalLinear for Doubling {
        fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
            x * 2.0
        }
    }

    /// Records the names it is asked for and serves only one of them.
    struct OneName(Mutex<Vec<String>>);
    impl LinearSource for OneName {
        fn linear(
            &self,
            name: &str,
            _in_dim: usize,
            _out_dim: usize,
        ) -> Result<Option<Arc<dyn ExternalLinear>>> {
            self.0.lock().unwrap().push(name.to_string());
            Ok(
                (name == "a.b.served.weight")
                    .then(|| Arc::new(Doubling) as Arc<dyn ExternalLinear>),
            )
        }
    }

    /// The source is asked by the full checkpoint name; what it serves
    /// replaces the tensor, which is never read (the checkpoint here has
    /// none for it), and what it declines comes from the checkpoint.
    #[test]
    fn a_source_is_asked_by_full_name_and_the_checkpoint_fills_the_rest() {
        let dev = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "a.b.kept.weight".to_string(),
            Tensor::ones((2, 2), DType::F32, &dev).unwrap(),
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        candle_core::safetensors::save(&tensors, &path).unwrap();
        let vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(&[&path], DType::F32, &dev)
                .unwrap()
        }
        .pp("a")
        .pp("b");
        let source = OneName(Mutex::new(Vec::new()));
        let x = Tensor::new(&[[1f32, 3.]], &dev).unwrap();

        let served = load_proj(&vb, "served", 2, 2, Some(&source)).unwrap();
        assert!(matches!(served, Proj::External(_)));
        assert_eq!(
            served.forward(&x).unwrap().to_vec2::<f32>().unwrap(),
            [[2., 6.]]
        );

        let kept = load_proj(&vb, "kept", 2, 2, Some(&source)).unwrap();
        assert!(matches!(kept, Proj::Dense(_)));
        assert_eq!(
            kept.forward(&x).unwrap().to_vec2::<f32>().unwrap(),
            [[4., 4.]]
        );

        assert_eq!(
            *source.0.lock().unwrap(),
            ["a.b.served.weight", "a.b.kept.weight"]
        );
    }
}
