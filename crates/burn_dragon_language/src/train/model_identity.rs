use std::collections::BTreeMap;

use anyhow::{Context, Result};
use burn::{module::Module, tensor::backend::Backend};
use burn_store::{Collector, TensorSnapshot};
use sha2::{Digest, Sha256};

pub(crate) const MODEL_TENSOR_FINGERPRINT_SCHEMA: &str = "burn-dragon-model-tensors-f32-v1";

/// Eagerly realizes lazy Burn parameters at an explicit RNG boundary.
///
/// Cloning an uninitialized module clones its initialization closures, so each
/// copy may otherwise draw an independent parameter set when first used.
pub(crate) fn materialize_model_parameters<B, M>(module: &M)
where
    B: Backend,
    M: Module<B>,
{
    let mut collector = Collector::default();
    module.visit(&mut collector);
}

/// Returns a deterministic digest of every floating-point parameter tensor.
///
/// Tensor values are transferred one parameter at a time, so host memory is
/// bounded by the largest parameter rather than the full model size.
pub(crate) fn model_tensor_fingerprint<B, M>(module: &M) -> Result<String>
where
    B: Backend,
    M: Module<B>,
{
    let mut collector = Collector::default();
    module.visit(&mut collector);
    let snapshots = collector
        .into_tensors()
        .into_iter()
        .filter(|snapshot| snapshot.dtype.is_float())
        .map(|snapshot| (snapshot.full_path(), snapshot))
        .collect::<BTreeMap<String, TensorSnapshot>>();

    let mut digest = Sha256::new();
    hash_bytes(&mut digest, MODEL_TENSOR_FINGERPRINT_SCHEMA.as_bytes());
    hash_u64(&mut digest, snapshots.len() as u64);
    for (path, snapshot) in snapshots {
        hash_bytes(&mut digest, path.as_bytes());
        hash_u64(&mut digest, snapshot.shape.len() as u64);
        for dimension in snapshot.shape.iter() {
            hash_u64(&mut digest, *dimension as u64);
        }
        let data = snapshot
            .to_data()
            .with_context(|| format!("read model parameter tensor {path}"))?;
        hash_u64(&mut digest, data.num_elements() as u64);
        for value in data.iter::<f64>() {
            digest.update((value as f32).to_bits().to_le_bytes());
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn hash_bytes(digest: &mut Sha256, value: &[u8]) {
    hash_u64(digest, value.len() as u64);
    digest.update(value);
}

fn hash_u64(digest: &mut Sha256, value: u64) {
    digest.update(value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::NdArray,
        module::Module,
        nn::{Linear, LinearConfig},
        tensor::{Device, backend::Backend},
    };

    use super::{materialize_model_parameters, model_tensor_fingerprint};

    type TestBackend = NdArray<f32>;

    #[derive(Module, Debug)]
    struct FingerprintModel<B: Backend> {
        projection: Linear<B>,
    }

    fn model(seed: u64, device: &Device<TestBackend>) -> FingerprintModel<TestBackend> {
        TestBackend::seed(device, seed);
        let model = FingerprintModel {
            projection: LinearConfig::new(5, 3).init(device),
        };
        materialize_model_parameters::<TestBackend, _>(&model);
        model
    }

    #[test]
    fn model_tensor_fingerprint_is_seed_stable_and_value_sensitive() {
        let device = Device::<TestBackend>::default();
        let first = model(17, &device);
        let repeated = model(17, &device);
        let changed = model(19, &device);

        let first = model_tensor_fingerprint::<TestBackend, _>(&first).unwrap();
        let repeated = model_tensor_fingerprint::<TestBackend, _>(&repeated).unwrap();
        let changed = model_tensor_fingerprint::<TestBackend, _>(&changed).unwrap();

        assert_eq!(first, repeated);
        assert_ne!(first, changed);
    }
}
