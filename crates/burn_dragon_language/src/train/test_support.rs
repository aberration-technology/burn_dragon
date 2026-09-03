use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// Replace backend-random matrix parameters with a deterministic test fixture.
///
/// Burn's NdArray seed is process-global, so parallel tests that reseed it can
/// interleave model initialization. Exact numerical-contract tests use this
/// mapper to keep their parameter values independent of test scheduling.
pub(crate) fn deterministic_matrix_parameters<B, M>(module: M) -> M
where
    B: Backend,
    M: Module<B>,
{
    module.map(&mut DeterministicMatrixMapper { parameter_index: 0 })
}

struct DeterministicMatrixMapper {
    parameter_index: u64,
}

impl<B: Backend> ModuleMapper<B> for DeterministicMatrixMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if D < 2 {
            return param;
        }

        let (id, tensor, mapper) = param.consume();
        let require_grad = tensor.is_require_grad();
        let dims = tensor.shape().dims::<D>();
        let elements = dims.iter().product::<usize>();
        let device = tensor.device();
        let parameter_seed = self.parameter_index.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.parameter_index = self.parameter_index.saturating_add(1);
        let mut values = (0..elements)
            .map(|index| {
                let mixed = splitmix64(parameter_seed ^ index as u64);
                let unit = (mixed >> 40) as f32 / ((1u32 << 24) - 1) as f32;
                unit * 2.0 - 1.0
            })
            .collect::<Vec<_>>();
        let fixture_rms = (values.iter().map(|value| value * value).sum::<f32>()
            / elements.max(1) as f32)
            .sqrt()
            .max(1.0e-12);
        let scale = 0.02 / fixture_rms;
        values.iter_mut().for_each(|value| *value *= scale);
        let tensor = Tensor::from_data(TensorData::new(values, dims), &device);
        let tensor = if require_grad {
            tensor.require_grad()
        } else {
            tensor
        };
        Param::from_mapped_value(id, tensor, mapper)
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
