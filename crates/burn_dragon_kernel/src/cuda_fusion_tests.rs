//! Regressions for dependency fusion contracts used by Dragon training.

use burn::tensor::{Tensor, TensorData, activation};
use burn_autodiff::Autodiff;
use burn_cubecl::{CubeBackend, cubecl::cuda::CudaRuntime};

type Direct = Autodiff<CubeBackend<CudaRuntime, f32, i32, u8>>;
type Fused = Autodiff<burn_cuda::Cuda<f32>>;

fn data(rows: usize, cols: usize, salt: usize) -> TensorData {
    TensorData::new(
        (0..rows * cols)
            .map(|i| ((i * 17 + salt) % 113) as f32 / 113.0 - 0.5)
            .collect::<Vec<_>>(),
        [rows, cols],
    )
}

fn run<B: burn::tensor::backend::AutodiffBackend>(
    input: TensorData,
    weight: TensorData,
) -> [Vec<f32>; 4] {
    let device = Default::default();
    let x = Tensor::<B, 2>::from_data(input, &device).require_grad();
    let w = Tensor::<B, 2>::from_data(weight, &device).require_grad();
    let y = x.clone().matmul(w.clone());
    let mask = y.clone().greater_elem(0.0);
    let activated = activation::relu(y.clone());
    let grads = activated.clone().square().mean().backward();
    let values = |data: TensorData| data.convert::<f32>().into_vec::<f32>().unwrap();
    [
        values(activated.into_data()),
        values(mask.float().into_data()),
        values(x.grad(&grads).unwrap().into_data()),
        values(w.grad(&grads).unwrap().into_data()),
    ]
}

#[test]
fn cuda_fused_matmul_byte_mask_matches_direct_values_and_gradients() {
    // Power-of-two dimensions admit the invalid 16-wide choice; 96-wide tests do not.
    for (rows, inner, cols) in [(256, 256, 256), (256, 256, 4096), (256, 4096, 256)] {
        let x = data(rows, inner, 5);
        let w = data(inner, cols, 11);
        let expected = run::<Direct>(x.clone(), w.clone());
        let actual = run::<Fused>(x, w);
        for (slot, (expected, actual)) in expected.into_iter().zip(actual).enumerate() {
            assert_eq!(expected.len(), actual.len());
            for (index, (expected, actual)) in expected.into_iter().zip(actual).enumerate() {
                let tolerance = if slot == 1 {
                    0.0
                } else {
                    2e-4 + 2e-4 * expected.abs()
                };
                assert!(
                    actual.is_finite() && (expected - actual).abs() <= tolerance,
                    "shape=({rows},{inner},{cols}) slot={slot} index={index}: {actual} vs {expected}"
                );
            }
        }
    }
}

fn long_lag_attention<B: burn::tensor::backend::AutodiffBackend>()
where
    B::FloatTensorPrimitive: 'static,
{
    use crate::api::attention::try_fused_dense_causal_attention_wgpu;

    let device = Default::default();
    let time = 257;
    let query = Tensor::<B, 4>::ones([1, 4, time, 1], &device);
    let mut pulse = vec![0.0f32; time];
    pulse[0] = 1.0;
    let value =
        Tensor::<B, 4>::from_data(TensorData::new(pulse, [1, 1, time, 1]), &device).require_grad();
    let slopes = [0.25f32, 0.0625, 0.015625, 0.00390625];
    let decay = Tensor::<B, 1>::from_data(
        TensorData::new(slopes.map(|slope| (-slope).exp()).to_vec(), [4]),
        &device,
    );
    let output = try_fused_dense_causal_attention_wgpu(&query, &value, &decay)
        .expect("CUDA fused attention must execute the long-lag contract")
        .slice_dim(2, time - 1..time);
    let actual = output.clone().into_data().to_vec::<f32>().unwrap();
    let grads = output.sum().backward();
    let input_grad = value
        .grad(&grads)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let expected = slopes.map(|slope| (-(slope as f64) * 256.0).exp());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (*actual as f64 - expected).abs() < 2e-5,
            "long-lag context {actual} vs {expected}"
        );
    }
    let expected_grad = expected.iter().sum::<f64>();
    // The VJP's CubeK matmul stages f32 operands as TF32 on CUDA (10 fraction
    // bits). Bound that rounding separately from the f32 forward kernel above.
    let grad_tolerance = 2e-5 + expected_grad * 5e-4;
    assert!(
        (input_grad[0] as f64 - expected_grad).abs() < grad_tolerance,
        "long-lag value VJP {} vs {expected_grad} ({})",
        input_grad[0],
        std::any::type_name::<B>(),
    );
}

#[test]
fn cuda_alibi_long_lag_pulse_and_vjp_match_analytic_decay() {
    long_lag_attention::<Direct>();
    long_lag_attention::<Fused>();
}
