use super::{recurrent_attention_dense_score_reference, recurrent_attention_reference};
use crate::kernel::linear_attention::{default_alibi_slopes, reference_alibi_slopes};
use burn::tensor::{Tensor, TensorData};

type B = burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>>;

#[test]
fn alibi_long_lag_carry_and_vjp_match_analytic_decay_across_chunks() {
    let device = Default::default();
    let time = 257;
    for slopes in [default_alibi_slopes(4), reference_alibi_slopes(4)] {
        let decay = Tensor::<B, 1>::from_data(
            TensorData::new(slopes.iter().map(|s| (-s).exp()).collect::<Vec<_>>(), [4]),
            &device,
        );
        let query = Tensor::<B, 4>::ones([1, 4, time, 1], &device);
        let value = Tensor::<B, 4>::zeros([1, 1, time, 1], &device);
        let initial = Tensor::<B, 4>::ones([1, 4, 1, 1], &device).require_grad();
        let (dense, _) = recurrent_attention_dense_score_reference(
            query.clone(),
            value.clone(),
            Some(initial.clone()),
            Some(decay.clone()),
            512,
            None,
        );
        let (tokenwise, _) = recurrent_attention_reference(
            query.clone(),
            value.clone(),
            Some(initial.clone()),
            Some(decay.clone()),
        );
        let mut state = initial.clone();
        let mut parts = Vec::new();
        for (start, end) in [(0, 64), (64, 256), (256, time)] {
            let (context, next) = recurrent_attention_dense_score_reference(
                query.clone().slice_dim(2, start..end),
                value.clone().slice_dim(2, start..end),
                Some(state),
                Some(decay.clone()),
                512,
                None,
            );
            state = next;
            parts.push(context);
        }
        let chunked = Tensor::cat(parts, 2);
        for output in [dense, tokenwise, chunked] {
            let terminal = output.slice_dim(2, time - 1..time);
            let values = terminal.clone().into_data().to_vec::<f32>().unwrap();
            let gradients = terminal.sum().backward();
            let vjp = initial
                .grad(&gradients)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            for head in 0..4 {
                let expected = (-(slopes[head] as f64) * 256.0).exp();
                for actual in [values[head], vjp[head]] {
                    assert!(
                        (actual as f64 - expected).abs() < 1e-6 + 1e-5 * expected,
                        "head={head}, slope={}, actual={actual}, expected={expected}",
                        slopes[head]
                    );
                }
            }
        }
    }
}
