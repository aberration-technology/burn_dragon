use burn::tensor::Tensor;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn_dragon_kernel::api::projection::{
    LowrankGradInputExecutor, try_fused_relu_lowrank_projection_wgpu_with_executor,
};

use super::block_sparse::BlockPattern1d;

fn single_stream_projection_flat<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
) -> Option<Tensor<B, 4>> {
    let [batch, streams, time, embd] = input.shape().dims::<4>();
    let [weight_batch, heads, weight_embd, latent] = weight.shape().dims::<4>();
    if streams != 1 || weight_batch != 1 || embd != weight_embd {
        return None;
    }

    let input_flat = input.reshape([batch * time, embd]);
    let weight_flat = weight
        .reshape([heads, embd, latent])
        .swap_dims(0, 1)
        .reshape([embd, heads * latent]);
    let projected = input_flat.matmul(weight_flat);
    Some(
        projected
            .reshape([batch, time, heads, latent])
            .swap_dims(1, 2),
    )
}

fn head_aligned_projection_flat<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
) -> Option<Tensor<B, 4>> {
    let [batch, heads, time, embd] = input.shape().dims::<4>();
    let [weight_batch, weight_heads, weight_embd, latent] = weight.shape().dims::<4>();
    if weight_batch != 1 || heads != weight_heads || embd != weight_embd {
        return None;
    }

    let input_by_head = input.swap_dims(0, 1).reshape([heads, batch * time, embd]);
    let weight_by_head = weight.reshape([heads, embd, latent]);
    let projected = input_by_head.matmul(weight_by_head);
    Some(
        projected
            .reshape([heads, batch, time, latent])
            .swap_dims(0, 1),
    )
}

fn grouped_single_stream_projection_flat<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
) -> Option<Tensor<B, 4>> {
    let [batch, streams, time, embd] = input.shape().dims::<4>();
    let [weight_batch, heads, weight_embd, latent] = weight.shape().dims::<4>();
    if streams != 1 || weight_batch <= 1 || batch % weight_batch != 0 || embd != weight_embd {
        return None;
    }
    let per_weight_batch = batch / weight_batch;
    let input_grouped = input.reshape([weight_batch, per_weight_batch * time, embd]);
    let weight_grouped = weight
        .swap_dims(1, 2)
        .reshape([weight_batch, embd, heads * latent]);
    Some(
        input_grouped
            .matmul(weight_grouped)
            .reshape([weight_batch, per_weight_batch, time, heads, latent])
            .swap_dims(2, 3)
            .reshape([batch, heads, time, latent]),
    )
}

fn grouped_head_aligned_projection_flat<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
) -> Option<Tensor<B, 4>> {
    let [batch, input_heads, time, embd] = input.shape().dims::<4>();
    let [weight_batch, heads, weight_embd, latent] = weight.shape().dims::<4>();
    if input_heads != heads || weight_batch <= 1 || batch % weight_batch != 0 || embd != weight_embd
    {
        return None;
    }
    let per_weight_batch = batch / weight_batch;
    let input_grouped = input
        .reshape([weight_batch, per_weight_batch, heads, time, embd])
        .swap_dims(1, 2)
        .reshape([weight_batch, heads, per_weight_batch * time, embd]);
    Some(
        input_grouped
            .matmul(weight)
            .reshape([weight_batch, heads, per_weight_batch, time, latent])
            .swap_dims(1, 2)
            .reshape([batch, heads, time, latent]),
    )
}

pub fn reference_forward<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 3>>,
    threshold: f32,
    layout: &BlockPattern1d,
    sparse_mask: Option<Tensor<B, 4>>,
) -> Tensor<B, 4> {
    let device = input.device();
    let latent = weight.shape().dims::<4>()[3];

    let mut projected = single_stream_projection_flat(input.clone(), weight.clone())
        .or_else(|| head_aligned_projection_flat(input.clone(), weight.clone()))
        .or_else(|| grouped_single_stream_projection_flat(input.clone(), weight.clone()))
        .or_else(|| grouped_head_aligned_projection_flat(input.clone(), weight.clone()))
        .unwrap_or_else(|| input.matmul(weight));

    if let Some(bias) = bias {
        let dims = bias.shape().dims::<3>();
        let bias = bias.reshape([1, dims[0], 1, dims[2]]);
        projected = projected + bias;
    }

    if threshold != 0.0 {
        projected = projected.sub_scalar(threshold);
    }

    let mut activated = activation::relu(projected);

    if sparse_mask.is_some() || layout.is_sparse() {
        let mask = sparse_mask.unwrap_or_else(|| layout.mask::<B>(latent, &device));
        activated = activated * mask;
    }

    activated
}

pub fn fused_forward<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 3>>,
    threshold: f32,
    layout: &BlockPattern1d,
    sparse_mask: Option<Tensor<B, 4>>,
) -> Tensor<B, 4>
where
    B::FloatTensorPrimitive: 'static,
{
    fused_forward_with_executor(
        input,
        weight,
        bias,
        threshold,
        layout,
        sparse_mask,
        LowrankGradInputExecutor::Auto,
    )
}

pub fn fused_forward_with_executor<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 3>>,
    threshold: f32,
    layout: &BlockPattern1d,
    sparse_mask: Option<Tensor<B, 4>>,
    grad_input_executor: LowrankGradInputExecutor,
) -> Tensor<B, 4>
where
    B::FloatTensorPrimitive: 'static,
{
    let kernel_mask = sparse_mask.or_else(|| {
        layout.is_sparse().then(|| {
            let latent = weight.shape().dims::<4>()[3];
            layout.mask::<B>(latent, &input.device())
        })
    });

    if let Some(fused) = try_wgpu_fused_forward_with_executor(
        &input,
        &weight,
        bias.as_ref(),
        threshold,
        kernel_mask.as_ref(),
        grad_input_executor,
    ) {
        return fused;
    }

    reference_forward(input, weight, bias, threshold, layout, kernel_mask)
}

pub fn try_wgpu_fused_forward<B: Backend>(
    input: &Tensor<B, 4>,
    weight: &Tensor<B, 4>,
    bias: Option<&Tensor<B, 3>>,
    threshold: f32,
    sparse_mask: Option<&Tensor<B, 4>>,
) -> Option<Tensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    try_wgpu_fused_forward_with_executor(
        input,
        weight,
        bias,
        threshold,
        sparse_mask,
        LowrankGradInputExecutor::Auto,
    )
}

pub fn try_wgpu_fused_forward_with_executor<B: Backend>(
    input: &Tensor<B, 4>,
    weight: &Tensor<B, 4>,
    bias: Option<&Tensor<B, 3>>,
    threshold: f32,
    sparse_mask: Option<&Tensor<B, 4>>,
    grad_input_executor: LowrankGradInputExecutor,
) -> Option<Tensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if bias.is_some() {
        return None;
    }
    try_fused_relu_lowrank_projection_wgpu_with_executor(
        input,
        weight,
        threshold,
        sparse_mask,
        grad_input_executor,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    #[test]
    fn fused_forward_matches_reference_single_stream_projection() {
        type Backend = NdArray<f32>;
        let device = burn::tensor::Device::<Backend>::default();
        let input = Tensor::<Backend, 4>::from_data(
            TensorData::new((1..=12).map(|v| v as f32).collect::<Vec<_>>(), [2, 1, 2, 3]),
            &device,
        );
        let weight = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (1..=24).map(|v| (v as f32) * 0.05).collect::<Vec<_>>(),
                [1, 2, 3, 4],
            ),
            &device,
        );

        let actual = fused_forward(
            input.clone(),
            weight.clone(),
            None,
            0.0,
            &BlockPattern1d::dense(4),
            None,
        );
        let expected = activation::relu(input.matmul(weight));
        let actual = actual.into_data().to_vec::<f32>().expect("actual");
        let expected = expected.into_data().to_vec::<f32>().expect("expected");
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.into_iter().zip(expected) {
            assert!((a - b).abs() <= 1e-6, "single-stream mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn fused_forward_matches_reference_head_aligned_projection() {
        type Backend = NdArray<f32>;
        let device = burn::tensor::Device::<Backend>::default();
        let input = Tensor::<Backend, 4>::from_data(
            TensorData::new((1..=24).map(|v| v as f32).collect::<Vec<_>>(), [2, 2, 2, 3]),
            &device,
        );
        let weight = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (1..=24).map(|v| (v as f32) * 0.03).collect::<Vec<_>>(),
                [1, 2, 3, 4],
            ),
            &device,
        );

        let actual = fused_forward(
            input.clone(),
            weight.clone(),
            None,
            0.0,
            &BlockPattern1d::dense(4),
            None,
        );
        let expected = activation::relu(input.matmul(weight));
        let actual = actual.into_data().to_vec::<f32>().expect("actual");
        let expected = expected.into_data().to_vec::<f32>().expect("expected");
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.into_iter().zip(expected) {
            assert!((a - b).abs() <= 1e-6, "head-aligned mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn explicit_mask_is_honored_with_dense_layout() {
        type Backend = NdArray<f32>;
        let device = burn::tensor::Device::<Backend>::default();
        let input = Tensor::<Backend, 4>::ones([1, 1, 2, 3], &device);
        let weight = Tensor::<Backend, 4>::ones([1, 2, 3, 4], &device);
        let mask = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![1.0_f32, 0.0, 1.0, 0.0], [1, 1, 1, 4]),
            &device,
        );

        let actual = fused_forward(
            input,
            weight,
            None,
            0.0,
            &BlockPattern1d::dense(4),
            Some(mask),
        )
        .into_data()
        .to_vec::<f32>()
        .expect("masked output");

        for channels in actual.chunks_exact(4) {
            assert_eq!(channels, &[3.0, 0.0, 3.0, 0.0]);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
        if let Some(message) = payload.downcast_ref::<String>() {
            return message.clone();
        }
        if let Some(message) = payload.downcast_ref::<&'static str>() {
            return (*message).to_owned();
        }
        "unknown panic payload".to_owned()
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn fused_forward_routes_to_wgpu_kernel_when_supported() {
        use burn::tensor::Distribution;
        use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};

        type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

        static INIT_FAILURE: OnceLock<Option<String>> = OnceLock::new();
        let device = burn::tensor::Device::<Backend>::default();
        if let Some(reason) = INIT_FAILURE.get_or_init(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(
                    &device,
                    RuntimeOptions::default(),
                );
            }))
            .err()
            .map(panic_message)
        }) {
            eprintln!("skipping WGPU fused-forward route test without adapter: {reason}");
            return;
        }

        let input = Tensor::<Backend, 4>::random([2, 1, 11, 16], Distribution::Default, &device);
        let weight = Tensor::<Backend, 4>::random([1, 4, 16, 12], Distribution::Default, &device);
        let mask = Tensor::<Backend, 1>::from_floats([1.0; 12], &device).reshape([1, 1, 1, 12]);
        let pattern = BlockPattern1d::from_blocks(2, [0, 2, 4]);

        let auto = fused_forward(
            input.clone(),
            weight.clone(),
            None,
            0.1,
            &pattern,
            Some(mask.clone()),
        );
        let direct = try_wgpu_fused_forward(&input, &weight, None, 0.1, Some(&mask))
            .expect("wgpu fused lowrank path");
        let auto = auto.into_data().to_vec::<f32>().expect("auto");
        let direct = direct.into_data().to_vec::<f32>().expect("direct");
        assert_eq!(auto.len(), direct.len());
        for (lhs, rhs) in auto.into_iter().zip(direct) {
            assert!(
                (lhs - rhs).abs() <= 1e-4,
                "wgpu auto-route mismatch: {lhs} vs {rhs}"
            );
        }
    }
}
