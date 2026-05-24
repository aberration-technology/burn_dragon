use std::any::Any;
use std::marker::PhantomData;
use std::sync::Once;

#[cfg(feature = "cuda")]
use super::runtime::{
    gdn2_backward_runtime, gdn2_forward_params, gdn2_forward_runtime, gdn2_wy_factors_runtime,
};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Tensor, TensorPrimitive};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::checkpoint::strategy::NoCheckpointing;
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::wgpu::WgpuRuntime;
use burn_cubecl::tensor::CubeTensor;
use burn_wgpu::CubeBackend;

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend = Autodiff<WgpuCubeBackend>;
type WgpuCubeAutodiffTensor = burn::tensor::ops::FloatTensor<WgpuCubeAutodiffBackend>;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffBackend = Autodiff<CudaCubeBackend>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffTensor = burn::tensor::ops::FloatTensor<CudaCubeAutodiffBackend>;

#[derive(Debug)]
pub struct GatedDeltaNet2CustomBackwardOutput<B: BackendTrait> {
    pub context: Tensor<B, 4>,
    pub state: Tensor<B, 4>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GatedDeltaNet2BackwardState<FT> {
    query: FT,
    key: FT,
    value: FT,
    erase: FT,
    write: FT,
    log_decay: FT,
    initial_state: FT,
    boundary_states: Option<FT>,
    runtime_params: Option<FT>,
    chunk_size: usize,
}

#[derive(Debug)]
struct GatedDeltaNet2ChunkWyBackward<B>(PhantomData<B>);

fn custom_backward_enabled() -> bool {
    match std::env::var("BURN_DRAGON_GDN2_CHUNK_WY_CUSTOM_BACKWARD")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        _ => true,
    }
}

#[cfg(feature = "cuda")]
fn cuda_tensor_core_backward_enabled() -> bool {
    match std::env::var("BURN_DRAGON_GDN2_CUDA_TENSOR_CORE_BACKWARD")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        _ => true,
    }
}

fn log_gdn2_path_selection_once(message: &str) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| eprintln!("{message}"));
}

fn try_cast_primitive<B: BackendTrait, T: 'static>(value: B::FloatTensorPrimitive) -> Option<T>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed.downcast::<T>().ok().map(|boxed| *boxed)
}

fn try_cast_backend<B: BackendTrait, T: 'static>(value: T) -> Option<B::FloatTensorPrimitive>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed
        .downcast::<B::FloatTensorPrimitive>()
        .ok()
        .map(|boxed| *boxed)
}

fn forward_impl<B: BackendTrait>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    erase: Tensor<B, 4>,
    write: Tensor<B, 4>,
    log_decay: Tensor<B, 4>,
    initial_state: Tensor<B, 4>,
    chunk_size: usize,
) -> GatedDeltaNet2CustomBackwardOutput<B> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let dense_dim = value.shape().dims::<4>()[3];
    let output_scale = (latent as f32).sqrt().recip();
    let mut state = initial_state;
    let mut outputs = Vec::with_capacity(time);
    let chunk_size = chunk_size.max(1);

    for chunk_start in (0..time).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(time);
        for t in chunk_start..chunk_end {
            let q_t = query.clone().slice_dim(2, t..t + 1);
            let k_t = key.clone().slice_dim(2, t..t + 1);
            let v_t = value.clone().slice_dim(2, t..t + 1);
            let b_t = erase.clone().slice_dim(2, t..t + 1);
            let w_t = write.clone().slice_dim(2, t..t + 1);
            let decay_bh1l = log_decay.clone().slice_dim(2, t..t + 1).exp();
            let decay_bhl1 = decay_bh1l.swap_dims(2, 3);

            let decayed = state * decay_bhl1;
            let erased_key = b_t * k_t.clone();
            let erased_value = (decayed.clone() * erased_key.swap_dims(2, 3))
                .sum_dim(2)
                .reshape([batch, heads, 1, dense_dim]);
            let update = w_t * v_t - erased_value;
            state = decayed + k_t.swap_dims(2, 3) * update;
            let output = (state.clone() * q_t.swap_dims(2, 3))
                .sum_dim(2)
                .reshape([batch, heads, 1, dense_dim])
                .mul_scalar(output_scale);
            outputs.push(output);
        }
    }

    GatedDeltaNet2CustomBackwardOutput {
        context: Tensor::cat(outputs, 2),
        state,
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
fn backward_chunk_wy_tensor_core_impl<B: BackendTrait>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    erase: Tensor<B, 4>,
    write: Tensor<B, 4>,
    boundary_states: Tensor<B, 5>,
    cumulative_decay: Tensor<B, 5>,
    wy_lower: Tensor<B, 5>,
    grad_output: Tensor<B, 4>,
    chunk_size: usize,
) -> (
    Tensor<B, 4>,
    Tensor<B, 4>,
    Tensor<B, 4>,
    Tensor<B, 4>,
    Tensor<B, 4>,
    Tensor<B, 4>,
    Tensor<B, 4>,
) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let dense_dim = value.shape().dims::<4>()[3];
    let batch_heads = batch * heads;
    let device = query.device();
    let chunk_size = chunk_size.max(1);
    let output_scale = (latent as f32).sqrt().recip();

    let mut grad_query_chunks = Vec::with_capacity(time.div_ceil(chunk_size));
    let mut grad_key_chunks = Vec::with_capacity(time.div_ceil(chunk_size));
    let mut grad_value_chunks = Vec::with_capacity(time.div_ceil(chunk_size));
    let mut grad_erase_chunks = Vec::with_capacity(time.div_ceil(chunk_size));
    let mut grad_write_chunks = Vec::with_capacity(time.div_ceil(chunk_size));
    let mut grad_log_decay_chunks = Vec::with_capacity(time.div_ceil(chunk_size));
    let mut grad_state_carry = Tensor::<B, 3>::zeros([batch_heads, latent, dense_dim], &device);

    let num_chunks = time.div_ceil(chunk_size);
    for chunk_index_rev in 0..num_chunks {
        let chunk_index = num_chunks - 1 - chunk_index_rev;
        let chunk_start = chunk_index * chunk_size;
        let chunk_end = (chunk_start + chunk_size).min(time);
        let chunk_len = chunk_end - chunk_start;

        let q_chunk = query.clone().slice_dim(2, chunk_start..chunk_end).reshape([
            batch_heads,
            chunk_len,
            latent,
        ]);
        let k_chunk = key.clone().slice_dim(2, chunk_start..chunk_end).reshape([
            batch_heads,
            chunk_len,
            latent,
        ]);
        let value_chunk = value.clone().slice_dim(2, chunk_start..chunk_end).reshape([
            batch_heads,
            chunk_len,
            dense_dim,
        ]);
        let erase_chunk = erase.clone().slice_dim(2, chunk_start..chunk_end).reshape([
            batch_heads,
            chunk_len,
            latent,
        ]);
        let write_chunk = write.clone().slice_dim(2, chunk_start..chunk_end).reshape([
            batch_heads,
            chunk_len,
            dense_dim,
        ]);
        let grad_output_chunk = grad_output
            .clone()
            .slice_dim(2, chunk_start..chunk_end)
            .reshape([batch_heads, chunk_len, dense_dim]);
        let chunk_initial_state = boundary_states
            .clone()
            .slice_dim(2, chunk_index..chunk_index + 1)
            .reshape([batch_heads, latent, dense_dim]);

        let cumulative = cumulative_decay
            .clone()
            .slice_dim(2, chunk_index..chunk_index + 1)
            .reshape([batch_heads, chunk_size, latent])
            .slice_dim(1, 0..chunk_len);
        let m_basis = erase_chunk.clone() * k_chunk.clone() * cumulative.clone();
        let w_basis = k_chunk.clone() / cumulative.clone();
        let lower = wy_lower
            .clone()
            .slice_dim(2, chunk_index..chunk_index + 1)
            .reshape([batch_heads, chunk_size, chunk_size])
            .slice_dim(1, 0..chunk_len)
            .slice_dim(2, 0..chunk_len);
        let rhs = write_chunk.clone() * value_chunk.clone()
            - m_basis.clone().matmul(chunk_initial_state.clone());

        let eye = Tensor::<B, 2>::eye(chunk_len, &device)
            .reshape([1, chunk_len, chunk_len])
            .repeat_dim(0, batch_heads);
        let mut inverse_rows = Vec::with_capacity(chunk_len);
        for local in 0..chunk_len {
            let diagonal_row = eye.clone().slice_dim(1, local..local + 1);
            let inverse_row = if local == 0 {
                diagonal_row
            } else {
                let previous_inverse = Tensor::cat(inverse_rows.clone(), 1);
                let lower_row = lower
                    .clone()
                    .slice_dim(1, local..local + 1)
                    .slice_dim(2, 0..local);
                diagonal_row - lower_row.matmul(previous_inverse)
            };
            inverse_rows.push(inverse_row);
        }
        let inverse_lower = Tensor::cat(inverse_rows, 1);
        let solved_update = inverse_lower.clone().matmul(rhs);

        let update_outer = w_basis.clone().reshape([batch_heads, chunk_len, latent, 1])
            * solved_update
                .clone()
                .reshape([batch_heads, chunk_len, 1, dense_dim]);
        let transformed = update_outer.cumsum(1)
            + chunk_initial_state
                .clone()
                .reshape([batch_heads, 1, latent, dense_dim]);
        let state_values = transformed.clone()
            * cumulative
                .clone()
                .reshape([batch_heads, chunk_len, latent, 1]);

        let grad_query_chunk = grad_output_chunk
            .clone()
            .reshape([batch_heads * chunk_len, 1, dense_dim])
            .matmul(
                state_values
                    .clone()
                    .reshape([batch_heads * chunk_len, latent, dense_dim])
                    .swap_dims(1, 2),
            )
            .reshape([batch, heads, chunk_len, latent])
            .mul_scalar(output_scale);

        let grad_state_from_output = q_chunk.clone().reshape([batch_heads, chunk_len, latent, 1])
            * grad_output_chunk
                .clone()
                .reshape([batch_heads, chunk_len, 1, dense_dim])
                .mul_scalar(output_scale);
        let grad_state_carry_chunk = if chunk_len == 1 {
            grad_state_carry
                .clone()
                .reshape([batch_heads, 1, latent, dense_dim])
        } else {
            Tensor::cat(
                vec![
                    Tensor::<B, 4>::zeros([batch_heads, chunk_len - 1, latent, dense_dim], &device),
                    grad_state_carry
                        .clone()
                        .reshape([batch_heads, 1, latent, dense_dim]),
                ],
                1,
            )
        };
        let grad_state_total = grad_state_from_output + grad_state_carry_chunk;
        let mut grad_cumulative = (grad_state_total.clone() * transformed)
            .sum_dim(3)
            .reshape([batch_heads, chunk_len, latent]);
        let grad_transformed = grad_state_total
            * cumulative
                .clone()
                .reshape([batch_heads, chunk_len, latent, 1]);
        let grad_update_outer = grad_transformed.flip([1]).cumsum(1).flip([1]);
        let mut grad_w = (grad_update_outer.clone()
            * solved_update
                .clone()
                .reshape([batch_heads, chunk_len, 1, dense_dim]))
        .sum_dim(3)
        .reshape([batch_heads, chunk_len, latent]);
        let grad_solved = (grad_update_outer.clone()
            * w_basis.clone().reshape([batch_heads, chunk_len, latent, 1]))
        .sum_dim(2)
        .reshape([batch_heads, chunk_len, dense_dim]);

        let grad_rhs = inverse_lower.swap_dims(1, 2).matmul(grad_solved);
        let grad_lower = grad_rhs
            .clone()
            .matmul(solved_update.swap_dims(1, 2))
            .mul_scalar(-1.0)
            .tril(-1);

        let grad_write_chunk = grad_rhs.clone() * value_chunk.clone();
        let grad_value_chunk = grad_rhs.clone() * write_chunk;

        let mut grad_m = grad_rhs
            .clone()
            .matmul(chunk_initial_state.clone().swap_dims(1, 2))
            .mul_scalar(-1.0);
        let grad_transformed_initial =
            grad_update_outer
                .slice_dim(1, 0..1)
                .reshape([batch_heads, latent, dense_dim]);
        grad_state_carry =
            grad_transformed_initial - m_basis.clone().swap_dims(1, 2).matmul(grad_rhs.clone());

        grad_m = grad_m + grad_lower.clone().matmul(w_basis.clone());
        grad_w = grad_w + grad_lower.swap_dims(1, 2).matmul(m_basis.clone());

        let grad_erase_chunk = grad_m.clone() * k_chunk.clone() * cumulative.clone();
        let grad_key_chunk = grad_m.clone() * erase_chunk.clone() * cumulative.clone()
            + grad_w.clone() / cumulative.clone();
        grad_cumulative = grad_cumulative + grad_m * erase_chunk * k_chunk.clone()
            - grad_w * k_chunk / (cumulative.clone() * cumulative.clone());

        let grad_cumulative_terms = grad_cumulative * cumulative;
        let grad_log_decay_chunk = grad_cumulative_terms.flip([1]).cumsum(1).flip([1]);

        grad_query_chunks.push(grad_query_chunk);
        grad_key_chunks.push(grad_key_chunk.reshape([batch, heads, chunk_len, latent]));
        grad_value_chunks.push(grad_value_chunk.reshape([batch, heads, chunk_len, dense_dim]));
        grad_erase_chunks.push(grad_erase_chunk.reshape([batch, heads, chunk_len, latent]));
        grad_write_chunks.push(grad_write_chunk.reshape([batch, heads, chunk_len, dense_dim]));
        grad_log_decay_chunks.push(grad_log_decay_chunk.reshape([batch, heads, chunk_len, latent]));
    }

    grad_query_chunks.reverse();
    grad_key_chunks.reverse();
    grad_value_chunks.reverse();
    grad_erase_chunks.reverse();
    grad_write_chunks.reverse();
    grad_log_decay_chunks.reverse();

    (
        Tensor::cat(grad_query_chunks, 2),
        Tensor::cat(grad_key_chunks, 2),
        Tensor::cat(grad_value_chunks, 2),
        Tensor::cat(grad_erase_chunks, 2),
        Tensor::cat(grad_write_chunks, 2),
        Tensor::cat(grad_log_decay_chunks, 2),
        grad_state_carry.reshape([batch, heads, latent, dense_dim]),
    )
}

#[allow(clippy::type_complexity)]
fn recompute_chunk_states<B: BackendTrait>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    erase: Tensor<B, 4>,
    write: Tensor<B, 4>,
    log_decay: Tensor<B, 4>,
    initial_state: Tensor<B, 4>,
    chunk_start: usize,
    chunk_end: usize,
) -> (
    Vec<Tensor<B, 4>>,
    Vec<Tensor<B, 4>>,
    Vec<Tensor<B, 4>>,
    Tensor<B, 4>,
) {
    let [batch, heads, _time, _latent] = query.shape().dims::<4>();
    let dense_dim = value.shape().dims::<4>()[3];
    let mut state = initial_state;
    let mut prev_states = Vec::with_capacity(chunk_end - chunk_start);
    let mut decayed_states = Vec::with_capacity(chunk_end - chunk_start);
    let mut next_states = Vec::with_capacity(chunk_end - chunk_start);

    for t in chunk_start..chunk_end {
        prev_states.push(state.clone());
        let k_t = key.clone().slice_dim(2, t..t + 1);
        let v_t = value.clone().slice_dim(2, t..t + 1);
        let b_t = erase.clone().slice_dim(2, t..t + 1);
        let w_t = write.clone().slice_dim(2, t..t + 1);
        let decay_t = log_decay
            .clone()
            .slice_dim(2, t..t + 1)
            .exp()
            .swap_dims(2, 3);
        let decayed = state * decay_t;
        decayed_states.push(decayed.clone());
        let erased_key = b_t * k_t.clone();
        let erased_value = (decayed.clone() * erased_key.swap_dims(2, 3))
            .sum_dim(2)
            .reshape([batch, heads, 1, dense_dim]);
        let update = w_t * v_t - erased_value;
        state = decayed + k_t.swap_dims(2, 3) * update;
        next_states.push(state.clone());
    }

    (prev_states, decayed_states, next_states, state)
}

fn backward_impl<B>(
    ops: Ops<GatedDeltaNet2BackwardState<B::FloatTensorPrimitive>, 7>,
    grads: &mut Gradients,
) where
    B: BackendTrait,
{
    let grad_output =
        Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(grads.consume::<B>(&ops.node)));
    let state = ops.state;
    let parents = ops.parents;

    let query = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.query));
    let key = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.key));
    let value = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.value));
    let erase = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.erase));
    let write = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.write));
    let log_decay = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.log_decay));
    let initial_state = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.initial_state));

    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let dense_dim = value.shape().dims::<4>()[3];
    let device = query.device();
    let chunk_size = state.chunk_size.max(1);
    let output_scale = (latent as f32).sqrt().recip();

    let mut chunk_initial_states = Vec::new();
    let mut chunk_ranges = Vec::new();
    let mut running_state = initial_state.clone();
    for chunk_start in (0..time).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(time);
        chunk_initial_states.push(running_state.clone());
        chunk_ranges.push((chunk_start, chunk_end));
        let (_, _, _, next_state) = recompute_chunk_states(
            query.clone(),
            key.clone(),
            value.clone(),
            erase.clone(),
            write.clone(),
            log_decay.clone(),
            running_state,
            chunk_start,
            chunk_end,
        );
        running_state = next_state;
    }

    let mut grad_query_chunks = Vec::with_capacity(chunk_ranges.len());
    let mut grad_key_chunks = Vec::with_capacity(chunk_ranges.len());
    let mut grad_value_chunks = Vec::with_capacity(chunk_ranges.len());
    let mut grad_erase_chunks = Vec::with_capacity(chunk_ranges.len());
    let mut grad_write_chunks = Vec::with_capacity(chunk_ranges.len());
    let mut grad_log_decay_chunks = Vec::with_capacity(chunk_ranges.len());
    let mut grad_state_carry = Tensor::<B, 4>::zeros([batch, heads, latent, dense_dim], &device);

    for (chunk_index, (chunk_start, chunk_end)) in chunk_ranges.iter().enumerate().rev() {
        let chunk_initial = chunk_initial_states[chunk_index].clone();
        let (prev_states, decayed_states, next_states, _) = recompute_chunk_states(
            query.clone(),
            key.clone(),
            value.clone(),
            erase.clone(),
            write.clone(),
            log_decay.clone(),
            chunk_initial,
            *chunk_start,
            *chunk_end,
        );
        let chunk_len = chunk_end - chunk_start;
        let mut grad_q_rev = Vec::with_capacity(chunk_len);
        let mut grad_k_rev = Vec::with_capacity(chunk_len);
        let mut grad_v_rev = Vec::with_capacity(chunk_len);
        let mut grad_erase_rev = Vec::with_capacity(chunk_len);
        let mut grad_write_rev = Vec::with_capacity(chunk_len);
        let mut grad_log_decay_rev = Vec::with_capacity(chunk_len);

        for local_rev in 0..chunk_len {
            let local = chunk_len - 1 - local_rev;
            let t = chunk_start + local;
            let q_t = query.clone().slice_dim(2, t..t + 1);
            let k_t = key.clone().slice_dim(2, t..t + 1);
            let v_t = value.clone().slice_dim(2, t..t + 1);
            let b_t = erase.clone().slice_dim(2, t..t + 1);
            let w_t = write.clone().slice_dim(2, t..t + 1);
            let decay_bh1l = log_decay.clone().slice_dim(2, t..t + 1).exp();
            let decay_bhl1 = decay_bh1l.clone().swap_dims(2, 3);
            let prev_state = prev_states[local].clone();
            let decayed_state = decayed_states[local].clone();
            let next_state = next_states[local].clone();
            let grad_o = grad_output.clone().slice_dim(2, t..t + 1);

            let grad_q = (next_state.clone() * grad_o.clone())
                .sum_dim(3)
                .reshape([batch, heads, 1, latent])
                .mul_scalar(output_scale);
            let grad_next_state =
                grad_state_carry + q_t.swap_dims(2, 3) * grad_o.mul_scalar(output_scale);

            let erased_key = b_t.clone() * k_t.clone();
            let erased_value = (decayed_state.clone() * erased_key.clone().swap_dims(2, 3))
                .sum_dim(2)
                .reshape([batch, heads, 1, dense_dim]);
            let update = w_t.clone() * v_t.clone() - erased_value;

            let grad_k_from_outer = (grad_next_state.clone() * update.clone())
                .sum_dim(3)
                .reshape([batch, heads, 1, latent]);
            let grad_update = (grad_next_state.clone() * k_t.clone().swap_dims(2, 3))
                .sum_dim(2)
                .reshape([batch, heads, 1, dense_dim]);
            let mut grad_decayed = grad_next_state;
            let grad_w = grad_update.clone() * v_t.clone();
            let grad_v = grad_update.clone() * w_t;
            let grad_erased_value = grad_update.mul_scalar(-1.0);
            grad_decayed = grad_decayed + erased_key.swap_dims(2, 3) * grad_erased_value.clone();
            let grad_erased_key = (decayed_state.clone() * grad_erased_value)
                .sum_dim(3)
                .reshape([batch, heads, 1, latent]);
            let grad_erase = grad_erased_key.clone() * k_t.clone();
            let grad_k = grad_k_from_outer + grad_erased_key * b_t;
            let grad_decay = (grad_decayed.clone() * prev_state.clone())
                .sum_dim(3)
                .reshape([batch, heads, 1, latent]);
            let grad_log_decay = grad_decay * decay_bh1l;
            grad_state_carry = grad_decayed * decay_bhl1;

            grad_q_rev.push(grad_q);
            grad_k_rev.push(grad_k);
            grad_v_rev.push(grad_v);
            grad_erase_rev.push(grad_erase);
            grad_write_rev.push(grad_w);
            grad_log_decay_rev.push(grad_log_decay);
        }

        grad_q_rev.reverse();
        grad_k_rev.reverse();
        grad_v_rev.reverse();
        grad_erase_rev.reverse();
        grad_write_rev.reverse();
        grad_log_decay_rev.reverse();
        grad_query_chunks.push(Tensor::cat(grad_q_rev, 2));
        grad_key_chunks.push(Tensor::cat(grad_k_rev, 2));
        grad_value_chunks.push(Tensor::cat(grad_v_rev, 2));
        grad_erase_chunks.push(Tensor::cat(grad_erase_rev, 2));
        grad_write_chunks.push(Tensor::cat(grad_write_rev, 2));
        grad_log_decay_chunks.push(Tensor::cat(grad_log_decay_rev, 2));
    }

    grad_query_chunks.reverse();
    grad_key_chunks.reverse();
    grad_value_chunks.reverse();
    grad_erase_chunks.reverse();
    grad_write_chunks.reverse();
    grad_log_decay_chunks.reverse();

    let grad_query = Tensor::cat(grad_query_chunks, 2);
    let grad_key = Tensor::cat(grad_key_chunks, 2);
    let grad_value = Tensor::cat(grad_value_chunks, 2);
    let grad_erase = Tensor::cat(grad_erase_chunks, 2);
    let grad_write = Tensor::cat(grad_write_chunks, 2);
    let grad_log_decay = Tensor::cat(grad_log_decay_chunks, 2);

    if let Some(parent) = &parents[0] {
        grads.register::<B>(parent.id, grad_query.into_primitive().tensor());
    }
    if let Some(parent) = &parents[1] {
        grads.register::<B>(parent.id, grad_key.into_primitive().tensor());
    }
    if let Some(parent) = &parents[2] {
        grads.register::<B>(parent.id, grad_value.into_primitive().tensor());
    }
    if let Some(parent) = &parents[3] {
        grads.register::<B>(parent.id, grad_erase.into_primitive().tensor());
    }
    if let Some(parent) = &parents[4] {
        grads.register::<B>(parent.id, grad_write.into_primitive().tensor());
    }
    if let Some(parent) = &parents[5] {
        grads.register::<B>(parent.id, grad_log_decay.into_primitive().tensor());
    }
    if let Some(parent) = &parents[6] {
        grads.register::<B>(parent.id, grad_state_carry.into_primitive().tensor());
    }
}

#[allow(clippy::too_many_arguments)]
fn try_custom_backward_wgpu<B: BackendTrait>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    erase: Tensor<B, 4>,
    write: Tensor<B, 4>,
    log_decay: Tensor<B, 4>,
    initial_state: Tensor<B, 4>,
    chunk_size: usize,
) -> Option<GatedDeltaNet2CustomBackwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let query_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(query.into_primitive().tensor())?;
    let key_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(key.into_primitive().tensor())?;
    let value_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(value.into_primitive().tensor())?;
    let erase_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(erase.into_primitive().tensor())?;
    let write_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(write.into_primitive().tensor())?;
    let log_decay_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(log_decay.into_primitive().tensor())?;
    let initial_state_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(initial_state.into_primitive().tensor())?;

    let query_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(query_ad.clone());
    let key_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(key_ad.clone());
    let value_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(value_ad.clone());
    let erase_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(erase_ad.clone());
    let write_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(write_ad.clone());
    let log_decay_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(log_decay_ad.clone());
    let initial_state_inner =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(initial_state_ad.clone());

    let output = forward_impl(
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(query_inner.clone())),
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(key_inner.clone())),
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(value_inner.clone())),
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(erase_inner.clone())),
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(write_inner.clone())),
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            log_decay_inner.clone(),
        )),
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            initial_state_inner.clone(),
        )),
        chunk_size,
    );
    let context_inner = output.context.into_primitive().tensor();
    let state_inner = output.state.into_primitive().tensor();
    let context_ad = match GatedDeltaNet2ChunkWyBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            query_ad.node.clone(),
            key_ad.node.clone(),
            value_ad.node.clone(),
            erase_ad.node.clone(),
            write_ad.node.clone(),
            log_decay_ad.node.clone(),
            initial_state_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            GatedDeltaNet2BackwardState {
                query: query_inner,
                key: key_inner,
                value: value_inner,
                erase: erase_inner,
                write: write_inner,
                log_decay: log_decay_inner,
                initial_state: initial_state_inner,
                boundary_states: None,
                runtime_params: None,
                chunk_size,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };

    Some(GatedDeltaNet2CustomBackwardOutput {
        context: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
            context_ad,
        )?)),
        state: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
            <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(state_inner),
        )?)),
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_custom_backward_cuda<B: BackendTrait>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    erase: Tensor<B, 4>,
    write: Tensor<B, 4>,
    log_decay: Tensor<B, 4>,
    initial_state: Tensor<B, 4>,
    chunk_size: usize,
) -> Option<GatedDeltaNet2CustomBackwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let query_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(query.into_primitive().tensor())?;
    let key_ad: CudaCubeAutodiffTensor = try_cast_primitive::<B, _>(key.into_primitive().tensor())?;
    let value_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(value.into_primitive().tensor())?;
    let erase_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(erase.into_primitive().tensor())?;
    let write_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(write.into_primitive().tensor())?;
    let log_decay_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(log_decay.into_primitive().tensor())?;
    let initial_state_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(initial_state.into_primitive().tensor())?;

    let query_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(query_ad.clone());
    let key_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(key_ad.clone());
    let value_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(value_ad.clone());
    let erase_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(erase_ad.clone());
    let write_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(write_ad.clone());
    let log_decay_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(log_decay_ad.clone());
    let initial_state_inner =
        <CudaCubeAutodiffBackend as AutodiffBackend>::inner(initial_state_ad.clone());

    let query_tensor =
        Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(query_inner.clone()));
    let value_tensor =
        Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(value_inner.clone()));
    let params_tensor = gdn2_forward_params(&query_tensor, &value_tensor, chunk_size);
    let time = query_tensor.shape().dims::<4>()[2];
    let num_chunks = time.div_ceil(chunk_size.max(1));
    let runtime_params_inner = params_tensor.into_primitive().tensor();
    let output = gdn2_forward_runtime::<CudaRuntime>(
        query_inner.clone(),
        key_inner.clone(),
        value_inner.clone(),
        erase_inner.clone(),
        write_inner.clone(),
        log_decay_inner.clone(),
        initial_state_inner.clone(),
        runtime_params_inner.clone(),
        num_chunks,
    );
    let context_inner = output.context;
    let state_inner = output.final_state;
    let boundary_states_inner = output.boundary_states;
    let context_ad = match GatedDeltaNet2ChunkWyBackward::<CudaCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            query_ad.node.clone(),
            key_ad.node.clone(),
            value_ad.node.clone(),
            erase_ad.node.clone(),
            write_ad.node.clone(),
            log_decay_ad.node.clone(),
            initial_state_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            GatedDeltaNet2BackwardState {
                query: query_inner,
                key: key_inner,
                value: value_inner,
                erase: erase_inner,
                write: write_inner,
                log_decay: log_decay_inner,
                initial_state: initial_state_inner,
                boundary_states: Some(boundary_states_inner),
                runtime_params: Some(runtime_params_inner),
                chunk_size,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };

    Some(GatedDeltaNet2CustomBackwardOutput {
        context: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
            context_ad,
        )?)),
        state: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
            <CudaCubeAutodiffBackend as AutodiffBackend>::from_inner(state_inner),
        )?)),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn try_gdn2_chunk_wy_custom_backward<B: BackendTrait>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    erase: Tensor<B, 4>,
    write: Tensor<B, 4>,
    log_decay: Tensor<B, 4>,
    initial_state: Tensor<B, 4>,
    chunk_size: usize,
) -> Option<GatedDeltaNet2CustomBackwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !custom_backward_enabled() {
        return None;
    }
    if let Some(output) = try_custom_backward_wgpu(
        query.clone(),
        key.clone(),
        value.clone(),
        erase.clone(),
        write.clone(),
        log_decay.clone(),
        initial_state.clone(),
        chunk_size,
    ) {
        log_gdn2_path_selection_once(
            "gated_deltanet2 chunk-WY path: using chunked custom analytic backward on WGPU",
        );
        return Some(output);
    }
    #[cfg(feature = "cuda")]
    if let Some(output) = try_custom_backward_cuda(
        query,
        key,
        value,
        erase,
        write,
        log_decay,
        initial_state,
        chunk_size,
    ) {
        log_gdn2_path_selection_once(
            "gated_deltanet2 chunk-WY path: using CUDA fused WY solver/backward",
        );
        return Some(output);
    }
    None
}

impl Backward<WgpuCubeBackend, 7> for GatedDeltaNet2ChunkWyBackward<WgpuCubeBackend> {
    type State = GatedDeltaNet2BackwardState<CubeTensor<WgpuRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 7>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        backward_impl::<WgpuCubeBackend>(ops, grads);
    }
}

#[cfg(feature = "cuda")]
fn backward_cuda_runtime_impl(
    ops: Ops<GatedDeltaNet2BackwardState<CubeTensor<CudaRuntime>>, 7>,
    grads: &mut Gradients,
) {
    let state = ops.state;
    let boundary_states = state
        .boundary_states
        .expect("gdn2 CUDA custom backward requires Cube runtime boundary states");
    let runtime_params = state
        .runtime_params
        .expect("gdn2 CUDA custom backward requires Cube runtime params");
    let grad_output = grads.consume::<CudaCubeBackend>(&ops.node);
    let parents = ops.parents;

    if cuda_tensor_core_backward_enabled() {
        let (cumulative_decay, wy_lower) = gdn2_wy_factors_runtime::<CudaRuntime>(
            state.key.clone(),
            state.erase.clone(),
            state.log_decay.clone(),
            runtime_params.clone(),
            state.chunk_size,
        );
        let query =
            Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(state.query));
        let key = Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(state.key));
        let value =
            Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(state.value));
        let erase =
            Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(state.erase));
        let write =
            Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(state.write));
        let boundary_states =
            Tensor::<CudaCubeBackend, 5>::from_primitive(TensorPrimitive::Float(boundary_states));
        let cumulative_decay =
            Tensor::<CudaCubeBackend, 5>::from_primitive(TensorPrimitive::Float(cumulative_decay));
        let wy_lower =
            Tensor::<CudaCubeBackend, 5>::from_primitive(TensorPrimitive::Float(wy_lower));
        let grad_output =
            Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(grad_output));
        let (
            grad_query,
            grad_key,
            grad_value,
            grad_erase,
            grad_write,
            grad_log_decay,
            grad_initial_state,
        ) = backward_chunk_wy_tensor_core_impl(
            query,
            key,
            value,
            erase,
            write,
            boundary_states,
            cumulative_decay,
            wy_lower,
            grad_output,
            state.chunk_size,
        );

        if let Some(parent) = &parents[0] {
            grads.register::<CudaCubeBackend>(parent.id, grad_query.into_primitive().tensor());
        }
        if let Some(parent) = &parents[1] {
            grads.register::<CudaCubeBackend>(parent.id, grad_key.into_primitive().tensor());
        }
        if let Some(parent) = &parents[2] {
            grads.register::<CudaCubeBackend>(parent.id, grad_value.into_primitive().tensor());
        }
        if let Some(parent) = &parents[3] {
            grads.register::<CudaCubeBackend>(parent.id, grad_erase.into_primitive().tensor());
        }
        if let Some(parent) = &parents[4] {
            grads.register::<CudaCubeBackend>(parent.id, grad_write.into_primitive().tensor());
        }
        if let Some(parent) = &parents[5] {
            grads.register::<CudaCubeBackend>(parent.id, grad_log_decay.into_primitive().tensor());
        }
        if let Some(parent) = &parents[6] {
            grads.register::<CudaCubeBackend>(
                parent.id,
                grad_initial_state.into_primitive().tensor(),
            );
        }
        return;
    }

    let output = gdn2_backward_runtime::<CudaRuntime>(
        state.query,
        state.key,
        state.value,
        state.erase,
        state.write,
        state.log_decay,
        state.initial_state,
        boundary_states,
        grad_output,
        runtime_params,
        state.chunk_size,
    );

    if let Some(parent) = &parents[0] {
        grads.register::<CudaCubeBackend>(parent.id, output.grad_query);
    }
    if let Some(parent) = &parents[1] {
        grads.register::<CudaCubeBackend>(parent.id, output.grad_key);
    }
    if let Some(parent) = &parents[2] {
        grads.register::<CudaCubeBackend>(parent.id, output.grad_value);
    }
    if let Some(parent) = &parents[3] {
        grads.register::<CudaCubeBackend>(parent.id, output.grad_erase);
    }
    if let Some(parent) = &parents[4] {
        grads.register::<CudaCubeBackend>(parent.id, output.grad_write);
    }
    if let Some(parent) = &parents[5] {
        grads.register::<CudaCubeBackend>(parent.id, output.grad_log_decay);
    }
    if let Some(parent) = &parents[6] {
        grads.register::<CudaCubeBackend>(parent.id, output.grad_initial_state);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use burn::prelude::ElementConversion;
    use burn::tensor::TensorData;

    type TestBackend = CudaCubeAutodiffBackend;

    fn tensor_data(shape: [usize; 4], stride: usize, modulus: usize, offset: f32) -> TensorData {
        let len = shape.iter().product::<usize>();
        TensorData::new(
            (0..len)
                .map(|idx| ((idx * stride) % modulus) as f32 / modulus as f32 + offset)
                .collect::<Vec<_>>(),
            shape,
        )
    }

    fn assert_close<B: BackendTrait, const D: usize>(
        label: &str,
        lhs: Tensor<B, D>,
        rhs: Tensor<B, D>,
    ) {
        let max_rhs = rhs.clone().abs().max().into_scalar().elem::<f32>();
        let max_diff = lhs.sub(rhs).abs().max().into_scalar().elem::<f32>();
        let max_tol = 3.0e-3 + 3.0e-3 * max_rhs;
        assert!(
            max_diff <= max_tol,
            "{label} max difference {max_diff} exceeds tolerance {max_tol} (rhs max {max_rhs})"
        );
    }

    fn run_chunk_wy_custom_backward_matches_direct_graph_on_cuda(
        batch: usize,
        heads: usize,
        time: usize,
        latent: usize,
        dense: usize,
        chunk_size: usize,
    ) {
        let device = burn::tensor::Device::<TestBackend>::default();

        let query_data = tensor_data([batch, heads, time, latent], 3, 23, -0.35);
        let key_data = tensor_data([batch, heads, time, latent], 5, 29, -0.25);
        let value_data = tensor_data([batch, heads, time, dense], 7, 31, -0.2);
        let erase_data = tensor_data([batch, heads, time, latent], 11, 37, 0.1);
        let write_data = tensor_data([batch, heads, time, dense], 13, 41, 0.2);
        let log_decay_data = tensor_data([batch, heads, time, latent], 17, 43, -1.2);
        let initial_state_data = tensor_data([batch, heads, latent, dense], 19, 47, -0.1);
        let output_weight_data = tensor_data([batch, heads, time, dense], 23, 53, -0.3);

        let graph_query =
            Tensor::<TestBackend, 4>::from_data(query_data.clone(), &device).require_grad();
        let graph_key =
            Tensor::<TestBackend, 4>::from_data(key_data.clone(), &device).require_grad();
        let graph_value =
            Tensor::<TestBackend, 4>::from_data(value_data.clone(), &device).require_grad();
        let graph_erase =
            Tensor::<TestBackend, 4>::from_data(erase_data.clone(), &device).require_grad();
        let graph_write =
            Tensor::<TestBackend, 4>::from_data(write_data.clone(), &device).require_grad();
        let graph_log_decay =
            Tensor::<TestBackend, 4>::from_data(log_decay_data.clone(), &device).require_grad();
        let graph_initial_state =
            Tensor::<TestBackend, 4>::from_data(initial_state_data.clone(), &device).require_grad();

        let wrapped_query = Tensor::<TestBackend, 4>::from_data(query_data, &device).require_grad();
        let wrapped_key = Tensor::<TestBackend, 4>::from_data(key_data, &device).require_grad();
        let wrapped_value = Tensor::<TestBackend, 4>::from_data(value_data, &device).require_grad();
        let wrapped_erase = Tensor::<TestBackend, 4>::from_data(erase_data, &device).require_grad();
        let wrapped_write = Tensor::<TestBackend, 4>::from_data(write_data, &device).require_grad();
        let wrapped_log_decay =
            Tensor::<TestBackend, 4>::from_data(log_decay_data, &device).require_grad();
        let wrapped_initial_state =
            Tensor::<TestBackend, 4>::from_data(initial_state_data, &device).require_grad();

        let graph = forward_impl(
            graph_query.clone(),
            graph_key.clone(),
            graph_value.clone(),
            graph_erase.clone(),
            graph_write.clone(),
            graph_log_decay.clone(),
            graph_initial_state.clone(),
            chunk_size,
        );
        let wrapped = try_gdn2_chunk_wy_custom_backward(
            wrapped_query.clone(),
            wrapped_key.clone(),
            wrapped_value.clone(),
            wrapped_erase.clone(),
            wrapped_write.clone(),
            wrapped_log_decay.clone(),
            wrapped_initial_state.clone(),
            chunk_size,
        )
        .expect("cuda custom backward path");
        let _ = <TestBackend as BackendTrait>::sync(&device);
        assert_close("context", graph.context.clone(), wrapped.context.clone());
        assert_close("state", graph.state.clone(), wrapped.state.clone());

        let output_weights = Tensor::<TestBackend, 4>::from_data(output_weight_data, &device);
        let graph_grads = (graph.context * output_weights.clone()).sum().backward();
        let wrapped_grads = (wrapped.context * output_weights).sum().backward();
        let _ = <TestBackend as BackendTrait>::sync(&device);

        assert_close(
            "query grad",
            graph_query.grad(&graph_grads).expect("graph query grad"),
            wrapped_query
                .grad(&wrapped_grads)
                .expect("wrapped query grad"),
        );
        assert_close(
            "key grad",
            graph_key.grad(&graph_grads).expect("graph key grad"),
            wrapped_key.grad(&wrapped_grads).expect("wrapped key grad"),
        );
        assert_close(
            "value grad",
            graph_value.grad(&graph_grads).expect("graph value grad"),
            wrapped_value
                .grad(&wrapped_grads)
                .expect("wrapped value grad"),
        );
        assert_close(
            "erase grad",
            graph_erase.grad(&graph_grads).expect("graph erase grad"),
            wrapped_erase
                .grad(&wrapped_grads)
                .expect("wrapped erase grad"),
        );
        assert_close(
            "write grad",
            graph_write.grad(&graph_grads).expect("graph write grad"),
            wrapped_write
                .grad(&wrapped_grads)
                .expect("wrapped write grad"),
        );
        assert_close(
            "log_decay grad",
            graph_log_decay
                .grad(&graph_grads)
                .expect("graph log_decay grad"),
            wrapped_log_decay
                .grad(&wrapped_grads)
                .expect("wrapped log_decay grad"),
        );
        assert_close(
            "initial state grad",
            graph_initial_state
                .grad(&graph_grads)
                .expect("graph initial state grad"),
            wrapped_initial_state
                .grad(&wrapped_grads)
                .expect("wrapped initial state grad"),
        );
    }

    #[test]
    fn chunk_wy_custom_backward_matches_direct_graph_on_cuda() {
        run_chunk_wy_custom_backward_matches_direct_graph_on_cuda(1, 2, 5, 3, 4, 2);
    }

    #[test]
    fn chunk_wy_custom_backward_matches_direct_graph_on_cuda_multi_block_dense() {
        run_chunk_wy_custom_backward_matches_direct_graph_on_cuda(1, 2, 9, 5, 37, 4);
    }

    #[test]
    fn chunk_wy_custom_backward_matches_direct_graph_on_cuda_training_geometry() {
        run_chunk_wy_custom_backward_matches_direct_graph_on_cuda(1, 4, 12, 64, 128, 6);
    }

    #[test]
    fn chunk_wy_custom_backward_matches_direct_graph_on_cuda_full_chunk() {
        run_chunk_wy_custom_backward_matches_direct_graph_on_cuda(1, 2, 64, 16, 32, 64);
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 7> for GatedDeltaNet2ChunkWyBackward<CudaCubeBackend> {
    type State = GatedDeltaNet2BackwardState<CubeTensor<CudaRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 7>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        if ops.state.boundary_states.is_some() && ops.state.runtime_params.is_some() {
            backward_cuda_runtime_impl(ops, grads);
        } else {
            backward_impl::<CudaCubeBackend>(ops, grads);
        }
    }
}
