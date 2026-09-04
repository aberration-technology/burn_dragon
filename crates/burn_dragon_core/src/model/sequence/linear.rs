use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::model::attention::DenseScoreDecayCache;
use crate::model::backend_float_dtype;

pub fn expand_attention_values_to_heads<B: Backend>(
    value: Tensor<B, 4>,
    heads: usize,
) -> Tensor<B, 4> {
    match value.shape().dims::<4>()[1] {
        1 => value.repeat_dim(1, heads),
        existing if existing == heads => value,
        existing => panic!("value heads {existing} must be 1 or {heads}"),
    }
}

pub fn recurrent_attention_reference<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    decay: Option<Tensor<B, 1>>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, heads, time, latent] = query.shape().dims();
    let n_embd = value.shape().dims::<4>()[3];
    let device = value.device();
    let decay = decay.map(|tensor| tensor.reshape([1, heads, 1, 1]));

    let mut rho = match rho_state {
        Some(existing) => {
            let dims = existing.shape().dims::<4>();
            if dims == [batch, heads, latent, n_embd] {
                existing
            } else {
                Tensor::<B, 4>::zeros([batch, heads, latent, n_embd], &device)
            }
        }
        None => Tensor::<B, 4>::zeros([batch, heads, latent, n_embd], &device),
    };

    let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(time);

    for t in 0..time {
        let x_t = query.clone().slice_dim(2, t..t + 1);
        let v_t = value.clone().slice_dim(2, t..t + 1).repeat_dim(1, heads);
        let x_t_latent = x_t.swap_dims(2, 3);

        let attn_t = (rho.clone() * x_t_latent.clone())
            .sum_dim(2)
            .reshape([batch, heads, 1, n_embd]);
        outputs.push(attn_t);

        rho = rho + x_t_latent * v_t;
        if let Some(decay) = &decay {
            rho = rho * decay.clone();
        }
    }

    (Tensor::cat(outputs, 2), rho)
}

pub fn recurrent_attention_dense_score_reference<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    decay: Option<Tensor<B, 1>>,
    row_chunk: usize,
    decay_cache: Option<DenseScoreDecayCache<B>>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let (context, rho) = recurrent_attention_dense_score_reference_maybe_final(
        query,
        value,
        rho_state,
        decay,
        row_chunk,
        decay_cache,
        true,
    );
    (
        context,
        rho.expect("dense-score attention requested terminal sequence state"),
    )
}

pub fn recurrent_attention_dense_score_context_reference<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    decay: Option<Tensor<B, 1>>,
    row_chunk: usize,
    decay_cache: Option<DenseScoreDecayCache<B>>,
) -> Tensor<B, 4> {
    recurrent_attention_dense_score_reference_maybe_final(
        query,
        value,
        rho_state,
        decay,
        row_chunk,
        decay_cache,
        false,
    )
    .0
}

fn recurrent_attention_dense_score_reference_maybe_final<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    decay: Option<Tensor<B, 1>>,
    row_chunk: usize,
    decay_cache: Option<DenseScoreDecayCache<B>>,
    retain_final_rho: bool,
) -> (Tensor<B, 4>, Option<Tensor<B, 4>>) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let n_embd = value.shape().dims::<4>()[3];
    let device = value.device();

    let row_chunk = row_chunk.max(1);
    if time <= row_chunk {
        return recurrent_attention_dense_score_reference_full_maybe_final(
            query,
            value,
            rho_state,
            decay,
            decay_cache,
            retain_final_rho,
        );
    }

    let value = expand_attention_values_to_heads(value, heads);
    let rho_state =
        rho_state.filter(|state| state.shape().dims::<4>() == [batch, heads, latent, n_embd]);
    let query_key = query.clone().swap_dims(2, 3);
    let pos_col = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
        .cast(backend_float_dtype::<B>())
        .reshape([1, 1, 1, time]);
    let decay_heads = decay.clone().map(|tensor| tensor.reshape([1, heads, 1, 1]));

    let rho = retain_final_rho.then(|| {
        recurrent_attention_dense_score_final_rho_reference(
            query.clone(),
            value.clone(),
            rho_state.clone(),
            decay.clone(),
        )
    });

    let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(time.div_ceil(row_chunk));
    for start in (0..time).step_by(row_chunk) {
        let end = (start + row_chunk).min(time);
        let rows = end.saturating_sub(start);
        let q_chunk = query.clone().slice_dim(2, start..end);
        let mut score_chunk = q_chunk
            .clone()
            .matmul(query_key.clone())
            .tril(start as i64 - 1);
        let initial_context_chunk = if let Some(decay_heads) = decay_heads.clone() {
            let pos_row = Tensor::<B, 1, Int>::arange(start as i64..end as i64, &device)
                .cast(backend_float_dtype::<B>())
                .reshape([1, 1, rows, 1]);
            let diff = (pos_row.clone() - pos_col.clone())
                .tril(start as i64 - 1)
                .repeat_dim(1, heads);
            let decay_score = decay_heads.clone().repeat_dim(2, rows).repeat_dim(3, time);
            score_chunk = score_chunk * decay_score.powf(diff);

            if let Some(rho_state) = rho_state.clone() {
                let decay_state = decay_heads
                    .clone()
                    .repeat_dim(2, rows)
                    .powf(pos_row.repeat_dim(1, heads));
                q_chunk
                    .clone()
                    .mul(decay_state)
                    .matmul(rho_state)
                    .reshape([batch, heads, rows, n_embd])
            } else {
                Tensor::<B, 4>::zeros([batch, heads, rows, n_embd], &device)
            }
        } else if let Some(rho_state) = rho_state.clone() {
            q_chunk
                .clone()
                .matmul(rho_state)
                .reshape([batch, heads, rows, n_embd])
        } else {
            Tensor::<B, 4>::zeros([batch, heads, rows, n_embd], &device)
        };

        let chunk_context = initial_context_chunk
            + score_chunk
                .matmul(value.clone())
                .reshape([batch, heads, rows, n_embd]);
        outputs.push(chunk_context);
    }

    (Tensor::cat(outputs, 2), rho)
}

#[cfg(test)]
fn recurrent_attention_dense_score_reference_full<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    decay: Option<Tensor<B, 1>>,
    decay_cache: Option<DenseScoreDecayCache<B>>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let (context, rho) = recurrent_attention_dense_score_reference_full_maybe_final(
        query,
        value,
        rho_state,
        decay,
        decay_cache,
        true,
    );
    (
        context,
        rho.expect("dense-score attention requested terminal sequence state"),
    )
}

fn recurrent_attention_dense_score_reference_full_maybe_final<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    decay: Option<Tensor<B, 1>>,
    decay_cache: Option<DenseScoreDecayCache<B>>,
    retain_final_rho: bool,
) -> (Tensor<B, 4>, Option<Tensor<B, 4>>) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let n_embd = value.shape().dims::<4>()[3];
    let device = value.device();
    let value = expand_attention_values_to_heads(value, heads);
    let rho_state =
        rho_state.filter(|state| state.shape().dims::<4>() == [batch, heads, latent, n_embd]);

    let pos_row = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
        .cast(backend_float_dtype::<B>())
        .reshape([1, 1, time, 1]);
    let pos_col = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
        .cast(backend_float_dtype::<B>())
        .reshape([1, 1, 1, time]);

    let mut scores = query.clone().matmul(query.clone().swap_dims(2, 3)).tril(-1);
    let (initial_context, rho) = if let Some(decay) = decay {
        let (decay_score, decay_state, decay_final, decay_carry) = if let Some(cache) = decay_cache
        {
            (
                cache.score,
                cache.initial_state,
                cache.final_state,
                cache.carry,
            )
        } else {
            let diff = (pos_row.clone() - pos_col.clone())
                .tril(-1)
                .repeat_dim(1, heads);
            let decay_heads = decay.clone().reshape([1, heads, 1, 1]);
            let decay_score = decay_heads
                .clone()
                .repeat_dim(2, time)
                .repeat_dim(3, time)
                .powf(diff);
            let decay_state = decay_heads
                .clone()
                .repeat_dim(2, time)
                .powf(pos_row.clone().repeat_dim(1, heads));
            let final_exponents = pos_row
                .clone()
                .mul_scalar(-1.0)
                .add_scalar(time as f32)
                .repeat_dim(1, heads);
            let decay_final = decay_heads
                .clone()
                .repeat_dim(2, time)
                .powf(final_exponents);
            let decay_carry = decay_heads.powf_scalar(time as f32);
            (decay_score, decay_state, decay_final, decay_carry)
        };
        scores = scores * decay_score;

        let initial_context = if let Some(rho_state) = rho_state.clone() {
            query
                .clone()
                .mul(decay_state.clone())
                .matmul(rho_state)
                .reshape([batch, heads, time, n_embd])
        } else {
            Tensor::<B, 4>::zeros([batch, heads, time, n_embd], &device)
        };

        let rho = retain_final_rho.then(|| {
            if let Some(rho_state) = rho_state {
                rho_state.mul(decay_carry)
                    + query.mul(decay_final).swap_dims(2, 3).matmul(value.clone())
            } else {
                query.mul(decay_final).swap_dims(2, 3).matmul(value.clone())
            }
        });

        (initial_context, rho)
    } else {
        let initial_context = if let Some(rho_state) = rho_state.clone() {
            query
                .clone()
                .matmul(rho_state)
                .reshape([batch, heads, time, n_embd])
        } else {
            Tensor::<B, 4>::zeros([batch, heads, time, n_embd], &device)
        };
        let rho = retain_final_rho.then(|| {
            if let Some(rho_state) = rho_state {
                rho_state + query.swap_dims(2, 3).matmul(value.clone())
            } else {
                query.swap_dims(2, 3).matmul(value.clone())
            }
        });
        (initial_context, rho)
    };

    let context = initial_context + scores.matmul(value).reshape([batch, heads, time, n_embd]);
    (context, rho)
}

pub fn recurrent_attention_dense_score_final_rho_reference<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    decay: Option<Tensor<B, 1>>,
) -> Tensor<B, 4> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let n_embd = value.shape().dims::<4>()[3];
    let device = value.device();
    let value = expand_attention_values_to_heads(value, heads);
    let rho_state =
        rho_state.filter(|state| state.shape().dims::<4>() == [batch, heads, latent, n_embd]);

    if let Some(decay) = decay {
        let pos_row = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
            .cast(backend_float_dtype::<B>())
            .reshape([1, 1, time, 1]);
        let final_exponents = pos_row
            .mul_scalar(-1.0)
            .add_scalar(time as f32)
            .repeat_dim(1, heads);
        let decay_final = decay
            .clone()
            .reshape([1, heads, 1, 1])
            .repeat_dim(2, time)
            .powf(final_exponents);
        let contribution = query.mul(decay_final).swap_dims(2, 3).matmul(value);
        if let Some(rho_state) = rho_state {
            rho_state.mul(decay.reshape([1, heads, 1, 1]).powf_scalar(time as f32)) + contribution
        } else {
            contribution
        }
    } else {
        let contribution = query.swap_dims(2, 3).matmul(value);
        if let Some(rho_state) = rho_state {
            rho_state + contribution
        } else {
            contribution
        }
    }
}

pub fn recurrent_attention_dense_score_initial_context_reference<B: Backend>(
    query: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    decay: Option<Tensor<B, 1>>,
    n_embd: usize,
) -> Tensor<B, 4> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let device = query.device();
    let rho_state =
        rho_state.filter(|state| state.shape().dims::<4>() == [batch, heads, latent, n_embd]);

    if let Some(decay) = decay {
        let Some(rho_state) = rho_state else {
            return Tensor::<B, 4>::zeros([batch, heads, time, n_embd], &device);
        };
        let pos_row = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
            .cast(backend_float_dtype::<B>())
            .reshape([1, 1, time, 1]);
        let decay_state = decay
            .reshape([1, heads, 1, 1])
            .repeat_dim(2, time)
            .powf(pos_row.repeat_dim(1, heads));
        query
            .mul(decay_state)
            .matmul(rho_state)
            .reshape([batch, heads, time, n_embd])
    } else {
        let Some(rho_state) = rho_state else {
            return Tensor::<B, 4>::zeros([batch, heads, time, n_embd], &device);
        };
        query
            .matmul(rho_state)
            .reshape([batch, heads, time, n_embd])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;
    type TestAutodiffBackend = Autodiff<TestBackend>;

    fn tensor4(values: Vec<f32>, shape: [usize; 4]) -> Tensor<TestBackend, 4> {
        Tensor::<TestBackend, 4>::from_data(TensorData::new(values, shape), &Default::default())
    }

    fn max_abs_diff(lhs: Tensor<TestBackend, 4>, rhs: Tensor<TestBackend, 4>) -> f32 {
        let lhs = lhs
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("lhs vec");
        let rhs = rhs
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rhs vec");
        lhs.into_iter()
            .zip(rhs)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn recurrent_attention_rho_can_be_signed_with_nonnegative_query() {
        let query = tensor4(vec![0.5, 1.0], [1, 1, 1, 2]);
        let value = tensor4(vec![2.0, -3.0, 4.0], [1, 1, 1, 3]);

        let (_context, rho) = recurrent_attention_reference(query, value, None, None);
        let rho = rho
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rho values");

        assert!(
            rho.iter().any(|value| *value < 0.0),
            "rho should remain signed because linear attention stores query^T * value"
        );
    }

    #[test]
    fn dense_score_matches_recurrent_attention_with_decay_and_state() {
        let batch = 2;
        let heads = 3;
        let time = 16;
        let latent = 5;
        let n_embd = 7;
        let query_shape = [batch, heads, time, latent];
        let value_shape = [batch, 1, time, n_embd];
        let rho_shape = [batch, heads, latent, n_embd];
        let query = tensor4(
            (0..query_shape.iter().product::<usize>())
                .map(|index| ((index * 5) % 97) as f32 / 97.0)
                .collect(),
            query_shape,
        );
        let value = tensor4(
            (0..value_shape.iter().product::<usize>())
                .map(|index| (((index * 7) % 89) as f32 - 44.0) / 89.0)
                .collect(),
            value_shape,
        );
        let rho_state = tensor4(
            (0..rho_shape.iter().product::<usize>())
                .map(|index| (((index * 11) % 83) as f32 - 41.0) / 83.0)
                .collect(),
            rho_shape,
        );
        let decay = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.91f32, 0.95, 0.98], [heads]),
            &Default::default(),
        );

        let (recurrent_context, recurrent_rho) = recurrent_attention_reference(
            query.clone(),
            value.clone(),
            Some(rho_state.clone()),
            Some(decay.clone()),
        );
        let (dense_context, dense_rho) = recurrent_attention_dense_score_reference(
            query,
            value,
            Some(rho_state),
            Some(decay),
            64,
            None,
        );

        assert!(max_abs_diff(recurrent_context, dense_context) < 2.0e-4);
        assert!(max_abs_diff(recurrent_rho, dense_rho) < 2.0e-4);
    }

    #[test]
    fn chunked_dense_score_reference_matches_full_without_decay() {
        let row_chunk = 256;
        let time = row_chunk + 64;
        let shape = [2, 3, time, 8];
        let value_shape = [2, 1, time, 8];
        let query = tensor4(
            (0..shape.iter().product::<usize>())
                .map(|index| (index % 97) as f32 / 97.0)
                .collect(),
            shape,
        );
        let value = tensor4(
            (0..value_shape.iter().product::<usize>())
                .map(|index| ((index * 3) % 89) as f32 / 89.0)
                .collect(),
            value_shape,
        );

        let (chunked_context, chunked_rho) = recurrent_attention_dense_score_reference(
            query.clone(),
            value.clone(),
            None,
            None,
            row_chunk,
            None,
        );
        let (full_context, full_rho) =
            recurrent_attention_dense_score_reference_full(query, value, None, None, None);

        assert!(max_abs_diff(chunked_context, full_context) < 1.0e-4);
        assert!(max_abs_diff(chunked_rho, full_rho) < 1.0e-4);
    }

    #[test]
    fn chunked_dense_score_reference_matches_full_with_decay_and_state() {
        let row_chunk = 256;
        let time = row_chunk + 64;
        let shape = [1, 4, time, 6];
        let value_shape = [1, 1, time, 5];
        let rho_shape = [1, 4, 6, 5];
        let query = tensor4(
            (0..shape.iter().product::<usize>())
                .map(|index| ((index * 5) % 113) as f32 / 113.0)
                .collect(),
            shape,
        );
        let value = tensor4(
            (0..value_shape.iter().product::<usize>())
                .map(|index| ((index * 7) % 101) as f32 / 101.0)
                .collect(),
            value_shape,
        );
        let rho_state = tensor4(
            (0..rho_shape.iter().product::<usize>())
                .map(|index| ((index * 11) % 79) as f32 / 79.0)
                .collect(),
            rho_shape,
        );
        let decay = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.91f32, 0.93, 0.95, 0.97], [4]),
            &Default::default(),
        );

        let (chunked_context, chunked_rho) = recurrent_attention_dense_score_reference(
            query.clone(),
            value.clone(),
            Some(rho_state.clone()),
            Some(decay.clone()),
            row_chunk,
            None,
        );
        let (full_context, full_rho) = recurrent_attention_dense_score_reference_full(
            query,
            value,
            Some(rho_state),
            Some(decay),
            None,
        );

        assert!(max_abs_diff(chunked_context, full_context) < 2.0e-4);
        assert!(max_abs_diff(chunked_rho, full_rho) < 2.0e-4);
    }

    #[test]
    fn cached_decay_matches_dynamic_dense_score_reference() {
        let time = 16;
        let shape = [2, 2, time, 6];
        let value_shape = [2, 1, time, 5];
        let rho_shape = [2, 2, 6, 5];
        let query = tensor4(
            (0..shape.iter().product::<usize>())
                .map(|index| ((index * 5) % 97) as f32 / 97.0)
                .collect(),
            shape,
        );
        let value = tensor4(
            (0..value_shape.iter().product::<usize>())
                .map(|index| ((index * 7) % 89) as f32 / 89.0)
                .collect(),
            value_shape,
        );
        let rho_state = tensor4(
            (0..rho_shape.iter().product::<usize>())
                .map(|index| ((index * 11) % 83) as f32 / 83.0)
                .collect(),
            rho_shape,
        );
        let slopes = [0.25f32, 0.75];
        let decay = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(
                slopes.iter().map(|slope| (-slope).exp()).collect(),
                [slopes.len()],
            ),
            &Default::default(),
        );
        let cache = DenseScoreDecayCache::<TestBackend>::new(&slopes, time, &Default::default())
            .expect("bounded decay cache");

        let (cached_context, cached_rho) = recurrent_attention_dense_score_reference_full(
            query.clone(),
            value.clone(),
            Some(rho_state.clone()),
            Some(decay.clone()),
            Some(cache),
        );
        let (dynamic_context, dynamic_rho) = recurrent_attention_dense_score_reference_full(
            query,
            value,
            Some(rho_state),
            Some(decay),
            None,
        );

        assert!(max_abs_diff(cached_context, dynamic_context) < 1.0e-5);
        assert!(max_abs_diff(cached_rho, dynamic_rho) < 1.0e-5);
    }

    #[test]
    fn context_only_dense_score_matches_stateful_context_and_gradients() {
        let device = burn::tensor::Device::<TestAutodiffBackend>::default();
        let query = Tensor::<TestAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(2 * 2 * 8 * 4))
                    .map(|index| ((index * 5) % 97) as f32 / 97.0)
                    .collect(),
                [2, 2, 8, 4],
            ),
            &device,
        )
        .require_grad();
        let value = Tensor::<TestAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(2 * 8 * 3))
                    .map(|index| ((index * 7) % 89) as f32 / 89.0)
                    .collect(),
                [2, 1, 8, 3],
            ),
            &device,
        )
        .require_grad();
        let rho_state = Tensor::<TestAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(2 * 2 * 4 * 3))
                    .map(|index| ((index * 11) % 83) as f32 / 83.0)
                    .collect(),
                [2, 2, 4, 3],
            ),
            &device,
        )
        .require_grad();
        let decay = Tensor::<TestAutodiffBackend, 1>::from_data(
            TensorData::new(vec![0.91f32, 0.97], [2]),
            &device,
        );
        let weight = Tensor::<TestAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(2 * 2 * 8 * 3))
                    .map(|index| ((index * 13) % 79) as f32 / 79.0)
                    .collect(),
                [2, 2, 8, 3],
            ),
            &device,
        );

        let (stateful_context, _rho) = recurrent_attention_dense_score_reference(
            query.clone(),
            value.clone(),
            Some(rho_state.clone()),
            Some(decay.clone()),
            512,
            None,
        );
        let context_only = recurrent_attention_dense_score_context_reference(
            query.clone(),
            value.clone(),
            Some(rho_state.clone()),
            Some(decay),
            512,
            None,
        );
        assert!(
            max_abs_diff(
                stateful_context.clone().inner(),
                context_only.clone().inner()
            ) < 1.0e-6
        );

        let stateful_grads = (stateful_context * weight.clone()).sum().backward();
        let context_only_grads = (context_only * weight).sum().backward();
        assert!(
            max_abs_diff(
                query
                    .grad(&stateful_grads)
                    .expect("stateful query gradient"),
                query
                    .grad(&context_only_grads)
                    .expect("context-only query gradient"),
            ) < 1.0e-6
        );
        assert!(
            max_abs_diff(
                value
                    .grad(&stateful_grads)
                    .expect("stateful value gradient"),
                value
                    .grad(&context_only_grads)
                    .expect("context-only value gradient"),
            ) < 1.0e-6
        );
        assert!(
            max_abs_diff(
                rho_state
                    .grad(&stateful_grads)
                    .expect("stateful rho gradient"),
                rho_state
                    .grad(&context_only_grads)
                    .expect("context-only rho gradient"),
            ) < 1.0e-6
        );
    }

    #[test]
    fn cached_decay_matches_dynamic_query_value_and_rho_gradients() {
        let device = burn::tensor::Device::<TestAutodiffBackend>::default();
        let time = 8;
        let query = Tensor::<TestAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(2 * 2 * time * 4))
                    .map(|index| ((index * 5) % 97) as f32 / 97.0)
                    .collect(),
                [2, 2, time, 4],
            ),
            &device,
        )
        .require_grad();
        let value = Tensor::<TestAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(2 * time * 3))
                    .map(|index| ((index * 7) % 89) as f32 / 89.0)
                    .collect(),
                [2, 1, time, 3],
            ),
            &device,
        )
        .require_grad();
        let rho_state = Tensor::<TestAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(2 * 2 * 4 * 3))
                    .map(|index| ((index * 11) % 83) as f32 / 83.0)
                    .collect(),
                [2, 2, 4, 3],
            ),
            &device,
        )
        .require_grad();
        let slopes = [0.25f32, 0.75];
        let decay = Tensor::<TestAutodiffBackend, 1>::from_data(
            TensorData::new(
                slopes.iter().map(|slope| (-slope).exp()).collect(),
                [slopes.len()],
            ),
            &device,
        );
        let cache = DenseScoreDecayCache::<TestAutodiffBackend>::new(&slopes, time, &device)
            .expect("bounded decay cache");
        let context_weight = Tensor::<TestAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(2 * 2 * time * 3))
                    .map(|index| ((index * 13) % 79) as f32 / 79.0)
                    .collect(),
                [2, 2, time, 3],
            ),
            &device,
        );
        let rho_weight = Tensor::<TestAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(2 * 2 * 4 * 3))
                    .map(|index| ((index * 17) % 73) as f32 / 73.0)
                    .collect(),
                [2, 2, 4, 3],
            ),
            &device,
        );

        let (cached_context, cached_rho) = recurrent_attention_dense_score_reference_full(
            query.clone(),
            value.clone(),
            Some(rho_state.clone()),
            Some(decay.clone()),
            Some(cache),
        );
        let (dynamic_context, dynamic_rho) = recurrent_attention_dense_score_reference_full(
            query.clone(),
            value.clone(),
            Some(rho_state.clone()),
            Some(decay),
            None,
        );
        let cached_grads = ((cached_context * context_weight.clone()).sum()
            + (cached_rho * rho_weight.clone()).sum())
        .backward();
        let dynamic_grads = ((dynamic_context * context_weight).sum()
            + (dynamic_rho * rho_weight).sum())
        .backward();

        assert!(
            max_abs_diff(
                query.grad(&cached_grads).expect("cached query gradient"),
                query.grad(&dynamic_grads).expect("dynamic query gradient"),
            ) < 2.0e-5
        );
        assert!(
            max_abs_diff(
                value.grad(&cached_grads).expect("cached value gradient"),
                value.grad(&dynamic_grads).expect("dynamic value gradient"),
            ) < 2.0e-5
        );
        assert!(
            max_abs_diff(
                rho_state.grad(&cached_grads).expect("cached rho gradient"),
                rho_state
                    .grad(&dynamic_grads)
                    .expect("dynamic rho gradient"),
            ) < 2.0e-5
        );
    }
}
