use std::f32::consts::PI;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData, activation};

use super::config::FusedKernelConfig;
use crate::kernel::{BlockPattern2d, linear_attention};
use crate::model::backend_float_dtype;
use crate::positional::RotaryEmbedding;

const ROW_NORM_EPS: f32 = 1e-6;
const MAX_DENSE_SCORE_DECAY_CACHE_TIME: usize = 1024;

#[derive(Debug, Clone)]
pub(crate) struct DenseScoreDecayCache<B: Backend> {
    pub score: Tensor<B, 4>,
    pub initial_state: Tensor<B, 4>,
    pub final_state: Tensor<B, 4>,
    pub carry: Tensor<B, 4>,
    time: usize,
}

impl<B: Backend> DenseScoreDecayCache<B> {
    pub(crate) fn new(slopes: &[f32], time: usize, device: &B::Device) -> Option<Self> {
        if slopes.is_empty() || time == 0 || time > MAX_DENSE_SCORE_DECAY_CACHE_TIME {
            return None;
        }

        let heads = slopes.len();
        let mut score = Vec::with_capacity(heads * time * time);
        let mut initial_state = Vec::with_capacity(heads * time);
        let mut final_state = Vec::with_capacity(heads * time);
        let mut carry = Vec::with_capacity(heads);
        for slope in slopes {
            let decay = (-slope).exp();
            for row in 0..time {
                for col in 0..time {
                    let exponent = row.saturating_sub(col) as f32;
                    score.push(if col < row { decay.powf(exponent) } else { 1.0 });
                }
            }
            for position in 0..time {
                initial_state.push(decay.powf(position as f32));
                final_state.push(decay.powf((time - position) as f32));
            }
            carry.push(decay.powf(time as f32));
        }

        Some(Self {
            score: Tensor::<B, 4>::from_data(
                TensorData::new(score, [1, heads, time, time]),
                device,
            )
            .cast(backend_float_dtype::<B>()),
            initial_state: Tensor::<B, 4>::from_data(
                TensorData::new(initial_state, [1, heads, time, 1]),
                device,
            )
            .cast(backend_float_dtype::<B>()),
            final_state: Tensor::<B, 4>::from_data(
                TensorData::new(final_state, [1, heads, time, 1]),
                device,
            )
            .cast(backend_float_dtype::<B>()),
            carry: Tensor::<B, 4>::from_data(TensorData::new(carry, [1, heads, 1, 1]), device)
                .cast(backend_float_dtype::<B>()),
            time,
        })
    }
}

#[derive(Default, Debug, Clone)]
pub struct AttentionCache<B: Backend> {
    q_rot: Option<Tensor<B, 4>>,
    value: Option<Tensor<B, 4>>,
    #[cfg(feature = "viz")]
    last_attention: Option<Tensor<B, 3>>,
}

impl<B: Backend> AttentionCache<B> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.q_rot
            .as_ref()
            .map(|tensor| tensor.shape().dims::<4>()[2])
            .unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.q_rot = None;
        self.value = None;
        #[cfg(feature = "viz")]
        {
            self.last_attention = None;
        }
    }

    pub fn append(&mut self, q_rot: Tensor<B, 4>, value: Tensor<B, 4>) {
        self.q_rot = Some(match self.q_rot.take() {
            Some(prev) => Tensor::cat(vec![prev, q_rot], 2),
            None => q_rot,
        });
        self.value = Some(match self.value.take() {
            Some(prev) => Tensor::cat(vec![prev, value], 2),
            None => value,
        });
        #[cfg(feature = "viz")]
        {
            self.last_attention = None;
        }
    }

    pub fn retain_last(&mut self, max_len: usize) {
        if max_len == 0 {
            self.reset();
            return;
        }

        if let Some(existing) = self.q_rot.take() {
            let time = existing.shape().dims::<4>()[2];
            let trimmed = if time > max_len {
                let start = time - max_len;
                existing.slice_dim(2, start..time)
            } else {
                existing
            };
            self.q_rot = Some(trimmed);
        }

        if let Some(existing) = self.value.take() {
            let time = existing.shape().dims::<4>()[2];
            let trimmed = if time > max_len {
                let start = time - max_len;
                existing.slice_dim(2, start..time)
            } else {
                existing
            };
            self.value = Some(trimmed);
        }
        #[cfg(feature = "viz")]
        {
            self.last_attention = None;
        }
    }

    #[cfg(feature = "viz")]
    pub fn set_last_attention(&mut self, row: Tensor<B, 3>) {
        self.last_attention = Some(row);
    }

    #[cfg(feature = "viz")]
    pub fn last_attention(&self) -> Option<Tensor<B, 3>> {
        self.last_attention.clone()
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    freqs: Tensor<B, 4>,
    n_head: usize,
    fused: bool,
    block_pattern: BlockPattern2d,
    use_alibi: bool,
    alibi_slopes: Tensor<B, 1>,
    alibi_decay: Tensor<B, 1>,
    #[module(skip)]
    dense_score_decay_score: Option<Tensor<B, 4>>,
    #[module(skip)]
    dense_score_decay_initial_state: Option<Tensor<B, 4>>,
    #[module(skip)]
    dense_score_decay_final_state: Option<Tensor<B, 4>>,
    #[module(skip)]
    dense_score_decay_carry: Option<Tensor<B, 4>>,
    dense_score_decay_time: usize,
    rotary_embedding: RotaryEmbedding,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        latent: usize,
        n_head: usize,
        device: &B::Device,
        kernel: &FusedKernelConfig,
    ) -> Self {
        let freqs = Self::build_freqs(latent, kernel.rope_theta, kernel.rotary_embedding, device);
        let use_alibi = matches!(kernel.rotary_embedding, RotaryEmbedding::Alibi);
        let (use_alibi, slopes) = if use_alibi {
            let slopes = kernel
                .alibi_slopes
                .clone()
                .filter(|slopes| !slopes.is_empty())
                .unwrap_or_else(|| linear_attention::default_alibi_slopes(n_head));
            (true, slopes)
        } else {
            (false, vec![0.0; n_head])
        };
        let alibi_slopes =
            Tensor::<B, 1>::from_floats(slopes.as_slice(), device).cast(backend_float_dtype::<B>());
        let decay_values = slopes
            .iter()
            .map(|slope| (-slope).exp())
            .collect::<Vec<_>>();
        let alibi_decay = Tensor::<B, 1>::from_floats(decay_values.as_slice(), device)
            .cast(backend_float_dtype::<B>());
        let dense_score_decay_cache = use_alibi
            .then(|| {
                DenseScoreDecayCache::new(
                    slopes.as_slice(),
                    kernel.block_sparse.time.block_size(),
                    device,
                )
            })
            .flatten();
        let (
            dense_score_decay_score,
            dense_score_decay_initial_state,
            dense_score_decay_final_state,
            dense_score_decay_carry,
            dense_score_decay_time,
        ) = match dense_score_decay_cache {
            Some(cache) => (
                Some(cache.score),
                Some(cache.initial_state),
                Some(cache.final_state),
                Some(cache.carry),
                cache.time,
            ),
            None => (None, None, None, None, 0),
        };

        Self {
            freqs,
            n_head,
            fused: kernel.enabled,
            block_pattern: kernel.block_sparse.time.clone(),
            use_alibi,
            alibi_slopes,
            alibi_decay,
            dense_score_decay_score,
            dense_score_decay_initial_state,
            dense_score_decay_final_state,
            dense_score_decay_carry,
            dense_score_decay_time,
            rotary_embedding: kernel.rotary_embedding,
        }
    }

    pub(crate) fn widened_from_prefix(
        &self,
        fresh: &Self,
        old_latent: usize,
        new_latent: usize,
    ) -> Result<Self, String> {
        let current_shape = self.freqs.shape().dims::<4>();
        let fresh_shape = fresh.freqs.shape().dims::<4>();
        if self.n_head != fresh.n_head
            || self.fused != fresh.fused
            || self.use_alibi != fresh.use_alibi
            || self.rotary_embedding != fresh.rotary_embedding
        {
            return Err(format!(
                "cannot widen attention with incompatible config (current_heads={} fresh_heads={} current_rotary={:?} fresh_rotary={:?})",
                self.n_head, fresh.n_head, self.rotary_embedding, fresh.rotary_embedding
            ));
        }
        if current_shape != [1, 1, 1, old_latent]
            || fresh_shape != [1, 1, 1, new_latent]
            || old_latent > new_latent
        {
            return Err(format!(
                "cannot widen attention frequencies with incompatible latent shape (current={current_shape:?}, fresh={fresh_shape:?}, old={old_latent}, new={new_latent})"
            ));
        }
        if old_latent == new_latent {
            return Ok(self.clone());
        }
        let mut widened = fresh.clone();
        widened.freqs = Tensor::cat(
            vec![
                self.freqs.clone(),
                fresh
                    .freqs
                    .clone()
                    .slice([0..1, 0..1, 0..1, old_latent..new_latent]),
            ],
            3,
        )
        .detach();
        widened.alibi_slopes = self.alibi_slopes.clone();
        widened.alibi_decay = self.alibi_decay.clone();
        widened.dense_score_decay_score = self.dense_score_decay_score.clone();
        widened.dense_score_decay_initial_state = self.dense_score_decay_initial_state.clone();
        widened.dense_score_decay_final_state = self.dense_score_decay_final_state.clone();
        widened.dense_score_decay_carry = self.dense_score_decay_carry.clone();
        widened.dense_score_decay_time = self.dense_score_decay_time;
        widened.block_pattern = self.block_pattern.clone();
        Ok(widened)
    }

    pub fn forward(&self, query: Tensor<B, 4>, value: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.fused {
            return linear_attention::fused_state_aligned(
                query,
                value,
                self.freqs.clone(),
                self.use_alibi.then_some(self.alibi_slopes.clone()),
                &self.block_pattern,
                self.rotary_embedding,
            );
        }

        let q_rot = self.rotate(query, 0);
        let k_rot = q_rot.clone();

        let mut scores = q_rot.clone().matmul(k_rot.swap_dims(2, 3)).tril(-1);
        if self.use_alibi {
            let device = q_rot.device();
            let slopes = self.alibi_slopes.clone().reshape([1, self.n_head, 1, 1]);
            let time = q_rot.shape().dims::<4>()[2];
            let pos_row = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
                .cast(backend_float_dtype::<B>())
                .reshape([1, 1, time, 1]);
            let pos_new = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
                .cast(backend_float_dtype::<B>())
                .reshape([1, 1, 1, time]);
            let alibi = slopes * (pos_new - pos_row).tril(-1);
            scores = scores + alibi;
        }
        let scores = Self::row_normalize(scores);
        let value = value.repeat_dim(1, self.n_head);

        scores.matmul(value)
    }

    pub(crate) fn rotate_positions(&self, values: Tensor<B, 4>, start: usize) -> Tensor<B, 4> {
        self.rotate(values, start)
    }

    pub(crate) fn rotate_positions_fixed(
        &self,
        values: Tensor<B, 4>,
        position: usize,
    ) -> Tensor<B, 4> {
        self.rotate_fixed(values, position)
    }

    pub(crate) fn alibi_decay(&self) -> Option<Tensor<B, 1>> {
        if !self.use_alibi {
            return None;
        }
        Some(self.alibi_decay.clone())
    }

    pub(crate) fn dense_score_decay_cache(&self, time: usize) -> Option<DenseScoreDecayCache<B>> {
        if self.dense_score_decay_time != time {
            return None;
        }
        Some(DenseScoreDecayCache {
            score: self.dense_score_decay_score.clone()?,
            initial_state: self.dense_score_decay_initial_state.clone()?,
            final_state: self.dense_score_decay_final_state.clone()?,
            carry: self.dense_score_decay_carry.clone()?,
            time,
        })
    }

    pub fn forward_cached(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        cache: &mut AttentionCache<B>,
    ) -> Tensor<B, 4> {
        let time_new = query.shape().dims::<4>()[2];
        let position = cache.len();

        let q_rot = self.rotate(query, position);
        let k_rot = q_rot.clone();
        let value_rep = value.repeat_dim(1, self.n_head);

        #[cfg(feature = "viz")]
        let mut attn_row: Option<Tensor<B, 3>> = None;

        let context = if let (Some(prev_q), Some(prev_v)) = (&cache.q_rot, &cache.value) {
            let scores_prev = q_rot.clone().matmul(prev_q.clone().swap_dims(2, 3));
            let mut scores_self = q_rot.clone().matmul(k_rot.clone().swap_dims(2, 3)).tril(-1);

            let scores_prev = if self.use_alibi {
                let device = q_rot.device();
                let slopes = self.alibi_slopes.clone().reshape([1, self.n_head, 1, 1]);
                let prev_len = position;

                let pos_row = Tensor::<B, 1, Int>::arange(
                    position as i64..(position + time_new) as i64,
                    &device,
                )
                .cast(backend_float_dtype::<B>())
                .reshape([1, 1, time_new, 1]);

                let pos_prev = Tensor::<B, 1, Int>::arange(0..prev_len as i64, &device)
                    .cast(backend_float_dtype::<B>())
                    .reshape([1, 1, 1, prev_len]);
                let alibi_prev = slopes.clone() * (pos_prev - pos_row.clone());

                let pos_new = Tensor::<B, 1, Int>::arange(
                    position as i64..(position + time_new) as i64,
                    &device,
                )
                .cast(backend_float_dtype::<B>())
                .reshape([1, 1, 1, time_new]);
                let alibi_self = slopes * (pos_new - pos_row).tril(-1);

                scores_self = scores_self + alibi_self;
                scores_prev + alibi_prev
            } else {
                scores_prev
            };

            let scores = Tensor::cat(vec![scores_prev, scores_self], 3);
            let scores = Self::row_normalize(scores);

            #[cfg(feature = "viz")]
            {
                let dims = scores.shape().dims::<4>();
                if dims[2] > 0 {
                    let row = scores
                        .clone()
                        .slice_dim(2, (dims[2] - 1)..dims[2])
                        .reshape([dims[0], dims[1], dims[3]]);
                    attn_row = Some(row);
                }
            }
            let value_all = Tensor::cat(vec![prev_v.clone(), value_rep.clone()], 2);
            scores.matmul(value_all)
        } else {
            let mut scores = q_rot.clone().matmul(k_rot.clone().swap_dims(2, 3)).tril(-1);
            if self.use_alibi {
                let device = q_rot.device();
                let slopes = self.alibi_slopes.clone().reshape([1, self.n_head, 1, 1]);
                let pos_row = Tensor::<B, 1, Int>::arange(
                    position as i64..(position + time_new) as i64,
                    &device,
                )
                .cast(backend_float_dtype::<B>())
                .reshape([1, 1, time_new, 1]);
                let pos_new = Tensor::<B, 1, Int>::arange(
                    position as i64..(position + time_new) as i64,
                    &device,
                )
                .cast(backend_float_dtype::<B>())
                .reshape([1, 1, 1, time_new]);
                let alibi = slopes * (pos_new - pos_row).tril(-1);
                scores = scores + alibi;
            }
            scores = Self::row_normalize(scores);
            #[cfg(feature = "viz")]
            {
                let dims = scores.shape().dims::<4>();
                if dims[2] > 0 {
                    let row = scores
                        .clone()
                        .slice_dim(2, (dims[2] - 1)..dims[2])
                        .reshape([dims[0], dims[1], dims[3]]);
                    attn_row = Some(row);
                }
            }
            scores.matmul(value_rep.clone())
        };

        cache.append(k_rot.clone(), value_rep.clone());

        #[cfg(feature = "viz")]
        if let Some(row) = attn_row {
            cache.set_last_attention(row);
        }

        context
    }

    fn row_normalize(scores: Tensor<B, 4>) -> Tensor<B, 4> {
        let denom = scores.clone().abs().sum_dim(3).add_scalar(ROW_NORM_EPS);
        scores / denom
    }

    fn rope(&self, phases: Tensor<B, 4>, values: Tensor<B, 4>) -> Tensor<B, 4> {
        let cos = phases.clone().cos();
        let sin = phases.sin();

        let [b, h, t, n] = values.shape().dims();
        let pairs = values.clone().reshape([b, h, t, n / 2, 2]);

        let even = pairs.clone().slice_dim(4, 0..1).squeeze_dim::<4>(4);
        let odd = pairs.slice_dim(4, 1..2).squeeze_dim::<4>(4);

        let rotated = Tensor::stack::<5>(vec![odd.clone().neg(), even], 4).reshape([b, h, t, n]);

        values * cos + rotated * sin
    }

    fn pope(&self, phases: Tensor<B, 4>, values: Tensor<B, 4>) -> Tensor<B, 4> {
        let magnitude = activation::softplus(values, 1.0);
        let cos = phases.clone().cos();
        let sin = phases.sin();
        let real = magnitude.clone() * cos;
        let imag = magnitude * sin;
        Tensor::cat(vec![real, imag], 3)
    }

    fn rotate(&self, values: Tensor<B, 4>, start: usize) -> Tensor<B, 4> {
        if self.rotary_embedding == RotaryEmbedding::Alibi {
            return values;
        }
        let time = values.shape().dims::<4>()[2];
        let device = values.device();
        let positions = Tensor::<B, 1, Int>::arange(start as i64..(start + time) as i64, &device)
            .cast(backend_float_dtype::<B>())
            .reshape([1, 1, time, 1]);

        self.rotate_with_positions(values, positions)
    }

    fn rotate_fixed(&self, values: Tensor<B, 4>, position: usize) -> Tensor<B, 4> {
        if self.rotary_embedding == RotaryEmbedding::Alibi {
            return values;
        }
        let time = values.shape().dims::<4>()[2];
        let device = values.device();
        let positions = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
            .cast(backend_float_dtype::<B>())
            .mul_scalar(0.0)
            .add_scalar(position as f32)
            .reshape([1, 1, time, 1]);

        self.rotate_with_positions(values, positions)
    }

    fn rotate_with_positions(&self, values: Tensor<B, 4>, positions: Tensor<B, 4>) -> Tensor<B, 4> {
        let latent = values.shape().dims::<4>()[3];
        let freqs = self.freqs.clone().slice([0..1, 0..1, 0..1, 0..latent]);
        let raw = positions * freqs;
        let phases = (raw.clone() - raw.clone().detach().floor()) * (2.0 * PI);
        match self.rotary_embedding {
            RotaryEmbedding::Rope => self.rope(phases, values),
            RotaryEmbedding::Pope => self.pope(phases, values),
            RotaryEmbedding::Alibi => values,
        }
    }

    fn build_freqs(
        latent: usize,
        theta: f32,
        rotary_embedding: RotaryEmbedding,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let mut data = Vec::with_capacity(latent);
        for idx in 0..latent {
            let value = match rotary_embedding {
                RotaryEmbedding::Rope => {
                    let exponent = ((idx as f32 / 2.0).floor() * 2.0) / latent as f32;
                    1.0 / theta.powf(exponent) / (2.0 * PI)
                }
                RotaryEmbedding::Pope => {
                    let exponent = idx as f32 / latent as f32;
                    1.0 / theta.powf(exponent) / (2.0 * PI)
                }
                RotaryEmbedding::Alibi => 0.0,
            };
            data.push(value);
        }
        Tensor::<B, 1>::from_floats(data.as_slice(), device)
            .cast(backend_float_dtype::<B>())
            .reshape([1, 1, 1, latent])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn alibi_decay_matches_exp_neg_slope() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut kernel = FusedKernelConfig::default();
        kernel.set_rotary_embedding(RotaryEmbedding::Alibi);
        kernel.set_alibi_slopes(vec![0.5, 1.0]);

        let attention = Attention::<TestBackend>::new(1, 2, &device, &kernel);
        let decay = attention.alibi_decay().expect("alibi decay");
        let values = decay
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("decay values");

        let expected = [(-0.5_f32).exp(), (-1.0_f32).exp()];
        for (value, exp) in values.iter().zip(expected.iter()) {
            assert!((*value - *exp).abs() < 1e-6);
        }
    }

    #[test]
    fn alibi_bias_applies_in_forward() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut kernel = FusedKernelConfig::default();
        kernel.set_rotary_embedding(RotaryEmbedding::Alibi);
        kernel.set_alibi_slopes(vec![0.5]);

        let attention = Attention::<TestBackend>::new(1, 1, &device, &kernel);
        let query = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0_f32, 1.0_f32], [1, 1, 2, 1]),
            &device,
        );
        let value = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0_f32, 3.0_f32], [1, 1, 2, 1]),
            &device,
        );

        let output = attention.forward(query, value);
        let data = output
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("output values");

        assert_eq!(data.len(), 2);
        assert!(data[0].abs() < 1e-6);
        assert!((data[1] - 1.0).abs() < 1e-4);
    }
}
