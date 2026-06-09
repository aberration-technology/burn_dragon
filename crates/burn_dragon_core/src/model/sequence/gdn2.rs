use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Tensor, activation};
pub use burn_gdn::GatedDeltaNet2GateMode;
use serde::{Deserialize, Serialize};

use super::linear::expand_attention_values_to_heads;

fn default_chunk_size() -> usize {
    64
}

fn default_true() -> bool {
    true
}

fn default_state_epsilon() -> f32 {
    1.0e-6
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GatedDeltaNet2StatePrecision {
    #[default]
    F32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct GatedDeltaNet2Config {
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default)]
    pub implementation: GatedDeltaNet2Implementation,
    #[serde(default = "default_true")]
    pub qk_l2_norm: bool,
    #[serde(default)]
    pub allow_neg_eigval: bool,
    #[serde(default)]
    pub erase_gate: GatedDeltaNet2GateMode,
    #[serde(default)]
    pub write_gate: GatedDeltaNet2GateMode,
    #[serde(default)]
    pub decay_gate: GatedDeltaNet2GateMode,
    #[serde(default)]
    pub state_precision: GatedDeltaNet2StatePrecision,
    #[serde(default = "default_state_epsilon")]
    pub state_epsilon: f32,
    #[serde(default = "default_output_scale")]
    pub output_scale: f32,
}

impl Default for GatedDeltaNet2Config {
    fn default() -> Self {
        Self {
            chunk_size: default_chunk_size(),
            implementation: GatedDeltaNet2Implementation::default(),
            qk_l2_norm: default_true(),
            allow_neg_eigval: false,
            erase_gate: GatedDeltaNet2GateMode::Channel,
            write_gate: GatedDeltaNet2GateMode::Channel,
            decay_gate: GatedDeltaNet2GateMode::Channel,
            state_precision: GatedDeltaNet2StatePrecision::F32,
            state_epsilon: default_state_epsilon(),
            output_scale: default_output_scale(),
        }
    }
}

fn default_output_scale() -> f32 {
    1.0
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GatedDeltaNet2Implementation {
    #[default]
    BdhAdapterLegacy,
    #[serde(alias = "upstream")]
    UpstreamFull,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedGatedDeltaNet2Config {
    pub n_head: usize,
    pub dense_dim: usize,
    pub max_latent_per_head: usize,
    pub chunk_size: usize,
    pub implementation: GatedDeltaNet2Implementation,
    pub qk_l2_norm: bool,
    pub allow_neg_eigval: bool,
    pub erase_gate: GatedDeltaNet2GateMode,
    pub write_gate: GatedDeltaNet2GateMode,
    pub decay_gate: GatedDeltaNet2GateMode,
    pub state_precision: GatedDeltaNet2StatePrecision,
    pub state_epsilon: f32,
    pub output_scale: f32,
}

impl GatedDeltaNet2Config {
    pub fn validate(
        &self,
        n_head: usize,
        dense_dim: usize,
        max_latent_per_head: usize,
    ) -> Result<(), String> {
        if n_head == 0 {
            return Err("gated_deltanet2 requires n_head > 0".to_string());
        }
        if dense_dim == 0 {
            return Err("gated_deltanet2 requires n_embd > 0".to_string());
        }
        if max_latent_per_head == 0 {
            return Err("gated_deltanet2 requires latent_per_head > 0".to_string());
        }
        if self.chunk_size == 0 {
            return Err("gated_deltanet2.chunk_size must be > 0".to_string());
        }
        if self.state_epsilon <= 0.0 || !self.state_epsilon.is_finite() {
            return Err("gated_deltanet2.state_epsilon must be finite and > 0".to_string());
        }
        if !self.output_scale.is_finite() {
            return Err("gated_deltanet2.output_scale must be finite".to_string());
        }
        Ok(())
    }

    pub fn resolve(
        &self,
        n_head: usize,
        dense_dim: usize,
        max_latent_per_head: usize,
    ) -> ResolvedGatedDeltaNet2Config {
        self.validate(n_head, dense_dim, max_latent_per_head)
            .unwrap_or_else(|message| panic!("{message}"));
        ResolvedGatedDeltaNet2Config {
            n_head,
            dense_dim,
            max_latent_per_head,
            chunk_size: self.chunk_size,
            implementation: self.implementation,
            qk_l2_norm: self.qk_l2_norm,
            allow_neg_eigval: self.allow_neg_eigval,
            erase_gate: self.erase_gate,
            write_gate: self.write_gate,
            decay_gate: self.decay_gate,
            state_precision: self.state_precision,
            state_epsilon: self.state_epsilon.max(1.0e-12),
            output_scale: self.output_scale,
        }
    }
}

impl ResolvedGatedDeltaNet2Config {
    pub fn upstream_config(
        self,
        executor: burn_gdn::GatedDeltaNet2Executor,
    ) -> burn_gdn::GatedDeltaNet2Config {
        burn_gdn::GatedDeltaNet2Config {
            heads: self.n_head,
            latent_per_head: self.max_latent_per_head,
            chunk_size: self.chunk_size,
            qk_l2_norm: self.qk_l2_norm,
            allow_neg_eigval: self.allow_neg_eigval,
            erase_gate: self.erase_gate,
            write_gate: self.write_gate,
            decay_gate: self.decay_gate,
            state_epsilon: self.state_epsilon,
            output_scale: self.output_scale,
            executor,
        }
    }
}

#[derive(Module, Debug)]
pub struct GatedDeltaNet2Parameters<B: Backend> {
    n_head: usize,
    dense_dim: usize,
    max_latent_per_head: usize,
    key_proj: Param<Tensor<B, 3>>,
    erase_proj: Param<Tensor<B, 3>>,
    erase_bias: Param<Tensor<B, 2>>,
    decay_proj: Param<Tensor<B, 3>>,
    decay_log: Param<Tensor<B, 2>>,
    decay_bias: Param<Tensor<B, 2>>,
    write_proj: Param<Tensor<B, 3>>,
    write_bias: Param<Tensor<B, 2>>,
}

#[derive(Debug)]
pub struct GatedDeltaNet2Inputs<B: Backend> {
    pub key: Tensor<B, 4>,
    pub erase: Tensor<B, 4>,
    pub write: Tensor<B, 4>,
    pub log_decay: Tensor<B, 4>,
}

impl<B: Backend> GatedDeltaNet2Parameters<B> {
    pub fn new(config: ResolvedGatedDeltaNet2Config, device: &B::Device) -> Self {
        let latent_std = (1.0 / config.dense_dim.max(1) as f32).sqrt().min(0.02);
        let write_std = (1.0 / config.dense_dim.max(1) as f32).sqrt().min(0.02);
        let zeros_latent =
            Tensor::<B, 2>::zeros([config.n_head, config.max_latent_per_head], device);
        let decay_log = Tensor::<B, 2>::zeros([config.n_head, config.max_latent_per_head], device);
        let decay_bias = Tensor::<B, 2>::ones([config.n_head, config.max_latent_per_head], device)
            .mul_scalar(-4.6);
        Self {
            n_head: config.n_head,
            dense_dim: config.dense_dim,
            max_latent_per_head: config.max_latent_per_head,
            key_proj: Param::from_tensor(Tensor::<B, 3>::random(
                [config.n_head, config.dense_dim, config.max_latent_per_head],
                TensorDistribution::Normal(0.0, latent_std as f64),
                device,
            )),
            erase_proj: Param::from_tensor(Tensor::<B, 3>::random(
                [config.n_head, config.dense_dim, config.max_latent_per_head],
                TensorDistribution::Normal(0.0, latent_std as f64),
                device,
            )),
            erase_bias: Param::from_tensor(zeros_latent.clone()),
            decay_proj: Param::from_tensor(Tensor::<B, 3>::random(
                [config.n_head, config.dense_dim, config.max_latent_per_head],
                TensorDistribution::Normal(0.0, latent_std as f64),
                device,
            )),
            decay_log: Param::from_tensor(decay_log),
            decay_bias: Param::from_tensor(decay_bias),
            write_proj: Param::from_tensor(Tensor::<B, 3>::random(
                [config.n_head, config.dense_dim, config.dense_dim],
                TensorDistribution::Normal(0.0, write_std as f64),
                device,
            )),
            write_bias: Param::from_tensor(Tensor::<B, 2>::zeros(
                [config.n_head, config.dense_dim],
                device,
            )),
        }
    }

    pub fn project_inputs(
        &self,
        dense: Tensor<B, 4>,
        latent_per_head: usize,
        config: ResolvedGatedDeltaNet2Config,
    ) -> GatedDeltaNet2Inputs<B> {
        assert!(
            latent_per_head <= self.max_latent_per_head,
            "gated_deltanet2 latent_per_head {} exceeds configured max {}",
            latent_per_head,
            self.max_latent_per_head
        );
        let key = self.project_latent(dense.clone(), self.key_proj.val(), None, latent_per_head);
        let erase_logits = self.project_latent(
            dense.clone(),
            self.erase_proj.val(),
            Some(self.erase_bias.val()),
            latent_per_head,
        );
        let decay_logits = self.project_latent(
            dense.clone(),
            self.decay_proj.val(),
            Some(self.decay_bias.val()),
            latent_per_head,
        );
        let write_logits = self.project_dense(dense);

        let erase = apply_gate_mode(erase_logits, config.erase_gate, config.allow_neg_eigval);
        let write = apply_gate_mode(write_logits, config.write_gate, false);
        let log_decay = self.apply_decay_mode(decay_logits, latent_per_head, config.decay_gate);
        GatedDeltaNet2Inputs {
            key,
            erase,
            write,
            log_decay,
        }
    }

    fn project_latent(
        &self,
        dense: Tensor<B, 4>,
        weight: Tensor<B, 3>,
        bias: Option<Tensor<B, 2>>,
        latent_per_head: usize,
    ) -> Tensor<B, 4> {
        let weight = weight
            .slice([0..self.n_head, 0..self.dense_dim, 0..latent_per_head])
            .reshape([1, self.n_head, self.dense_dim, latent_per_head]);
        let mut projected = dense.matmul(weight);
        if let Some(bias) = bias {
            projected = projected
                + bias.slice([0..self.n_head, 0..latent_per_head]).reshape([
                    1,
                    self.n_head,
                    1,
                    latent_per_head,
                ]);
        }
        projected
    }

    fn project_dense(&self, dense: Tensor<B, 4>) -> Tensor<B, 4> {
        let weight =
            self.write_proj
                .val()
                .reshape([1, self.n_head, self.dense_dim, self.dense_dim]);
        dense.matmul(weight)
            + self
                .write_bias
                .val()
                .reshape([1, self.n_head, 1, self.dense_dim])
    }

    fn apply_decay_mode(
        &self,
        logits: Tensor<B, 4>,
        latent_per_head: usize,
        mode: GatedDeltaNet2GateMode,
    ) -> Tensor<B, 4> {
        let [batch, heads, time, latent] = logits.shape().dims::<4>();
        let device = logits.device();
        if matches!(mode, GatedDeltaNet2GateMode::Disabled) {
            return Tensor::<B, 4>::zeros([batch, heads, time, latent], &device);
        }
        let logits = match mode {
            GatedDeltaNet2GateMode::Channel => logits,
            GatedDeltaNet2GateMode::Scalar => logits.mean_dim(3).repeat_dim(3, latent),
            GatedDeltaNet2GateMode::Disabled => unreachable!(),
        };
        let decay_rate = self
            .decay_log
            .val()
            .slice([0..self.n_head, 0..latent_per_head])
            .exp()
            .reshape([1, self.n_head, 1, latent_per_head]);
        activation::softplus(logits, 1.0)
            .mul(decay_rate)
            .mul_scalar(-1.0)
    }
}

fn apply_gate_mode<B: Backend>(
    logits: Tensor<B, 4>,
    mode: GatedDeltaNet2GateMode,
    allow_neg_eigval: bool,
) -> Tensor<B, 4> {
    let [batch, heads, time, channels] = logits.shape().dims::<4>();
    let device = logits.device();
    let gate = match mode {
        GatedDeltaNet2GateMode::Channel => activation::sigmoid(logits),
        GatedDeltaNet2GateMode::Scalar => {
            activation::sigmoid(logits.mean_dim(3)).repeat_dim(3, channels)
        }
        GatedDeltaNet2GateMode::Disabled => {
            Tensor::<B, 4>::ones([batch, heads, time, channels], &device)
        }
    };
    if allow_neg_eigval && !matches!(mode, GatedDeltaNet2GateMode::Disabled) {
        gate.mul_scalar(2.0)
    } else {
        gate
    }
}

pub(crate) fn l2_normalize_last<B: Backend>(values: Tensor<B, 4>, epsilon: f32) -> Tensor<B, 4> {
    burn_gdn::l2_normalize_last(values, epsilon as f64)
}

#[allow(clippy::too_many_arguments)]
pub fn gated_deltanet2_reference<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    erase: Tensor<B, 4>,
    write: Tensor<B, 4>,
    log_decay: Tensor<B, 4>,
    state: Option<Tensor<B, 4>>,
    qk_l2_norm: bool,
    epsilon: f32,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let value = expand_attention_values_to_heads(value, heads);
    let dense_dim = value.shape().dims::<4>()[3];
    let device = value.device();
    let query = if qk_l2_norm {
        l2_normalize_last(query, epsilon)
    } else {
        query
    };
    let key = if qk_l2_norm {
        l2_normalize_last(key, epsilon)
    } else {
        key
    };
    let state = match state {
        Some(existing) if existing.shape().dims::<4>() == [batch, heads, latent, dense_dim] => {
            existing
        }
        _ => Tensor::<B, 4>::zeros([batch, heads, latent, dense_dim], &device),
    };
    burn_gdn::gated_deltanet2_reference(
        query,
        key,
        value,
        erase,
        write,
        log_decay,
        state,
        time.max(1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

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

    #[allow(clippy::too_many_arguments)]
    fn wy_chunk_reference(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        erase: &[f32],
        write: &[f32],
        log_decay: &[f32],
        initial_state: &[f32],
        time: usize,
        latent: usize,
        dense: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let latent_index = |t: usize, l: usize| t * latent + l;
        let dense_index = |t: usize, d: usize| t * dense + d;
        let state_index = |l: usize, d: usize| l * dense + d;
        let output_scale = (latent as f32).sqrt().recip();

        let mut cumulative_decay = vec![0.0; time * latent];
        for t in 0..time {
            for l in 0..latent {
                let decay = log_decay[latent_index(t, l)].exp();
                cumulative_decay[latent_index(t, l)] = if t == 0 {
                    decay
                } else {
                    cumulative_decay[latent_index(t - 1, l)] * decay
                };
            }
        }

        let mut w_basis = vec![0.0; time * latent];
        let mut m_basis = vec![0.0; time * latent];
        for t in 0..time {
            for l in 0..latent {
                let c = cumulative_decay[latent_index(t, l)];
                w_basis[latent_index(t, l)] = key[latent_index(t, l)] / c;
                m_basis[latent_index(t, l)] =
                    erase[latent_index(t, l)] * key[latent_index(t, l)] * c;
            }
        }

        let mut lower = vec![0.0; time * time];
        for i in 0..time {
            for j in 0..i {
                let mut acc = 0.0;
                for l in 0..latent {
                    acc += m_basis[latent_index(i, l)] * w_basis[latent_index(j, l)];
                }
                lower[i * time + j] = acc;
            }
        }

        let mut z = vec![0.0; time * dense];
        for d in 0..dense {
            for i in 0..time {
                let mut rhs = write[dense_index(i, d)] * value[dense_index(i, d)];
                for l in 0..latent {
                    rhs -= m_basis[latent_index(i, l)] * initial_state[state_index(l, d)];
                }
                for j in 0..i {
                    rhs -= lower[i * time + j] * z[dense_index(j, d)];
                }
                z[dense_index(i, d)] = rhs;
            }
        }

        let mut outputs = vec![0.0; time * dense];
        let mut final_state = vec![0.0; latent * dense];
        for t in 0..time {
            for d in 0..dense {
                let mut output = 0.0;
                for l in 0..latent {
                    let mut transformed_state = initial_state[state_index(l, d)];
                    for j in 0..=t {
                        transformed_state += w_basis[latent_index(j, l)] * z[dense_index(j, d)];
                    }
                    let state_value = cumulative_decay[latent_index(t, l)] * transformed_state;
                    output += query[latent_index(t, l)] * state_value;
                    if t + 1 == time {
                        final_state[state_index(l, d)] = state_value;
                    }
                }
                outputs[dense_index(t, d)] = output * output_scale;
            }
        }

        (outputs, final_state)
    }

    #[test]
    fn reference_matches_one_step_paper_update() {
        let query = tensor4(vec![0.7, -0.3], [1, 1, 1, 2]);
        let key = tensor4(vec![0.4, 0.6], [1, 1, 1, 2]);
        let value = tensor4(vec![0.8, -0.5, 0.25], [1, 1, 1, 3]);
        let erase = tensor4(vec![0.25, 0.75], [1, 1, 1, 2]);
        let write = tensor4(vec![0.5, 0.2, 0.9], [1, 1, 1, 3]);
        let log_decay = tensor4(vec![0.5f32.ln(), 0.25f32.ln()], [1, 1, 1, 2]);
        let state = tensor4(vec![0.2, -0.1, 0.05, 0.3, 0.4, -0.2], [1, 1, 2, 3]);

        let (output, next_state) = gated_deltanet2_reference(
            query.clone(),
            key.clone(),
            value.clone(),
            erase.clone(),
            write.clone(),
            log_decay.clone(),
            Some(state),
            false,
            1.0e-6,
        );

        let decayed_state = [[0.1f32, -0.05, 0.025], [0.075, 0.1, -0.05]];
        let erased_key = [0.25f32 * 0.4, 0.75 * 0.6];
        let erased_value = [
            decayed_state[0][0] * erased_key[0] + decayed_state[1][0] * erased_key[1],
            decayed_state[0][1] * erased_key[0] + decayed_state[1][1] * erased_key[1],
            decayed_state[0][2] * erased_key[0] + decayed_state[1][2] * erased_key[1],
        ];
        let write_value = [
            0.5f32 * 0.8 - erased_value[0],
            0.2 * -0.5 - erased_value[1],
            0.9 * 0.25 - erased_value[2],
        ];
        let expected_state = [
            decayed_state[0][0] + 0.4 * write_value[0],
            decayed_state[0][1] + 0.4 * write_value[1],
            decayed_state[0][2] + 0.4 * write_value[2],
            decayed_state[1][0] + 0.6 * write_value[0],
            decayed_state[1][1] + 0.6 * write_value[1],
            decayed_state[1][2] + 0.6 * write_value[2],
        ];
        let output_scale = 2.0f32.sqrt().recip();
        let expected_output = [
            (0.7 * expected_state[0] - 0.3 * expected_state[3]) * output_scale,
            (0.7 * expected_state[1] - 0.3 * expected_state[4]) * output_scale,
            (0.7 * expected_state[2] - 0.3 * expected_state[5]) * output_scale,
        ];

        let expected_state = tensor4(expected_state.to_vec(), [1, 1, 2, 3]);
        let expected_output = tensor4(expected_output.to_vec(), [1, 1, 1, 3]);

        assert!(max_abs_diff(next_state, expected_state) < 1.0e-6);
        assert!(max_abs_diff(output, expected_output) < 1.0e-6);
    }

    #[test]
    fn reference_is_continuous_across_chunk_boundaries() {
        let query_shape = [2, 3, 7, 4];
        let value_shape = [2, 1, 7, 5];
        let query = tensor4(
            (0..query_shape.iter().product::<usize>())
                .map(|index| ((index * 3) % 23) as f32 / 23.0)
                .collect(),
            query_shape,
        );
        let key = tensor4(
            (0..query_shape.iter().product::<usize>())
                .map(|index| ((index * 5) % 29) as f32 / 29.0)
                .collect(),
            query_shape,
        );
        let value = tensor4(
            (0..value_shape.iter().product::<usize>())
                .map(|index| ((index * 7) % 31) as f32 / 31.0)
                .collect(),
            value_shape,
        );
        let erase =
            Tensor::<TestBackend, 4>::ones(query_shape, &Default::default()).mul_scalar(0.5);
        let write = Tensor::<TestBackend, 4>::ones([2, 3, 7, 5], &Default::default());
        let log_decay = Tensor::<TestBackend, 4>::zeros(query_shape, &Default::default());

        let (full_context, full_state) = gated_deltanet2_reference(
            query.clone(),
            key.clone(),
            value.clone(),
            erase.clone(),
            write.clone(),
            log_decay.clone(),
            None,
            true,
            1.0e-6,
        );
        let (first_context, first_state) = gated_deltanet2_reference(
            query.clone().slice_dim(2, 0..3),
            key.clone().slice_dim(2, 0..3),
            value.clone().slice_dim(2, 0..3),
            erase.clone().slice_dim(2, 0..3),
            write.clone().slice_dim(2, 0..3),
            log_decay.clone().slice_dim(2, 0..3),
            None,
            true,
            1.0e-6,
        );
        let (second_context, chunked_state) = gated_deltanet2_reference(
            query.slice_dim(2, 3..7),
            key.slice_dim(2, 3..7),
            value.slice_dim(2, 3..7),
            erase.slice_dim(2, 3..7),
            write.slice_dim(2, 3..7),
            log_decay.slice_dim(2, 3..7),
            Some(first_state),
            true,
            1.0e-6,
        );
        let chunked_context = Tensor::cat(vec![first_context, second_context], 2);

        assert!(max_abs_diff(full_context, chunked_context) < 1.0e-5);
        assert!(max_abs_diff(full_state, chunked_state) < 1.0e-5);
    }

    #[test]
    fn wy_chunk_reference_matches_recurrent_update() {
        let time = 5;
        let latent = 4;
        let dense = 3;
        let query = (0..time * latent)
            .map(|index| ((index * 3) % 19) as f32 / 19.0 - 0.45)
            .collect::<Vec<_>>();
        let key = (0..time * latent)
            .map(|index| ((index * 5) % 23) as f32 / 23.0 - 0.35)
            .collect::<Vec<_>>();
        let value = (0..time * dense)
            .map(|index| ((index * 7) % 29) as f32 / 29.0 - 0.2)
            .collect::<Vec<_>>();
        let erase = (0..time * latent)
            .map(|index| 0.2 + ((index * 11) % 31) as f32 / 62.0)
            .collect::<Vec<_>>();
        let write = (0..time * dense)
            .map(|index| 0.1 + ((index * 13) % 37) as f32 / 74.0)
            .collect::<Vec<_>>();
        let log_decay = (0..time * latent)
            .map(|index| -0.05 - ((index * 17) % 41) as f32 / 200.0)
            .collect::<Vec<_>>();
        let initial_state = (0..latent * dense)
            .map(|index| ((index * 19) % 43) as f32 / 43.0 - 0.5)
            .collect::<Vec<_>>();

        let (wy_context, wy_state) = wy_chunk_reference(
            &query,
            &key,
            &value,
            &erase,
            &write,
            &log_decay,
            &initial_state,
            time,
            latent,
            dense,
        );
        let (recurrent_context, recurrent_state) = gated_deltanet2_reference(
            tensor4(query, [1, 1, time, latent]),
            tensor4(key, [1, 1, time, latent]),
            tensor4(value, [1, 1, time, dense]),
            tensor4(erase, [1, 1, time, latent]),
            tensor4(write, [1, 1, time, dense]),
            tensor4(log_decay, [1, 1, time, latent]),
            Some(tensor4(initial_state, [1, 1, latent, dense])),
            false,
            1.0e-6,
        );

        assert!(max_abs_diff(tensor4(wy_context, [1, 1, time, dense]), recurrent_context) < 1.0e-5);
        assert!(max_abs_diff(tensor4(wy_state, [1, 1, latent, dense]), recurrent_state) < 1.0e-5);
    }

    #[test]
    fn gate_modes_match_expected_shapes() {
        let logits = tensor4(
            (0..24).map(|index| index as f32 / 24.0).collect(),
            [1, 2, 3, 4],
        );
        let channel = apply_gate_mode(logits.clone(), GatedDeltaNet2GateMode::Channel, false);
        let scalar = apply_gate_mode(logits.clone(), GatedDeltaNet2GateMode::Scalar, false);
        let disabled = apply_gate_mode(logits, GatedDeltaNet2GateMode::Disabled, false);

        assert_eq!(channel.shape().dims::<4>(), [1, 2, 3, 4]);
        assert_eq!(scalar.shape().dims::<4>(), [1, 2, 3, 4]);
        assert_eq!(disabled.shape().dims::<4>(), [1, 2, 3, 4]);
    }
}
