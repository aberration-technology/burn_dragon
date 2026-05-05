use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use super::{
    DragonInitializer, DragonNeuronGainKind, DragonProjectionRole, DragonTopologyLatentAxis,
    DragonTopologyPriorConfig, DragonTopologyPriorKind,
};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct DragonReservoirInitializationConfig {
    /// Deterministic CPU-side generator seed.
    #[serde(default = "default_reservoir_seed")]
    pub seed: u64,
    /// Probability of a nonzero basis weight before topology shaping.
    #[serde(default = "default_reservoir_density")]
    pub density: f64,
    /// Scale applied to encoder_v relative to the base projection RMS.
    #[serde(default = "default_reservoir_encoder_value_scale")]
    pub encoder_value_scale: f64,
    /// Scale applied to decoder relative to the base projection RMS.
    #[serde(default = "default_reservoir_decoder_scale")]
    pub decoder_scale: f64,
}

impl Default for DragonReservoirInitializationConfig {
    fn default() -> Self {
        Self {
            seed: default_reservoir_seed(),
            density: default_reservoir_density(),
            encoder_value_scale: default_reservoir_encoder_value_scale(),
            decoder_scale: default_reservoir_decoder_scale(),
        }
    }
}

impl DragonReservoirInitializationConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.density.is_finite() || !(0.0..=1.0).contains(&self.density) {
            return Err(format!(
                "model.initialization.reservoir.density must be finite and in (0, 1] (got {})",
                self.density
            ));
        }
        if self.density == 0.0 {
            return Err("model.initialization.reservoir.density must be > 0".to_string());
        }
        validate_positive_finite_f64(
            self.encoder_value_scale,
            "model.initialization.reservoir.encoder_value_scale",
        )?;
        validate_positive_finite_f64(
            self.decoder_scale,
            "model.initialization.reservoir.decoder_scale",
        )?;
        Ok(())
    }
}

impl<'a> DragonInitializer<'a> {
    pub(crate) fn reservoir_headwise_projection_tensor<B: Backend>(
        &self,
        role: DragonProjectionRole,
        heads: usize,
        fan_in: usize,
        fan_out: usize,
        residual_depth: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let target_rms = self.reservoir_target_rms(role, fan_in, fan_out, residual_depth);
        let latent_axis = reservoir_latent_axis(role);
        let mut values = Vec::with_capacity(heads * fan_in * fan_out);
        for head in 0..heads {
            let seed = mix_seed(
                self.config.reservoir.seed,
                &[
                    0x34b4_49fd_a4df_9925,
                    role_seed(role),
                    head as u64,
                    fan_in as u64,
                    fan_out as u64,
                    residual_depth as u64,
                ],
            );
            values.extend(reservoir_projection_values(
                fan_in,
                fan_out,
                latent_axis,
                target_rms,
                &self.config.reservoir,
                &self.config.topology_prior,
                seed,
            ));
        }
        apply_headwise_neuron_gains_in_place(
            &mut values,
            heads,
            fan_in,
            fan_out,
            &self.config.neuron_gains,
            mix_seed(
                self.config.reservoir.seed,
                &[0xcf25_f12d_0b6a_0c3d, role_seed(role), fan_out as u64],
            ),
        );
        Tensor::<B, 3>::from_data(TensorData::new(values, [heads, fan_in, fan_out]), device)
    }

    pub(crate) fn reservoir_projection_tensor<B: Backend>(
        &self,
        role: DragonProjectionRole,
        fan_in: usize,
        fan_out: usize,
        residual_depth: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let target_rms = self.reservoir_target_rms(role, fan_in, fan_out, residual_depth);
        let seed = mix_seed(
            self.config.reservoir.seed,
            &[
                0x1aa5_d2bc_6b91_0f01,
                role_seed(role),
                fan_in as u64,
                fan_out as u64,
                residual_depth as u64,
            ],
        );
        let mut values = reservoir_projection_values(
            fan_in,
            fan_out,
            reservoir_latent_axis(role),
            target_rms,
            &self.config.reservoir,
            &self.config.topology_prior,
            seed,
        );
        apply_projection_neuron_gains_in_place(
            &mut values,
            fan_in,
            fan_out,
            role,
            &self.config.neuron_gains,
            mix_seed(
                self.config.reservoir.seed,
                &[0x52a1_3f91_bacd_1421, role_seed(role), fan_in as u64],
            ),
        );
        Tensor::<B, 2>::from_data(TensorData::new(values, [fan_in, fan_out]), device)
    }

    fn reservoir_target_rms(
        &self,
        role: DragonProjectionRole,
        fan_in: usize,
        fan_out: usize,
        residual_depth: usize,
    ) -> f64 {
        let scale = match role {
            DragonProjectionRole::Encoder => 1.0,
            DragonProjectionRole::EncoderValue => self.config.reservoir.encoder_value_scale,
            DragonProjectionRole::Decoder => self.config.reservoir.decoder_scale,
            DragonProjectionRole::LmHead => 1.0,
        };
        self.projection_std(role, fan_in, fan_out, residual_depth) * scale
    }
}

fn default_reservoir_seed() -> u64 {
    0x0BAD_C0FF_EEC0_2026
}

fn default_reservoir_density() -> f64 {
    0.08
}

fn default_reservoir_encoder_value_scale() -> f64 {
    0.70
}

fn default_reservoir_decoder_scale() -> f64 {
    1.00
}

fn reservoir_projection_values(
    rows: usize,
    cols: usize,
    latent_axis: DragonTopologyLatentAxis,
    target_rms: f64,
    reservoir: &DragonReservoirInitializationConfig,
    topology: &DragonTopologyPriorConfig,
    seed: u64,
) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut values = vec![0.0f32; rows * cols];
    let context = ReservoirProjectionContext {
        rows,
        cols,
        latent_axis,
        target_rms,
        reservoir,
        topology,
    };
    fill_reservoir_values(&mut values, context, &mut rng);
    ensure_latent_axis_coverage(&mut values, context, &mut rng);
    normalize_latent_axis_rms_in_place(&mut values, rows, cols, latent_axis, target_rms);
    values
}

#[derive(Clone, Copy)]
struct ReservoirProjectionContext<'a> {
    rows: usize,
    cols: usize,
    latent_axis: DragonTopologyLatentAxis,
    target_rms: f64,
    reservoir: &'a DragonReservoirInitializationConfig,
    topology: &'a DragonTopologyPriorConfig,
}

fn fill_reservoir_values(
    values: &mut [f32],
    context: ReservoirProjectionContext<'_>,
    rng: &mut StdRng,
) {
    for row in 0..context.rows {
        for col in 0..context.cols {
            let (probability, gain) = topology_probability_and_gain(row, col, context);
            if rng.r#gen::<f64>() > probability {
                continue;
            }
            values[row * context.cols + col] =
                (sample_standard_normal(rng) * context.target_rms * gain) as f32;
        }
    }
}

fn ensure_latent_axis_coverage(
    values: &mut [f32],
    context: ReservoirProjectionContext<'_>,
    rng: &mut StdRng,
) {
    match context.latent_axis {
        DragonTopologyLatentAxis::Rows => {
            for row in 0..context.rows {
                if (0..context.cols).any(|col| values[row * context.cols + col] != 0.0) {
                    continue;
                }
                let col = rng.gen_range(0..context.cols.max(1));
                let (_, gain) = topology_probability_and_gain(row, col, context);
                values[row * context.cols + col] = coverage_value(context.target_rms, gain, rng);
            }
        }
        DragonTopologyLatentAxis::Cols => {
            for col in 0..context.cols {
                if (0..context.rows).any(|row| values[row * context.cols + col] != 0.0) {
                    continue;
                }
                let row = rng.gen_range(0..context.rows.max(1));
                let (_, gain) = topology_probability_and_gain(row, col, context);
                values[row * context.cols + col] = coverage_value(context.target_rms, gain, rng);
            }
        }
    }
}

fn normalize_latent_axis_rms_in_place(
    values: &mut [f32],
    rows: usize,
    cols: usize,
    latent_axis: DragonTopologyLatentAxis,
    target_rms: f64,
) {
    let target_rms = target_rms as f32;
    match latent_axis {
        DragonTopologyLatentAxis::Rows => {
            for row in 0..rows {
                let mut sum_sq = 0.0f32;
                for col in 0..cols {
                    let value = values[row * cols + col];
                    sum_sq += value * value;
                }
                let rms = (sum_sq / cols.max(1) as f32).sqrt();
                let scale = target_rms / rms.max(1.0e-12);
                for col in 0..cols {
                    values[row * cols + col] *= scale;
                }
            }
        }
        DragonTopologyLatentAxis::Cols => {
            for col in 0..cols {
                let mut sum_sq = 0.0f32;
                for row in 0..rows {
                    let value = values[row * cols + col];
                    sum_sq += value * value;
                }
                let rms = (sum_sq / rows.max(1) as f32).sqrt();
                let scale = target_rms / rms.max(1.0e-12);
                for row in 0..rows {
                    values[row * cols + col] *= scale;
                }
            }
        }
    }
}

fn topology_probability_and_gain(
    row: usize,
    col: usize,
    context: ReservoirProjectionContext<'_>,
) -> (f64, f64) {
    match context.topology.kind {
        DragonTopologyPriorKind::Iid => (context.reservoir.density, 1.0),
        DragonTopologyPriorKind::ModularBridges => {
            let latent_size = match context.latent_axis {
                DragonTopologyLatentAxis::Rows => context.rows,
                DragonTopologyLatentAxis::Cols => context.cols,
            }
            .max(1);
            let latent_index = match context.latent_axis {
                DragonTopologyLatentAxis::Rows => row,
                DragonTopologyLatentAxis::Cols => col,
            };
            let bridge_count = ((latent_size as f64 * context.topology.bridge_fraction).round()
                as usize)
                .min(latent_size);
            let is_bridge = latent_index >= latent_size.saturating_sub(bridge_count);
            let community_count = context
                .topology
                .community_count
                .max(1)
                .min(context.rows.min(context.cols).max(1));
            let row_community = (row * community_count) / context.rows.max(1);
            let col_community = (col * community_count) / context.cols.max(1);
            if is_bridge {
                (context.reservoir.density, context.topology.bridge_gain)
            } else if row_community == col_community {
                (
                    context.reservoir.density,
                    context.topology.intra_community_gain,
                )
            } else {
                (
                    context.reservoir.density * context.topology.bridge_fraction,
                    context.topology.inter_community_gain,
                )
            }
        }
    }
}

fn apply_headwise_neuron_gains_in_place(
    values: &mut [f32],
    heads: usize,
    rows: usize,
    cols: usize,
    config: &super::DragonNeuronGainConfig,
    seed: u64,
) {
    let gains = neuron_gain_values(heads * cols, config, seed);
    for head in 0..heads {
        for col in 0..cols {
            let gain = gains[head * cols + col];
            for row in 0..rows {
                values[head * rows * cols + row * cols + col] *= gain;
            }
        }
    }
}

fn apply_projection_neuron_gains_in_place(
    values: &mut [f32],
    rows: usize,
    cols: usize,
    role: DragonProjectionRole,
    config: &super::DragonNeuronGainConfig,
    seed: u64,
) {
    if !matches!(role, DragonProjectionRole::Decoder) {
        return;
    }
    let gains = neuron_gain_values(rows, config, seed);
    for row in 0..rows {
        for col in 0..cols {
            values[row * cols + col] *= gains[row];
        }
    }
}

fn neuron_gain_values(count: usize, config: &super::DragonNeuronGainConfig, seed: u64) -> Vec<f32> {
    match config.kind {
        DragonNeuronGainKind::Iid => vec![1.0; count],
        DragonNeuronGainKind::HeavyTailedLogNormal => {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut gains = (0..count)
                .map(|_| {
                    (sample_standard_normal(&mut rng) * config.log_sigma)
                        .exp()
                        .min(config.max_gain) as f32
                })
                .collect::<Vec<_>>();
            let rms = (gains.iter().copied().map(|gain| gain * gain).sum::<f32>()
                / gains.len().max(1) as f32)
                .sqrt()
                .max(1.0e-6);
            gains.iter_mut().for_each(|gain| *gain /= rms);
            gains
        }
    }
}

fn reservoir_latent_axis(role: DragonProjectionRole) -> DragonTopologyLatentAxis {
    match role {
        DragonProjectionRole::Decoder => DragonTopologyLatentAxis::Rows,
        DragonProjectionRole::Encoder
        | DragonProjectionRole::EncoderValue
        | DragonProjectionRole::LmHead => DragonTopologyLatentAxis::Cols,
    }
}

fn coverage_value(target_rms: f64, gain: f64, rng: &mut StdRng) -> f32 {
    let sign = if rng.r#gen::<bool>() { 1.0 } else { -1.0 };
    (sign * target_rms * gain.max(1.0e-12)) as f32
}

fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng
        .r#gen::<f64>()
        .clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    let u2 = rng.r#gen::<f64>();
    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    radius * theta.cos()
}

fn mix_seed(seed: u64, values: &[u64]) -> u64 {
    let mut mixed = splitmix64(seed);
    for value in values {
        mixed = splitmix64(mixed ^ splitmix64(*value));
    }
    mixed
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn role_seed(role: DragonProjectionRole) -> u64 {
    match role {
        DragonProjectionRole::Encoder => 0x0e11_c0de_0000_0001,
        DragonProjectionRole::EncoderValue => 0x0e11_c0de_0000_0002,
        DragonProjectionRole::Decoder => 0x0e11_c0de_0000_0003,
        DragonProjectionRole::LmHead => 0x0e11_c0de_0000_0004,
    }
}

fn validate_positive_finite_f64(value: f64, field: &str) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{field} must be finite and > 0 (got {value})"));
    }
    Ok(())
}
