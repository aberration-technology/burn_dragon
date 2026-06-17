use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    #[default]
    Adamw,
    Eggroll,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerScheduleMode {
    #[default]
    DragonReference,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum EggrollPopulationExecutionBackend {
    #[default]
    Auto,
    Reference,
    Cuda,
    Factorized,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum EggrollPerturbationScope {
    #[default]
    DragonCoreProjection,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct EggrollPopulationExecutionConfig {
    pub backend: EggrollPopulationExecutionBackend,
    pub perturbation_scope: EggrollPerturbationScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_tile_size: Option<usize>,
}

impl Default for EggrollPopulationExecutionConfig {
    fn default() -> Self {
        Self {
            backend: EggrollPopulationExecutionBackend::Auto,
            perturbation_scope: EggrollPerturbationScope::DragonCoreProjection,
            population_tile_size: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct EggrollAutoPopulationConfig {
    pub enabled: bool,
    pub min_population_size: usize,
    pub max_population_size: usize,
    pub population_per_batch: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_chunk_size: Option<usize>,
    pub chunk_autotune: EggrollChunkAutotuneConfig,
    pub prefer_power_of_two: bool,
}

impl Default for EggrollAutoPopulationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_population_size: 8,
            max_population_size: 4096,
            population_per_batch: 128,
            population_chunk_size: None,
            chunk_autotune: EggrollChunkAutotuneConfig::default(),
            prefer_power_of_two: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct EggrollChunkAutotuneConfig {
    pub enabled: bool,
    pub candidates: Vec<usize>,
    pub max_probe_population_size: usize,
}

impl Default for EggrollChunkAutotuneConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            candidates: Vec::new(),
            max_probe_population_size: 128,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedEggrollAutoPopulation {
    pub configured_population_size: usize,
    pub configured_population_chunk_size: usize,
    pub configured_rank: usize,
    pub batch_size: usize,
    pub population_per_batch: usize,
    pub min_population_size: usize,
    pub max_population_size: usize,
    pub resolved_population_size: usize,
    pub resolved_population_chunk_size: usize,
}

impl EggrollAutoPopulationConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        validate_even_population(
            "optimizer.eggroll_auto_population.min_population_size",
            self.min_population_size,
        )?;
        validate_even_population(
            "optimizer.eggroll_auto_population.max_population_size",
            self.max_population_size,
        )?;
        if self.max_population_size < self.min_population_size {
            return Err(anyhow!(
                "optimizer.eggroll_auto_population.max_population_size must be >= min_population_size"
            ));
        }
        if self.population_per_batch == 0 {
            return Err(anyhow!(
                "optimizer.eggroll_auto_population.population_per_batch must be > 0"
            ));
        }
        if let Some(population_chunk_size) = self.population_chunk_size {
            validate_even_population(
                "optimizer.eggroll_auto_population.population_chunk_size",
                population_chunk_size,
            )?;
        }
        self.chunk_autotune.validate()?;
        Ok(())
    }
}

impl EggrollPopulationExecutionConfig {
    pub fn validate(&self) -> Result<()> {
        if let Some(population_tile_size) = self.population_tile_size {
            validate_even_population(
                "optimizer.eggroll_population_execution.population_tile_size",
                population_tile_size,
            )?;
        }
        Ok(())
    }
}

impl EggrollChunkAutotuneConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        if self.max_probe_population_size < 2 {
            return Err(anyhow!(
                "optimizer.eggroll_auto_population.chunk_autotune.max_probe_population_size must be >= 2"
            ));
        }
        if !self.max_probe_population_size.is_multiple_of(2) {
            return Err(anyhow!(
                "optimizer.eggroll_auto_population.chunk_autotune.max_probe_population_size must be even"
            ));
        }
        for candidate in &self.candidates {
            validate_even_population(
                "optimizer.eggroll_auto_population.chunk_autotune.candidates",
                *candidate,
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct OptimizerConfig {
    #[serde(default)]
    pub name: OptimizerKind,
    pub learning_rate: f64,
    pub weight_decay: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_decay_final: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lr_schedule: Option<LearningRateScheduleConfig>,
    #[serde(default)]
    pub schedule_mode: OptimizerScheduleMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_clip_norm: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_clip_value: Option<f32>,
    #[serde(default)]
    pub eggroll: burn_eggroll::EggrollConfig,
    #[serde(default)]
    pub eggroll_population_execution: EggrollPopulationExecutionConfig,
    #[serde(default)]
    pub eggroll_auto_population: EggrollAutoPopulationConfig,
}

impl OptimizerConfig {
    pub fn effective_eggroll_config(&self) -> burn_eggroll::EggrollConfig {
        let mut eggroll = self.eggroll.clone();
        eggroll.update.learning_rate = self.learning_rate;
        eggroll.update.weight_decay = self.weight_decay;
        eggroll
    }

    pub fn apply_effective_eggroll_config(&mut self) {
        self.eggroll = self.effective_eggroll_config();
    }

    pub fn apply_auto_eggroll_population(
        &mut self,
        batch_size: usize,
    ) -> Option<ResolvedEggrollAutoPopulation> {
        if !matches!(self.name, OptimizerKind::Eggroll) || !self.eggroll_auto_population.enabled {
            return None;
        }

        let auto = &self.eggroll_auto_population;
        let configured_population_size = self.eggroll.population.population_size;
        let configured_population_chunk_size = self.eggroll.population.population_chunk_size;
        let configured_rank = self.eggroll.population.rank;
        let batch_size = batch_size.max(1);
        let requested = configured_population_size
            .max(auto.min_population_size)
            .max(batch_size.saturating_mul(auto.population_per_batch));
        let resolved_population_size = resolve_population_size(requested, auto);
        let requested_chunk = auto
            .population_chunk_size
            .unwrap_or(configured_population_chunk_size)
            .min(resolved_population_size)
            .max(2);
        let resolved_population_chunk_size = make_even(requested_chunk)
            .min(resolved_population_size)
            .max(2);

        self.eggroll.population.population_size = resolved_population_size;
        self.eggroll.population.population_chunk_size = resolved_population_chunk_size;

        Some(ResolvedEggrollAutoPopulation {
            configured_population_size,
            configured_population_chunk_size,
            configured_rank,
            batch_size,
            population_per_batch: auto.population_per_batch,
            min_population_size: auto.min_population_size,
            max_population_size: auto.max_population_size,
            resolved_population_size,
            resolved_population_chunk_size,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 {
            return Err(anyhow!("optimizer.learning_rate must be > 0"));
        }
        if self.weight_decay < 0.0 {
            return Err(anyhow!("optimizer.weight_decay must be >= 0"));
        }
        if let Some(weight_decay_final) = self.weight_decay_final
            && weight_decay_final < 0.0
        {
            return Err(anyhow!("optimizer.weight_decay_final must be >= 0"));
        }
        if let Some(clip) = self.grad_clip_norm
            && clip <= 0.0
        {
            return Err(anyhow!("optimizer.grad_clip_norm must be > 0"));
        }
        if let Some(clip) = self.grad_clip_value
            && clip <= 0.0
        {
            return Err(anyhow!("optimizer.grad_clip_value must be > 0"));
        }
        if self.grad_clip_norm.is_some() && self.grad_clip_value.is_some() {
            return Err(anyhow!(
                "optimizer.grad_clip_norm and optimizer.grad_clip_value are mutually exclusive"
            ));
        }
        self.eggroll_population_execution.validate()?;
        self.eggroll_auto_population.validate()?;
        if self.eggroll_auto_population.enabled && !matches!(self.name, OptimizerKind::Eggroll) {
            return Err(anyhow!(
                "optimizer.eggroll_auto_population.enabled requires optimizer.name=eggroll"
            ));
        }
        if matches!(self.name, OptimizerKind::Eggroll) {
            self.eggroll.validate()?;
            if self.grad_clip_norm.is_some() || self.grad_clip_value.is_some() {
                return Err(anyhow!(
                    "optimizer grad clipping is not supported by optimizer.name=eggroll"
                ));
            }
        }
        Ok(())
    }
}

fn validate_even_population(name: &str, value: usize) -> Result<()> {
    if value < 2 {
        return Err(anyhow!("{name} must be >= 2"));
    }
    if !value.is_multiple_of(2) {
        return Err(anyhow!("{name} must be even for antithetic pairs"));
    }
    Ok(())
}

fn make_even(value: usize) -> usize {
    value.saturating_sub(value % 2).max(2)
}

fn resolve_population_size(requested: usize, auto: &EggrollAutoPopulationConfig) -> usize {
    let min_population = make_even(auto.min_population_size);
    let max_population = make_even(auto.max_population_size.max(min_population));
    let requested = make_even(requested).clamp(min_population, max_population);
    if !auto.prefer_power_of_two {
        return requested;
    }

    let Some(next_power) = requested.checked_next_power_of_two() else {
        return requested;
    };
    if next_power <= max_population && next_power >= min_population {
        return next_power;
    }
    let previous_power = 1usize << (usize::BITS - requested.leading_zeros() - 1);
    if previous_power >= min_population {
        previous_power
    } else {
        requested
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LearningRateScheduleConfig {
    Constant {
        #[serde(default)]
        initial_lr: Option<f64>,
    },
    Cosine {
        #[serde(default)]
        initial_lr: Option<f64>,
        #[serde(default)]
        min_lr: Option<f64>,
        #[serde(default)]
        warmup_steps: Option<usize>,
        #[serde(default)]
        num_iters: Option<usize>,
    },
    Linear {
        #[serde(default)]
        initial_lr: Option<f64>,
        final_lr: f64,
        #[serde(default)]
        num_iters: Option<usize>,
    },
    Exponential {
        #[serde(default)]
        initial_lr: Option<f64>,
        gamma: f64,
    },
    Step {
        #[serde(default)]
        initial_lr: Option<f64>,
        #[serde(default = "default_step_gamma")]
        gamma: f64,
        #[serde(default)]
        step_size: Option<usize>,
    },
    Noam {
        #[serde(default)]
        initial_lr: Option<f64>,
        #[serde(default)]
        warmup_steps: Option<usize>,
        #[serde(default)]
        model_size: Option<usize>,
    },
}

fn default_step_gamma() -> f64 {
    0.1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_optimizer() -> OptimizerConfig {
        OptimizerConfig {
            name: OptimizerKind::default(),
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: OptimizerScheduleMode::default(),
            grad_clip_norm: None,
            grad_clip_value: None,
            eggroll: burn_eggroll::EggrollConfig::default(),
            eggroll_population_execution: EggrollPopulationExecutionConfig::default(),
            eggroll_auto_population: EggrollAutoPopulationConfig::default(),
        }
    }

    #[test]
    fn weight_decay_final_must_be_non_negative() {
        let config = OptimizerConfig {
            weight_decay_final: Some(-0.1),
            ..base_optimizer()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("weight_decay_final"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn eggroll_config_is_validated() {
        let config = OptimizerConfig {
            name: OptimizerKind::Eggroll,
            eggroll: burn_eggroll::EggrollConfig {
                sigma: 0.0,
                ..burn_eggroll::EggrollConfig::default()
            },
            ..base_optimizer()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("eggroll.sigma"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn eggroll_rejects_gradient_clipping() {
        let config = OptimizerConfig {
            name: OptimizerKind::Eggroll,
            grad_clip_norm: Some(1.0),
            ..base_optimizer()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("grad clipping"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn eggroll_population_execution_rejects_odd_tile_size() {
        let config = OptimizerConfig {
            name: OptimizerKind::Eggroll,
            eggroll_population_execution: EggrollPopulationExecutionConfig {
                perturbation_scope: EggrollPerturbationScope::DragonCoreProjection,
                population_tile_size: Some(3),
                ..EggrollPopulationExecutionConfig::default()
            },
            ..base_optimizer()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("population_tile_size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn eggroll_population_execution_allows_cuda_tensorized_backend() {
        let config = OptimizerConfig {
            name: OptimizerKind::Eggroll,
            eggroll_population_execution: EggrollPopulationExecutionConfig {
                backend: EggrollPopulationExecutionBackend::Cuda,
                perturbation_scope: EggrollPerturbationScope::DragonCoreProjection,
                ..EggrollPopulationExecutionConfig::default()
            },
            ..base_optimizer()
        };
        config.validate().expect("cuda tensorized config validates");
    }

    #[test]
    fn eggroll_population_execution_allows_factorized_backend() {
        let config = OptimizerConfig {
            name: OptimizerKind::Eggroll,
            eggroll_population_execution: EggrollPopulationExecutionConfig {
                backend: EggrollPopulationExecutionBackend::Factorized,
                perturbation_scope: EggrollPerturbationScope::DragonCoreProjection,
                ..EggrollPopulationExecutionConfig::default()
            },
            ..base_optimizer()
        };
        config
            .validate()
            .expect("factorized tensorized config validates");
    }

    #[test]
    fn eggroll_auto_population_resolves_from_batch_size() {
        let mut config = OptimizerConfig {
            name: OptimizerKind::Eggroll,
            eggroll: burn_eggroll::EggrollConfig {
                population: burn_eggroll::PopulationConfig {
                    population_size: 64,
                    population_chunk_size: 16,
                    rank: 2,
                    seed: 7,
                    matrix_noise: burn_eggroll::MatrixNoiseMode::default(),
                },
                ..burn_eggroll::EggrollConfig::default()
            },
            eggroll_auto_population: EggrollAutoPopulationConfig {
                enabled: true,
                min_population_size: 8,
                max_population_size: 1024,
                population_per_batch: 96,
                population_chunk_size: None,
                chunk_autotune: Default::default(),
                prefer_power_of_two: true,
            },
            ..base_optimizer()
        };

        config.validate().expect("valid auto population config");
        let report = config
            .apply_auto_eggroll_population(6)
            .expect("auto population applied");

        assert_eq!(report.configured_population_size, 64);
        assert_eq!(report.configured_population_chunk_size, 16);
        assert_eq!(report.configured_rank, 2);
        assert_eq!(report.batch_size, 6);
        assert_eq!(report.resolved_population_size, 1024);
        assert_eq!(report.resolved_population_chunk_size, 16);
        assert_eq!(config.eggroll.population.population_size, 1024);
        assert_eq!(config.eggroll.population.population_chunk_size, 16);
    }

    #[test]
    fn eggroll_auto_population_respects_explicit_chunk_and_bounds() {
        let mut config = OptimizerConfig {
            name: OptimizerKind::Eggroll,
            eggroll_auto_population: EggrollAutoPopulationConfig {
                enabled: true,
                min_population_size: 8,
                max_population_size: 256,
                population_per_batch: 1024,
                population_chunk_size: Some(32),
                chunk_autotune: Default::default(),
                prefer_power_of_two: false,
            },
            ..base_optimizer()
        };

        let report = config
            .apply_auto_eggroll_population(8)
            .expect("auto population applied");

        assert_eq!(report.resolved_population_size, 256);
        assert_eq!(report.resolved_population_chunk_size, 32);
        assert_eq!(config.eggroll.population.population_size, 256);
        assert_eq!(config.eggroll.population.population_chunk_size, 32);
    }

    #[test]
    fn eggroll_auto_population_rejects_adamw_noop() {
        let config = OptimizerConfig {
            eggroll_auto_population: EggrollAutoPopulationConfig {
                enabled: true,
                ..EggrollAutoPopulationConfig::default()
            },
            ..base_optimizer()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("requires optimizer.name=eggroll"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn eggroll_chunk_autotune_rejects_odd_candidates() {
        let config = OptimizerConfig {
            name: OptimizerKind::Eggroll,
            eggroll_auto_population: EggrollAutoPopulationConfig {
                enabled: true,
                chunk_autotune: EggrollChunkAutotuneConfig {
                    enabled: true,
                    candidates: vec![32, 63],
                    max_probe_population_size: 128,
                },
                ..EggrollAutoPopulationConfig::default()
            },
            ..base_optimizer()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("chunk_autotune.candidates"),
            "unexpected error: {err}"
        );
    }
}
