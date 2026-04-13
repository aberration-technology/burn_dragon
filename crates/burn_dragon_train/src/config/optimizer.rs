use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    #[default]
    Adamw,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerScheduleMode {
    #[default]
    DragonReference,
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
}

impl OptimizerConfig {
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
        Ok(())
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
}
