use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContinualBackpropTarget {
    #[default]
    SharedLowrankLatents,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContinualBackpropLrCoupling {
    #[default]
    None,
    GlobalRatio,
    TargetGroupRatio,
}

fn default_continual_backprop_utility_decay() -> f32 {
    0.99
}

fn default_continual_backprop_replacement_rate() -> f32 {
    1.0e-4
}

fn default_continual_backprop_maturity_steps() -> usize {
    100
}

fn default_continual_backprop_sample_interval_steps() -> usize {
    8
}

fn default_continual_backprop_replace_interval_steps() -> usize {
    64
}

fn default_continual_backprop_utility_epsilon() -> f32 {
    1.0e-6
}

fn default_continual_backprop_lr_coupling_power() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct ContinualBackpropConfig {
    pub enabled: bool,
    pub target: ContinualBackpropTarget,
    #[serde(default = "default_continual_backprop_utility_decay")]
    pub utility_decay: f32,
    #[serde(default = "default_continual_backprop_replacement_rate")]
    pub replacement_rate: f32,
    #[serde(default = "default_continual_backprop_maturity_steps")]
    pub maturity_steps: usize,
    #[serde(default = "default_continual_backprop_sample_interval_steps")]
    pub sample_interval_steps: usize,
    #[serde(default = "default_continual_backprop_replace_interval_steps")]
    pub replace_interval_steps: usize,
    #[serde(default = "default_continual_backprop_utility_epsilon")]
    pub utility_epsilon: f32,
    #[serde(default)]
    pub lr_coupling: ContinualBackpropLrCoupling,
    #[serde(default = "default_continual_backprop_lr_coupling_power")]
    pub lr_coupling_power: f32,
}

impl Default for ContinualBackpropConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target: ContinualBackpropTarget::default(),
            utility_decay: default_continual_backprop_utility_decay(),
            replacement_rate: default_continual_backprop_replacement_rate(),
            maturity_steps: default_continual_backprop_maturity_steps(),
            sample_interval_steps: default_continual_backprop_sample_interval_steps(),
            replace_interval_steps: default_continual_backprop_replace_interval_steps(),
            utility_epsilon: default_continual_backprop_utility_epsilon(),
            lr_coupling: ContinualBackpropLrCoupling::default(),
            lr_coupling_power: default_continual_backprop_lr_coupling_power(),
        }
    }
}
