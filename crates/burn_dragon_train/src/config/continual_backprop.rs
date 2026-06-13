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
    None,
    #[default]
    GlobalRatio,
    TargetGroupRatio,
}

fn default_continual_backprop_utility_decay() -> f32 {
    0.99
}

fn default_continual_backprop_replacement_rate() -> f32 {
    1.0e-5
}

fn default_continual_backprop_maturity_steps() -> usize {
    1_024
}

fn default_continual_backprop_sample_interval_steps() -> usize {
    8
}

fn default_continual_backprop_replace_interval_steps() -> usize {
    256
}

fn default_continual_backprop_utility_epsilon() -> f32 {
    1.0e-6
}

fn default_continual_backprop_lr_coupling_power() -> f32 {
    1.0
}

fn default_continual_backprop_warmup_steps() -> usize {
    1_024
}

fn default_continual_backprop_cooldown_steps() -> usize {
    256
}

fn default_continual_backprop_max_replacements_per_interval() -> usize {
    1
}

fn default_continual_backprop_regression_pause_steps() -> usize {
    1_024
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
    #[serde(default = "default_continual_backprop_warmup_steps")]
    pub warmup_steps: usize,
    #[serde(default = "default_continual_backprop_cooldown_steps")]
    pub cooldown_steps: usize,
    #[serde(default = "default_continual_backprop_max_replacements_per_interval")]
    pub max_replacements_per_interval: usize,
    #[serde(default = "default_continual_backprop_regression_pause_steps")]
    pub regression_pause_steps: usize,
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
            warmup_steps: default_continual_backprop_warmup_steps(),
            cooldown_steps: default_continual_backprop_cooldown_steps(),
            max_replacements_per_interval: default_continual_backprop_max_replacements_per_interval(
            ),
            regression_pause_steps: default_continual_backprop_regression_pause_steps(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_conservative_for_long_running_training() {
        let config = ContinualBackpropConfig::default();

        assert_eq!(config.replacement_rate, 1.0e-5);
        assert_eq!(config.maturity_steps, 1_024);
        assert_eq!(config.replace_interval_steps, 256);
        assert_eq!(config.lr_coupling, ContinualBackpropLrCoupling::GlobalRatio);
        assert_eq!(config.max_replacements_per_interval, 1);
        assert_eq!(config.regression_pause_steps, 1_024);
    }

    #[test]
    fn serde_defaults_fill_new_policy_fields() {
        let config: ContinualBackpropConfig = serde_json::from_str("{}").expect("config");

        assert_eq!(config.warmup_steps, 1_024);
        assert_eq!(config.cooldown_steps, 256);
        assert_eq!(config.max_replacements_per_interval, 1);
        assert_eq!(config.lr_coupling, ContinualBackpropLrCoupling::GlobalRatio);
    }
}
