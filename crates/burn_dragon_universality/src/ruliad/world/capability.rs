use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq)]
pub struct BernoulliEvidence {
    pub success_mass: f64,
    pub observation_mass: f64,
}

impl BernoulliEvidence {
    pub fn observe_rate(&mut self, rate: f32, item_count: usize) {
        if item_count == 0 {
            return;
        }
        let count = item_count as f64;
        self.success_mass += f64::from(rate.clamp(0.0, 1.0)) * count;
        self.observation_mass += count;
    }

    pub fn mean(self) -> f64 {
        if self.observation_mass <= f64::EPSILON {
            return 0.0;
        }
        (self.success_mass / self.observation_mass).clamp(0.0, 1.0)
    }

    pub fn wilson_lower(self, z: f64) -> f64 {
        self.wilson_interval(z).0
    }

    pub fn wilson_upper(self, z: f64) -> f64 {
        self.wilson_interval(z).1
    }

    fn wilson_interval(self, z: f64) -> (f64, f64) {
        let n = self.observation_mass;
        if n <= f64::EPSILON {
            return (0.0, 1.0);
        }
        let p = self.mean();
        let z = z.max(0.0);
        let z2 = z * z;
        let denominator = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denominator;
        let radius = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).max(0.0)).sqrt() / denominator;
        (
            (center - radius).clamp(0.0, 1.0),
            (center + radius).clamp(0.0, 1.0),
        )
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct RuliadCapabilityMasteryThresholds {
    pub minimum_items: usize,
    pub confidence_z: f64,
    pub verifier_min: f64,
    pub completion_health_min: f64,
    pub schema_wrong_max: f64,
    pub malformed_max: f64,
    pub missing_max: f64,
}

impl Default for RuliadCapabilityMasteryThresholds {
    fn default() -> Self {
        Self {
            minimum_items: 16,
            confidence_z: 1.0,
            verifier_min: 0.50,
            completion_health_min: 0.75,
            schema_wrong_max: 0.25,
            malformed_max: 0.05,
            missing_max: 0.05,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq)]
pub struct RuliadCapabilityPosterior {
    pub verifier: BernoulliEvidence,
    pub partial_credit: BernoulliEvidence,
    pub completion_health: BernoulliEvidence,
    pub schema_wrong: BernoulliEvidence,
    pub malformed: BernoulliEvidence,
    pub missing: BernoulliEvidence,
}

impl RuliadCapabilityPosterior {
    pub fn observation_count(self) -> usize {
        self.verifier.observation_mass.round() as usize
    }

    pub fn mastered(self, thresholds: RuliadCapabilityMasteryThresholds) -> bool {
        self.observation_count() >= thresholds.minimum_items
            && self.verifier.wilson_lower(thresholds.confidence_z) >= thresholds.verifier_min
            && self.completion_health.wilson_lower(thresholds.confidence_z)
                >= thresholds.completion_health_min
            && self.schema_wrong.wilson_upper(thresholds.confidence_z)
                <= thresholds.schema_wrong_max
            && self.malformed.wilson_upper(thresholds.confidence_z) <= thresholds.malformed_max
            && self.missing.wilson_upper(thresholds.confidence_z) <= thresholds.missing_max
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq)]
pub struct RuliadCapabilityCoverage {
    pub difficulty_level: usize,
    pub candidate_coverage: f32,
    pub family_coverage: f32,
    pub task_coverage: f32,
    pub contract_coverage: f32,
    pub observed_items: usize,
}

impl RuliadCapabilityCoverage {
    pub fn mastered(self, minimum_coverage: f32) -> bool {
        let minimum = minimum_coverage.clamp(0.0, 1.0);
        self.candidate_coverage >= minimum
            && self.family_coverage >= minimum
            && self.task_coverage >= minimum
            && self.contract_coverage >= minimum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_bound_blocks_small_lucky_panels() {
        let mut posterior = RuliadCapabilityPosterior::default();
        posterior.verifier.observe_rate(1.0, 1);
        posterior.completion_health.observe_rate(1.0, 1);
        posterior.schema_wrong.observe_rate(0.0, 1);
        posterior.malformed.observe_rate(0.0, 1);
        posterior.missing.observe_rate(0.0, 1);
        assert!(!posterior.mastered(RuliadCapabilityMasteryThresholds::default()));
    }

    #[test]
    fn healthy_supported_panel_reaches_mastery() {
        let mut posterior = RuliadCapabilityPosterior::default();
        posterior.verifier.observe_rate(0.90, 128);
        posterior.completion_health.observe_rate(0.98, 128);
        posterior.schema_wrong.observe_rate(0.01, 128);
        posterior.malformed.observe_rate(0.0, 128);
        posterior.missing.observe_rate(0.0, 128);
        assert!(posterior.mastered(RuliadCapabilityMasteryThresholds::default()));
    }
}
