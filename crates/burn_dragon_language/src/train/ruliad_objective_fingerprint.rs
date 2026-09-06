//! Stable identity for fully materialized Ruliad proof-policy objectives.
//!
//! Source-batch identity is insufficient once DAgger adds model-visited states.
//! This builder hashes the semantic rows consumed by either global backprop or
//! local predictive coding so paired experiments can distinguish exogenous
//! stream parity from endogenous policy divergence.

const FNV1A_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RuliadObjectiveSequenceKind {
    CompletionLikelihood,
    SemanticEnergy,
    ResidualEnergy,
}

impl RuliadObjectiveSequenceKind {
    fn tag(self) -> u64 {
        match self {
            Self::CompletionLikelihood => 1,
            Self::SemanticEnergy => 2,
            Self::ResidualEnergy => 3,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RuliadObjectivePanelFingerprint {
    state: u64,
    expected_rows: usize,
    observed_rows: usize,
}

impl RuliadObjectivePanelFingerprint {
    pub(crate) fn new(expected_rows: usize) -> Self {
        Self {
            state: hash_u64(FNV1A_OFFSET_BASIS, expected_rows as u64),
            expected_rows,
            observed_rows: 0,
        }
    }

    pub(crate) fn push_prefix(
        &mut self,
        inputs: &[i64],
        position: usize,
        support_tokens: &[i64],
        valid_tokens: &[i64],
        weight: f32,
    ) {
        self.state = hash_u64(self.state, 0);
        self.state = hash_i64_slice(self.state, inputs);
        self.state = hash_u64(self.state, position as u64);
        self.state = hash_i64_slice(self.state, support_tokens);
        self.state = hash_i64_slice(self.state, valid_tokens);
        self.state = hash_u64(self.state, u64::from(weight.to_bits()));
        self.observed_rows = self.observed_rows.saturating_add(1);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_sequence(
        &mut self,
        kind: RuliadObjectiveSequenceKind,
        prompt: &[i64],
        candidates: &[Vec<i64>],
        valid_indices: &[usize],
        target_action_weights: Option<&[f32]>,
        target_group: usize,
        weight: f32,
    ) {
        self.state = hash_u64(self.state, kind.tag());
        self.state = hash_i64_slice(self.state, prompt);
        self.state = hash_u64(self.state, candidates.len() as u64);
        for candidate in candidates {
            self.state = hash_i64_slice(self.state, candidate);
        }
        self.state = hash_usize_slice(self.state, valid_indices);
        self.state = hash_u64(self.state, u64::from(target_action_weights.is_some()));
        if let Some(weights) = target_action_weights {
            self.state = hash_u64(self.state, weights.len() as u64);
            for &value in weights {
                self.state = hash_u64(self.state, u64::from(value.to_bits()));
            }
        }
        self.state = hash_u64(self.state, target_group as u64);
        self.state = hash_u64(self.state, u64::from(weight.to_bits()));
        self.observed_rows = self.observed_rows.saturating_add(1);
    }

    pub(crate) fn finish(self) -> Option<u64> {
        (self.observed_rows == self.expected_rows).then_some(self.state)
    }
}

fn hash_u64(mut state: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        state ^= u64::from(byte);
        state = state.wrapping_mul(FNV1A_PRIME);
    }
    state
}

fn hash_i64_slice(mut state: u64, values: &[i64]) -> u64 {
    state = hash_u64(state, values.len() as u64);
    for &value in values {
        state = hash_u64(state, value as u64);
    }
    state
}

fn hash_usize_slice(mut state: u64, values: &[usize]) -> u64 {
    state = hash_u64(state, values.len() as u64);
    for &value in values {
        state = hash_u64(state, value as u64);
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint(valid_indices: &[usize]) -> u64 {
        let mut panel = RuliadObjectivePanelFingerprint::new(1);
        panel.push_sequence(
            RuliadObjectiveSequenceKind::ResidualEnergy,
            &[1, 2],
            &[vec![3], vec![4]],
            valid_indices,
            Some(&[1.0, 0.0]),
            7,
            0.5,
        );
        panel.finish().expect("complete panel")
    }

    #[test]
    fn fingerprint_is_stable_and_label_sensitive() {
        assert_eq!(fingerprint(&[0]), fingerprint(&[0]));
        assert_ne!(fingerprint(&[0]), fingerprint(&[1]));
    }

    #[test]
    fn incomplete_panel_is_rejected() {
        assert_eq!(RuliadObjectivePanelFingerprint::new(1).finish(), None);
    }
}
