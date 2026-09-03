//! Backend-neutral token supervision for Ruliad documents.

use serde::{Deserialize, Serialize};

const SYMBOLIC_ANSWER_TOKEN: u32 = 264;
const SYMBOLIC_DOCUMENT_END_TOKEN: u32 = 265;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuliadTokenSupervisionMode {
    #[default]
    FullDocument,
    AnswerWindow,
    AnswerCompletion,
    /// Supervise answer schema and termination while masking semantic values.
    AnswerStructure,
    AnswerValues,
    /// Alternate unit-normalized structure and value objectives.
    FactorizedAnswer,
    TraceAndAnswer,
    Mixed,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct RuliadTokenSupervisionConfig {
    pub mode: RuliadTokenSupervisionMode,
    pub mask_high_entropy_spans: bool,
    /// Normalize trace and answer targets to equal aggregate weight when both occur in one
    /// `TraceAndAnswer` window. The scale is derived from observed target counts and role weights.
    pub balance_trace_answer_mass: bool,
    pub answer_close_marker_stride: usize,
    pub answer_close_marker_weight: i64,
    pub answer_schema_token_weight: i64,
    pub answer_schema_start_token_weight: i64,
    pub answer_value_token_weight: i64,
}

impl Default for RuliadTokenSupervisionConfig {
    fn default() -> Self {
        Self {
            mode: RuliadTokenSupervisionMode::FullDocument,
            mask_high_entropy_spans: false,
            balance_trace_answer_mass: false,
            answer_close_marker_stride: 1,
            answer_close_marker_weight: 1,
            answer_schema_token_weight: 1,
            answer_schema_start_token_weight: 1,
            answer_value_token_weight: 1,
        }
    }
}

impl RuliadTokenSupervisionConfig {
    pub fn effective_for(self, validation: bool, progress_index: usize) -> Self {
        let mode = match self.mode {
            RuliadTokenSupervisionMode::Mixed => {
                if validation || progress_index & 1 == 0 {
                    RuliadTokenSupervisionMode::AnswerCompletion
                } else {
                    RuliadTokenSupervisionMode::FullDocument
                }
            }
            RuliadTokenSupervisionMode::FactorizedAnswer => {
                if validation {
                    RuliadTokenSupervisionMode::AnswerCompletion
                } else if progress_index & 1 == 0 {
                    RuliadTokenSupervisionMode::AnswerStructure
                } else {
                    RuliadTokenSupervisionMode::AnswerValues
                }
            }
            mode => mode,
        };
        Self { mode, ..self }
    }

    pub fn uses_answer_target_mask(self) -> bool {
        matches!(
            self.mode,
            RuliadTokenSupervisionMode::AnswerCompletion
                | RuliadTokenSupervisionMode::AnswerStructure
                | RuliadTokenSupervisionMode::AnswerValues
                | RuliadTokenSupervisionMode::FactorizedAnswer
                | RuliadTokenSupervisionMode::TraceAndAnswer
                | RuliadTokenSupervisionMode::Mixed
        )
    }

    pub fn uses_trace_answer_target_mask(self) -> bool {
        self.mode == RuliadTokenSupervisionMode::TraceAndAnswer
    }

    pub fn uses_target_loss_mask(self) -> bool {
        self.uses_answer_target_mask() || self.mask_high_entropy_spans
    }
}

pub fn ruliad_token_loss_mask(
    window: &[u32],
    mask: &mut [i64],
    supervision: RuliadTokenSupervisionConfig,
) -> bool {
    if window.len() < mask.len().saturating_add(1) {
        mask.fill(0);
        return false;
    }
    let mut any = if supervision.uses_trace_answer_target_mask() {
        answer_target_loss_mask(window, mask, supervision, 1)
    } else if supervision.uses_answer_target_mask() {
        answer_target_loss_mask(window, mask, supervision, 0)
    } else {
        mask.fill(1);
        !mask.is_empty()
    };
    if supervision.mask_high_entropy_spans {
        mask_high_entropy_targets(window, mask);
        any = mask.iter().any(|value| *value != 0);
    }
    if supervision.balance_trace_answer_mass
        && supervision.mode == RuliadTokenSupervisionMode::TraceAndAnswer
    {
        balance_trace_answer_mass(mask);
        any = mask.iter().any(|value| *value != 0);
    }
    any
}

fn answer_target_loss_mask(
    window: &[u32],
    mask: &mut [i64],
    supervision: RuliadTokenSupervisionConfig,
    base_weight: i64,
) -> bool {
    let value_only = supervision.mode == RuliadTokenSupervisionMode::AnswerValues;
    let structure_only = supervision.mode == RuliadTokenSupervisionMode::AnswerStructure;
    mask.fill(base_weight);
    let mut in_answer = false;
    let mut in_close_marker = false;
    let mut in_answer_value = false;
    let mut semantic_action = false;
    let mut semantic_action_segment = 0usize;
    let mut semantic_action_segment_offset = 0usize;
    let mut answer_hash = 0u64;
    let close_weight = supervision.answer_close_marker_weight.clamp(1, 16);
    let schema_weight = supervision.answer_schema_token_weight.clamp(1, 16);
    let schema_start_weight = supervision.answer_schema_start_token_weight.clamp(1, 16);
    let value_weight = supervision.answer_value_token_weight.clamp(1, 16);
    let mut any = false;
    let mut schema_key_start_pending = false;
    for t in 0..mask.len() {
        let input = window[t];
        let target = window[t + 1];
        if input == SYMBOLIC_ANSWER_TOKEN
            || (input == u32::from(b':') && t > 0 && window.get(t - 1) == Some(&u32::from(b'!')))
        {
            in_answer = true;
            in_answer_value = false;
            semantic_action = semantic_proof_action_at(window, t.saturating_add(1));
            semantic_action_segment = 0;
            semantic_action_segment_offset = 0;
            schema_key_start_pending = true;
        }
        if in_answer && !in_close_marker && token_byte(input) == Some(b'=') {
            in_answer_value = true;
            schema_key_start_pending = false;
        }
        let close_marker_start = target == SYMBOLIC_DOCUMENT_END_TOKEN
            || (target == u32::from(b'[') && window.get(t + 2) == Some(&u32::from(b'/')));
        let close_marker_end =
            target == SYMBOLIC_DOCUMENT_END_TOKEN || (in_close_marker && target == u32::from(b']'));
        let supervise_close_marker = supervision.answer_close_marker_stride > 0
            && (supervision.answer_close_marker_stride == 1
                || answer_hash.is_multiple_of(supervision.answer_close_marker_stride as u64));
        if in_answer && target != SYMBOLIC_ANSWER_TOKEN {
            if close_marker_start && !supervise_close_marker {
                in_answer = false;
                in_close_marker = false;
                in_answer_value = false;
            } else {
                let target_byte = token_byte(target);
                let close_marker_target = close_marker_start || in_close_marker;
                let semantic_value_target = semantic_action
                    && semantic_action_value_position(
                        semantic_action_segment,
                        semantic_action_segment_offset,
                    );
                let value_target = !close_marker_target
                    && (in_answer_value || semantic_value_target)
                    && target_byte.is_some_and(is_answer_value_byte);
                let schema_target = !close_marker_target
                    && !value_target
                    && target_byte.is_some_and(is_answer_schema_byte);
                let schema_start_target = schema_target
                    && schema_key_start_pending
                    && target_byte.is_some_and(is_answer_key_start_byte);
                let weight = if (value_only && !value_target) || (structure_only && value_target) {
                    0
                } else if close_marker_target {
                    close_weight
                } else if schema_start_target {
                    schema_start_weight.max(schema_weight)
                } else if value_target {
                    value_weight
                } else if schema_target {
                    schema_weight
                } else {
                    1
                };
                // Negative values are an allocation-free internal tag. They are converted back to
                // positive loss weights by `balance_trace_answer_mass` before this function returns.
                mask[t] = if supervision.balance_trace_answer_mass
                    && supervision.mode == RuliadTokenSupervisionMode::TraceAndAnswer
                    && weight > 0
                {
                    -weight
                } else {
                    weight
                };
                any |= mask[t] != 0;
                if close_marker_start {
                    in_close_marker = true;
                } else if !in_close_marker {
                    answer_hash = answer_hash
                        .wrapping_mul(1_099_511_628_211)
                        .wrapping_add(u64::from(target).wrapping_add(1));
                    if matches!(
                        token_byte(target),
                        Some(b';') | Some(b'\n') | Some(b'\r') | Some(b'|')
                    ) {
                        in_answer_value = false;
                        schema_key_start_pending = true;
                    } else if schema_start_target {
                        schema_key_start_pending = false;
                    }
                    if semantic_action {
                        if target_byte == Some(b'|') {
                            semantic_action_segment = semantic_action_segment.saturating_add(1);
                            semantic_action_segment_offset = 0;
                        } else {
                            semantic_action_segment_offset =
                                semantic_action_segment_offset.saturating_add(1);
                        }
                    }
                }
            }
        }
        if close_marker_end {
            in_answer = false;
            in_close_marker = false;
            in_answer_value = false;
            semantic_action = false;
            semantic_action_segment = 0;
            semantic_action_segment_offset = 0;
            schema_key_start_pending = false;
        }
        if target == SYMBOLIC_ANSWER_TOKEN {
            in_answer = true;
            in_close_marker = false;
            in_answer_value = false;
            semantic_action = semantic_proof_action_at(window, t.saturating_add(2));
            semantic_action_segment = 0;
            semantic_action_segment_offset = 0;
            schema_key_start_pending = true;
            answer_hash = 0;
        }
    }
    any
}

fn balance_trace_answer_mass(mask: &mut [i64]) {
    let trace_mass = mask
        .iter()
        .copied()
        .filter(|weight| *weight > 0)
        .map(i128::from)
        .sum::<i128>();
    let answer_mass = mask
        .iter()
        .copied()
        .filter(|weight| *weight < 0)
        .map(|weight| i128::from(weight).saturating_neg())
        .sum::<i128>();
    if answer_mass == 0 {
        return;
    }

    for weight in mask.iter_mut().filter(|weight| **weight < 0) {
        let role_weight = i128::from(*weight).saturating_neg();
        let balanced = if trace_mass == 0 {
            role_weight
        } else {
            role_weight
                .saturating_mul(trace_mass)
                .saturating_add(answer_mass / 2)
                .checked_div(answer_mass)
                .unwrap_or(1)
                .max(1)
        };
        *weight = i64::try_from(balanced).unwrap_or(i64::MAX);
    }
}

fn mask_high_entropy_targets(window: &[u32], mask: &mut [i64]) {
    let mut index = 0usize;
    while index < window.len() {
        if !token_byte(window[index]).is_some_and(|byte| byte.is_ascii_hexdigit()) {
            index = index.saturating_add(1);
            continue;
        }
        let start = index;
        let mut end = index;
        while end < window.len()
            && token_byte(window[end]).is_some_and(|byte| byte.is_ascii_hexdigit())
        {
            end = end.saturating_add(1);
        }
        if end.saturating_sub(start) >= 8 {
            for token_index in start..end {
                if let Some(mask_index) = token_index.checked_sub(1)
                    && let Some(slot) = mask.get_mut(mask_index)
                {
                    *slot = 0;
                }
            }
        }
        index = end.max(index.saturating_add(1));
    }
}

fn token_byte(token: u32) -> Option<u8> {
    (token <= u8::MAX as u32).then_some(token as u8)
}

fn is_answer_value_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'+' | b'.')
}

fn is_answer_schema_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'=' | b';' | b',' | b'|')
}

fn is_answer_key_start_byte(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn semantic_proof_action_at(window: &[u32], start: usize) -> bool {
    if window.get(start).and_then(|token| token_byte(*token)) != Some(b'g') {
        return false;
    }
    let mut index = start.saturating_add(1);
    let digit_start = index;
    while window
        .get(index)
        .and_then(|token| token_byte(*token))
        .is_some_and(|byte| byte.is_ascii_digit())
    {
        index = index.saturating_add(1);
    }
    if index == digit_start || window.get(index).and_then(|token| token_byte(*token)) != Some(b'|')
    {
        return false;
    }
    index = index.saturating_add(1);
    if !matches!(
        window.get(index).and_then(|token| token_byte(*token)),
        Some(b'a') | Some(b'l')
    ) {
        return false;
    }
    window
        .get(index.saturating_add(1))
        .and_then(|token| token_byte(*token))
        == Some(b':')
}

fn semantic_action_value_position(segment: usize, offset: usize) -> bool {
    match segment {
        0 => offset >= 1,
        1 => offset >= 2,
        2 | 3 => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r3_answer_completion_masks_prefix_and_supervises_certificate_and_close() {
        let window = b"[R3 x]\nP:p\n?:root=1\n!:certificate-wire\n[/R3]"
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        let mut mask = vec![0; window.len() - 1];
        assert!(ruliad_token_loss_mask(
            &window,
            &mut mask,
            RuliadTokenSupervisionConfig {
                mode: RuliadTokenSupervisionMode::AnswerCompletion,
                ..RuliadTokenSupervisionConfig::default()
            }
        ));
        let supervised = window
            .iter()
            .skip(1)
            .zip(&mask)
            .filter_map(|(token, weight)| (*weight > 0).then_some(*token as u8 as char))
            .collect::<String>();
        assert_eq!(supervised, "certificate-wire\n[/R3]");
    }

    #[test]
    fn answer_value_supervision_excludes_schema_and_document_close_tokens() {
        let window = b"[R3 x]\nP:p\n?:root=1\n!:c=2\n[/R3]"
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        let mut mask = vec![0; window.len() - 1];
        assert!(ruliad_token_loss_mask(
            &window,
            &mut mask,
            RuliadTokenSupervisionConfig {
                mode: RuliadTokenSupervisionMode::AnswerValues,
                ..RuliadTokenSupervisionConfig::default()
            }
        ));
        let supervised = window
            .iter()
            .skip(1)
            .zip(&mask)
            .filter_map(|(token, weight)| (*weight > 0).then_some(*token as u8 as char))
            .collect::<String>();
        assert_eq!(supervised, "2");
    }

    #[test]
    fn answer_structure_supervision_excludes_semantic_values() {
        let window = b"[R3 x]\nP:p\n?:root=1\n!:c=2\n[/R3]"
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        let mut mask = vec![0; window.len() - 1];
        assert!(ruliad_token_loss_mask(
            &window,
            &mut mask,
            RuliadTokenSupervisionConfig {
                mode: RuliadTokenSupervisionMode::AnswerStructure,
                ..RuliadTokenSupervisionConfig::default()
            }
        ));
        let supervised = window
            .iter()
            .skip(1)
            .zip(&mask)
            .filter_map(|(token, weight)| (*weight > 0).then_some(*token as u8 as char))
            .collect::<String>();
        assert_eq!(supervised, "c=\n[/R3]");
    }

    #[test]
    fn semantic_proof_action_marks_goal_source_direction_and_path_as_values() {
        let window = b"?:q\n!:g12|a:r3|f|1.1\n[/R3]"
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        let mut mask = vec![0; window.len() - 1];
        assert!(ruliad_token_loss_mask(
            &window,
            &mut mask,
            RuliadTokenSupervisionConfig {
                mode: RuliadTokenSupervisionMode::AnswerCompletion,
                answer_value_token_weight: 3,
                ..Default::default()
            },
        ));
        let weighted = window
            .iter()
            .skip(1)
            .zip(mask.iter())
            .filter_map(|(token, weight)| (*weight == 3).then_some(*token as u8 as char))
            .collect::<String>();
        assert_eq!(weighted, "12r3f1.1");

        let mut value_only_mask = vec![0; window.len() - 1];
        assert!(ruliad_token_loss_mask(
            &window,
            &mut value_only_mask,
            RuliadTokenSupervisionConfig {
                mode: RuliadTokenSupervisionMode::AnswerValues,
                ..Default::default()
            },
        ));
        let values = window
            .iter()
            .skip(1)
            .zip(value_only_mask.iter())
            .filter_map(|(token, weight)| (*weight == 1).then_some(*token as u8 as char))
            .collect::<String>();
        assert_eq!(values, "12r3f1.1");
    }

    #[test]
    fn mixed_supervision_is_deterministic_from_progress() {
        let mixed = RuliadTokenSupervisionConfig {
            mode: RuliadTokenSupervisionMode::Mixed,
            ..RuliadTokenSupervisionConfig::default()
        };
        assert_eq!(
            mixed.effective_for(false, 0).mode,
            RuliadTokenSupervisionMode::AnswerCompletion
        );
        assert_eq!(
            mixed.effective_for(false, 1).mode,
            RuliadTokenSupervisionMode::FullDocument
        );
        assert_eq!(
            mixed.effective_for(true, 1).mode,
            RuliadTokenSupervisionMode::AnswerCompletion
        );
    }

    #[test]
    fn factorized_answer_balances_structure_and_values_by_step() {
        let factorized = RuliadTokenSupervisionConfig {
            mode: RuliadTokenSupervisionMode::FactorizedAnswer,
            ..RuliadTokenSupervisionConfig::default()
        };
        assert_eq!(
            factorized.effective_for(false, 0).mode,
            RuliadTokenSupervisionMode::AnswerStructure
        );
        assert_eq!(
            factorized.effective_for(false, 1).mode,
            RuliadTokenSupervisionMode::AnswerValues
        );
        assert_eq!(
            factorized.effective_for(true, 1).mode,
            RuliadTokenSupervisionMode::AnswerCompletion
        );
    }

    #[test]
    fn balanced_trace_and_answer_uses_observed_segment_mass() {
        let text = format!("[R3 x]\nP:{}\n?:root=1\n!:ok=1\n[/R3]", "trace".repeat(32));
        let window = text.bytes().map(u32::from).collect::<Vec<_>>();
        let mut balanced = vec![0; window.len() - 1];
        assert!(ruliad_token_loss_mask(
            &window,
            &mut balanced,
            RuliadTokenSupervisionConfig {
                mode: RuliadTokenSupervisionMode::TraceAndAnswer,
                balance_trace_answer_mass: true,
                ..Default::default()
            },
        ));
        assert!(balanced.iter().all(|weight| *weight >= 0));

        let mut answer_targets = vec![0; window.len() - 1];
        assert!(ruliad_token_loss_mask(
            &window,
            &mut answer_targets,
            RuliadTokenSupervisionConfig {
                mode: RuliadTokenSupervisionMode::AnswerCompletion,
                ..Default::default()
            },
        ));
        let trace_mass = balanced
            .iter()
            .zip(&answer_targets)
            .filter(|(_, answer)| **answer == 0)
            .map(|(weight, _)| *weight)
            .sum::<i64>();
        let answer_mass = balanced
            .iter()
            .zip(&answer_targets)
            .filter(|(_, answer)| **answer > 0)
            .map(|(weight, _)| *weight)
            .sum::<i64>();
        let rounding_bound = answer_targets.iter().filter(|weight| **weight > 0).count() as i64;
        assert!(trace_mass > 0);
        assert!(answer_mass > 0);
        assert!(
            (trace_mass - answer_mass).abs() <= rounding_bound,
            "trace_mass={trace_mass} answer_mass={answer_mass} rounding_bound={rounding_bound}"
        );
    }

    #[test]
    fn high_entropy_mask_covers_wire_hashes_and_random_symbol_suffixes() {
        let window = b"[R3 0123456789abcdef]\nP:atom_89abcdef\nC:[3,\"fedcba9876543210\"]\n!:ok=1"
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        let mut mask = vec![0; window.len() - 1];
        assert!(ruliad_token_loss_mask(
            &window,
            &mut mask,
            RuliadTokenSupervisionConfig {
                mode: RuliadTokenSupervisionMode::TraceAndAnswer,
                mask_high_entropy_spans: true,
                ..RuliadTokenSupervisionConfig::default()
            }
        ));
        let supervised = window
            .iter()
            .skip(1)
            .zip(&mask)
            .filter_map(|(token, weight)| (*weight > 0).then_some(*token as u8 as char))
            .collect::<String>();
        assert!(!supervised.contains("0123456789abcdef"), "{supervised}");
        assert!(!supervised.contains("89abcdef"), "{supervised}");
        assert!(!supervised.contains("fedcba9876543210"), "{supervised}");
        assert!(supervised.contains("!:ok=1"), "{supervised}");
    }
}
