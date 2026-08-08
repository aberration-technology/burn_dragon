use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::config::{
    LocalPredictiveCodingTerminalCriterion, RuliadProofPolicyNormalization,
    RuliadProofPolicyTrainingConfig,
};

use super::criterion::LocalPcTerminalCriterion;

#[derive(Debug)]
pub(crate) struct PreparedRuliadVerifierTerminal<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub criterion: LocalPcTerminalCriterion<B>,
    pub semantic_states: usize,
    pub decision_rows: usize,
}

#[derive(Debug)]
struct VerifierDecisionRow {
    inputs: Vec<i64>,
    position: usize,
    support_tokens: Vec<i64>,
    valid_tokens: Vec<i64>,
    weight: f32,
}

pub(crate) fn verifier_terminal_due(
    terminal: LocalPredictiveCodingTerminalCriterion,
    policy: RuliadProofPolicyTrainingConfig,
    absolute_step: usize,
) -> bool {
    matches!(
        terminal,
        LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet
    ) && policy.enabled
        && policy.every_steps > 0
        && absolute_step >= policy.start_after_steps
        && absolute_step.is_multiple_of(policy.every_steps)
}

/// Materialize a bounded static-expert verifier panel as sparse terminal rows.
///
/// Only actual decision points in the candidate trie become model rows;
/// deterministic action syntax is context. Every semantic state has unit total
/// weight regardless of presentation count or trie depth.
pub(crate) fn prepare_ruliad_verifier_terminal<B: Backend>(
    policy_batch: &crate::dataset::RuliadPolicyBatch,
    config: RuliadProofPolicyTrainingConfig,
    block_size: usize,
    vocab: usize,
    device: &B::Device,
) -> Option<PreparedRuliadVerifierTerminal<B>> {
    let tokenizer = burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
        &policy_batch.tokenization,
    )
    .ok()?;
    let state_budget = config.base_semantic_rows_per_update().max(1);
    let row_budget = config.max_presentation_rows_per_update.max(1);
    let mut rows = Vec::<VerifierDecisionRow>::new();
    let mut semantic_states = 0usize;

    for sample in &policy_batch.samples {
        if semantic_states >= state_budget {
            break;
        }
        let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
            problem,
            certificate,
            proof_step_index,
            action_answer_contract,
            task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
            ..
        }) = sample.item.spec.as_ref()
        else {
            continue;
        };
        // A malformed generated state must not discard every otherwise-valid
        // state in the batch. Treat generation/serialization failures as
        // state-local ineligibility and continue filling the bounded panel.
        let state_rows = (|| -> Option<Vec<VerifierDecisionRow>> {
            let state =
                burn_dragon_universality::ruliad::RuliadProofPolicyState::from_certificate_prefix(
                    problem,
                    certificate,
                    proof_step_index.unwrap_or_default(),
                )
                .ok()?;
            let actions = state.action_set(problem, config.candidates).ok()?;
            let rotations = crate::train::ruliad_policy::candidate_presentation_rotations(
                config.candidate_symmetry,
                actions.selected_index,
                actions.candidates.len(),
                semantic_states,
            )
            .ok()?;
            let mut state_rows = Vec::<VerifierDecisionRow>::new();
            for rotation in &rotations {
                let presented = actions.rotate_left(*rotation).ok()?;
                let prompt = burn_dragon_universality::ruliad::ruliad_proof_action_prompt(
                    problem, &presented,
                )
                .ok()?;
                let prompt_tokens = tokenizer
                    .encode_payload(&prompt)
                    .into_iter()
                    .map(i64::from)
                    .collect::<Vec<_>>();
                let candidates = (0..presented.candidates.len())
                    .map(|candidate_index| {
                        let answer = burn_dragon_universality::ruliad::proof_action_answer(
                            &presented,
                            candidate_index,
                            *action_answer_contract,
                        )
                        .ok()?;
                        let tokens = tokenizer
                            .encode_payload(&answer)
                            .into_iter()
                            .map(i64::from)
                            .collect::<Vec<_>>();
                        (!tokens.is_empty() && tokens.len() <= config.max_completion_tokens)
                            .then_some(tokens)
                    })
                    .collect::<Option<Vec<_>>>()?;
                let branches = crate::train::ruliad_policy::semantic_candidate_trie_branches(
                    &candidates,
                    &presented.equivalent_indices,
                )
                .ok()?;
                let branch_weight =
                    1.0 / rotations.len().max(1) as f32 / branches.len().max(1) as f32;
                for branch in branches {
                    let max_prompt = block_size.saturating_sub(branch.prefix.len()).max(1);
                    let prompt_start = prompt_tokens.len().saturating_sub(max_prompt);
                    let mut inputs = prompt_tokens[prompt_start..].to_vec();
                    inputs.extend_from_slice(&branch.prefix);
                    if inputs.is_empty() || inputs.len() > block_size {
                        return None;
                    }
                    let support_tokens = match config.normalization {
                        RuliadProofPolicyNormalization::VocabularyMarginal => Vec::new(),
                        RuliadProofPolicyNormalization::CandidateConditional
                        | RuliadProofPolicyNormalization::PrefixConditional => {
                            branch.candidate_tokens
                        }
                    };
                    state_rows.push(VerifierDecisionRow {
                        position: inputs.len() - 1,
                        inputs,
                        support_tokens,
                        valid_tokens: branch.equivalent_tokens,
                        weight: branch_weight,
                    });
                }
            }
            Some(state_rows)
        })();
        let Some(state_rows) = state_rows else {
            continue;
        };
        if state_rows.is_empty() || rows.len().saturating_add(state_rows.len()) > row_budget {
            continue;
        }
        rows.extend(state_rows);
        semantic_states = semantic_states.saturating_add(1);
    }
    if rows.is_empty() {
        return None;
    }

    let row_count = rows.len();
    let sequence_len = rows.iter().map(|row| row.inputs.len()).max()?.max(1);
    let mut input_values = vec![0_i64; row_count * sequence_len];
    let mut positions = Vec::with_capacity(row_count);
    let mut support = vec![0.0_f32; row_count * vocab];
    let mut valid = vec![0.0_f32; row_count * vocab];
    let mut weights = Vec::with_capacity(row_count);
    for (row_index, row) in rows.into_iter().enumerate() {
        let offset = row_index * sequence_len;
        input_values[offset..offset + row.inputs.len()].copy_from_slice(&row.inputs);
        positions.push(i64::try_from(row.position).ok()?);
        weights.push(row.weight);
        if row.support_tokens.is_empty() {
            support[row_index * vocab..(row_index + 1) * vocab].fill(1.0);
        } else {
            for token in row.support_tokens {
                let token = usize::try_from(token).ok()?;
                if token >= vocab {
                    return None;
                }
                support[row_index * vocab + token] = 1.0;
            }
        }
        for token in row.valid_tokens {
            let token = usize::try_from(token).ok()?;
            if token >= vocab || support[row_index * vocab + token] == 0.0 {
                return None;
            }
            valid[row_index * vocab + token] = 1.0;
        }
    }

    Some(PreparedRuliadVerifierTerminal {
        inputs: Tensor::from_data(
            TensorData::new(input_values, [row_count, sequence_len]),
            device,
        ),
        criterion: LocalPcTerminalCriterion::CategoricalSetAtPositions {
            positions: Tensor::from_data(TensorData::new(positions, [row_count]), device),
            support_action_mask: Tensor::from_data(
                TensorData::new(support, [row_count, vocab]),
                device,
            ),
            valid_action_mask: Tensor::from_data(
                TensorData::new(valid, [row_count, vocab]),
                device,
            ),
            row_weights: Tensor::from_data(TensorData::new(weights, [row_count]), device),
            eps: 1.0e-12,
        },
        semantic_states,
        decision_rows: row_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use std::path::PathBuf;

    type TestBackend = NdArray<f32>;

    fn formal_policy_batch(
        answer_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
    ) -> crate::dataset::RuliadPolicyBatch {
        let bundle = burn_dragon_universality::ruliad::formal::generate_formal_bundle(
            29,
            burn_dragon_universality::ruliad::formal::RuliadFormalGeneratorConfig {
                rewrite_depth: 2,
                leaf_count: 3,
                context_depth: 1,
                distractor_axioms: 1,
                ..Default::default()
            },
        )
        .expect("formal bundle");
        let proof_step_index = 1.min(bundle.certificate.step_count().saturating_sub(1));
        let actions = burn_dragon_universality::ruliad::oracle_proof_action_set(
            &bundle.problem,
            &bundle.certificate,
            proof_step_index,
            4,
        )
        .expect("oracle action set");
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: bundle.problem.canonical_hash().expect("problem hash"),
            sample_index: 29,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "formal_proof".to_string(),
            task_kind: burn_dragon_universality::RuliadTaskKind::SelectProofAction
                .label()
                .to_string(),
            math_domains: vec!["formal_proof".to_string()],
            reasoning_modes: vec!["proof_construction".to_string()],
            prompt: burn_dragon_universality::ruliad::ruliad_proof_action_prompt(
                &bundle.problem,
                &actions,
            )
            .expect("policy prompt"),
            expected_answer: format!("c={}", actions.selected_index),
            difficulty_level: Some(0),
            spec: Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                problem: bundle.problem,
                certificate: bundle.certificate,
                candidate: None,
                proof_step_index: Some(proof_step_index),
                action_presentation_rotation: Some(0),
                action_answer_contract: answer_contract,
                task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
            }),
        };
        crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::StructuredSymbolic {
                vocab_size: 272,
                eos_id: Some(271),
            },
            stop_token_id: Some(271),
        }
    }

    fn policy(normalization: RuliadProofPolicyNormalization) -> RuliadProofPolicyTrainingConfig {
        RuliadProofPolicyTrainingConfig {
            enabled: true,
            mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
            scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
            gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
            normalization,
            candidate_symmetry:
                crate::config::RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage,
            presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
            weight: 1.0,
            every_steps: 1,
            start_after_steps: 0,
            max_rows_per_update: 1,
            max_presentation_rows_per_update: 64,
            candidates: 4,
            max_completion_tokens: 128,
            ..RuliadProofPolicyTrainingConfig::default()
        }
    }

    #[test]
    fn verifier_schedule_is_explicit_and_periodic() {
        let policy = RuliadProofPolicyTrainingConfig {
            enabled: true,
            every_steps: 4,
            start_after_steps: 8,
            ..RuliadProofPolicyTrainingConfig::default()
        };
        for step in 0..16 {
            assert_eq!(
                verifier_terminal_due(
                    LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet,
                    policy,
                    step,
                ),
                matches!(step, 8 | 12)
            );
        }
        assert!(!verifier_terminal_due(
            LocalPredictiveCodingTerminalCriterion::NextToken,
            policy,
            8,
        ));
    }

    #[test]
    fn semantic_verifier_panel_materializes_sparse_normalized_trie_rows() {
        let device = Default::default();
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let prepared = prepare_ruliad_verifier_terminal::<TestBackend>(
            &batch,
            policy(RuliadProofPolicyNormalization::CandidateConditional),
            512,
            272,
            &device,
        )
        .expect("semantic verifier terminal");
        assert_eq!(prepared.semantic_states, 1);
        assert!(
            prepared.decision_rows > 1,
            "semantic answers should form a trie"
        );
        let [rows, time] = prepared.inputs.shape().dims::<2>();
        assert_eq!(rows, prepared.decision_rows);
        assert!(time <= 512);

        let LocalPcTerminalCriterion::CategoricalSetAtPositions {
            positions,
            support_action_mask,
            valid_action_mask,
            row_weights,
            ..
        } = prepared.criterion
        else {
            panic!("verifier panel must use sparse positions");
        };
        let positions = positions.into_data().to_vec::<i64>().expect("positions");
        assert!(
            positions
                .iter()
                .all(|position| *position >= 0 && *position < time as i64)
        );
        let support = support_action_mask
            .into_data()
            .to_vec::<f32>()
            .expect("support mask");
        let valid = valid_action_mask
            .into_data()
            .to_vec::<f32>()
            .expect("valid mask");
        let weights = row_weights.into_data().to_vec::<f32>().expect("weights");
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
        for row in 0..rows {
            let range = row * 272..(row + 1) * 272;
            let support_count = support[range.clone()].iter().filter(|v| **v > 0.0).count();
            let valid_count = valid[range.clone()].iter().filter(|v| **v > 0.0).count();
            assert!(support_count >= valid_count && valid_count >= 1);
            assert!(
                support_count < 272,
                "conditional support must stay candidate-local"
            );
            assert!(
                valid[range]
                    .iter()
                    .zip(&support[row * 272..(row + 1) * 272])
                    .all(|(valid, support)| *valid <= *support)
            );
        }
    }

    #[test]
    fn vocabulary_marginal_panel_exposes_the_full_vocabulary_support() {
        let device = Default::default();
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::PresentationIndex,
        );
        let prepared = prepare_ruliad_verifier_terminal::<TestBackend>(
            &batch,
            policy(RuliadProofPolicyNormalization::VocabularyMarginal),
            512,
            272,
            &device,
        )
        .expect("vocabulary-marginal verifier terminal");
        let LocalPcTerminalCriterion::CategoricalSetAtPositions {
            support_action_mask,
            ..
        } = prepared.criterion
        else {
            panic!("verifier panel must use sparse positions");
        };
        let support = support_action_mask
            .into_data()
            .to_vec::<f32>()
            .expect("support mask");
        assert!(
            support
                .iter()
                .all(|value| (*value - 1.0).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn local_pc_profile_materializes_source_selected_verifier_rows() {
        use crate::dataset::{DatasetSplit, TokenSequenceDataset};

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let profile =
            root.join("config/language/experiments/predictive_coding/local-pc-verifier-1m.toml");
        let mut training =
            crate::config::load_training_config(&[profile]).expect("load local PC profile");
        if let crate::config::DatasetSourceConfig::UniversalityRuliad { config } =
            &mut training.dataset.source
        {
            *config = root.join(&*config);
        } else {
            panic!("local PC profile must use the Ruliad corpus");
        }
        let datasets = crate::train::utils::prepare_datasets(&training.dataset, &training.training)
            .expect("prepare local PC datasets");
        let vocab = datasets.train.tokenizer().len();
        let batch = TokenSequenceDataset::source_selected_ruliad_policy_batch(
            datasets.train.as_ref(),
            DatasetSplit::Train,
            0,
            0,
            32,
            4,
        )
        .expect("source-selected proof-policy batch");
        assert!(batch.samples.iter().all(|sample| matches!(
            sample.item.spec,
            Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
                ..
            })
        )));

        let mut config = policy(RuliadProofPolicyNormalization::PrefixConditional);
        config.candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
        config.max_rows_per_update = 8;
        config.max_presentation_rows_per_update = 64;
        config.stratified_difficulty_levels = 4;
        let prepared = prepare_ruliad_verifier_terminal::<TestBackend>(
            &batch,
            config,
            training.training.block_size,
            vocab,
            &Default::default(),
        )
        .expect("real source-selected batch must yield verifier rows");
        assert!(prepared.semantic_states > 0);
        assert!(prepared.decision_rows <= config.max_presentation_rows_per_update);
    }
}
