#[cfg(any(feature = "native", test))]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
#[cfg(feature = "native")]
use std::fs;

#[cfg(feature = "native")]
use anyhow::Context;
#[cfg(any(feature = "native", test))]
use anyhow::ensure;
use anyhow::{Result, bail};
use burn::module::Module;
use burn::tensor::backend::Backend;
#[cfg(any(feature = "native", test))]
use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_core::DragonModel;
#[cfg(any(feature = "native", test))]
use burn_dragon_core::ModelState;
use burn_dragon_eggroll::collect_matrix_params;
#[cfg(any(feature = "native", test))]
use burn_dragon_eggroll::{
    AntitheticFitness, EggrollModuleOptimizerState, apply_antithetic_update_with_allowed_param_ids,
    perturb_module_with_allowed_param_ids,
};
#[cfg(feature = "native")]
use burn_p2p::CachedMicroShard;
use burn_p2p::ContentId;
#[cfg(any(feature = "native", test))]
use burn_p2p::{
    CompactScalarEncoding, CompactUpdateBody, SeededFitnessReplayPolicy, UpdateReplayStats,
};
#[cfg(any(feature = "native", test))]
use burn_p2p_workload::ValidatedCompactUpdate;

#[cfg(any(feature = "native", test))]
use crate::config::TokenWindowRecord;

#[derive(Clone, Debug)]
pub(crate) struct DragonSeededFitnessCatalog {
    pub allowed_param_ids: BTreeSet<u64>,
    pub parameter_catalog_hash: ContentId,
    pub parameter_count: u64,
    pub perturbation_generator_hash: ContentId,
}

pub(crate) fn dragon_seeded_fitness_catalog<B: Backend>(
    model: &DragonModel<B>,
    config: &burn_eggroll::EggrollConfig,
) -> Result<DragonSeededFitnessCatalog>
where
    DragonModel<B>: Module<B>,
{
    config.validate()?;
    let mut catalog = collect_matrix_params::<B, _>(model)
        .params
        .into_iter()
        .filter(|parameter| matches!(parameter.path.as_str(), "encoder" | "encoder_v" | "decoder"))
        .collect::<Vec<_>>();
    catalog.sort_by(|left, right| left.path.cmp(&right.path));
    if catalog.len() != 3 {
        bail!(
            "Dragon seeded-fitness contract expected encoder, encoder_v, and decoder; found {:?}",
            catalog
                .iter()
                .map(|parameter| parameter.path.as_str())
                .collect::<Vec<_>>()
        );
    }
    let allowed_param_ids = catalog.iter().map(|parameter| parameter.param_id).collect();
    let parameter_count = catalog
        .iter()
        .map(|parameter| parameter.shape.iter().product::<usize>() as u64)
        .sum();
    let wire_catalog = catalog
        .iter()
        .map(|parameter| {
            (
                parameter.path.clone(),
                parameter.perturbation_key,
                parameter.rank,
                parameter.shape.clone(),
            )
        })
        .collect::<Vec<_>>();
    let parameter_catalog_hash = ContentId::derive(&wire_catalog)?;
    let perturbation_generator_hash = ContentId::derive(&(
        "burn-dragon-eggroll-generator-v2",
        config.population.seed,
        config.population.rank,
        config.population.matrix_noise,
        &parameter_catalog_hash,
    ))?;
    Ok(DragonSeededFitnessCatalog {
        allowed_param_ids,
        parameter_catalog_hash,
        parameter_count,
        perturbation_generator_hash,
    })
}

#[cfg(any(feature = "native", test))]
pub(crate) fn replay_dragon_seeded_fitness_update<B: Backend>(
    mut model: DragonModel<B>,
    config: &burn_eggroll::EggrollConfig,
    optimizer_update_hash: &ContentId,
    update: &ValidatedCompactUpdate,
) -> Result<DragonModel<B>>
where
    DragonModel<B>: Module<B>,
{
    let catalog = dragon_seeded_fitness_catalog(&model, config)?;
    ensure!(
        update.payload.parameter_catalog_hash == catalog.parameter_catalog_hash,
        "seeded-fitness parameter catalog hash mismatch"
    );
    ensure!(
        update.payload.parameter_count == catalog.parameter_count,
        "seeded-fitness parameter count mismatch"
    );
    let CompactUpdateBody::SeededFitness {
        perturbation_generator_hash,
        optimizer_update_hash: payload_optimizer_hash,
        generations,
        ..
    } = &update.payload.body
    else {
        bail!("Dragon seeded-fitness replay received a different compact update codec");
    };
    ensure!(
        perturbation_generator_hash == &catalog.perturbation_generator_hash,
        "seeded-fitness perturbation generator hash mismatch"
    );
    ensure!(
        payload_optimizer_hash == optimizer_update_hash,
        "seeded-fitness optimizer update hash mismatch"
    );

    let mut optimizer_state = EggrollModuleOptimizerState::new();
    for generation in generations {
        let values = generation.fitness.decode()?;
        ensure!(
            values.len().is_multiple_of(2),
            "seeded-fitness population must contain antithetic pairs"
        );
        let fitness = values
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
            .map(|(pair_index, pair)| AntitheticFitness {
                pair_index: pair_index as u64,
                plus: pair[0],
                minus: pair[1],
            })
            .collect::<Vec<_>>();
        (model, _) = apply_antithetic_update_with_allowed_param_ids(
            model,
            config,
            generation.generation,
            &fitness,
            &mut optimizer_state,
            Some(&catalog.allowed_param_ids),
        )?;
    }
    Ok(model)
}

#[cfg(feature = "native")]
pub(crate) fn load_replay_token_window_records(
    cached_microshards: &[CachedMicroShard],
) -> Result<Vec<TokenWindowRecord>> {
    let mut records = Vec::new();
    for shard in cached_microshards {
        let bytes = fs::read(&shard.path)
            .with_context(|| format!("read replay shard {}", shard.path.display()))?;
        let mut shard_records = serde_json::from_slice::<Vec<TokenWindowRecord>>(&bytes)
            .with_context(|| format!("decode replay shard {}", shard.path.display()))?;
        records.append(&mut shard_records);
    }
    ensure!(
        !records.is_empty(),
        "seeded-fitness replay lease produced no token-window records"
    );
    Ok(records)
}

#[cfg(any(feature = "native", test))]
struct DragonReplayBatch<B: Backend> {
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    batch_digest: ContentId,
    reset_stream_state: bool,
}

#[cfg(any(feature = "native", test))]
fn build_replay_record_catalog(
    records: Vec<TokenWindowRecord>,
) -> Result<BTreeMap<ContentId, TokenWindowRecord>> {
    let mut catalog = BTreeMap::new();
    for record in records {
        let digest = ContentId::derive(&record)?;
        if let Some(existing) = catalog.insert(digest.clone(), record.clone()) {
            ensure!(
                existing == record,
                "token-window record digest collision for {}",
                digest.as_str()
            );
        }
    }
    Ok(catalog)
}

#[cfg(any(feature = "native", test))]
fn replay_batch_from_generation<B: Backend>(
    generation: &burn_p2p::SeededFitnessGeneration,
    record_catalog: &BTreeMap<ContentId, TokenWindowRecord>,
    device: &B::Device,
) -> Result<DragonReplayBatch<B>> {
    let records = generation
        .record_digests
        .iter()
        .map(|digest| {
            record_catalog.get(digest).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "seeded-fitness replay record {} was not present in the authenticated lease",
                    digest.as_str()
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let block_size = records
        .first()
        .map(|record| record.inputs.len())
        .unwrap_or_default();
    ensure!(
        block_size > 0
            && records.iter().all(|record| {
                record.inputs.len() == block_size && record.targets.len() == block_size
            }),
        "seeded-fitness replay records have inconsistent token-window shapes"
    );
    let batch_digest = ContentId::derive(&(
        "dragon-token-window-batch-v2",
        records.iter().collect::<Vec<_>>(),
        block_size,
        generation.reset_stream_state,
    ))?;
    ensure!(
        batch_digest == generation.batch_digest,
        "seeded-fitness replay batch digest mismatch"
    );
    let mut inputs = Vec::with_capacity(records.len() * block_size);
    let mut targets = Vec::with_capacity(records.len() * block_size);
    for record in &records {
        inputs.extend(record.inputs.iter().copied());
        targets.extend(record.targets.iter().copied());
    }
    Ok(DragonReplayBatch {
        inputs: Tensor::<B, 2, Int>::from_data(
            TensorData::new(inputs, [records.len(), block_size]),
            device,
        ),
        targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(targets, [records.len(), block_size]),
            device,
        ),
        batch_digest,
        reset_stream_state: generation.reset_stream_state,
    })
}

#[cfg(any(feature = "native", test))]
fn visit_replay_next_token_chunks<B: Backend>(
    model: &DragonModel<B>,
    batch: &DragonReplayBatch<B>,
    state: &mut ModelState<B>,
    tbptt_chunk_size: Option<usize>,
    mut visit: impl FnMut(Tensor<B, 1>),
) {
    let [batch_size, block_size] = batch.inputs.shape().dims();
    let chunk_size = tbptt_chunk_size
        .filter(|chunk_size| *chunk_size > 0)
        .unwrap_or(block_size.max(1))
        .min(block_size.max(1));
    for start in (0..block_size).step_by(chunk_size) {
        let end = (start + chunk_size).min(block_size);
        let inputs = batch.inputs.clone().slice([0..batch_size, start..end]);
        let targets = batch.targets.clone().slice([0..batch_size, start..end]);
        let hidden = model.forward_hidden_with_state(inputs, state);
        let chunk_weight = (end - start) as f32 / block_size.max(1) as f32;
        visit(
            model
                .language_loss_from_hidden(hidden, targets)
                .mul_scalar(chunk_weight),
        );
        if end < block_size {
            state.detach_in_place();
        }
    }
}

#[cfg(any(feature = "native", test))]
fn replay_fitness<B: Backend>(
    model: &DragonModel<B>,
    batch: &DragonReplayBatch<B>,
    state: &ModelState<B>,
    tbptt_chunk_size: Option<usize>,
) -> Result<f64> {
    let mut state = state.detached_clone();
    let mut losses = Vec::new();
    visit_replay_next_token_chunks(model, batch, &mut state, tbptt_chunk_size, |loss| {
        losses.push(loss)
    });
    let values = Tensor::cat(losses, 0)
        .sum()
        .reshape([1])
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|error| anyhow::anyhow!("decode replay loss scalar: {error}"))?;
    Ok(-f64::from(values.first().copied().unwrap_or_default()))
}

#[cfg(any(feature = "native", test))]
fn replay_pair_indices(
    payload_id: &ContentId,
    generation: u64,
    pair_count: usize,
    requested: usize,
) -> Vec<usize> {
    let mut seed = 0xcbf29ce484222325_u64;
    for byte in payload_id.as_str().bytes().chain(generation.to_le_bytes()) {
        seed ^= u64::from(byte);
        seed = seed.wrapping_mul(0x100000001b3);
    }
    let mut indices = (0..pair_count).collect::<Vec<_>>();
    indices.sort_by_key(|index| splitmix64(seed ^ *index as u64));
    indices.truncate(requested.min(pair_count));
    indices.sort_unstable();
    indices
}

#[cfg(any(feature = "native", test))]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

#[cfg(any(feature = "native", test))]
fn fitness_matches(
    observed: f64,
    expected: f64,
    policy: &SeededFitnessReplayPolicy,
    quantization_tolerance: f64,
) -> (bool, f64, f64) {
    let absolute_error = (observed - expected).abs();
    let scale = observed.abs().max(expected.abs()).max(1.0e-12);
    let relative_error = absolute_error / scale;
    let allowed =
        policy.absolute_tolerance() + policy.relative_tolerance() * scale + quantization_tolerance;
    (absolute_error <= allowed, absolute_error, relative_error)
}

#[cfg(any(feature = "native", test))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_and_replay_dragon_seeded_fitness_update<B: Backend>(
    mut model: DragonModel<B>,
    config: &burn_eggroll::EggrollConfig,
    optimizer_update_hash: &ContentId,
    update: &ValidatedCompactUpdate,
    records: Vec<TokenWindowRecord>,
    replay_policy: &SeededFitnessReplayPolicy,
    tbptt_chunk_size: Option<usize>,
    persist_across_steps: bool,
    device: &B::Device,
) -> Result<(DragonModel<B>, UpdateReplayStats)>
where
    DragonModel<B>: Module<B>,
{
    let catalog = dragon_seeded_fitness_catalog(&model, config)?;
    ensure!(
        update.payload.parameter_catalog_hash == catalog.parameter_catalog_hash,
        "seeded-fitness parameter catalog hash mismatch"
    );
    ensure!(
        update.payload.parameter_count == catalog.parameter_count,
        "seeded-fitness parameter count mismatch"
    );
    let CompactUpdateBody::SeededFitness {
        perturbation_generator_hash,
        optimizer_update_hash: payload_optimizer_hash,
        generations,
        ..
    } = &update.payload.body
    else {
        bail!("Dragon seeded-fitness replay received a different compact update codec");
    };
    ensure!(
        perturbation_generator_hash == &catalog.perturbation_generator_hash,
        "seeded-fitness perturbation generator hash mismatch"
    );
    ensure!(
        payload_optimizer_hash == optimizer_update_hash,
        "seeded-fitness optimizer update hash mismatch"
    );
    let record_catalog = build_replay_record_catalog(records)?;
    let pair_count = config.population.population_size / 2;
    let mut optimizer_state = EggrollModuleOptimizerState::new();
    let mut state_slot = None;
    let mut pairs_checked = 0_u32;
    let mut max_absolute_error = 0.0_f64;
    let mut max_relative_error = 0.0_f64;

    for generation in generations {
        let batch = replay_batch_from_generation::<B>(generation, &record_catalog, device)?;
        ensure!(
            batch.batch_digest == generation.batch_digest,
            "seeded-fitness replay batch changed after reconstruction"
        );
        let base_state = if persist_across_steps {
            if batch.reset_stream_state {
                state_slot = None;
            }
            state_slot.take().unwrap_or_else(|| model.init_state())
        } else {
            model.init_state_ephemeral()
        };
        let values = generation.fitness.decode()?;
        ensure!(
            values.len() == pair_count * 2,
            "seeded-fitness population does not match replay configuration"
        );
        let selected_pairs = replay_pair_indices(
            &update.payload_id,
            generation.generation,
            pair_count,
            replay_policy.pairs_per_generation as usize,
        );
        let quantization_tolerance = match generation.fitness.encoding {
            CompactScalarEncoding::Fp32 => 0.0,
            CompactScalarEncoding::SymmetricInt8 | CompactScalarEncoding::SymmetricInt16 => {
                f64::from(generation.fitness.scale.abs()) * 0.51
            }
        };
        for pair_index in selected_pairs {
            for (offset, sign) in [
                burn_eggroll::AntitheticSign::Plus,
                burn_eggroll::AntitheticSign::Minus,
            ]
            .into_iter()
            .enumerate()
            {
                let candidate = perturb_module_with_allowed_param_ids(
                    model.clone(),
                    config,
                    generation.generation,
                    pair_index as u64,
                    sign,
                    Some(&catalog.allowed_param_ids),
                );
                let observed = replay_fitness(&candidate, &batch, &base_state, tbptt_chunk_size)?;
                let expected = f64::from(values[pair_index * 2 + offset]);
                let (matches, absolute_error, relative_error) =
                    fitness_matches(observed, expected, replay_policy, quantization_tolerance);
                max_absolute_error = max_absolute_error.max(absolute_error);
                max_relative_error = max_relative_error.max(relative_error);
                ensure!(
                    matches,
                    "seeded-fitness replay mismatch at generation {} pair {} sign {:?}: observed={} transmitted={} absolute_error={} relative_error={}",
                    generation.generation,
                    pair_index,
                    sign,
                    observed,
                    expected,
                    absolute_error,
                    relative_error,
                );
            }
            pairs_checked = pairs_checked.saturating_add(1);
        }
        let fitness = values
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
            .map(|(pair_index, pair)| AntitheticFitness {
                pair_index: pair_index as u64,
                plus: pair[0],
                minus: pair[1],
            })
            .collect::<Vec<_>>();
        (model, _) = apply_antithetic_update_with_allowed_param_ids(
            model,
            config,
            generation.generation,
            &fitness,
            &mut optimizer_state,
            Some(&catalog.allowed_param_ids),
        )?;
        if persist_across_steps {
            let mut next_state = base_state;
            visit_replay_next_token_chunks(&model, &batch, &mut next_state, tbptt_chunk_size, drop);
            next_state.detach_in_place();
            state_slot = Some(next_state);
        }
    }
    Ok((
        model,
        UpdateReplayStats {
            generations_checked: generations.len() as u32,
            pairs_checked,
            total_pairs: (generations.len() * pair_count) as u32,
            max_absolute_error,
            max_relative_error,
        },
    ))
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_dragon_core::DragonConfig;
    use burn_p2p::{
        COMPACT_UPDATE_PAYLOAD_VERSION, CompactScalarEncoding, CompactScalarVector,
        CompactUpdatePayload, SeededFitnessGeneration,
    };

    use super::*;

    fn fixture(
        model: &DragonModel<NdArray<f32>>,
        config: &burn_eggroll::EggrollConfig,
        optimizer_hash: &ContentId,
    ) -> ValidatedCompactUpdate {
        let catalog = dragon_seeded_fitness_catalog(model, config).expect("seeded-fitness catalog");
        let fitness = (0..config.population.population_size)
            .map(|index| if index.is_multiple_of(2) { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let payload = CompactUpdatePayload {
            version: COMPACT_UPDATE_PAYLOAD_VERSION,
            training_contract_id: ContentId::new("contract"),
            model_schema_hash: ContentId::new("schema"),
            parameter_catalog_hash: catalog.parameter_catalog_hash,
            parameter_count: catalog.parameter_count,
            body: CompactUpdateBody::SeededFitness {
                population: config.population.population_size as u32,
                rank: config.population.rank as u32,
                seed: config.population.seed,
                perturbation_generator_hash: catalog.perturbation_generator_hash,
                optimizer_update_hash: optimizer_hash.clone(),
                generations: vec![SeededFitnessGeneration {
                    generation: 3,
                    batch_digest: ContentId::new("batch"),
                    record_digests: vec![ContentId::new("record")],
                    reset_stream_state: true,
                    fitness: CompactScalarVector::encode(&fitness, CompactScalarEncoding::Fp32)
                        .expect("fitness encoding"),
                }],
            },
        };
        ValidatedCompactUpdate {
            payload_id: ContentId::derive(&payload).expect("payload id"),
            payload,
        }
    }

    #[test]
    fn seeded_fitness_replay_is_deterministic_and_changes_outputs() {
        let device = burn::tensor::Device::<NdArray<f32>>::default();
        let config = burn_eggroll::EggrollConfig::default();
        let model_config = DragonConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            ..DragonConfig::default()
        };
        let model = DragonModel::new(model_config, &device);
        let optimizer_hash = ContentId::new("optimizer");
        let update = fixture(&model, &config, &optimizer_hash);
        let inputs = Tensor::<NdArray<f32>, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
            &device,
        );
        let before = model
            .forward(inputs.clone())
            .into_data()
            .into_vec::<f32>()
            .expect("before logits");

        let left =
            replay_dragon_seeded_fitness_update(model.clone(), &config, &optimizer_hash, &update)
                .expect("left replay");
        let right = replay_dragon_seeded_fitness_update(model, &config, &optimizer_hash, &update)
            .expect("right replay");
        let left = left
            .forward(inputs.clone())
            .into_data()
            .into_vec::<f32>()
            .expect("left logits");
        let right = right
            .forward(inputs)
            .into_data()
            .into_vec::<f32>()
            .expect("right logits");

        assert_eq!(left, right);
        assert_ne!(before, left);
    }

    #[test]
    fn seeded_fitness_replay_rejects_optimizer_semantic_mismatch() {
        let device = burn::tensor::Device::<NdArray<f32>>::default();
        let config = burn_eggroll::EggrollConfig::default();
        let model = DragonModel::new(
            DragonConfig {
                n_layer: 1,
                n_embd: 8,
                n_head: 1,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 32,
                dropout: 0.0,
                ..DragonConfig::default()
            },
            &device,
        );
        let update = fixture(&model, &config, &ContentId::new("optimizer-a"));

        assert!(
            replay_dragon_seeded_fitness_update(
                model,
                &config,
                &ContentId::new("optimizer-b"),
                &update,
            )
            .expect_err("optimizer mismatch")
            .to_string()
            .contains("optimizer update hash mismatch")
        );
    }

    fn independently_replayable_fixture(
        model: &DragonModel<NdArray<f32>>,
        config: &burn_eggroll::EggrollConfig,
        optimizer_hash: &ContentId,
        record: &TokenWindowRecord,
    ) -> ValidatedCompactUpdate {
        let catalog = dragon_seeded_fitness_catalog(model, config).expect("catalog");
        let record_digest = ContentId::derive(record).expect("record digest");
        let batch_digest = ContentId::derive(&(
            "dragon-token-window-batch-v2",
            vec![record],
            record.inputs.len(),
            true,
        ))
        .expect("batch digest");
        let generation = 7;
        let placeholder = SeededFitnessGeneration {
            generation,
            batch_digest,
            record_digests: vec![record_digest.clone()],
            reset_stream_state: true,
            fitness: CompactScalarVector::encode(&[0.0, 0.0], CompactScalarEncoding::Fp32)
                .expect("placeholder fitness"),
        };
        let records = BTreeMap::from([(record_digest, record.clone())]);
        let device = burn::tensor::Device::<NdArray<f32>>::default();
        let batch = replay_batch_from_generation::<NdArray<f32>>(&placeholder, &records, &device)
            .expect("replay batch");
        let state = model.init_state_ephemeral();
        let fitness = [
            burn_eggroll::AntitheticSign::Plus,
            burn_eggroll::AntitheticSign::Minus,
        ]
        .into_iter()
        .map(|sign| {
            let candidate = perturb_module_with_allowed_param_ids(
                model.clone(),
                config,
                generation,
                0,
                sign,
                Some(&catalog.allowed_param_ids),
            );
            replay_fitness(&candidate, &batch, &state, None).expect("candidate fitness") as f32
        })
        .collect::<Vec<_>>();
        let payload = CompactUpdatePayload {
            version: COMPACT_UPDATE_PAYLOAD_VERSION,
            training_contract_id: ContentId::new("contract"),
            model_schema_hash: ContentId::new("schema"),
            parameter_catalog_hash: catalog.parameter_catalog_hash,
            parameter_count: catalog.parameter_count,
            body: CompactUpdateBody::SeededFitness {
                population: 2,
                rank: config.population.rank as u32,
                seed: config.population.seed,
                perturbation_generator_hash: catalog.perturbation_generator_hash,
                optimizer_update_hash: optimizer_hash.clone(),
                generations: vec![SeededFitnessGeneration {
                    fitness: CompactScalarVector::encode(&fitness, CompactScalarEncoding::Fp32)
                        .expect("fitness"),
                    ..placeholder
                }],
            },
        };
        ValidatedCompactUpdate {
            payload_id: ContentId::derive(&payload).expect("payload id"),
            payload,
        }
    }

    #[test]
    fn independent_seeded_fitness_replay_accepts_exact_observations_and_rejects_tampering() {
        let device = burn::tensor::Device::<NdArray<f32>>::default();
        let mut config = burn_eggroll::EggrollConfig::default();
        config.population.population_size = 2;
        config.population.population_chunk_size = 2;
        let model = DragonModel::new(
            DragonConfig {
                n_layer: 1,
                n_embd: 8,
                n_head: 1,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 32,
                dropout: 0.0,
                ..DragonConfig::default()
            },
            &device,
        );
        let record = TokenWindowRecord {
            inputs: vec![1, 2, 3, 4],
            targets: vec![2, 3, 4, 5],
            reset_stream_state: true,
            ..TokenWindowRecord::default()
        };
        let optimizer_hash = ContentId::new("optimizer");
        let exact = independently_replayable_fixture(&model, &config, &optimizer_hash, &record);
        let (_, evidence) = validate_and_replay_dragon_seeded_fitness_update(
            model.clone(),
            &config,
            &optimizer_hash,
            &exact,
            vec![record.clone()],
            &SeededFitnessReplayPolicy::default(),
            None,
            false,
            &device,
        )
        .expect("exact replay");
        assert_eq!(evidence.generations_checked, 1);
        assert_eq!(evidence.pairs_checked, 1);
        assert_eq!(evidence.total_pairs, 1);

        let mut wrong_batch = exact.clone();
        let CompactUpdateBody::SeededFitness { generations, .. } = &mut wrong_batch.payload.body
        else {
            panic!("seeded fitness payload");
        };
        generations[0].batch_digest = ContentId::new("invented-batch");
        wrong_batch.payload_id =
            ContentId::derive(&wrong_batch.payload).expect("wrong batch payload id");
        let error = validate_and_replay_dragon_seeded_fitness_update(
            model.clone(),
            &config,
            &optimizer_hash,
            &wrong_batch,
            vec![record.clone()],
            &SeededFitnessReplayPolicy::default(),
            None,
            false,
            &device,
        )
        .expect_err("invented batch digest must fail");
        assert!(
            error.to_string().contains("batch digest mismatch"),
            "{error:#}"
        );

        let mut unleased_record = exact.clone();
        let CompactUpdateBody::SeededFitness { generations, .. } =
            &mut unleased_record.payload.body
        else {
            panic!("seeded fitness payload");
        };
        generations[0].record_digests = vec![ContentId::new("unleased-record")];
        unleased_record.payload_id =
            ContentId::derive(&unleased_record.payload).expect("unleased record payload id");
        let error = validate_and_replay_dragon_seeded_fitness_update(
            model.clone(),
            &config,
            &optimizer_hash,
            &unleased_record,
            vec![record.clone()],
            &SeededFitnessReplayPolicy::default(),
            None,
            false,
            &device,
        )
        .expect_err("record outside the authenticated lease must fail");
        assert!(
            error
                .to_string()
                .contains("was not present in the authenticated lease"),
            "{error:#}"
        );

        let mut tampered = exact;
        let CompactUpdateBody::SeededFitness { generations, .. } = &mut tampered.payload.body
        else {
            panic!("seeded fitness payload");
        };
        let mut fitness = generations[0].fitness.decode().expect("decode fitness");
        fitness[0] += 0.25;
        generations[0].fitness = CompactScalarVector::encode(&fitness, CompactScalarEncoding::Fp32)
            .expect("tampered fitness");
        tampered.payload_id = ContentId::derive(&tampered.payload).expect("tampered payload id");
        let error = validate_and_replay_dragon_seeded_fitness_update(
            model,
            &config,
            &optimizer_hash,
            &tampered,
            vec![record],
            &SeededFitnessReplayPolicy::default(),
            None,
            false,
            &device,
        )
        .expect_err("invented fitness must fail");
        assert!(error.to_string().contains("replay mismatch"), "{error:#}");
    }
}
