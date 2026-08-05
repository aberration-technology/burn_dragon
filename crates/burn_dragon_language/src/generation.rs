use anyhow::{Result, anyhow};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor, TensorData};
use burn_dragon_time::Instant;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::cmp::Ordering;
use std::mem::size_of;
use std::sync::{Mutex, OnceLock};

use burn_dragon_core::{DragonModel, ModelState};

use crate::GenerationConfig;
use crate::config::ContextStrategyConfig;
use crate::summary_events::summary_event_mask_tensor;
use crate::tokenizer::Tokenizer;

type TokenChunkCallback<'a> = Option<&'a mut dyn FnMut(&[i64])>;

#[derive(Clone, Copy, Debug)]
pub enum ContextStrategy {
    Infinite,
    Sliding { window: usize },
}

#[derive(Clone, Copy, Debug)]
pub struct GenerationSettings {
    pub max_new_tokens: Option<usize>,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub strategy: ContextStrategy,
    pub stop_on_token: Option<i64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenerationProfileSnapshot {
    pub prefill_forward_ns: u128,
    pub token_forward_ns: u128,
    pub sample_host_transfer_ns: u128,
    pub sample_cpu_ns: u128,
    pub token_tensor_copy_ns: u128,
    pub chunk_flush_ns: u128,
    pub token_steps: u64,
    pub prefill_tokens: u64,
    pub host_sync_points: u64,
    pub chunk_flushes: u64,
    pub chunk_flushed_tokens: u64,
    pub host_to_device_copy_bytes: u128,
    pub device_to_host_copy_bytes: u128,
}

#[derive(Clone, Copy, Debug, Default)]
struct GenerationProfileState {
    prefill_forward_ns: u128,
    token_forward_ns: u128,
    sample_host_transfer_ns: u128,
    sample_cpu_ns: u128,
    token_tensor_copy_ns: u128,
    chunk_flush_ns: u128,
    token_steps: u64,
    prefill_tokens: u64,
    host_sync_points: u64,
    chunk_flushes: u64,
    chunk_flushed_tokens: u64,
    host_to_device_copy_bytes: u128,
    device_to_host_copy_bytes: u128,
}

static GENERATION_PROFILE: OnceLock<Mutex<GenerationProfileState>> = OnceLock::new();

fn generation_profile_enabled() -> bool {
    std::env::var_os("DragonModel_STAGE_PROFILE").is_some()
}

fn generation_profile_state() -> &'static Mutex<GenerationProfileState> {
    GENERATION_PROFILE.get_or_init(|| Mutex::new(GenerationProfileState::default()))
}

fn generation_profile_record(mutator: impl FnOnce(&mut GenerationProfileState)) {
    if let Ok(mut state) = generation_profile_state().lock() {
        mutator(&mut state);
    }
}

pub fn generation_profile_reset() {
    if let Ok(mut state) = generation_profile_state().lock() {
        *state = GenerationProfileState::default();
    }
}

pub fn generation_profile_snapshot() -> GenerationProfileSnapshot {
    if let Ok(state) = generation_profile_state().lock() {
        return GenerationProfileSnapshot {
            prefill_forward_ns: state.prefill_forward_ns,
            token_forward_ns: state.token_forward_ns,
            sample_host_transfer_ns: state.sample_host_transfer_ns,
            sample_cpu_ns: state.sample_cpu_ns,
            token_tensor_copy_ns: state.token_tensor_copy_ns,
            chunk_flush_ns: state.chunk_flush_ns,
            token_steps: state.token_steps,
            prefill_tokens: state.prefill_tokens,
            host_sync_points: state.host_sync_points,
            chunk_flushes: state.chunk_flushes,
            chunk_flushed_tokens: state.chunk_flushed_tokens,
            host_to_device_copy_bytes: state.host_to_device_copy_bytes,
            device_to_host_copy_bytes: state.device_to_host_copy_bytes,
        };
    }
    GenerationProfileSnapshot::default()
}

fn sample_from_logits_values_with_rng<R: Rng + ?Sized>(
    mut logits_values: Vec<f32>,
    top_k: Option<usize>,
    rng: &mut R,
) -> Result<i64> {
    let vocab = logits_values.len();
    if vocab == 0 {
        return Err(anyhow!("logits are empty"));
    }

    if let Some(k) = top_k
        && k > 0
        && k < vocab
    {
        let mut sorted = logits_values.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let threshold = sorted[k - 1];
        for value in logits_values.iter_mut() {
            if *value < threshold {
                *value = f32::NEG_INFINITY;
            }
        }
    }

    let max_logit = logits_values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits_values
        .iter()
        .map(|value| (value - max_logit).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum == 0.0 || sum.is_nan() {
        let uniform = 1.0 / vocab as f32;
        for p in probs.iter_mut() {
            *p = uniform;
        }
    } else {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    let dist = WeightedIndex::new(&probs).map_err(|err| anyhow!(err.to_string()))?;
    Ok(dist.sample(rng) as i64)
}

fn sample_from_logits_values(logits_values: Vec<f32>, top_k: Option<usize>) -> Result<i64> {
    let mut rng = thread_rng();
    sample_from_logits_values_with_rng(logits_values, top_k, &mut rng)
}

fn sample_argmax_token<B: Backend>(logits_temp: Tensor<B, 1>) -> Result<i64> {
    let values = logits_temp
        .argmax(0)
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|err| anyhow!("{err:?}"))?;
    values
        .first()
        .copied()
        .ok_or_else(|| anyhow!("argmax output is empty"))
}

fn sample_argmax_token_tensor<B: Backend>(logits_temp: Tensor<B, 1>) -> Tensor<B, 2, Int> {
    logits_temp.argmax(0).reshape([1, 1])
}

fn flush_pending_token_tensors<B: Backend>(
    pending: &mut Vec<Tensor<B, 2, Int>>,
    full_tokens: &mut Vec<i64>,
    on_chunk: &mut TokenChunkCallback<'_>,
    stop_on_token: Option<i64>,
) -> Result<bool> {
    if pending.is_empty() {
        return Ok(false);
    }

    let prof_enabled = generation_profile_enabled();
    let tokens = Tensor::cat(std::mem::take(pending), 1);
    let host_start = prof_enabled.then(Instant::now);
    let chunk = tokens
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|err| anyhow!("{err:?}"))?;
    let chunk_len = chunk.len();
    if let Some(start) = host_start {
        let elapsed = start.elapsed().as_nanos();
        generation_profile_record(|profile| {
            profile.sample_host_transfer_ns =
                profile.sample_host_transfer_ns.saturating_add(elapsed);
            profile.chunk_flush_ns = profile.chunk_flush_ns.saturating_add(elapsed);
            profile.host_sync_points = profile.host_sync_points.saturating_add(1);
            profile.chunk_flushes = profile.chunk_flushes.saturating_add(1);
            profile.chunk_flushed_tokens = profile
                .chunk_flushed_tokens
                .saturating_add(chunk_len as u64);
            profile.device_to_host_copy_bytes = profile
                .device_to_host_copy_bytes
                .saturating_add((chunk_len.saturating_mul(size_of::<i64>())) as u128);
        });
    }

    let visible_len = stop_on_token
        .and_then(|stop| {
            chunk
                .iter()
                .position(|&token| token == stop)
                .map(|idx| idx + 1)
        })
        .unwrap_or(chunk_len);
    let visible_chunk = &chunk[..visible_len];

    if let Some(callback) = on_chunk.as_mut() {
        (**callback)(visible_chunk);
    }
    full_tokens.extend_from_slice(visible_chunk);
    Ok(visible_len < chunk_len)
}

pub fn prefill_state<B: Backend>(
    model: &DragonModel<B>,
    prompt_tokens: &[i64],
    device: &B::Device,
) -> Result<(ModelState<B>, Tensor<B, 1>)> {
    let prompt_len = prompt_tokens.len();
    if prompt_len == 0 {
        return Err(anyhow!("prompt must contain at least one token"));
    }

    let prof_enabled = generation_profile_enabled();
    if prof_enabled {
        let prompt_bytes = (prompt_len.saturating_mul(size_of::<i64>())) as u128;
        generation_profile_record(|profile| {
            profile.prefill_tokens = profile.prefill_tokens.saturating_add(prompt_len as u64);
            profile.host_to_device_copy_bytes = profile
                .host_to_device_copy_bytes
                .saturating_add(prompt_bytes);
        });
    }

    let prompt_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(prompt_tokens.to_vec(), [1, prompt_len]),
        device,
    );

    let mut state = model.init_state();
    let prefill_start = prof_enabled.then(Instant::now);
    let logits = match summary_event_mask_tensor::<B>(
        prompt_tokens,
        1,
        prompt_len,
        model.summary_memory_write_trigger_token_ids(),
        device,
    ) {
        Some(mask) => {
            model.forward_with_state_and_summary_event_mask(prompt_tensor, mask, &mut state)
        }
        None => model.forward_with_state(prompt_tensor, &mut state),
    };
    if let Some(start) = prefill_start {
        let elapsed = start.elapsed().as_nanos();
        generation_profile_record(|profile| {
            profile.prefill_forward_ns = profile.prefill_forward_ns.saturating_add(elapsed);
        });
    }
    let [_, time, vocab] = logits.shape().dims::<3>();
    if time != prompt_len {
        return Err(anyhow!(
            "prefill produced mismatched length: expected {prompt_len}, got {time}"
        ));
    }

    let last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);

    #[cfg(feature = "viz")]
    state.clear_viz();

    Ok((state, last_logits))
}

/// Generate exact greedy continuations for a ragged batch of prompts.
///
/// Every row advances at the same absolute recurrent position. A row still inside its prompt
/// consumes its next ground-truth token while a completed prompt consumes its device-side argmax.
/// This avoids padding-state corruption without requiring per-row positions. Generated tensors
/// remain on the device for `device_buffer_tokens` steps before a batched host read resolves stop
/// tokens and budgets; speculative tokens after a row's stop are discarded and cannot affect any
/// other row because recurrent state is batch-separable.
pub fn generate_greedy_batch_ragged<B: Backend>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    max_new_tokens: &[usize],
    device: &B::Device,
    strategy: ContextStrategy,
    stop_on_token: Option<i64>,
    device_buffer_tokens: usize,
) -> Result<Vec<Vec<i64>>> {
    if prompt_tokens.len() != max_new_tokens.len() {
        return Err(anyhow!(
            "batched greedy generation requires one token budget per prompt"
        ));
    }
    if prompt_tokens.is_empty() {
        return Ok(Vec::new());
    }
    if prompt_tokens.iter().any(Vec::is_empty) {
        return Err(anyhow!(
            "batched greedy generation requires non-empty prompts"
        ));
    }

    let mut generated = (0..prompt_tokens.len())
        .map(|_| Vec::new())
        .collect::<Vec<Vec<i64>>>();
    let mut active_rows = max_new_tokens
        .iter()
        .enumerate()
        .filter_map(|(row, budget)| (*budget > 0).then_some(row))
        .collect::<Vec<_>>();
    if active_rows.is_empty() {
        return Ok(generated);
    }

    let mut state = model.init_state();
    let mut last_logits = None;
    let prof_enabled = generation_profile_enabled();
    let device_buffer_tokens = device_buffer_tokens.max(1);

    while !active_rows.is_empty() {
        if active_rows
            .iter()
            .all(|row| state.position < prompt_tokens[*row].len())
        {
            let next_boundary = active_rows
                .iter()
                .map(|row| prompt_tokens[*row].len())
                .min()
                .expect("active prompt rows");
            let segment_len = next_boundary.saturating_sub(state.position);
            let active_count = active_rows.len();
            let values = active_rows
                .iter()
                .flat_map(|row| {
                    prompt_tokens[*row][state.position..next_boundary]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            let input = Tensor::<B, 2, Int>::from_data(
                TensorData::new(values.clone(), [active_count, segment_len]),
                device,
            );
            if prof_enabled {
                generation_profile_record(|profile| {
                    profile.prefill_tokens = profile
                        .prefill_tokens
                        .saturating_add(active_count.saturating_mul(segment_len) as u64);
                    profile.host_to_device_copy_bytes = profile
                        .host_to_device_copy_bytes
                        .saturating_add((values.len().saturating_mul(size_of::<i64>())) as u128);
                });
            }
            let prefill_start = prof_enabled.then(Instant::now);
            let logits = match summary_event_mask_tensor::<B>(
                &values,
                active_count,
                segment_len,
                model.summary_memory_write_trigger_token_ids(),
                device,
            ) {
                Some(mask) => {
                    model.forward_with_state_and_summary_event_mask(input, mask, &mut state)
                }
                None => model.forward_with_state(input, &mut state),
            };
            if let Some(start) = prefill_start {
                let elapsed = start.elapsed().as_nanos();
                generation_profile_record(|profile| {
                    profile.prefill_forward_ns = profile.prefill_forward_ns.saturating_add(elapsed);
                });
            }
            let [_, time, vocab] = logits.shape().dims::<3>();
            last_logits = Some(
                logits
                    .slice([0..active_count, (time - 1)..time, 0..vocab])
                    .reshape([active_count, vocab]),
            );
            continue;
        }

        let active_count = active_rows.len();
        let mut pending = Vec::with_capacity(device_buffer_tokens);
        let mut generated_masks = Vec::with_capacity(device_buffer_tokens);
        for _ in 0..device_buffer_tokens {
            let position = state.position;
            let prompt_mask = active_rows
                .iter()
                .map(|row| position < prompt_tokens[*row].len())
                .collect::<Vec<_>>();
            let prompt_values = active_rows
                .iter()
                .map(|row| {
                    prompt_tokens[*row]
                        .get(position)
                        .copied()
                        .unwrap_or_default()
                })
                .collect::<Vec<_>>();
            let argmax = last_logits
                .as_ref()
                .expect("ragged generation must prefill before decoding")
                .clone()
                .argmax(1);
            let prompt_mask_tensor = Tensor::<B, 2, Bool>::from_data(
                TensorData::new(prompt_mask.clone(), [active_count, 1]),
                device,
            );
            let prompt_tensor = Tensor::<B, 2, Int>::from_data(
                TensorData::new(prompt_values.clone(), [active_count, 1]),
                device,
            );
            let next_tokens = argmax.mask_where(prompt_mask_tensor, prompt_tensor);
            let forward_start = prof_enabled.then(Instant::now);
            let logits = if let Some(trigger_ids) = model.summary_memory_write_trigger_token_ids() {
                let summary_mask = prompt_mask
                    .iter()
                    .zip(&prompt_values)
                    .map(|(is_prompt, token)| {
                        *is_prompt && *token >= 0 && trigger_ids.contains(&(*token as u32))
                    })
                    .map(i64::from)
                    .collect::<Vec<_>>();
                let summary_mask = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(summary_mask, [active_count, 1]),
                    device,
                );
                model.forward_with_state_and_summary_event_mask(
                    next_tokens.clone(),
                    summary_mask,
                    &mut state,
                )
            } else {
                model.forward_with_state(next_tokens.clone(), &mut state)
            };
            if let Some(start) = forward_start {
                let elapsed = start.elapsed().as_nanos();
                generation_profile_record(|profile| {
                    profile.token_forward_ns = profile.token_forward_ns.saturating_add(elapsed);
                    profile.token_steps = profile.token_steps.saturating_add(1);
                });
            }
            let [_, time, vocab] = logits.shape().dims::<3>();
            last_logits = Some(
                logits
                    .slice([0..active_count, (time - 1)..time, 0..vocab])
                    .reshape([active_count, vocab]),
            );
            pending.push(next_tokens);
            generated_masks.push(
                prompt_mask
                    .into_iter()
                    .map(|is_prompt| !is_prompt)
                    .collect::<Vec<_>>(),
            );
            if let ContextStrategy::Sliding { window } = strategy
                && window > 0
                && state.position > window
            {
                state.trim(window);
            }
        }

        let host_start = prof_enabled.then(Instant::now);
        let buffered_tokens = Tensor::cat(pending, 1)
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .map_err(|err| anyhow!("{err:?}"))?;
        if let Some(start) = host_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_host_transfer_ns =
                    profile.sample_host_transfer_ns.saturating_add(elapsed);
                profile.chunk_flush_ns = profile.chunk_flush_ns.saturating_add(elapsed);
                profile.host_sync_points = profile.host_sync_points.saturating_add(1);
                profile.chunk_flushes = profile.chunk_flushes.saturating_add(1);
                profile.chunk_flushed_tokens = profile
                    .chunk_flushed_tokens
                    .saturating_add(active_count.saturating_mul(device_buffer_tokens) as u64);
                profile.device_to_host_copy_bytes =
                    profile.device_to_host_copy_bytes.saturating_add(
                        (active_count
                            .saturating_mul(device_buffer_tokens)
                            .saturating_mul(size_of::<i64>())) as u128,
                    );
            });
        }

        let mut surviving_batch_rows = Vec::new();
        let mut surviving_original_rows = Vec::new();
        for (batch_row, original_row) in active_rows.iter().copied().enumerate() {
            let remaining =
                max_new_tokens[original_row].saturating_sub(generated[original_row].len());
            let row_start = batch_row.saturating_mul(device_buffer_tokens);
            let row_tokens = &buffered_tokens[row_start..row_start + device_buffer_tokens];
            let mut accepted = 0usize;
            let mut stopped = false;
            for (step, token) in row_tokens.iter().copied().enumerate() {
                if !generated_masks[step][batch_row] || accepted >= remaining {
                    continue;
                }
                generated[original_row].push(token);
                accepted = accepted.saturating_add(1);
                if stop_on_token == Some(token) {
                    stopped = true;
                    break;
                }
            }
            if !stopped && generated[original_row].len() < max_new_tokens[original_row] {
                surviving_batch_rows.push(batch_row as i64);
                surviving_original_rows.push(original_row);
            }
        }

        if surviving_batch_rows.is_empty() {
            break;
        }
        if surviving_batch_rows.len() < active_rows.len() {
            let indices = Tensor::<B, 1, Int>::from_data(
                TensorData::new(surviving_batch_rows, [surviving_original_rows.len()]),
                device,
            );
            state = state.select_batch(indices.clone());
            last_logits = last_logits.map(|logits| logits.select(0, indices));
        }
        active_rows = surviving_original_rows;
    }

    Ok(generated)
}

/// Generate greedy continuations for prompts that share one recurrent position.
///
/// Dragon's recurrent state currently stores one position for the whole batch, so callers must
/// group prompts by exact token length before using this path. Generated token tensors remain on
/// the device for `device_buffer_tokens` steps before one batched host read determines stop-token
/// and per-row budget completion. Extra speculative steps for a row that stopped inside the buffer
/// are discarded and cannot affect any other row because recurrent state is batch-separable.
pub fn generate_greedy_batch_equal_position<B: Backend>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    max_new_tokens: &[usize],
    device: &B::Device,
    strategy: ContextStrategy,
    stop_on_token: Option<i64>,
    device_buffer_tokens: usize,
) -> Result<Vec<Vec<i64>>> {
    if prompt_tokens.len() != max_new_tokens.len() {
        return Err(anyhow!(
            "batched greedy generation requires one token budget per prompt"
        ));
    }
    if prompt_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let prompt_len = prompt_tokens[0].len();
    if prompt_len == 0
        || prompt_tokens
            .iter()
            .any(|prompt| prompt.len() != prompt_len)
    {
        return Err(anyhow!(
            "batched greedy generation requires non-empty prompts with equal token lengths"
        ));
    }

    let mut generated = (0..prompt_tokens.len())
        .map(|_| Vec::new())
        .collect::<Vec<Vec<i64>>>();
    let mut active_rows = max_new_tokens
        .iter()
        .enumerate()
        .filter_map(|(row, budget)| (*budget > 0).then_some(row))
        .collect::<Vec<_>>();
    if active_rows.is_empty() {
        return Ok(generated);
    }

    let active_prompt_values = active_rows
        .iter()
        .flat_map(|row| prompt_tokens[*row].iter().copied())
        .collect::<Vec<_>>();
    let mut state = model.init_state();
    let active_count = active_rows.len();
    let prompt_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(active_prompt_values.clone(), [active_count, prompt_len]),
        device,
    );
    let prof_enabled = generation_profile_enabled();
    if prof_enabled {
        generation_profile_record(|profile| {
            profile.prefill_tokens = profile
                .prefill_tokens
                .saturating_add((active_count.saturating_mul(prompt_len)) as u64);
            profile.host_to_device_copy_bytes = profile.host_to_device_copy_bytes.saturating_add(
                (active_count
                    .saturating_mul(prompt_len)
                    .saturating_mul(size_of::<i64>())) as u128,
            );
        });
    }
    let prefill_start = prof_enabled.then(Instant::now);
    let logits = match summary_event_mask_tensor::<B>(
        &active_prompt_values,
        active_count,
        prompt_len,
        model.summary_memory_write_trigger_token_ids(),
        device,
    ) {
        Some(mask) => {
            model.forward_with_state_and_summary_event_mask(prompt_tensor, mask, &mut state)
        }
        None => model.forward_with_state(prompt_tensor, &mut state),
    };
    if let Some(start) = prefill_start {
        let elapsed = start.elapsed().as_nanos();
        generation_profile_record(|profile| {
            profile.prefill_forward_ns = profile.prefill_forward_ns.saturating_add(elapsed);
        });
    }
    let [_, time, vocab] = logits.shape().dims::<3>();
    let mut last_logits = logits
        .slice([0..active_count, (time - 1)..time, 0..vocab])
        .reshape([active_count, vocab]);
    let device_buffer_tokens = device_buffer_tokens.max(1);

    if let ContextStrategy::Sliding { window } = strategy
        && window > 0
        && state.position > window
    {
        state.trim(window);
    }

    while !active_rows.is_empty() {
        let buffered_steps = active_rows
            .iter()
            .map(|row| max_new_tokens[*row].saturating_sub(generated[*row].len()))
            .max()
            .unwrap_or_default()
            .min(device_buffer_tokens);
        if buffered_steps == 0 {
            break;
        }

        let active_count = active_rows.len();
        let mut pending = Vec::with_capacity(buffered_steps);
        for _ in 0..buffered_steps {
            let next_tokens = last_logits.clone().argmax(1);
            let forward_start = prof_enabled.then(Instant::now);
            let logits = model.forward_with_state(next_tokens.clone(), &mut state);
            if let Some(start) = forward_start {
                let elapsed = start.elapsed().as_nanos();
                generation_profile_record(|profile| {
                    profile.token_forward_ns = profile.token_forward_ns.saturating_add(elapsed);
                    profile.token_steps = profile.token_steps.saturating_add(1);
                });
            }
            let [_, time, vocab] = logits.shape().dims::<3>();
            last_logits = logits
                .slice([0..active_count, (time - 1)..time, 0..vocab])
                .reshape([active_count, vocab]);
            pending.push(next_tokens);
            if let ContextStrategy::Sliding { window } = strategy
                && window > 0
                && state.position > window
            {
                state.trim(window);
            }
        }

        let host_start = prof_enabled.then(Instant::now);
        let buffered_tokens = Tensor::cat(pending, 1)
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .map_err(|err| anyhow!("{err:?}"))?;
        if let Some(start) = host_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_host_transfer_ns =
                    profile.sample_host_transfer_ns.saturating_add(elapsed);
                profile.chunk_flush_ns = profile.chunk_flush_ns.saturating_add(elapsed);
                profile.host_sync_points = profile.host_sync_points.saturating_add(1);
                profile.chunk_flushes = profile.chunk_flushes.saturating_add(1);
                profile.chunk_flushed_tokens = profile
                    .chunk_flushed_tokens
                    .saturating_add(active_count.saturating_mul(buffered_steps) as u64);
                profile.device_to_host_copy_bytes =
                    profile.device_to_host_copy_bytes.saturating_add(
                        (active_count
                            .saturating_mul(buffered_steps)
                            .saturating_mul(size_of::<i64>())) as u128,
                    );
            });
        }

        let mut surviving_batch_rows = Vec::new();
        let mut surviving_original_rows = Vec::new();
        for (batch_row, original_row) in active_rows.iter().copied().enumerate() {
            let remaining =
                max_new_tokens[original_row].saturating_sub(generated[original_row].len());
            let row_start = batch_row.saturating_mul(buffered_steps);
            let row_tokens = &buffered_tokens[row_start..row_start + buffered_steps];
            let mut stopped = false;
            for token in row_tokens.iter().copied().take(remaining) {
                generated[original_row].push(token);
                if stop_on_token == Some(token) {
                    stopped = true;
                    break;
                }
            }
            if !stopped && generated[original_row].len() < max_new_tokens[original_row] {
                surviving_batch_rows.push(batch_row as i64);
                surviving_original_rows.push(original_row);
            }
        }

        if surviving_batch_rows.is_empty() {
            break;
        }
        if surviving_batch_rows.len() < active_rows.len() {
            let indices = Tensor::<B, 1, Int>::from_data(
                TensorData::new(surviving_batch_rows, [surviving_original_rows.len()]),
                device,
            );
            state = state.select_batch(indices.clone());
            last_logits = last_logits.select(0, indices);
        }
        active_rows = surviving_original_rows;
    }

    Ok(generated)
}

pub fn sample_next_token<B: Backend>(
    model: &DragonModel<B>,
    state: &mut ModelState<B>,
    last_logits: Tensor<B, 1>,
    temperature: f32,
    top_k: Option<usize>,
    device: &B::Device,
) -> Result<(i64, Tensor<B, 1>)> {
    sample_next_token_with_rng(model, state, last_logits, temperature, top_k, device, None)
}

fn sample_next_token_with_rng<B: Backend>(
    model: &DragonModel<B>,
    state: &mut ModelState<B>,
    last_logits: Tensor<B, 1>,
    temperature: f32,
    top_k: Option<usize>,
    device: &B::Device,
    rng: Option<&mut StdRng>,
) -> Result<(i64, Tensor<B, 1>)> {
    let prof_enabled = generation_profile_enabled();
    let logits_temp = last_logits.clone().div_scalar(temperature);
    let next = if top_k == Some(1) {
        let host_start = prof_enabled.then(Instant::now);
        let token = sample_argmax_token(logits_temp)?;
        if let Some(start) = host_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_host_transfer_ns =
                    profile.sample_host_transfer_ns.saturating_add(elapsed);
                profile.host_sync_points = profile.host_sync_points.saturating_add(1);
                profile.device_to_host_copy_bytes = profile
                    .device_to_host_copy_bytes
                    .saturating_add(size_of::<i64>() as u128);
            });
        }
        token
    } else {
        let host_start = prof_enabled.then(Instant::now);
        let logits_values = logits_temp
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|err| anyhow!("{err:?}"))?;
        if let Some(start) = host_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_host_transfer_ns =
                    profile.sample_host_transfer_ns.saturating_add(elapsed);
                profile.host_sync_points = profile.host_sync_points.saturating_add(1);
                profile.device_to_host_copy_bytes = profile
                    .device_to_host_copy_bytes
                    .saturating_add((logits_values.len().saturating_mul(size_of::<f32>())) as u128);
            });
        }
        let sample_start = prof_enabled.then(Instant::now);
        let token = if let Some(rng) = rng {
            sample_from_logits_values_with_rng(logits_values, top_k, rng)?
        } else {
            sample_from_logits_values(logits_values, top_k)?
        };
        if let Some(start) = sample_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_cpu_ns = profile.sample_cpu_ns.saturating_add(elapsed);
            });
        }
        token
    };

    let tensor_copy_start = prof_enabled.then(Instant::now);
    let next_tensor = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next], [1, 1]), device);
    if let Some(start) = tensor_copy_start {
        let elapsed = start.elapsed().as_nanos();
        generation_profile_record(|profile| {
            profile.token_tensor_copy_ns = profile.token_tensor_copy_ns.saturating_add(elapsed);
            profile.host_to_device_copy_bytes = profile
                .host_to_device_copy_bytes
                .saturating_add(size_of::<i64>() as u128);
        });
    }

    let forward_start = prof_enabled.then(Instant::now);
    let logits = model.forward_with_state(next_tensor, state);
    if let Some(start) = forward_start {
        let elapsed = start.elapsed().as_nanos();
        generation_profile_record(|profile| {
            profile.token_forward_ns = profile.token_forward_ns.saturating_add(elapsed);
            profile.token_steps = profile.token_steps.saturating_add(1);
        });
    }
    let [_, time, vocab] = logits.shape().dims::<3>();
    let new_last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);

    Ok((next, new_last_logits))
}

#[cfg(feature = "web")]
pub async fn sample_next_token_async<B: Backend>(
    model: &DragonModel<B>,
    state: &mut ModelState<B>,
    last_logits: Tensor<B, 1>,
    temperature: f32,
    top_k: Option<usize>,
    device: &B::Device,
) -> Result<(i64, Tensor<B, 1>)> {
    let prof_enabled = generation_profile_enabled();
    let logits_temp = last_logits.clone().div_scalar(temperature);
    let next = if top_k == Some(1) {
        let host_start = prof_enabled.then(Instant::now);
        let values = logits_temp
            .argmax(0)
            .to_data_async()
            .await
            .map_err(|err| anyhow!("{err:?}"))?
            .convert::<i64>()
            .into_vec::<i64>()
            .map_err(|err| anyhow!("{err:?}"))?;
        if let Some(start) = host_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_host_transfer_ns =
                    profile.sample_host_transfer_ns.saturating_add(elapsed);
                profile.host_sync_points = profile.host_sync_points.saturating_add(1);
                profile.device_to_host_copy_bytes = profile
                    .device_to_host_copy_bytes
                    .saturating_add(size_of::<i64>() as u128);
            });
        }
        values
            .first()
            .copied()
            .ok_or_else(|| anyhow!("argmax output is empty"))?
    } else {
        let host_start = prof_enabled.then(Instant::now);
        let logits_values = logits_temp
            .to_data_async()
            .await
            .map_err(|err| anyhow!("{err:?}"))?
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|err| anyhow!("{err:?}"))?;
        if let Some(start) = host_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_host_transfer_ns =
                    profile.sample_host_transfer_ns.saturating_add(elapsed);
                profile.host_sync_points = profile.host_sync_points.saturating_add(1);
                profile.device_to_host_copy_bytes = profile
                    .device_to_host_copy_bytes
                    .saturating_add((logits_values.len().saturating_mul(size_of::<f32>())) as u128);
            });
        }
        let sample_start = prof_enabled.then(Instant::now);
        let token = sample_from_logits_values(logits_values, top_k)?;
        if let Some(start) = sample_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.sample_cpu_ns = profile.sample_cpu_ns.saturating_add(elapsed);
            });
        }
        token
    };

    let tensor_copy_start = prof_enabled.then(Instant::now);
    let next_tensor = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next], [1, 1]), device);
    if let Some(start) = tensor_copy_start {
        let elapsed = start.elapsed().as_nanos();
        generation_profile_record(|profile| {
            profile.token_tensor_copy_ns = profile.token_tensor_copy_ns.saturating_add(elapsed);
            profile.host_to_device_copy_bytes = profile
                .host_to_device_copy_bytes
                .saturating_add(size_of::<i64>() as u128);
        });
    }

    let forward_start = prof_enabled.then(Instant::now);
    let logits = model.forward_with_state(next_tensor, state);
    if let Some(start) = forward_start {
        let elapsed = start.elapsed().as_nanos();
        generation_profile_record(|profile| {
            profile.token_forward_ns = profile.token_forward_ns.saturating_add(elapsed);
            profile.token_steps = profile.token_steps.saturating_add(1);
        });
    }
    let [_, time, vocab] = logits.shape().dims::<3>();
    let new_last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);

    Ok((next, new_last_logits))
}

pub fn generate_tokens<B: Backend>(
    model: &DragonModel<B>,
    prompt_tokens: Vec<i64>,
    device: &B::Device,
    settings: GenerationSettings,
    mut on_token: Option<&mut dyn FnMut(i64)>,
) -> Result<Vec<i64>> {
    generate_tokens_with_optional_seed(model, prompt_tokens, device, settings, None, &mut on_token)
}

/// Generate a greedy continuation through one predictive-context subnetwork.
///
/// Routed predictive coding has a distinct recurrent state and fixed neuron/activity masks for
/// each context. Keeping this path explicit prevents verifier probes from accidentally evaluating
/// the dense model while training updates only the selected subnetwork.
#[cfg(any(feature = "train", test))]
pub(crate) fn generate_greedy_tokens_with_subnetwork_masks<B: Backend>(
    model: &DragonModel<B>,
    prompt_tokens: Vec<i64>,
    device: &B::Device,
    settings: GenerationSettings,
    neuron_mask: Tensor<B, 4>,
    activity_mask: Tensor<B, 4>,
) -> Result<Vec<i64>>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    if prompt_tokens.is_empty() {
        return Err(anyhow!("prompt must contain at least one token"));
    }
    if settings.top_k != Some(1) {
        return Err(anyhow!(
            "predictive-context verifier generation currently requires greedy top_k=1"
        ));
    }

    let prompt_len = prompt_tokens.len();
    let prompt_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(prompt_tokens.clone(), [1, prompt_len]),
        device,
    );
    let prof_enabled = generation_profile_enabled();
    if prof_enabled {
        generation_profile_record(|profile| {
            profile.prefill_tokens = profile.prefill_tokens.saturating_add(prompt_len as u64);
            profile.host_to_device_copy_bytes = profile
                .host_to_device_copy_bytes
                .saturating_add((prompt_len.saturating_mul(size_of::<i64>())) as u128);
        });
    }
    let mut state = model.init_state();
    let prefill_start = prof_enabled.then(Instant::now);
    let logits = model
        .predictive_coding_forward_with_subnetwork_masks_and_state(
            prompt_tensor,
            neuron_mask.clone(),
            activity_mask.clone(),
            &mut state,
        )
        .map_err(anyhow::Error::msg)?;
    if let Some(start) = prefill_start {
        generation_profile_record(|profile| {
            profile.prefill_forward_ns = profile
                .prefill_forward_ns
                .saturating_add(start.elapsed().as_nanos());
        });
    }
    let [_, time, vocab] = logits.shape().dims::<3>();
    let mut last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);
    let mut full_tokens = prompt_tokens;
    let mut generated = 0usize;

    if let ContextStrategy::Sliding { window } = settings.strategy
        && window > 0
        && state.position > window
    {
        state.trim(window);
    }

    while settings.max_new_tokens.is_none_or(|max| generated < max) {
        let host_start = prof_enabled.then(Instant::now);
        let next = sample_argmax_token(last_logits.div_scalar(settings.temperature))?;
        if let Some(start) = host_start {
            generation_profile_record(|profile| {
                profile.sample_host_transfer_ns = profile
                    .sample_host_transfer_ns
                    .saturating_add(start.elapsed().as_nanos());
                profile.host_sync_points = profile.host_sync_points.saturating_add(1);
                profile.device_to_host_copy_bytes = profile
                    .device_to_host_copy_bytes
                    .saturating_add(size_of::<i64>() as u128);
            });
        }
        full_tokens.push(next);
        generated = generated.saturating_add(1);
        if settings.stop_on_token == Some(next) {
            break;
        }

        let copy_start = prof_enabled.then(Instant::now);
        let next_tensor =
            Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next], [1, 1]), device);
        if let Some(start) = copy_start {
            generation_profile_record(|profile| {
                profile.token_tensor_copy_ns = profile
                    .token_tensor_copy_ns
                    .saturating_add(start.elapsed().as_nanos());
                profile.host_to_device_copy_bytes = profile
                    .host_to_device_copy_bytes
                    .saturating_add(size_of::<i64>() as u128);
            });
        }
        let forward_start = prof_enabled.then(Instant::now);
        let logits = model
            .predictive_coding_forward_with_subnetwork_masks_and_state(
                next_tensor,
                neuron_mask.clone(),
                activity_mask.clone(),
                &mut state,
            )
            .map_err(anyhow::Error::msg)?;
        if let Some(start) = forward_start {
            generation_profile_record(|profile| {
                profile.token_forward_ns = profile
                    .token_forward_ns
                    .saturating_add(start.elapsed().as_nanos());
                profile.token_steps = profile.token_steps.saturating_add(1);
            });
        }
        let [_, time, vocab] = logits.shape().dims::<3>();
        last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);

        if let ContextStrategy::Sliding { window } = settings.strategy
            && window > 0
            && state.position > window
        {
            state.trim(window);
        }
    }

    Ok(full_tokens)
}

pub fn generate_tokens_seeded<B: Backend>(
    model: &DragonModel<B>,
    prompt_tokens: Vec<i64>,
    device: &B::Device,
    settings: GenerationSettings,
    seed: u64,
    mut on_token: Option<&mut dyn FnMut(i64)>,
) -> Result<Vec<i64>> {
    generate_tokens_with_optional_seed(
        model,
        prompt_tokens,
        device,
        settings,
        Some(seed),
        &mut on_token,
    )
}

fn generate_tokens_with_optional_seed<B: Backend>(
    model: &DragonModel<B>,
    prompt_tokens: Vec<i64>,
    device: &B::Device,
    settings: GenerationSettings,
    seed: Option<u64>,
    on_token: &mut Option<&mut dyn FnMut(i64)>,
) -> Result<Vec<i64>> {
    let GenerationSettings {
        max_new_tokens,
        temperature,
        top_k,
        strategy,
        stop_on_token,
    } = settings;

    let mut full_tokens = prompt_tokens;
    let (mut state, mut last_logits) = prefill_state(model, &full_tokens, device)?;
    let mut generated = 0usize;
    let mut rng = seed.map(StdRng::seed_from_u64);

    if let ContextStrategy::Sliding { window } = strategy
        && window > 0
        && state.position > window
    {
        state.trim(window);
    }

    while max_new_tokens.is_none_or(|max| generated < max) {
        let (next, logits) = sample_next_token_with_rng(
            model,
            &mut state,
            last_logits,
            temperature,
            top_k,
            device,
            rng.as_mut(),
        )?;
        full_tokens.push(next);
        last_logits = logits;
        generated = generated.saturating_add(1);

        if let Some(callback) = on_token.as_deref_mut() {
            callback(next);
        }
        if stop_on_token == Some(next) {
            break;
        }

        if let ContextStrategy::Sliding { window } = strategy
            && window > 0
            && state.position > window
        {
            state.trim(window);
        }
    }

    Ok(full_tokens)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_tokens_chunked<B: Backend>(
    model: &DragonModel<B>,
    prompt_tokens: Vec<i64>,
    device: &B::Device,
    settings: GenerationSettings,
    chunk_tokens: usize,
    device_buffer_tokens: usize,
    stop_on_token: Option<i64>,
    mut on_chunk: TokenChunkCallback<'_>,
) -> Result<Vec<i64>> {
    let GenerationSettings {
        max_new_tokens,
        temperature,
        top_k,
        strategy,
        stop_on_token: settings_stop_on_token,
    } = settings;
    let stop_on_token = stop_on_token.or(settings_stop_on_token);

    let chunk_tokens = chunk_tokens.max(1);
    let device_buffer_tokens = device_buffer_tokens.max(chunk_tokens);

    if top_k != Some(1) {
        let prompt_len = prompt_tokens.len();
        let full_tokens = generate_tokens(
            model,
            prompt_tokens,
            device,
            GenerationSettings {
                max_new_tokens,
                temperature,
                top_k,
                strategy,
                stop_on_token,
            },
            None,
        )?;
        if let Some(callback) = on_chunk.as_mut() {
            (**callback)(&full_tokens[prompt_len..]);
        }
        return Ok(full_tokens);
    }

    let mut full_tokens = prompt_tokens;
    let (mut state, mut last_logits) = prefill_state(model, &full_tokens, device)?;
    let mut generated = 0usize;
    let prof_enabled = generation_profile_enabled();
    let mut pending: Vec<Tensor<B, 2, Int>> =
        Vec::with_capacity(chunk_tokens.min(device_buffer_tokens));

    if let ContextStrategy::Sliding { window } = strategy
        && window > 0
        && state.position > window
    {
        state.trim(window);
    }

    while max_new_tokens.is_none_or(|max| generated < max) {
        let logits_temp = last_logits.clone().div_scalar(temperature);
        let next_tensor = sample_argmax_token_tensor(logits_temp);

        let forward_start = prof_enabled.then(Instant::now);
        let logits = model.forward_with_state(next_tensor.clone(), &mut state);
        if let Some(start) = forward_start {
            let elapsed = start.elapsed().as_nanos();
            generation_profile_record(|profile| {
                profile.token_forward_ns = profile.token_forward_ns.saturating_add(elapsed);
                profile.token_steps = profile.token_steps.saturating_add(1);
            });
        }

        let [_, time, vocab] = logits.shape().dims::<3>();
        last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);

        pending.push(next_tensor);
        generated = generated.saturating_add(1);

        if pending.len() >= chunk_tokens || pending.len() >= device_buffer_tokens {
            let stop_reached = flush_pending_token_tensors(
                &mut pending,
                &mut full_tokens,
                &mut on_chunk,
                stop_on_token,
            )?;
            if stop_reached {
                break;
            }
        }

        if let ContextStrategy::Sliding { window } = strategy
            && window > 0
            && state.position > window
        {
            state.trim(window);
        }
    }

    let _ =
        flush_pending_token_tensors(&mut pending, &mut full_tokens, &mut on_chunk, stop_on_token)?;
    Ok(full_tokens)
}

pub fn generate_text<B: Backend>(
    model: &DragonModel<B>,
    tokenizer: &dyn Tokenizer,
    device: &B::Device,
    block_size: usize,
    generation: &GenerationConfig,
) -> Result<String> {
    let strategy = resolve_context_strategy(&generation.context_strategy, block_size);
    let mut prompt_ids = tokenizer.encode(&generation.prompt, false, false);
    if let ContextStrategy::Sliding { window } = strategy
        && prompt_ids.len() > window
    {
        prompt_ids = prompt_ids[prompt_ids.len() - window..].to_vec();
    }

    let prompt_tokens: Vec<i64> = prompt_ids.iter().map(|&id| id as i64).collect();
    let max_new_tokens = normalize_max_tokens(generation.max_tokens);
    let settings = GenerationSettings {
        max_new_tokens,
        temperature: generation.temperature,
        top_k: generation.top_k,
        strategy,
        stop_on_token: None,
    };
    let tokens_all = generate_tokens(model, prompt_tokens, device, settings, None)?;

    let decoded_ids: Vec<u32> = tokens_all
        .iter()
        .filter_map(|&tok| (tok >= 0).then_some(tok as u32))
        .collect();

    Ok(tokenizer.decode(&decoded_ids))
}

fn normalize_max_tokens(max_tokens: Option<i64>) -> Option<usize> {
    match max_tokens {
        Some(value) if value >= 0 => Some(value as usize),
        _ => None,
    }
}

pub fn resolve_context_strategy(
    config: &ContextStrategyConfig,
    default_window: usize,
) -> ContextStrategy {
    match config {
        ContextStrategyConfig::Infinite => ContextStrategy::Infinite,
        ContextStrategyConfig::Sliding { window } => {
            let win = if *window == 0 {
                default_window.max(1)
            } else {
                *window
            };
            ContextStrategy::Sliding { window: win }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn test_model(device: &burn::tensor::Device<TestBackend>) -> DragonModel<TestBackend> {
        TestBackend::seed(device, 9_171);
        let mut config = burn_dragon_core::DragonConfig {
            n_layer: 2,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            ..Default::default()
        };
        config.sequence_kernel.executor =
            burn_dragon_core::SequenceTrainingExecutor::DenseScoreShortContext;
        config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
        DragonModel::new(config, device)
    }

    fn serial_greedy(
        model: &DragonModel<TestBackend>,
        prompts: &[Vec<i64>],
        budgets: &[usize],
        stop_on_token: Option<i64>,
        device: &burn::tensor::Device<TestBackend>,
    ) -> Vec<Vec<i64>> {
        prompts
            .iter()
            .zip(budgets)
            .map(|(prompt, budget)| {
                let tokens = generate_tokens(
                    model,
                    prompt.clone(),
                    device,
                    GenerationSettings {
                        max_new_tokens: Some(*budget),
                        temperature: 1.0,
                        top_k: Some(1),
                        strategy: ContextStrategy::Infinite,
                        stop_on_token,
                    },
                    None,
                )
                .expect("serial greedy generation");
                tokens[prompt.len()..].to_vec()
            })
            .collect()
    }

    #[test]
    fn batched_greedy_matches_serial_with_ragged_budgets() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = test_model(&device);
        let prompts = vec![vec![1, 2, 3, 4], vec![4, 3, 2, 1], vec![2, 5, 8, 11]];
        let budgets = vec![0, 3, 7];
        let serial = serial_greedy(&model, &prompts, &budgets, None, &device);
        let batched = generate_greedy_batch_equal_position(
            &model,
            &prompts,
            &budgets,
            &device,
            ContextStrategy::Infinite,
            None,
            4,
        )
        .expect("batched greedy generation");
        assert_eq!(batched, serial);
    }

    #[test]
    fn batched_greedy_matches_serial_stop_token_semantics() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = test_model(&device);
        let prompts = vec![vec![1, 7, 3], vec![4, 2, 9], vec![3, 8, 5]];
        let budgets = vec![8, 5, 7];
        let first_tokens = serial_greedy(&model, &prompts, &[1, 1, 1], None, &device);
        let stop_on_token = Some(first_tokens[0][0]);
        let serial = serial_greedy(&model, &prompts, &budgets, stop_on_token, &device);
        let batched = generate_greedy_batch_equal_position(
            &model,
            &prompts,
            &budgets,
            &device,
            ContextStrategy::Infinite,
            stop_on_token,
            4,
        )
        .expect("batched greedy generation");
        assert_eq!(batched, serial);
    }

    #[test]
    fn ragged_batched_greedy_matches_serial_across_prompt_positions() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = test_model(&device);
        let prompts = vec![vec![1, 7], vec![4, 2, 9, 3], vec![3, 8, 5, 6, 2, 1]];
        let budgets = vec![8, 5, 7];
        let first_tokens = serial_greedy(&model, &prompts, &[1, 1, 1], None, &device);
        let stop_on_token = Some(first_tokens[1][0]);
        let serial = serial_greedy(&model, &prompts, &budgets, stop_on_token, &device);
        let batched = generate_greedy_batch_ragged(
            &model,
            &prompts,
            &budgets,
            &device,
            ContextStrategy::Infinite,
            stop_on_token,
            4,
        )
        .expect("ragged batched greedy generation");
        assert_eq!(batched, serial);
    }

    #[test]
    fn batched_greedy_rejects_mismatched_positions_and_budgets() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = test_model(&device);
        assert!(
            generate_greedy_batch_equal_position(
                &model,
                &[vec![1, 2], vec![3]],
                &[1, 1],
                &device,
                ContextStrategy::Infinite,
                None,
                2,
            )
            .is_err()
        );
        assert!(
            generate_greedy_batch_equal_position(
                &model,
                &[vec![1, 2]],
                &[],
                &device,
                ContextStrategy::Infinite,
                None,
                2,
            )
            .is_err()
        );
    }

    #[test]
    fn all_active_predictive_context_generation_matches_dense_greedy() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = test_model(&device);
        model
            .predictive_coding_support()
            .expect("PC-compatible model");
        let prompt = vec![1, 7, 3, 9];
        let settings = GenerationSettings {
            max_new_tokens: Some(6),
            temperature: 1.0,
            top_k: Some(1),
            strategy: ContextStrategy::Infinite,
            stop_on_token: None,
        };
        let expected = generate_tokens(&model, prompt.clone(), &device, settings, None)
            .expect("dense greedy generation");
        let neuron_mask = Tensor::<TestBackend, 4>::ones([1, 2, 1, 16], &device);
        let activity_mask = Tensor::<TestBackend, 4>::ones([1, 1, 1, 16], &device);
        let actual = generate_greedy_tokens_with_subnetwork_masks(
            &model,
            prompt,
            &device,
            settings,
            neuron_mask,
            activity_mask,
        )
        .expect("context greedy generation");
        assert_eq!(actual, expected);
    }
}
