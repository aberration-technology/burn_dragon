//! Deterministic split seeds, document windows, and target masks.

use super::*;

pub(super) fn split_tag(split: burn_dragon_universality::SampleSplit) -> u8 {
    match split {
        burn_dragon_universality::SampleSplit::Train => 0,
        burn_dragon_universality::SampleSplit::Validation => 1,
    }
}

pub(super) fn live_source_selection_pending_limit() -> usize {
    std::env::var("DragonModel_RULIAD_SOURCE_SELECTION_PENDING_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4096)
}

pub(super) fn live_source_batch_cache_limit() -> usize {
    std::env::var("DragonModel_RULIAD_LIVE_BATCH_CACHE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_LIVE_SOURCE_BATCH_CACHE_LIMIT)
}

pub(super) fn live_source_batch_cache_bytes() -> usize {
    std::env::var("DragonModel_RULIAD_LIVE_BATCH_CACHE_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_LIVE_SOURCE_BATCH_CACHE_BYTES)
}

pub(super) fn live_source_selection_documents_per_step(
    batch_size: usize,
    configured: Option<usize>,
) -> usize {
    let environment_override =
        std::env::var("DragonModel_RULIAD_SOURCE_SELECTION_DOCUMENTS_PER_STEP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0);
    bounded_live_source_selection_documents_per_step(
        batch_size,
        environment_override.or(configured),
    )
}

pub(super) fn bounded_live_source_selection_documents_per_step(
    batch_size: usize,
    configured: Option<usize>,
) -> usize {
    configured
        .unwrap_or(batch_size.max(1))
        .min(batch_size.max(1))
        .max(1)
}

pub(super) fn live_source_selection_eos_window_probability() -> f64 {
    std::env::var("DragonModel_RULIAD_SOURCE_SELECTION_EOS_WINDOW_PROBABILITY")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .map(|value| value.clamp(0.0, 1.0))
        .unwrap_or(DEFAULT_SOURCE_SELECTED_EOS_WINDOW_PROBABILITY)
}

pub(super) fn source_selection_step_seed(
    epoch_index: usize,
    absolute_step: usize,
    salt: u64,
) -> u64 {
    mix_source_seed(
        0x8B8B_4D1A_51E5_E1ECu64
            ^ mix_source_seed(epoch_index as u64)
            ^ mix_source_seed((absolute_step as u64) ^ 0x9E37_79B9_7F4A_7C15)
            ^ mix_source_seed(salt ^ 0xD1B5_4A32_D192_ED03),
    )
}

fn mix_source_seed(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

pub(super) fn source_label_seed(label: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in label.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

pub(super) fn live_source_selection_sample_index(
    sample_count: usize,
    split: burn_dragon_universality::SampleSplit,
    epoch_index: usize,
    absolute_step: usize,
    bucket_label: &str,
    document_rank: usize,
) -> usize {
    let sample_count = sample_count.max(1);
    let split_salt = match split {
        burn_dragon_universality::SampleSplit::Train => split_tag(split) as usize,
        burn_dragon_universality::SampleSplit::Validation => {
            SOURCE_WEIGHTED_VALIDATION_SPLIT_TAG as usize
        }
    };
    let seed = source_selection_step_seed(
        epoch_index,
        absolute_step,
        source_label_seed(bucket_label)
            ^ (document_rank as u64).rotate_left(7)
            ^ (split_salt as u64).rotate_left(17),
    );
    let mut rng = StdRng::seed_from_u64(seed);
    rng.gen_range(0..sample_count)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct LiveSourceSampleCoordinate {
    pub epoch_index: usize,
    pub sample_index: usize,
}

/// Address an effectively unbounded online training stream without changing
/// the finite sample count used by offline materialization and validation.
/// Rows within a page form a deterministic permutation, so a large batch does
/// not sample the same generated document more than once merely because the
/// configured materialization panel is small.
pub(super) fn live_source_selection_sample_coordinate(
    sample_count: usize,
    split: burn_dragon_universality::SampleSplit,
    epoch_index: usize,
    absolute_step: usize,
    bucket_label: &str,
    document_rank: usize,
) -> LiveSourceSampleCoordinate {
    let sample_count = sample_count.max(1);
    if split == burn_dragon_universality::SampleSplit::Validation {
        return LiveSourceSampleCoordinate {
            epoch_index,
            sample_index: live_source_selection_sample_index(
                sample_count,
                split,
                epoch_index,
                absolute_step,
                bucket_label,
                document_rank,
            ),
        };
    }

    let page = document_rank / sample_count;
    let slot = document_rank % sample_count;
    let seed = source_selection_step_seed(
        epoch_index,
        absolute_step,
        source_label_seed(bucket_label) ^ (page as u64).rotate_left(29),
    );
    let count = sample_count as u64;
    let offset = (seed % count) as usize;
    let mut stride = ((seed.rotate_right(23) % count) as usize).max(1);
    while greatest_common_divisor(stride, sample_count) != 1 {
        stride = (stride + 1) % sample_count;
        if stride == 0 {
            stride = 1;
        }
    }
    let sample_index = offset.wrapping_add(slot.wrapping_mul(stride)) % sample_count;
    LiveSourceSampleCoordinate {
        // Keep native and wasm peers aligned by deriving the virtual epoch
        // from an explicitly fixed-width value.
        epoch_index: ((seed >> 32) as u32) as usize,
        sample_index,
    }
}

fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

pub(super) fn fixed_validation_probe_sample_index(
    sample_count: usize,
    bucket_label: Option<&str>,
    bucket_rank: usize,
) -> usize {
    let sample_count = sample_count.max(1);
    let bucket_seed = bucket_label
        .map(source_label_seed)
        .unwrap_or(0xA94E_195D_50C8_7A31);
    let seed = 0x66E3_5A9C_C9D4_17BFu64
        ^ bucket_seed.rotate_left(17)
        ^ (bucket_rank as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from(SOURCE_WEIGHTED_VALIDATION_SPLIT_TAG).rotate_left(31);
    let mut rng = StdRng::seed_from_u64(seed);
    rng.gen_range(0..sample_count)
}

pub(super) fn fixed_seeded_validation_probe_sample_index(
    sample_count: usize,
    panel_seed: u64,
    item_rank: usize,
) -> usize {
    let sample_count = sample_count.max(1);
    let seed = 0xF3A5_9C71_621B_4E0Du64
        ^ panel_seed.rotate_left(23)
        ^ (item_rank as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from(SOURCE_WEIGHTED_VALIDATION_SPLIT_TAG).rotate_left(31);
    let mut rng = StdRng::seed_from_u64(seed);
    rng.gen_range(0..sample_count)
}

pub(super) fn source_selected_windows_from_documents(
    documents: &[Arc<Vec<u32>>],
    eos_id: Option<u32>,
    bucket_label: &str,
    request: RuliadWindowRequest,
) -> Vec<Vec<u32>> {
    let RuliadWindowRequest {
        epoch_index,
        absolute_step,
        batch_size,
        block_size,
        prefer_answer_window,
        ..
    } = request;
    if documents.is_empty() {
        return Vec::new();
    }
    let document_count = documents.len();
    let pad_token = eos_id.unwrap_or(0);
    (0..batch_size)
        .map(|batch_index| {
            let mut rng = StdRng::seed_from_u64(source_selection_step_seed(
                epoch_index,
                absolute_step,
                source_label_seed(bucket_label) ^ (batch_index as u64).rotate_left(11),
            ));
            if prefer_answer_window {
                for attempt in 0..document_count {
                    let document = documents
                        .get((batch_index + attempt) % document_count)
                        .expect("source-selected document set must be non-empty");
                    let usable_len = valid_document_token_count(document, eos_id);
                    if let Some(window) = answer_window_from_document(
                        document, usable_len, block_size, pad_token, &mut rng,
                    ) {
                        return window;
                    }
                }
            }
            let document = documents
                .get(batch_index % document_count)
                .expect("source-selected document set must be non-empty");
            let usable_len = valid_document_token_count(document, eos_id);
            if usable_len <= block_size {
                return packed_valid_window_from_documents(
                    documents,
                    eos_id,
                    batch_index,
                    block_size,
                );
            }
            let max_start = usable_len.saturating_sub(block_size + 1);
            let start = if max_start == 0 {
                0
            } else {
                rng.gen_range(0..=max_start)
            };
            let start = selected_window_start(
                document,
                usable_len,
                block_size,
                start,
                &mut rng,
                prefer_answer_window,
            );
            document[start..start + block_size + 1].to_vec()
        })
        .collect()
}

pub(super) fn source_selected_stream_windows_from_documents(
    documents: &[Arc<Vec<u32>>],
    eos_id: Option<u32>,
    request: RuliadStreamWindowRequest,
) -> Vec<Vec<u32>> {
    let RuliadStreamWindowRequest {
        window:
            RuliadWindowRequest {
                epoch_index,
                absolute_step,
                batch_size,
                block_size,
                prefer_answer_window,
                ..
            },
        chunk_index_in_document,
    } = request;
    if documents.is_empty() {
        return Vec::new();
    }
    let pad_token = eos_id.unwrap_or(0);
    let start = chunk_index_in_document.saturating_mul(block_size);
    (0..batch_size)
        .map(|batch_index| {
            if prefer_answer_window {
                let mut rng = StdRng::seed_from_u64(source_selection_step_seed(
                    epoch_index,
                    absolute_step,
                    (batch_index as u64).rotate_left(13)
                        ^ (chunk_index_in_document as u64).rotate_left(23),
                ));
                for attempt in 0..documents.len() {
                    let document = documents
                        .get((batch_index + attempt) % documents.len())
                        .expect("source-selected stream document set must be non-empty");
                    let usable_len = valid_document_token_count(document, eos_id);
                    if let Some(window) = answer_window_from_document(
                        document, usable_len, block_size, pad_token, &mut rng,
                    ) {
                        return window;
                    }
                }
            }
            let document = documents
                .get(batch_index % documents.len())
                .expect("source-selected stream document set must be non-empty");
            let usable_len = valid_document_token_count(document, eos_id);
            let mut window = Vec::with_capacity(block_size + 1);
            for offset in 0..=block_size {
                let index = start.saturating_add(offset);
                window.push(
                    document
                        .get(index)
                        .copied()
                        .filter(|_| index < usable_len)
                        .unwrap_or(pad_token),
                );
            }
            window
        })
        .collect()
}

pub(super) fn source_selected_stream_document_complete(
    documents: &[Arc<Vec<u32>>],
    eos_id: Option<u32>,
    block_size: usize,
    chunk_index_in_document: usize,
) -> bool {
    let chunks = documents
        .iter()
        .map(|document| {
            valid_document_token_count(document, eos_id)
                .saturating_sub(1)
                .div_ceil(block_size.max(1))
                .max(1)
        })
        .max()
        .unwrap_or(1);
    chunk_index_in_document.saturating_add(1) >= chunks
}

pub(super) fn source_selected_stream_loss_masks_from_documents(
    documents: &[Arc<Vec<u32>>],
    eos_id: Option<u32>,
    batch_size: usize,
    block_size: usize,
    chunk_index_in_document: usize,
    supervision: RuliadSupervisionConfig,
) -> Vec<Vec<i64>> {
    if documents.is_empty() {
        return Vec::new();
    }
    let start = chunk_index_in_document.saturating_mul(block_size);
    (0..batch_size)
        .map(|batch_index| {
            let document = documents
                .get(batch_index % documents.len())
                .expect("source-selected stream document set must be non-empty");
            let usable_len = valid_document_token_count(document, eos_id);
            let mut mask = vec![0; block_size];
            ruliad_target_loss_mask_for_document_range(
                document,
                usable_len,
                start,
                block_size,
                &mut mask,
                supervision,
            );
            mask
        })
        .collect()
}

pub(super) fn ruliad_target_loss_mask_for_document_range(
    document: &[u32],
    usable_len: usize,
    start: usize,
    block_size: usize,
    mask: &mut [i64],
    supervision: RuliadSupervisionConfig,
) -> bool {
    mask.fill(0);
    let usable_len = usable_len.min(document.len());
    if usable_len < 2 || block_size == 0 {
        return false;
    }
    let mut document_mask = vec![0; usable_len - 1];
    if !ruliad_target_loss_mask(&document[..usable_len], &mut document_mask, supervision) {
        return false;
    }
    for (offset, slot) in mask.iter_mut().take(block_size).enumerate() {
        if let Some(value) = document_mask.get(start.saturating_add(offset)) {
            *slot = *value;
        }
    }
    mask.iter().any(|value| *value != 0)
}

pub(super) fn valid_document_token_count(document: &[u32], eos_id: Option<u32>) -> usize {
    eos_id
        .and_then(|eos_id| {
            document
                .iter()
                .position(|token| *token == eos_id)
                .map(|index| index.saturating_add(1))
        })
        .unwrap_or(document.len())
        .min(document.len())
}

pub(super) fn mask_fixed_document_eos_padding(
    window: &[u32],
    mask: &mut [i64],
    eos_id: Option<u32>,
) -> bool {
    let Some(eos_id) = eos_id else {
        return mask.iter().any(|value| *value != 0);
    };
    if window.len() < mask.len().saturating_add(1) {
        mask.fill(0);
        return false;
    }

    let mut document_ended = window.first() == Some(&eos_id);
    for (target_index, weight) in mask.iter_mut().enumerate() {
        if document_ended {
            *weight = 0;
            continue;
        }
        if window[target_index + 1] == eos_id {
            document_ended = true;
        }
    }
    mask.iter().any(|value| *value != 0)
}

pub(super) fn selected_window_start<R: Rng + ?Sized>(
    document: &[u32],
    usable_len: usize,
    block_size: usize,
    fallback_start: usize,
    rng: &mut R,
    prefer_answer_window: bool,
) -> usize {
    let max_start = usable_len.saturating_sub(block_size + 1);
    if max_start == 0 {
        return 0;
    }
    if prefer_answer_window {
        let candidates = answer_window_start_candidates(document, usable_len, block_size);
        if !candidates.is_empty() {
            return candidates[rng.gen_range(0..candidates.len())].min(max_start);
        }
    }
    if rng.gen_bool(live_source_selection_eos_window_probability()) {
        return max_start;
    }
    let candidates = semantic_window_start_candidates(document, usable_len, block_size);
    if candidates.is_empty() || !rng.gen_bool(0.85) {
        return fallback_start.min(max_start);
    }
    candidates[rng.gen_range(0..candidates.len())].min(max_start)
}

pub(super) fn answer_window_from_document<R: Rng + ?Sized>(
    document: &[u32],
    usable_len: usize,
    block_size: usize,
    pad_token: u32,
    rng: &mut R,
) -> Option<Vec<u32>> {
    if usable_len <= block_size + 1 {
        let mut window = vec![pad_token; block_size + 1];
        let copy_len = usable_len.min(document.len()).min(block_size + 1);
        window[..copy_len].copy_from_slice(&document[..copy_len]);
        let mut mask = vec![0; block_size];
        return ruliad_answer_target_loss_mask(&window, &mut mask).then_some(window);
    }
    let candidates = answer_window_start_candidates(document, usable_len, block_size);
    if candidates.is_empty() {
        return None;
    }
    let start = candidates[rng.gen_range(0..candidates.len())]
        .min(usable_len.saturating_sub(block_size + 1));
    Some(document[start..start + block_size + 1].to_vec())
}

pub(super) fn answer_window_start_candidates(
    document: &[u32],
    usable_len: usize,
    block_size: usize,
) -> Vec<usize> {
    if usable_len <= block_size + 1 {
        return Vec::new();
    }
    let max_start = usable_len.saturating_sub(block_size + 1);
    let lead = (block_size / 8).max(1);
    let mut starts = document
        .iter()
        .take(usable_len)
        .enumerate()
        .filter_map(|(index, _token)| {
            if !is_ruliad_answer_marker_at(document, usable_len, index) {
                return None;
            }
            let start = index.saturating_sub(lead).min(max_start);
            let end = start.saturating_add(block_size + 1).min(document.len());
            let mut mask = vec![0; end.saturating_sub(start).saturating_sub(1)];
            (index >= start
                && index < start.saturating_add(block_size)
                && ruliad_answer_target_loss_mask(&document[start..end], &mut mask))
            .then_some(start)
        })
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    starts
}

pub(super) fn semantic_window_start_candidates(
    document: &[u32],
    usable_len: usize,
    block_size: usize,
) -> Vec<usize> {
    if usable_len <= block_size + 1 {
        return Vec::new();
    }
    let max_start = usable_len.saturating_sub(block_size + 1);
    let lead = (block_size / 4).max(1);
    let mut starts = document
        .iter()
        .take(usable_len)
        .enumerate()
        .filter_map(|(index, token)| {
            if !is_semantic_window_anchor(document, index, *token) {
                return None;
            }
            Some(index.saturating_sub(lead).min(max_start))
        })
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    starts
}

pub(super) fn is_semantic_window_anchor(document: &[u32], index: usize, token: u32) -> bool {
    if matches!(
        token,
        RULIAD_SYMBOLIC_DATA_TOKEN
            | RULIAD_SYMBOLIC_QUERY_TOKEN
            | RULIAD_SYMBOLIC_PROOF_STEP_TOKEN
            | RULIAD_SYMBOLIC_ANSWER_TOKEN
            | RULIAD_SYMBOLIC_DOCUMENT_END_TOKEN
    ) {
        return true;
    }
    let marker = matches!(
        token,
        token if token == u32::from(b'?')
            || token == u32::from(b'>')
            || token == u32::from(b'!')
            || token == u32::from(b'G')
    );
    marker && (index == 0 || document.get(index - 1) == Some(&u32::from(b'\n')))
}

pub(super) fn ruliad_answer_target_loss_mask(window: &[u32], mask: &mut [i64]) -> bool {
    burn_dragon_universality::ruliad::ruliad_token_loss_mask(
        window,
        mask,
        burn_dragon_universality::ruliad::RuliadTokenSupervisionConfig {
            mode: burn_dragon_universality::ruliad::RuliadTokenSupervisionMode::AnswerCompletion,
            ..Default::default()
        },
    )
}

pub(super) fn ruliad_target_loss_mask(
    window: &[u32],
    mask: &mut [i64],
    supervision: RuliadSupervisionConfig,
) -> bool {
    burn_dragon_universality::ruliad::ruliad_token_loss_mask(
        window,
        mask,
        supervision.token_supervision(),
    )
}

pub(super) fn is_ruliad_answer_marker_at(
    document: &[u32],
    usable_len: usize,
    index: usize,
) -> bool {
    document
        .get(index)
        .is_some_and(|token| *token == RULIAD_SYMBOLIC_ANSWER_TOKEN)
        || (index + 1 < usable_len
            && document.get(index) == Some(&u32::from(b'!'))
            && document.get(index + 1) == Some(&u32::from(b':')))
}

pub(super) fn packed_valid_window_from_documents(
    documents: &[Arc<Vec<u32>>],
    eos_id: Option<u32>,
    first_document: usize,
    block_size: usize,
) -> Vec<u32> {
    let target_len = block_size.saturating_add(1);
    let mut window = Vec::with_capacity(target_len);
    if documents.is_empty() {
        window.resize(target_len, eos_id.unwrap_or(0));
        return window;
    }
    for offset in 0..documents.len().saturating_mul(2) {
        let document = documents
            .get((first_document + offset) % documents.len())
            .expect("source-selected document set must be non-empty");
        let usable_len = valid_document_token_count(document, eos_id);
        if usable_len == 0 {
            continue;
        }
        window.extend(document.iter().take(usable_len).copied());
        if window.len() >= target_len {
            break;
        }
    }
    let fill = eos_id.unwrap_or(0);
    while window.len() < target_len {
        window.push(fill);
    }
    window.truncate(target_len);
    window
}
