//! Pretokenized chunk access and manifest compatibility checks.

use super::*;

impl ChunkedTokens {
    pub(super) fn copy_into(&self, start: usize, dst: &mut [u32]) {
        let mut remaining = dst.len();
        let mut written = 0usize;
        let mut cursor = start;

        while remaining > 0 {
            let chunk_idx = self
                .chunks
                .partition_point(|chunk| chunk.token_offset + chunk.token_count <= cursor)
                .min(self.chunks.len().saturating_sub(1));
            let chunk = &self.chunks[chunk_idx];
            let chunk_data = self.chunk_data(chunk_idx);
            let chunk_tokens = mmap_as_u32_slice(&chunk_data, chunk.token_count);
            let chunk_start = cursor.saturating_sub(chunk.token_offset);
            let copy_len = chunk.token_count.saturating_sub(chunk_start).min(remaining);
            dst[written..written + copy_len]
                .copy_from_slice(&chunk_tokens[chunk_start..chunk_start + copy_len]);
            cursor += copy_len;
            written += copy_len;
            remaining -= copy_len;
        }
    }

    pub(super) fn chunk_data(&self, chunk_idx: usize) -> Arc<Mmap> {
        let chunk = &self.chunks[chunk_idx];
        load_cached_chunk_from_mutex(
            &self.cache,
            self.cache_limit,
            chunk_idx,
            &chunk.path,
            chunk.token_count,
            "universality",
        )
    }
}

pub(super) fn validate_pretokenized_tokenizer(
    tokenizer_cfg: &TokenizerConfig,
) -> io::Result<SharedTokenizer> {
    if !matches!(tokenizer_cfg.kind, TokenizerKind::Pretokenized(_)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "universality datasets require tokenizer.type = `pretokenized`",
        ));
    }
    tokenizer_cfg
        .fit(std::iter::empty())
        .map_err(io::Error::other)
}

pub(super) fn validate_tokenizer_against_manifest(
    tokenizer: &dyn crate::tokenizer::Tokenizer,
    manifest: &burn_dragon_universality::UniversalityTokenizerManifest,
) -> io::Result<()> {
    if tokenizer.len() != manifest.vocab_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "universality dataset tokenizer vocab mismatch (config={} manifest={})",
                tokenizer.len(),
                manifest.vocab_size
            ),
        ));
    }
    if tokenizer.bos_id() != manifest.bos_id
        || tokenizer.eos_id() != manifest.eos_id
        || tokenizer.pad_id() != manifest.pad_id
        || tokenizer.unk_id() != manifest.unk_id
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "universality dataset tokenizer special ids do not match manifest",
        ));
    }
    Ok(())
}

pub(super) fn config_file_display_name(file_stem: &str) -> &str {
    file_stem
}

pub(super) fn runtime_chunk_cache_limit() -> usize {
    std::env::var("DragonModel_PREPARED_TOKEN_CACHE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_RUNTIME_CHUNK_CACHE_LIMIT)
}

pub(super) fn runtime_document_cache_limit(
    batch_size: usize,
    train_samples: usize,
    validation_samples: usize,
) -> usize {
    std::env::var("DragonModel_UNIVERSALITY_RUNTIME_DOCUMENT_CACHE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            DEFAULT_RUNTIME_DOCUMENT_CACHE_LIMIT
                .max(batch_size.saturating_mul(8))
                .max(
                    train_samples
                        .saturating_mul(2)
                        .saturating_add(validation_samples),
                )
        })
}

pub(super) fn runtime_generation_worker_count(sample_count: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4);
    let configured = std::env::var("DragonModel_UNIVERSALITY_GENERATION_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| available.min(DEFAULT_RUNTIME_GENERATION_WORKER_LIMIT));
    configured.min(sample_count).max(1)
}

pub(super) fn fixed_manifest_logical_document_tokens(
    manifest: &burn_dragon_universality::UniversalityCorpusManifest,
) -> io::Result<Option<usize>> {
    let train_samples = manifest.stats.train_samples;
    let val_samples = manifest.stats.validation_samples;
    let train_doc_tokens = manifest
        .train_token_count
        .checked_div(train_samples)
        .and_then(|per_doc| {
            manifest
                .train_token_count
                .is_multiple_of(train_samples)
                .then_some(per_doc)
        });
    let val_doc_tokens = manifest
        .val_token_count
        .checked_div(val_samples)
        .and_then(|per_doc| {
            manifest
                .val_token_count
                .is_multiple_of(val_samples)
                .then_some(per_doc)
        });
    let document_token_count = match (train_doc_tokens, val_doc_tokens) {
        (Some(train), Some(val)) if train == val => Some(train),
        (Some(train), None) => Some(train),
        (None, Some(val)) => Some(val),
        (None, None) => None,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "universality manifest has inconsistent prepared document lengths across splits",
            ));
        }
    };

    match document_token_count {
        Some(0) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "universality manifest document token count must be > 0",
        )),
        Some(count) => Ok(Some(count.saturating_sub(1))),
        None => Ok(None),
    }
}
