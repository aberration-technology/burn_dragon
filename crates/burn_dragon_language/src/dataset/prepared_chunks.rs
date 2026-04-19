use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use memmap2::{Mmap, MmapOptions};

#[derive(Default)]
pub(super) struct ChunkRuntimeCache {
    pub(super) tick: u64,
    pub(super) entries: HashMap<usize, CachedChunk>,
}

pub(super) struct CachedChunk {
    mmap: Arc<Mmap>,
    last_used_tick: u64,
}

pub(super) fn load_cached_chunk(
    cache: &mut ChunkRuntimeCache,
    cache_limit: usize,
    chunk_idx: usize,
    chunk_path: &Path,
    token_count: usize,
    dataset_label: &'static str,
) -> Arc<Mmap> {
    cache.tick = cache.tick.wrapping_add(1);
    let tick = cache.tick;
    if let Some(entry) = cache.entries.get_mut(&chunk_idx) {
        entry.last_used_tick = tick;
        return Arc::clone(&entry.mmap);
    }

    let file = fs::File::open(chunk_path).unwrap_or_else(|err| {
        panic!(
            "failed to open {dataset_label} chunk {}: {err}",
            chunk_path.display()
        )
    });
    let mmap = unsafe { MmapOptions::new().map(&file) }.unwrap_or_else(|err| {
        panic!(
            "failed to mmap {dataset_label} chunk {}: {err}",
            chunk_path.display()
        )
    });
    let len = mmap.len() / 4;
    if mmap.len() % 4 != 0 || len != token_count {
        panic!(
            "{dataset_label} chunk {} size mismatch: bytes={} tokens={} expected_tokens={}",
            chunk_path.display(),
            mmap.len(),
            len,
            token_count
        );
    }
    let mmap = Arc::new(mmap);
    cache.entries.insert(
        chunk_idx,
        CachedChunk {
            mmap: Arc::clone(&mmap),
            last_used_tick: tick,
        },
    );
    while cache.entries.len() > cache_limit {
        let evict_key = cache
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_used_tick)
            .map(|(idx, _)| *idx)
            .expect("prepared chunk cache should not be empty");
        cache.entries.remove(&evict_key);
    }
    mmap
}

pub(super) fn load_cached_chunk_from_mutex(
    cache: &Mutex<ChunkRuntimeCache>,
    cache_limit: usize,
    chunk_idx: usize,
    chunk_path: &Path,
    token_count: usize,
    dataset_label: &'static str,
) -> Arc<Mmap> {
    let mut cache = cache
        .lock()
        .unwrap_or_else(|_| panic!("{dataset_label} chunk cache poisoned"));
    load_cached_chunk(
        &mut cache,
        cache_limit,
        chunk_idx,
        chunk_path,
        token_count,
        dataset_label,
    )
}

pub(super) fn mmap_as_u32_slice(mmap: &Mmap, len: usize) -> &[u32] {
    debug_assert_eq!(mmap.len(), len * 4);
    #[cfg(not(target_endian = "little"))]
    compile_error!("prepared token mmap currently assumes little-endian hosts");
    unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const u32, len) }
}
