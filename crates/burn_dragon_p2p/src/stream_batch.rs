use std::collections::BTreeMap;

use anyhow::{Result, bail};

use crate::config::TokenWindowRecord;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PlannedStreamBatch {
    pub record_indices: Vec<usize>,
    pub reset_stream_state: bool,
}

pub(crate) fn plan_stream_aligned_batches(
    records: &[TokenWindowRecord],
    batch_size: usize,
) -> Result<Vec<PlannedStreamBatch>> {
    let batch_size = batch_size.max(1);
    if records.iter().any(|record| {
        record.stream_group_id.is_none()
            || record.stream_row.is_none()
            || record.chunk_index.is_none()
    }) {
        return Ok((0..records.len())
            .collect::<Vec<_>>()
            .chunks(batch_size)
            .map(|indices| PlannedStreamBatch {
                record_indices: indices.to_vec(),
                reset_stream_state: true,
            })
            .collect());
    }

    let mut groups = BTreeMap::<(u64, usize), Vec<usize>>::new();
    for (record_index, record) in records.iter().enumerate() {
        let key = (
            record.stream_group_id.expect("checked stream group"),
            record.chunk_index.expect("checked chunk index"),
        );
        groups.entry(key).or_default().push(record_index);
    }

    let mut batches = Vec::with_capacity(groups.len());
    let mut previous_key: Option<(u64, usize)> = None;
    let mut previous_rows = Vec::new();
    for ((group_id, chunk_index), mut record_indices) in groups {
        record_indices.sort_by_key(|index| records[*index].stream_row.expect("checked stream row"));
        if record_indices.len() > batch_size {
            bail!(
                "stream group {group_id} chunk {chunk_index} contains {} rows, above batch size {batch_size}",
                record_indices.len()
            );
        }
        let rows = record_indices
            .iter()
            .map(|index| records[*index].stream_row.expect("checked stream row"))
            .collect::<Vec<_>>();
        if rows.windows(2).any(|window| window[0] == window[1]) {
            bail!("stream group {group_id} chunk {chunk_index} contains duplicate stream rows");
        }

        let contiguous = previous_key.is_some_and(|(previous_group, previous_chunk)| {
            previous_group == group_id
                && previous_chunk
                    .checked_add(1)
                    .is_some_and(|next_chunk| next_chunk == chunk_index)
        });
        let reset_stream_state = !contiguous || chunk_index == 0;
        if !reset_stream_state && rows != previous_rows {
            bail!(
                "stream group {group_id} changed row lanes between contiguous chunks {} and {chunk_index}",
                chunk_index.saturating_sub(1)
            );
        }
        batches.push(PlannedStreamBatch {
            record_indices,
            reset_stream_state,
        });
        previous_key = Some((group_id, chunk_index));
        previous_rows = rows;
    }
    Ok(batches)
}

pub(crate) fn plan_windowed_stream_batches(
    records: &[TokenWindowRecord],
    batch_size: usize,
    max_batches: Option<usize>,
    window_id: Option<u64>,
) -> Result<Vec<PlannedStreamBatch>> {
    let plan = plan_stream_aligned_batches(records, batch_size)?;
    let limit = max_batches.unwrap_or(plan.len()).min(plan.len());
    if limit == 0 || plan.is_empty() {
        return Ok(Vec::new());
    }
    if limit == plan.len() {
        return Ok(plan);
    }

    let window_index = u128::from(window_id.unwrap_or(1).saturating_sub(1));
    let start = ((window_index * limit as u128) % plan.len() as u128) as usize;
    let mut selected = Vec::with_capacity(limit);
    for offset in 0..limit {
        let plan_index = (start + offset) % plan.len();
        let mut batch = plan[plan_index].clone();
        if offset == 0 || plan_index == 0 {
            batch.reset_stream_state = true;
        }
        selected.push(batch);
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(
        group: Option<u64>,
        row: Option<usize>,
        chunk: Option<usize>,
        token: i64,
    ) -> TokenWindowRecord {
        TokenWindowRecord {
            inputs: vec![token],
            targets: vec![token + 1],
            reset_stream_state: chunk == Some(0),
            stream_group_id: group,
            stream_row: row,
            chunk_index: chunk,
        }
    }

    #[test]
    fn restores_group_chunk_and_row_order() {
        let records = vec![
            record(Some(7), Some(1), Some(1), 31),
            record(Some(8), Some(0), Some(5), 50),
            record(Some(7), Some(0), Some(0), 10),
            record(Some(7), Some(0), Some(1), 30),
            record(Some(7), Some(1), Some(0), 11),
        ];
        let batches = plan_stream_aligned_batches(&records, 2).expect("stream plan");

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].record_indices, vec![2, 4]);
        assert_eq!(batches[1].record_indices, vec![3, 0]);
        assert_eq!(batches[2].record_indices, vec![1]);
        assert!(batches[0].reset_stream_state);
        assert!(!batches[1].reset_stream_state);
        assert!(batches[2].reset_stream_state);
    }

    #[test]
    fn legacy_records_reset_every_batch() {
        let records = vec![
            record(None, None, None, 1),
            record(None, None, None, 2),
            record(None, None, None, 3),
        ];
        let batches = plan_stream_aligned_batches(&records, 2).expect("legacy plan");

        assert_eq!(batches.len(), 2);
        assert!(batches.iter().all(|batch| batch.reset_stream_state));
    }

    #[test]
    fn missing_chunk_resets_instead_of_reusing_stale_state() {
        let records = vec![
            record(Some(1), Some(0), Some(2), 1),
            record(Some(1), Some(0), Some(4), 2),
        ];
        let batches = plan_stream_aligned_batches(&records, 1).expect("gapped plan");

        assert!(batches[0].reset_stream_state);
        assert!(batches[1].reset_stream_state);
    }

    #[test]
    fn rejects_duplicate_or_changing_lanes() {
        let duplicate = vec![
            record(Some(1), Some(0), Some(0), 1),
            record(Some(1), Some(0), Some(0), 2),
        ];
        assert!(
            plan_stream_aligned_batches(&duplicate, 2)
                .expect_err("duplicate lanes")
                .to_string()
                .contains("duplicate")
        );

        let changed = vec![
            record(Some(1), Some(0), Some(0), 1),
            record(Some(1), Some(1), Some(0), 2),
            record(Some(1), Some(0), Some(1), 3),
        ];
        assert!(
            plan_stream_aligned_batches(&changed, 2)
                .expect_err("changed lanes")
                .to_string()
                .contains("changed row lanes")
        );
    }

    #[test]
    fn windowed_plan_rotates_disjoint_micro_epochs_and_resets_boundaries() {
        let records = (0..8)
            .map(|index| record(None, None, None, index))
            .collect::<Vec<_>>();

        let first =
            plan_windowed_stream_batches(&records, 1, Some(2), Some(1)).expect("first window");
        let second =
            plan_windowed_stream_batches(&records, 1, Some(2), Some(2)).expect("second window");
        let wrapped =
            plan_windowed_stream_batches(&records, 1, Some(2), Some(5)).expect("wrapped window");

        assert_eq!(
            first
                .iter()
                .map(|batch| batch.record_indices[0])
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(
            second
                .iter()
                .map(|batch| batch.record_indices[0])
                .collect::<Vec<_>>(),
            [2, 3]
        );
        assert_eq!(wrapped, first);
        assert!(first[0].reset_stream_state);
        assert!(second[0].reset_stream_state);
    }
}
