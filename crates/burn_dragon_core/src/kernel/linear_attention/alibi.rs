//! Host-side ALiBi schedules shared by recurrent and dense attention executors.

/// Historical Dragon schedule, retained for configs/checkpoints without explicit slopes.
///
/// This is not the reference ALiBi schedule. Under exponential recurrent decay it
/// has only sub-two-token direct-write half-lives, regardless of head count.
pub fn default_alibi_slopes(n_head: usize) -> Vec<f32> {
    (0..n_head)
        .map(|idx| 1.0 / 2.0_f32.powf(idx as f32 / n_head as f32))
        .collect()
}

/// Head ordering and geometric slopes from the original ALiBi implementation.
///
/// See https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py.
/// Non-power-of-two counts append the interleaved slopes of the next power of two.
pub fn reference_alibi_slopes(n_head: usize) -> Vec<f32> {
    if n_head == 0 {
        return Vec::new();
    }
    let base = 1usize << n_head.ilog2();
    (1..=base)
        .map(|index| (-8.0 * index as f32 / base as f32).exp2())
        .chain((0..n_head - base).map(|index| (-4.0 * (2 * index + 1) as f32 / base as f32).exp2()))
        .collect()
}

pub fn validate_alibi_slopes(slopes: &[f32], n_head: usize) -> Result<(), String> {
    if n_head == 0 || slopes.len() != n_head {
        return Err(format!(
            "alibi_slopes must contain one slope per head (got {} slopes for {n_head} heads)",
            slopes.len()
        ));
    }
    if slopes
        .iter()
        .any(|slope| !slope.is_finite() || *slope < 0.0)
    {
        return Err("alibi_slopes must be finite and nonnegative".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alibi_reference_schedule_matches_power_and_non_power_of_two_heads() {
        assert_eq!(reference_alibi_slopes(0), Vec::<f32>::new());
        assert_eq!(reference_alibi_slopes(1), vec![1.0 / 256.0]);
        assert_eq!(
            reference_alibi_slopes(4),
            vec![0.25, 0.0625, 0.015625, 0.00390625]
        );
        assert_eq!(reference_alibi_slopes(3), vec![0.0625, 0.00390625, 0.25]);
        assert_eq!(
            reference_alibi_slopes(6),
            vec![0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125]
        );
        for heads in 1..=32 {
            let slopes = reference_alibi_slopes(heads);
            validate_alibi_slopes(&slopes, heads).unwrap();
            assert!(slopes.contains(&(1.0 / 256.0)));
        }
    }

    #[test]
    fn alibi_default_retains_checkpoint_semantics() {
        assert!(default_alibi_slopes(0).is_empty());
        for heads in 1..=32 {
            for (index, slope) in default_alibi_slopes(heads).into_iter().enumerate() {
                assert_eq!(slope, 1.0 / 2.0_f32.powf(index as f32 / heads as f32));
            }
        }
        let old_slowest = default_alibi_slopes(4)[3];
        assert!((-old_slowest * 64.0).exp() < 3.0e-17);
        let reference_slowest = reference_alibi_slopes(4)[3];
        assert!((-reference_slowest * 256.0).exp() > 0.36);
    }

    #[test]
    fn alibi_explicit_schedule_rejects_unstable_or_misshaped_decay() {
        for slopes in [
            vec![],
            vec![1.0],
            vec![-0.1, 0.1],
            vec![f32::NAN, 0.1],
            vec![f32::INFINITY, 0.1],
        ] {
            assert!(validate_alibi_slopes(&slopes, 2).is_err());
        }
        assert!(validate_alibi_slopes(&[], 0).is_err());
        assert!(validate_alibi_slopes(&[0.0, 0.1], 2).is_ok());
    }
}
