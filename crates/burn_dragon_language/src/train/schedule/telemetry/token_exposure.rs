//! Epoch-local source exposure, measured from host metadata without GPU readback.
//! These counts exclude separate structured terminals and latent objectives.

#[derive(Default)]
pub(crate) struct TrainingTokenExposure {
    scheduled: usize,
    supervised: usize,
    supervised_batches: usize,
    zero_supervision_batches: usize,
    unknown_batches: usize,
}

impl TrainingTokenExposure {
    pub(crate) fn observe(&mut self, scheduled: usize, supervised: Option<usize>) {
        self.scheduled = self.scheduled.saturating_add(scheduled);
        if let Some(supervised) = supervised {
            assert!(
                supervised <= scheduled,
                "token count is not a loss-weight sum"
            );
            self.supervised = self.supervised.saturating_add(supervised);
            if supervised == 0 {
                self.zero_supervision_batches = self.zero_supervision_batches.saturating_add(1);
            } else {
                self.supervised_batches = self.supervised_batches.saturating_add(1);
            }
        } else {
            self.unknown_batches = self.unknown_batches.saturating_add(1);
        }
    }

    pub(crate) fn metrics(&self) -> Vec<(&'static str, f64)> {
        let mut metrics = vec![
            ("Epoch Scheduled Tokens", self.scheduled as f64),
            ("Epoch Source Supervised Tokens", self.supervised as f64),
            (
                "Epoch Unknown Supervision Batches",
                self.unknown_batches as f64,
            ),
            (
                "Epoch Source Supervised Batches",
                self.supervised_batches as f64,
            ),
            (
                "Epoch Zero Supervision Batches",
                self.zero_supervision_batches as f64,
            ),
        ];
        if self.unknown_batches == 0 && self.scheduled > 0 {
            metrics.push((
                "Source Supervised Token Fraction",
                self.supervised as f64 / self.scheduled as f64,
            ));
        }
        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_exposure_counts_padding_and_context_without_a_device_sync() {
        let mut exposure = TrainingTokenExposure::default();
        exposure.observe(64, Some(16));
        exposure.observe(64, Some(0));
        assert_eq!(exposure.metrics()[0].1, 128.0);
        assert_eq!(exposure.metrics()[1].1, 16.0);
        assert_eq!(exposure.metrics()[3].1, 1.0);
        assert_eq!(exposure.metrics()[4].1, 1.0);
        assert_eq!(exposure.metrics().last().unwrap().1, 0.125);
    }

    #[test]
    fn token_exposure_does_not_treat_unknown_masks_as_full_supervision() {
        let mut exposure = TrainingTokenExposure::default();
        exposure.observe(64, None);
        assert_eq!(exposure.metrics().len(), 5);
        assert_eq!(exposure.metrics()[2].1, 1.0);
        assert_eq!(exposure.metrics()[3].1, 0.0);
        assert_eq!(exposure.metrics()[4].1, 0.0);
    }
}
