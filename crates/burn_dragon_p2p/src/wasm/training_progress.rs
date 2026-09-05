#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum DragonBrowserTrainingPhase {
    #[default]
    Idle,
    PreparingRuntime,
    LoadingData,
    MaterializingBatches,
    SyncingCheckpoint,
    InitializingModel,
    LoadingCheckpoint,
    SubmittingTraining,
    MeasuringAdapter,
    SynchronizingLoss,
    Evaluating,
    PublishingUpdate,
    SubmittingReceipt,
    Stopping,
    Complete,
    Stopped,
    Failed,
}

impl DragonBrowserTrainingPhase {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Idle => "ready",
            Self::PreparingRuntime => "preparing runtime",
            Self::LoadingData => "loading data",
            Self::MaterializingBatches => "building batches",
            Self::SyncingCheckpoint => "syncing checkpoint",
            Self::InitializingModel => "initializing model",
            Self::LoadingCheckpoint => "loading checkpoint",
            Self::SubmittingTraining => "training",
            Self::MeasuringAdapter => "measuring GPU",
            Self::SynchronizingLoss => "synchronizing GPU",
            Self::Evaluating => "evaluating",
            Self::PublishingUpdate => "publishing update",
            Self::SubmittingReceipt => "submitting receipt",
            Self::Stopping => "stopping",
            Self::Complete => "window complete",
            Self::Stopped => "stopped",
            Self::Failed => "failed",
        }
    }

    pub(crate) const fn slug(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::PreparingRuntime => "preparing-runtime",
            Self::LoadingData => "loading-data",
            Self::MaterializingBatches => "materializing-batches",
            Self::SyncingCheckpoint => "syncing-checkpoint",
            Self::InitializingModel => "initializing-model",
            Self::LoadingCheckpoint => "loading-checkpoint",
            Self::SubmittingTraining => "submitting-training",
            Self::MeasuringAdapter => "measuring-adapter",
            Self::SynchronizingLoss => "synchronizing-loss",
            Self::Evaluating => "evaluating",
            Self::PublishingUpdate => "publishing-update",
            Self::SubmittingReceipt => "submitting-receipt",
            Self::Stopping => "stopping",
            Self::Complete => "complete",
            Self::Stopped => "stopped",
            Self::Failed => "failed",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DragonBrowserTrainingProgress {
    pub(crate) phase: DragonBrowserTrainingPhase,
    pub(crate) submitted_batches: usize,
    pub(crate) planned_batches: usize,
    pub(crate) submitted_tokens: usize,
}

impl DragonBrowserTrainingProgress {
    pub(crate) const fn phase(phase: DragonBrowserTrainingPhase) -> Self {
        Self {
            phase,
            submitted_batches: 0,
            planned_batches: 0,
            submitted_tokens: 0,
        }
    }

    pub(crate) const fn training(
        submitted_batches: usize,
        planned_batches: usize,
        submitted_tokens: usize,
    ) -> Self {
        Self {
            phase: DragonBrowserTrainingPhase::SubmittingTraining,
            submitted_batches,
            planned_batches,
            submitted_tokens,
        }
    }

    pub(crate) const fn synchronizing(
        submitted_batches: usize,
        planned_batches: usize,
        submitted_tokens: usize,
    ) -> Self {
        Self {
            phase: DragonBrowserTrainingPhase::SynchronizingLoss,
            submitted_batches,
            planned_batches,
            submitted_tokens,
        }
    }

    pub(crate) const fn measuring_adapter(
        submitted_batches: usize,
        planned_batches: usize,
        submitted_tokens: usize,
    ) -> Self {
        Self {
            phase: DragonBrowserTrainingPhase::MeasuringAdapter,
            submitted_batches,
            planned_batches,
            submitted_tokens,
        }
    }
}

pub(crate) trait DragonBrowserTrainingObserver {
    fn on_progress(&mut self, progress: DragonBrowserTrainingProgress);
}

impl<F> DragonBrowserTrainingObserver for F
where
    F: FnMut(DragonBrowserTrainingProgress),
{
    fn on_progress(&mut self, progress: DragonBrowserTrainingProgress) {
        self(progress);
    }
}

#[derive(Default)]
pub(crate) struct NoopDragonBrowserTrainingObserver;

impl DragonBrowserTrainingObserver for NoopDragonBrowserTrainingObserver {
    fn on_progress(&mut self, _progress: DragonBrowserTrainingProgress) {}
}

#[derive(Clone, Debug, Default)]
pub(crate) struct DragonBrowserTrainingUiState {
    pub(crate) current_window: Option<u64>,
    pub(crate) completed_windows: u64,
    pub(crate) progress: DragonBrowserTrainingProgress,
    pub(crate) failure: Option<String>,
}

impl Default for DragonBrowserTrainingProgress {
    fn default() -> Self {
        Self::phase(DragonBrowserTrainingPhase::Idle)
    }
}

impl DragonBrowserTrainingUiState {
    pub(crate) fn start_session(&mut self) {
        self.current_window = None;
        self.completed_windows = 0;
        self.progress =
            DragonBrowserTrainingProgress::phase(DragonBrowserTrainingPhase::PreparingRuntime);
        self.failure = None;
    }

    pub(crate) fn start_window(&mut self, window: u64) {
        self.current_window = Some(window);
        self.progress =
            DragonBrowserTrainingProgress::phase(DragonBrowserTrainingPhase::PreparingRuntime);
        self.failure = None;
    }

    pub(crate) fn observe(&mut self, progress: DragonBrowserTrainingProgress) {
        self.progress = progress;
    }

    pub(crate) fn complete_window(&mut self, window: u64) {
        self.current_window = Some(window);
        self.completed_windows = self.completed_windows.max(window);
        self.progress.phase = DragonBrowserTrainingPhase::Complete;
        self.failure = None;
    }

    pub(crate) fn stopping(&mut self) {
        self.progress.phase = DragonBrowserTrainingPhase::Stopping;
    }

    pub(crate) fn stopped(&mut self) {
        self.progress.phase = DragonBrowserTrainingPhase::Stopped;
    }

    pub(crate) fn fail(&mut self, message: String) {
        self.progress.phase = DragonBrowserTrainingPhase::Failed;
        self.failure = Some(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_window_keeps_completed_window_count() {
        let mut state = DragonBrowserTrainingUiState::default();
        state.completed_windows = 3;
        state.start_window(4);
        state.observe(DragonBrowserTrainingProgress::training(2, 8, 512));

        assert_eq!(state.current_window, Some(4));
        assert_eq!(state.completed_windows, 3);
        assert_eq!(state.progress.submitted_batches, 2);
        assert_eq!(state.progress.phase.label(), "training");
    }

    #[test]
    fn synchronization_is_distinct_from_command_submission() {
        let progress = DragonBrowserTrainingProgress::synchronizing(4, 8, 1024);

        assert_eq!(progress.phase.slug(), "synchronizing-loss");
        assert_eq!(progress.submitted_batches, 4);
        assert_eq!(progress.planned_batches, 8);
    }

    #[test]
    fn adapter_measurement_keeps_progress_visible() {
        let progress = DragonBrowserTrainingProgress::measuring_adapter(1, 64, 1536);

        assert_eq!(progress.phase.label(), "measuring GPU");
        assert_eq!(progress.submitted_batches, 1);
        assert_eq!(progress.submitted_tokens, 1536);
    }
}
