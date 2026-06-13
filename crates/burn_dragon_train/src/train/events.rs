pub use burn_ecs::{
    CheckpointEvent, ControlRequest, EventBusConfig, MetricAggregate, MonitorRunOptions,
    RuliadSourceSelectionSample, SourceSelectionSample, StepFinished, StepStarted,
    TrainingAppBuilder, TrainingAppConfig, TrainingControlHandle, TrainingControlResource,
    TrainingCorePlugin, TrainingDashboardResource, TrainingDashboardState, TrainingEcsRuntime,
    TrainingEcsThread, TrainingEpochSummary, TrainingEventBus, TrainingEventFiles,
    TrainingEventRuntimeConfig, TrainingEventsConfig, TrainingGateAction, TrainingGateEvent,
    TrainingGateSeverity, TrainingGatesConfig, TrainingJsonEvent, TrainingMetricSample,
    TrainingMetricSplit, TrainingRunConfig as TrainingRunContext, TrainingRunResource,
    TrainingRuntimeConfig, TrainingSet as TrainingEventSet, TrainingWindowFinished,
    TrainingWindowMode, TrainingWindowStarted, ValidationFinished, monitor_run, render_dashboard,
};

pub type TrainingEventRuntime = TrainingEcsRuntime;

#[cfg(feature = "train")]
pub use burn_ecs::burn_train::{
    BurnInterrupterControl, TrainingEventMetricLogger, TrainingMetricLogger,
};
