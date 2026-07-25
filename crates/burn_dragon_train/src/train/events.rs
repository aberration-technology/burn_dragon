pub use burn_ecs::prelude::{App, Plugin};
pub use burn_ecs::{
    CapabilityProbeExample, CapabilityProbeGroupMetric, CapabilityProbeSample,
    CapacityPlateauDetected, CapacityPlateauPlugin, CapacityPlateauState, CapacityScalingPolicy,
    CheckpointEvent, ContinualBackpropSample, ControlRequest, DynamicsControlEvent,
    DynamicsEquilibriumPlugin, DynamicsEquilibriumPolicy, DynamicsEquilibriumState, DynamicsMode,
    MetricAggregate, ModelCapacityConfig, ModelCapacityState, ModelScaleApplied, ModelScaleEvent,
    ModelScaleRequest, ModelScaleSkipped, MonitorRunOptions, OutputDegeneracySample,
    PredictiveCodingSample, RuliadSourceSelectionSample, SourceSelectionBucketMetric,
    SourceSelectionGroupMetric, SourceSelectionSample, StepFinished, StepStarted, TrainingAppExt,
    TrainingControlHandle, TrainingControlResource, TrainingCorePlugin, TrainingDashboard,
    TrainingDashboardState, TrainingEpochSummary, TrainingEventBus, TrainingEventBusConfig,
    TrainingEventBusStats, TrainingEventFiles, TrainingFileSinkConfig, TrainingGateAction,
    TrainingGateEvent, TrainingGatePolicy, TrainingGateSeverity, TrainingJsonEvent,
    TrainingMetricSample, TrainingMetricSplit, TrainingPlugins,
    TrainingRunConfig as TrainingRunContext, TrainingRunId, TrainingRunOptions, TrainingRuntime,
    TrainingRuntimeThread, TrainingSet as TrainingEventSet, TrainingWindowFinished,
    TrainingWindowMode, TrainingWindowStarted, ValidationFinished, monitor_run, render_dashboard,
};

/// Stable Dragon name for the shared file-sink configuration.
pub type TrainingEventsConfig = TrainingFileSinkConfig;
/// Stable Dragon name for the shared training-gate policy.
pub type TrainingGatesConfig = TrainingGatePolicy;
/// Stable Dragon name for the bounded event-bus configuration.
pub type EventBusConfig = TrainingEventBusConfig;
/// In-process event runtime used by tests and benchmarks.
pub type TrainingEventRuntime = TrainingRuntime;
/// Compatibility alias for downstream code migrating to [`TrainingRuntime`].
pub type TrainingEcsRuntime = TrainingRuntime;
/// Threaded event runtime used by Burn metric loggers and P2P observers.
pub type TrainingEcsThread = TrainingRuntimeThread;

#[cfg(feature = "train")]
pub use burn_ecs::burn_train::{
    BurnInterrupterControl, TrainingEventMetricLogger, TrainingMetricLogger,
};
