use crate::OptimizerScheduleMode;
use crate::train::prelude::*;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub enum ResolvedLrScheduler {
    Constant(LearningRate),
    Cosine(WarmupCosineLrScheduler),
    Linear(LinearLrScheduler),
    Exponential(ExponentialLrScheduler),
    Step(StepLrScheduler),
    Noam(NoamLrScheduler),
}

#[derive(Record, Clone, Debug)]
pub struct ResolvedLrSchedulerRecord<B: BackendTrait> {
    kind: u8,
    constant: Option<LearningRate>,
    cosine: Option<WarmupCosineLrSchedulerRecord>,
    linear: Option<<LinearLrScheduler as LrScheduler>::Record<B>>,
    exponential: Option<<ExponentialLrScheduler as LrScheduler>::Record<B>>,
    step: Option<<StepLrScheduler as LrScheduler>::Record<B>>,
    noam: Option<<NoamLrScheduler as LrScheduler>::Record<B>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleSource {
    Epochs,
    MaxIters,
}

impl ScheduleSource {
    pub fn as_str(self) -> &'static str {
        match self {
            ScheduleSource::Epochs => "epochs",
            ScheduleSource::MaxIters => "max_iters",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainSchedule {
    pub steps_per_epoch: usize,
    pub total_steps: usize,
    pub total_epochs: usize,
    pub source: ScheduleSource,
}

#[derive(Clone, Debug)]
pub struct WarmupCosineLrScheduler {
    peak_lr: LearningRate,
    min_lr: LearningRate,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

#[derive(Record, Clone, Debug)]
pub struct WarmupCosineLrSchedulerRecord {
    peak_lr: LearningRate,
    min_lr: LearningRate,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl WarmupCosineLrScheduler {
    fn new(
        peak_lr: LearningRate,
        min_lr: LearningRate,
        warmup_steps: usize,
        total_steps: usize,
    ) -> Result<Self, String> {
        if peak_lr <= 0.0 || peak_lr > 1.0 {
            return Err("Initial learning rate must be greater than 0 and at most 1".into());
        }
        if min_lr < 0.0 || min_lr > peak_lr {
            return Err(
                "Minimum learning rate must be at least 0 and at most equal to the initial learning rate"
                    .into(),
            );
        }
        if total_steps == 0 {
            return Err("Number of iterations must be at least 1".into());
        }

        Ok(Self {
            peak_lr,
            min_lr,
            warmup_steps: warmup_steps.min(total_steps),
            total_steps,
            current_step: 0,
        })
    }

    fn cosine_lr(&self, cosine_step: usize) -> LearningRate {
        let cosine_total_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let cosine_num_iters = cosine_total_steps.max(1);
        let current_iter = cosine_step % (cosine_num_iters + 1);

        self.min_lr
            + 0.5
                * (self.peak_lr - self.min_lr)
                * (1.0 + (current_iter as f64 / cosine_num_iters as f64 * PI).cos())
    }
}

impl LrScheduler for WarmupCosineLrScheduler {
    type Record<B: BackendTrait> = WarmupCosineLrSchedulerRecord;

    fn step(&mut self) -> LearningRate {
        self.current_step = self.current_step.saturating_add(1);

        if self.warmup_steps > 0 && self.current_step <= self.warmup_steps {
            return self.peak_lr * (self.current_step as f64 / self.warmup_steps as f64);
        }

        let cosine_step = self.current_step.saturating_sub(self.warmup_steps + 1);
        self.cosine_lr(cosine_step)
    }

    fn to_record<B: BackendTrait>(&self) -> Self::Record<B> {
        WarmupCosineLrSchedulerRecord {
            peak_lr: self.peak_lr,
            min_lr: self.min_lr,
            warmup_steps: self.warmup_steps,
            total_steps: self.total_steps,
            current_step: self.current_step,
        }
    }

    fn load_record<B: BackendTrait>(self, record: Self::Record<B>) -> Self {
        Self {
            peak_lr: record.peak_lr,
            min_lr: record.min_lr,
            warmup_steps: record.warmup_steps,
            total_steps: record.total_steps,
            current_step: record.current_step,
        }
    }
}

impl LrScheduler for ResolvedLrScheduler {
    type Record<B: BackendTrait> = ResolvedLrSchedulerRecord<B>;

    fn step(&mut self) -> LearningRate {
        match self {
            Self::Constant(lr) => *lr,
            Self::Cosine(scheduler) => scheduler.step(),
            Self::Linear(scheduler) => scheduler.step(),
            Self::Exponential(scheduler) => scheduler.step(),
            Self::Step(scheduler) => scheduler.step(),
            Self::Noam(scheduler) => scheduler.step(),
        }
    }

    fn to_record<B: BackendTrait>(&self) -> Self::Record<B> {
        match self {
            Self::Constant(lr) => ResolvedLrSchedulerRecord {
                kind: 0,
                constant: Some(*lr),
                cosine: None,
                linear: None,
                exponential: None,
                step: None,
                noam: None,
            },
            Self::Cosine(scheduler) => ResolvedLrSchedulerRecord {
                kind: 1,
                constant: None,
                cosine: Some(scheduler.to_record::<B>()),
                linear: None,
                exponential: None,
                step: None,
                noam: None,
            },
            Self::Linear(scheduler) => ResolvedLrSchedulerRecord {
                kind: 2,
                constant: None,
                cosine: None,
                linear: Some(scheduler.to_record::<B>()),
                exponential: None,
                step: None,
                noam: None,
            },
            Self::Exponential(scheduler) => ResolvedLrSchedulerRecord {
                kind: 3,
                constant: None,
                cosine: None,
                linear: None,
                exponential: Some(scheduler.to_record::<B>()),
                step: None,
                noam: None,
            },
            Self::Step(scheduler) => ResolvedLrSchedulerRecord {
                kind: 4,
                constant: None,
                cosine: None,
                linear: None,
                exponential: None,
                step: Some(scheduler.to_record::<B>()),
                noam: None,
            },
            Self::Noam(scheduler) => ResolvedLrSchedulerRecord {
                kind: 5,
                constant: None,
                cosine: None,
                linear: None,
                exponential: None,
                step: None,
                noam: Some(scheduler.to_record::<B>()),
            },
        }
    }

    fn load_record<B: BackendTrait>(self, record: Self::Record<B>) -> Self {
        match (self, record.kind) {
            (Self::Constant(_), 0) => {
                Self::Constant(record.constant.expect("constant lr scheduler record"))
            }
            (Self::Cosine(scheduler), 1) => Self::Cosine(
                scheduler.load_record::<B>(record.cosine.expect("cosine lr scheduler record")),
            ),
            (Self::Linear(scheduler), 2) => Self::Linear(
                scheduler.load_record::<B>(record.linear.expect("linear lr scheduler record")),
            ),
            (Self::Exponential(scheduler), 3) => Self::Exponential(
                scheduler
                    .load_record::<B>(record.exponential.expect("exponential lr scheduler record")),
            ),
            (Self::Step(scheduler), 4) => Self::Step(
                scheduler.load_record::<B>(record.step.expect("step lr scheduler record")),
            ),
            (Self::Noam(scheduler), 5) => Self::Noam(
                scheduler.load_record::<B>(record.noam.expect("noam lr scheduler record")),
            ),
            (variant, kind) => panic!(
                "resolved lr scheduler record kind {kind} does not match scheduler variant {}",
                match variant {
                    Self::Constant(_) => "constant",
                    Self::Cosine(_) => "cosine",
                    Self::Linear(_) => "linear",
                    Self::Exponential(_) => "exponential",
                    Self::Step(_) => "step",
                    Self::Noam(_) => "noam",
                }
            ),
        }
    }
}

pub fn resolve_valid_steps_per_epoch(
    total_steps: usize,
    log_frequency: usize,
    val_steps_per_epoch: usize,
) -> usize {
    let desired_valid_steps = usize::max(1, total_steps / log_frequency.max(1));
    desired_valid_steps.min(val_steps_per_epoch.max(1)).max(1)
}

pub fn resolve_lr_scheduler(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    override_num_iters: Option<usize>,
    default_model_size: usize,
) -> Result<ResolvedLrScheduler> {
    let base_lr = optimizer_cfg.learning_rate;
    let fallback_iters = total_steps.max(1);

    let schedule = match &optimizer_cfg.lr_schedule {
        None => match optimizer_cfg.schedule_mode {
            OptimizerScheduleMode::DragonReference => ResolvedLrScheduler::Constant(base_lr),
        },
        Some(LearningRateScheduleConfig::Constant { initial_lr }) => {
            ResolvedLrScheduler::Constant(initial_lr.unwrap_or(base_lr))
        }
        Some(LearningRateScheduleConfig::Cosine {
            initial_lr,
            min_lr,
            warmup_steps,
            num_iters,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = WarmupCosineLrScheduler::new(
                init_lr,
                min_lr.unwrap_or(0.0),
                warmup_steps.unwrap_or(0),
                override_num_iters
                    .unwrap_or_else(|| num_iters.unwrap_or(fallback_iters))
                    .max(1),
            )
            .map_err(|err| anyhow!("failed to initialize cosine lr scheduler: {err}"))?;
            ResolvedLrScheduler::Cosine(scheduler)
        }
        Some(LearningRateScheduleConfig::Linear {
            initial_lr,
            final_lr,
            num_iters,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = LinearLrSchedulerConfig::new(
                init_lr,
                *final_lr,
                override_num_iters
                    .unwrap_or_else(|| num_iters.unwrap_or(fallback_iters))
                    .max(1),
            )
            .init()
            .map_err(|err| anyhow!("failed to initialize linear lr scheduler: {err}"))?;
            ResolvedLrScheduler::Linear(scheduler)
        }
        Some(LearningRateScheduleConfig::Exponential { initial_lr, gamma }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = ExponentialLrSchedulerConfig::new(init_lr, *gamma)
                .init()
                .map_err(|err| anyhow!("failed to initialize exponential lr scheduler: {err}"))?;
            ResolvedLrScheduler::Exponential(scheduler)
        }
        Some(LearningRateScheduleConfig::Step {
            initial_lr,
            gamma,
            step_size,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler =
                StepLrSchedulerConfig::new(init_lr, step_size.unwrap_or(fallback_iters).max(1))
                    .with_gamma(*gamma)
                    .init()
                    .map_err(|err| anyhow!("failed to initialize step lr scheduler: {err}"))?;
            ResolvedLrScheduler::Step(scheduler)
        }
        Some(LearningRateScheduleConfig::Noam {
            initial_lr,
            warmup_steps,
            model_size,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let mut config = NoamLrSchedulerConfig::new(init_lr);
            config = config.with_warmup_steps(warmup_steps.unwrap_or(fallback_iters).max(1));
            config = config.with_model_size(model_size.unwrap_or(default_model_size).max(1));
            let scheduler = config
                .init()
                .map_err(|err| anyhow!("failed to initialize noam lr scheduler: {err}"))?;
            ResolvedLrScheduler::Noam(scheduler)
        }
    };

    Ok(schedule)
}

pub fn resolve_train_schedule(
    epochs: Option<usize>,
    max_iters: usize,
    steps_per_epoch: usize,
    label: &str,
) -> Result<TrainSchedule> {
    let steps_per_epoch = steps_per_epoch.max(1);
    match epochs {
        Some(epochs) => {
            let total_epochs = epochs.max(1);
            let total_steps = steps_per_epoch
                .checked_mul(total_epochs)
                .ok_or_else(|| {
                    anyhow!(
                        "{label}.epochs overflow: steps_per_epoch={steps_per_epoch}, epochs={total_epochs}"
                    )
                })?
                .max(1);
            Ok(TrainSchedule {
                steps_per_epoch,
                total_steps,
                total_epochs,
                source: ScheduleSource::Epochs,
            })
        }
        None => {
            let total_steps = max_iters.max(1);
            let total_epochs = usize::max(1, total_steps.div_ceil(steps_per_epoch));
            Ok(TrainSchedule {
                steps_per_epoch,
                total_steps,
                total_epochs,
                source: ScheduleSource::MaxIters,
            })
        }
    }
}
