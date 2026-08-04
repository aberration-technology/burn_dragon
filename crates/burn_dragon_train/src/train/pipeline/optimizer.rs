use crate::train::prelude::*;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::momentum::MomentumConfig;
use burn::optim::{MultiGradientsParams, Sgd, SgdConfig, SimpleOptimizer};
use burn::tensor::backend::Backend;

#[derive(Clone)]
pub struct PredictiveCodingDiagonalNatural {
    fisher_decay: f32,
    damping: f32,
    weight_decay: f32,
}

#[derive(Record, Clone)]
pub struct PredictiveCodingDiagonalNaturalState<B: Backend, const D: usize> {
    fisher: Tensor<B, D>,
}

impl PredictiveCodingDiagonalNatural {
    fn new(config: &PredictiveCodingOptimizerConfig, weight_decay: f32) -> Self {
        Self {
            fisher_decay: config.fisher_decay,
            damping: config.damping,
            weight_decay,
        }
    }
}

impl<B: BackendTrait> SimpleOptimizer<B> for PredictiveCodingDiagonalNatural {
    type State<const D: usize> = PredictiveCodingDiagonalNaturalState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        mut grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        if self.weight_decay > 0.0 {
            grad = grad + tensor.clone().mul_scalar(self.weight_decay);
        }
        let grad_square = grad.clone().square();
        let fisher = if let Some(state) = state {
            state.fisher.mul_scalar(self.fisher_decay)
                + grad_square.mul_scalar(1.0 - self.fisher_decay)
        } else {
            grad_square
        };
        let update = grad / fisher.clone().add_scalar(self.damping).sqrt();
        let tensor = tensor - update.mul_scalar(lr);
        (
            tensor,
            Some(PredictiveCodingDiagonalNaturalState { fisher }),
        )
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        state.fisher = state.fisher.to_device(device);
        state
    }
}

#[derive(Clone)]
pub enum ResolvedOptimizer<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    AdamW(OptimizerAdaptor<AdamW, M, B>),
    PredictiveCodingSgd(OptimizerAdaptor<Sgd<B::InnerBackend>, M, B>),
    PredictiveCodingMomentum(OptimizerAdaptor<Sgd<B::InnerBackend>, M, B>),
    PredictiveCodingAdamW(OptimizerAdaptor<AdamW, M, B>),
    PredictiveCodingDiagonalNatural(OptimizerAdaptor<PredictiveCodingDiagonalNatural, M, B>),
}

type ResolvedSgdOptimizerRecord<M, B> =
    <OptimizerAdaptor<Sgd<<B as AutodiffBackend>::InnerBackend>, M, B> as Optimizer<M, B>>::Record;

#[derive(Record, Clone)]
pub struct ResolvedOptimizerRecord<M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    kind: u8,
    adamw: <OptimizerAdaptor<AdamW, M, B> as Optimizer<M, B>>::Record,
    sgd: Option<ResolvedSgdOptimizerRecord<M, B>>,
    diagonal_natural: Option<
        <OptimizerAdaptor<PredictiveCodingDiagonalNatural, M, B> as Optimizer<M, B>>::Record,
    >,
}

impl<B, M> Optimizer<M, B> for ResolvedOptimizer<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    type Record = ResolvedOptimizerRecord<M, B>;

    fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M {
        match self {
            Self::AdamW(optimizer) => optimizer.step(lr, module, grads),
            Self::PredictiveCodingSgd(optimizer) => optimizer.step(lr, module, grads),
            Self::PredictiveCodingMomentum(optimizer) => optimizer.step(lr, module, grads),
            Self::PredictiveCodingAdamW(optimizer) => optimizer.step(lr, module, grads),
            Self::PredictiveCodingDiagonalNatural(optimizer) => optimizer.step(lr, module, grads),
        }
    }

    fn step_multi(&mut self, lr: LearningRate, module: M, grads: MultiGradientsParams) -> M {
        match self {
            Self::AdamW(optimizer) => optimizer.step_multi(lr, module, grads),
            Self::PredictiveCodingSgd(optimizer) => optimizer.step_multi(lr, module, grads),
            Self::PredictiveCodingMomentum(optimizer) => optimizer.step_multi(lr, module, grads),
            Self::PredictiveCodingAdamW(optimizer) => optimizer.step_multi(lr, module, grads),
            Self::PredictiveCodingDiagonalNatural(optimizer) => {
                optimizer.step_multi(lr, module, grads)
            }
        }
    }

    fn to_record(&self) -> Self::Record {
        match self {
            Self::AdamW(optimizer) => ResolvedOptimizerRecord {
                kind: 0,
                adamw: optimizer.to_record(),
                sgd: None,
                diagonal_natural: None,
            },
            Self::PredictiveCodingSgd(optimizer) => ResolvedOptimizerRecord {
                kind: 1,
                adamw: empty_adamw_record::<B, M>(),
                sgd: Some(optimizer.to_record()),
                diagonal_natural: None,
            },
            Self::PredictiveCodingMomentum(optimizer) => ResolvedOptimizerRecord {
                kind: 2,
                adamw: empty_adamw_record::<B, M>(),
                sgd: Some(optimizer.to_record()),
                diagonal_natural: None,
            },
            Self::PredictiveCodingAdamW(optimizer) => ResolvedOptimizerRecord {
                kind: 3,
                adamw: optimizer.to_record(),
                sgd: None,
                diagonal_natural: None,
            },
            Self::PredictiveCodingDiagonalNatural(optimizer) => ResolvedOptimizerRecord {
                kind: 4,
                adamw: empty_adamw_record::<B, M>(),
                sgd: None,
                diagonal_natural: Some(optimizer.to_record()),
            },
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        match (self, record.kind) {
            (Self::AdamW(optimizer), 0) => Self::AdamW(optimizer.load_record(record.adamw)),
            (Self::PredictiveCodingSgd(optimizer), 1) => Self::PredictiveCodingSgd(
                optimizer.load_record(record.sgd.expect("predictive coding sgd optimizer record")),
            ),
            (Self::PredictiveCodingMomentum(optimizer), 2) => Self::PredictiveCodingMomentum(
                optimizer.load_record(
                    record
                        .sgd
                        .expect("predictive coding momentum optimizer record"),
                ),
            ),
            (Self::PredictiveCodingAdamW(optimizer), 3) => {
                Self::PredictiveCodingAdamW(optimizer.load_record(record.adamw))
            }
            (Self::PredictiveCodingDiagonalNatural(optimizer), 4) => {
                Self::PredictiveCodingDiagonalNatural(
                    optimizer.load_record(
                        record
                            .diagonal_natural
                            .expect("predictive coding diagonal natural optimizer record"),
                    ),
                )
            }
            (variant, kind) => panic!(
                "optimizer record kind {kind} does not match optimizer variant {}",
                variant.name()
            ),
        }
    }
}

impl<B, M> ResolvedOptimizer<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    fn name(&self) -> &'static str {
        match self {
            Self::AdamW(_) => "adamw",
            Self::PredictiveCodingSgd(_) => "predictive_coding_sgd",
            Self::PredictiveCodingMomentum(_) => "predictive_coding_momentum",
            Self::PredictiveCodingAdamW(_) => "predictive_coding_adamw",
            Self::PredictiveCodingDiagonalNatural(_) => "predictive_coding_diagonal_natural",
        }
    }
}

fn empty_adamw_record<B, M>() -> <OptimizerAdaptor<AdamW, M, B> as Optimizer<M, B>>::Record
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    AdamWConfig::new().init::<B, M>().to_record()
}

fn sgd_config_from_predictive_coding(
    optimizer_cfg: &OptimizerConfig,
    use_momentum: bool,
) -> SgdConfig {
    let pc = &optimizer_cfg.predictive_coding;
    let mut config = SgdConfig::new();
    if optimizer_cfg.weight_decay > 0.0 {
        config = config.with_weight_decay(Some(WeightDecayConfig::new(optimizer_cfg.weight_decay)));
    }
    if use_momentum {
        config = config.with_momentum(Some(MomentumConfig {
            momentum: pc.momentum as f64,
            dampening: 0.0,
            nesterov: pc.nesterov,
        }));
    }
    if let Some(clip) = optimizer_cfg.grad_clip_norm {
        config = config.with_gradient_clipping(Some(GradientClippingConfig::Norm(clip)));
    } else if let Some(clip) = optimizer_cfg.grad_clip_value {
        config = config.with_gradient_clipping(Some(GradientClippingConfig::Value(clip)));
    }
    config
}

pub fn resolve_optimizer<B, M>(
    optimizer_cfg: &OptimizerConfig,
    _total_steps: usize,
) -> Result<ResolvedOptimizer<B, M>>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    match optimizer_cfg.name {
        OptimizerKind::Adamw => Ok(ResolvedOptimizer::AdamW(
            adamw_config_from_optimizer(optimizer_cfg).init::<B, M>(),
        )),
        OptimizerKind::Eggroll => Err(anyhow!(
            "optimizer.name=eggroll uses the EGGROLL evolution-strategy training path, not the Burn gradient optimizer resolver"
        )),
        OptimizerKind::PredictiveCoding => Err(anyhow!(
            "optimizer.name=predictive_coding is retired; predictive coding is a training algorithm, so use training.algorithm=predictive_coding with an ordinary parameter update transform"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EggrollAutoPopulationConfig, EggrollPopulationExecutionConfig, OptimizerScheduleMode,
    };
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray<f32>>;

    #[derive(Module, Debug)]
    struct TestModule<B: BackendTrait> {
        weight: Param<Tensor<B, 1>>,
    }

    fn base_optimizer_config(transform: PredictiveCodingOptimizerTransform) -> OptimizerConfig {
        OptimizerConfig {
            name: OptimizerKind::PredictiveCoding,
            learning_rate: 1.0e-2,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: OptimizerScheduleMode::DragonReference,
            grad_clip_norm: Some(1.0),
            grad_clip_value: None,
            eggroll: burn_eggroll::EggrollConfig::default(),
            eggroll_population_execution: EggrollPopulationExecutionConfig::default(),
            eggroll_auto_population: EggrollAutoPopulationConfig::default(),
            predictive_coding: PredictiveCodingOptimizerConfig {
                transform,
                ..PredictiveCodingOptimizerConfig::default()
            },
        }
    }

    fn test_module() -> TestModule<TestBackend> {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestModule {
            weight: Param::from_tensor(Tensor::<TestBackend, 1>::ones([2], &device)),
        }
    }

    fn assert_retired_transform_rejected(transform: PredictiveCodingOptimizerTransform) {
        let config = base_optimizer_config(transform);
        let error = resolve_optimizer::<TestBackend, TestModule<TestBackend>>(&config, 1)
            .err()
            .expect("legacy PC optimizer must not resolve");
        assert!(
            error
                .to_string()
                .contains("training.algorithm=predictive_coding")
        );
    }

    #[test]
    fn predictive_coding_optimizer_transforms_are_retired() {
        assert_retired_transform_rejected(PredictiveCodingOptimizerTransform::Sgd);
        assert_retired_transform_rejected(PredictiveCodingOptimizerTransform::Momentum);
        assert_retired_transform_rejected(PredictiveCodingOptimizerTransform::Adamw);
        assert_retired_transform_rejected(PredictiveCodingOptimizerTransform::DiagonalNatural);
    }
}
