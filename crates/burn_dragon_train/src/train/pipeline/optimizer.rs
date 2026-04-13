use crate::train::prelude::*;
use burn::optim::MultiGradientsParams;

#[derive(Clone)]
pub enum ResolvedOptimizer<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    AdamW(OptimizerAdaptor<AdamW, M, B>),
}

#[derive(Record, Clone)]
pub struct ResolvedOptimizerRecord<M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    adamw: <OptimizerAdaptor<AdamW, M, B> as Optimizer<M, B>>::Record,
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
        }
    }

    fn step_multi(&mut self, lr: LearningRate, module: M, grads: MultiGradientsParams) -> M {
        match self {
            Self::AdamW(optimizer) => optimizer.step_multi(lr, module, grads),
        }
    }

    fn to_record(&self) -> Self::Record {
        match self {
            Self::AdamW(optimizer) => ResolvedOptimizerRecord {
                adamw: optimizer.to_record(),
            },
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        match self {
            Self::AdamW(optimizer) => Self::AdamW(optimizer.load_record(record.adamw)),
        }
    }
}

pub fn resolve_optimizer<B, M>(
    optimizer_cfg: &OptimizerConfig,
    _total_steps: usize,
) -> Result<ResolvedOptimizer<B, M>>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    Ok(ResolvedOptimizer::AdamW(
        adamw_config_from_optimizer(optimizer_cfg).init::<B, M>(),
    ))
}
