use std::path::Path;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use burn::data::dataloader::{DataLoaderIterator, Progress};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

use crate::train::prelude::*;
use crate::train::schedule::{
    ForwardEggrollTrainEnvironment, TrainEnvironment, train_with_eggroll_forward_only,
    train_with_scheduler,
};

pub type OptimizerDynamicsBackend = Autodiff<NdArray<f32>>;
pub type OptimizerDynamicsEggrollBackend = NdArray<f32>;
pub type OptimizerDynamicsValidBackend = ValidBackend<OptimizerDynamicsBackend>;

fn dynamics_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerDynamicsKind {
    AdamW,
    Eggroll,
}

#[derive(Debug, Clone)]
pub struct OptimizerDynamicsConfig {
    pub epochs: usize,
    pub max_iters: usize,
    pub log_frequency: usize,
    pub seed: u64,
    pub adamw_learning_rate: f64,
    pub eggroll_learning_rate: f64,
    pub eggroll_sigma: f32,
    pub eggroll_interval_steps: usize,
    pub eggroll_population_size: usize,
    pub eggroll_rank: usize,
    pub eggroll_coefficient_clip: Option<f32>,
    pub eggroll_fitness_normalization: burn_eggroll::FitnessNormalization,
    pub eggroll_update_kind: burn_eggroll::EggrollUpdateKind,
    pub eggroll_max_delta_rms: Option<f32>,
    pub eggroll_matrix_noise: burn_eggroll::MatrixNoiseMode,
}

impl Default for OptimizerDynamicsConfig {
    fn default() -> Self {
        Self {
            epochs: 128,
            max_iters: 512,
            log_frequency: 4,
            seed: 29,
            adamw_learning_rate: 2.5e-2,
            eggroll_learning_rate: 1.0e-1,
            eggroll_sigma: 1.0e-2,
            eggroll_interval_steps: 1,
            eggroll_population_size: 256,
            eggroll_rank: 4,
            eggroll_coefficient_clip: Some(128.0),
            eggroll_fitness_normalization: burn_eggroll::FitnessNormalization::Rank,
            eggroll_update_kind: burn_eggroll::EggrollUpdateKind::Sgd,
            eggroll_max_delta_rms: Some(1.0e-1),
            eggroll_matrix_noise: burn_eggroll::MatrixNoiseMode::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimizerDynamicsReport {
    pub optimizer: OptimizerDynamicsKind,
    pub seed: u64,
    pub initial_train_loss: f32,
    pub final_train_loss: f32,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub elapsed_ms: u128,
    pub train_steps: usize,
    pub forward_evaluations: usize,
}

impl OptimizerDynamicsReport {
    pub fn loss_delta(&self) -> f32 {
        self.initial_loss - self.final_loss
    }

    pub fn train_loss_delta(&self) -> f32 {
        self.initial_train_loss - self.final_train_loss
    }

    pub fn evaluations_per_second(&self) -> f64 {
        if self.elapsed_ms == 0 {
            return f64::INFINITY;
        }
        self.forward_evaluations as f64 / (self.elapsed_ms as f64 / 1000.0)
    }

    pub fn milliseconds_per_train_step(&self) -> f64 {
        if self.train_steps == 0 {
            return f64::INFINITY;
        }
        self.elapsed_ms as f64 / self.train_steps as f64
    }

    pub fn loss_delta_per_forward_evaluation(&self) -> f64 {
        if self.forward_evaluations == 0 {
            return 0.0;
        }
        self.loss_delta() as f64 / self.forward_evaluations as f64
    }

    pub fn loss_delta_per_second(&self) -> f64 {
        if self.elapsed_ms == 0 {
            return f64::INFINITY;
        }
        self.loss_delta() as f64 / (self.elapsed_ms as f64 / 1000.0)
    }
}

#[derive(Debug, Clone)]
pub struct OptimizerDynamicsPairReport {
    pub adamw: OptimizerDynamicsReport,
    pub eggroll: OptimizerDynamicsReport,
}

#[derive(Debug, Clone)]
pub struct OptimizerDynamicsSuiteReport {
    pub pairs: Vec<OptimizerDynamicsPairReport>,
}

impl OptimizerDynamicsSuiteReport {
    pub fn mean_adamw_loss_delta(&self) -> f32 {
        mean_by(&self.pairs, |pair| pair.adamw.loss_delta())
    }

    pub fn mean_eggroll_loss_delta(&self) -> f32 {
        mean_by(&self.pairs, |pair| pair.eggroll.loss_delta())
    }

    pub fn mean_adamw_train_loss_delta(&self) -> f32 {
        mean_by(&self.pairs, |pair| pair.adamw.train_loss_delta())
    }

    pub fn mean_eggroll_train_loss_delta(&self) -> f32 {
        mean_by(&self.pairs, |pair| pair.eggroll.train_loss_delta())
    }

    pub fn mean_adamw_ms_per_step(&self) -> f64 {
        mean_by_f64(&self.pairs, |pair| pair.adamw.milliseconds_per_train_step())
    }

    pub fn mean_eggroll_ms_per_step(&self) -> f64 {
        mean_by_f64(&self.pairs, |pair| {
            pair.eggroll.milliseconds_per_train_step()
        })
    }
}

#[derive(Debug, Clone)]
pub struct EggrollDynamicsCandidate {
    pub name: String,
    pub config: OptimizerDynamicsConfig,
}

impl EggrollDynamicsCandidate {
    pub fn new(name: impl Into<String>, config: OptimizerDynamicsConfig) -> Self {
        Self {
            name: name.into(),
            config,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EggrollDynamicsCandidateReport {
    pub name: String,
    pub report: OptimizerDynamicsReport,
}

#[derive(Debug, Clone)]
pub struct EggrollDynamicsSweepReport {
    pub candidates: Vec<EggrollDynamicsCandidateReport>,
}

impl EggrollDynamicsSweepReport {
    pub fn get(&self, name: &str) -> Option<&EggrollDynamicsCandidateReport> {
        self.candidates
            .iter()
            .find(|candidate| candidate.name == name)
    }

    pub fn best_by_loss_delta(&self) -> Option<&EggrollDynamicsCandidateReport> {
        self.candidates.iter().max_by(|left, right| {
            left.report
                .loss_delta()
                .total_cmp(&right.report.loss_delta())
        })
    }

    pub fn best_by_loss_delta_per_second(&self) -> Option<&EggrollDynamicsCandidateReport> {
        self.candidates.iter().max_by(|left, right| {
            left.report
                .loss_delta_per_second()
                .total_cmp(&right.report.loss_delta_per_second())
        })
    }

    pub fn best_by_train_loss_delta(&self) -> Option<&EggrollDynamicsCandidateReport> {
        self.candidates.iter().max_by(|left, right| {
            left.report
                .train_loss_delta()
                .total_cmp(&right.report.train_loss_delta())
        })
    }
}

fn mean_by<T>(items: &[T], f: impl Fn(&T) -> f32) -> f32 {
    if items.is_empty() {
        return 0.0;
    }
    items.iter().map(f).sum::<f32>() / items.len() as f32
}

fn mean_by_f64<T>(items: &[T], f: impl Fn(&T) -> f64) -> f64 {
    if items.is_empty() {
        return 0.0;
    }
    items.iter().map(f).sum::<f64>() / items.len() as f64
}

#[derive(Clone)]
struct StaticSequenceLoader<B: BackendTrait> {
    items: Vec<SequenceBatch<B>>,
}

impl<B: BackendTrait> StaticSequenceLoader<B> {
    fn new(items: Vec<SequenceBatch<B>>) -> Self {
        Self { items }
    }
}

struct StaticSequenceIterator<B: BackendTrait> {
    items: Vec<SequenceBatch<B>>,
    index: usize,
}

impl<B: BackendTrait> Iterator for StaticSequenceIterator<B> {
    type Item = SequenceBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.items.get(self.index).cloned();
        if item.is_some() {
            self.index += 1;
        }
        item
    }
}

impl<B: BackendTrait> DataLoaderIterator<SequenceBatch<B>> for StaticSequenceIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(self.index, self.items.len())
    }
}

impl<B> DataLoader<B, SequenceBatch<B>> for StaticSequenceLoader<B>
where
    B: BackendTrait + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<SequenceBatch<B>> + 'a> {
        Box::new(StaticSequenceIterator {
            items: self.items.clone(),
            index: 0,
        })
    }

    fn num_items(&self) -> usize {
        self.items.len()
    }

    fn to_device(&self, _device: &B::Device) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        Arc::new(self.clone())
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        let len = self.items.len();
        let start = start.min(len);
        let end = end.min(len);
        Arc::new(Self {
            items: self.items[start..end].to_vec(),
        })
    }
}

fn make_batch<B: BackendTrait>(
    device: &B::Device,
    inputs: &[i64],
    targets: &[i64],
    shape: [usize; 2],
) -> SequenceBatch<B> {
    SequenceBatch::new(
        Tensor::<B, 2, Int>::from_data(TensorData::new(inputs.to_vec(), shape), device),
        Tensor::<B, 2, Int>::from_data(TensorData::new(targets.to_vec(), shape), device),
        None,
    )
}

fn tiny_model_config() -> DragonConfig {
    DragonConfig {
        n_layer: 1,
        n_embd: 8,
        n_head: 1,
        mlp_internal_dim_multiplier: 1,
        dropout: 0.0,
        vocab_size: 16,
        ..Default::default()
    }
}

fn tiny_training_hparams(config: &OptimizerDynamicsConfig) -> TrainingHyperparameters {
    TrainingHyperparameters {
        algorithm: TrainingAlgorithm::Auto,
        block_size: 4,
        tbptt_chunk_size: None,
        tbptt_persist_across_steps: false,
        sequence_batching: Default::default(),
        retain_ephemeral_terminal_sequence_state: false,
        min_logical_block_size: None,
        batch_size: 2,
        seed: 1337,
        gradient_accumulation_steps: 1,
        target_effective_batch_size: None,
        epochs: Some(config.epochs),
        max_iters: config.max_iters,
        checkpoint_interval_iters: 2000,
        log_frequency: config.log_frequency,
        launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
        resume_run_dir: None,
        resume_checkpoint_epoch: None,
        init_checkpoint_path: None,
        init_checkpoint_epoch: None,
        source_selection_state_path: None,
        init_transfer: Default::default(),
        continual_backprop: Default::default(),
        input_corruption: Default::default(),
        logit_entropy_floor: Default::default(),
        repeat_unlikelihood: Default::default(),
        greedy_rollout_unlikelihood: Default::default(),
        dynamics_anchor: Default::default(),
        predictive_coding: Default::default(),
        local_predictive_coding: Default::default(),
        predictive_context_routing: Default::default(),
        latent_reasoning: Default::default(),
        ruliad_supervision: Default::default(),
        ruliad_probe_generation: Default::default(),
        ruliad_policy_probe: Default::default(),
        module_lr_scales: Vec::new(),
        context_strategy: ContextStrategyConfig::Infinite,
        sequence_kernel_override: None,
        objective: Default::default(),
        gdpo: None,
        events: Default::default(),
        validation: Default::default(),
        sequence_state_probe: Default::default(),
        gates: Default::default(),
        dynamics: Default::default(),
        neuron_scaling: Default::default(),
        auto_batch_size: Default::default(),
    }
}

fn train_batches<B: BackendTrait>(device: &B::Device) -> Vec<SequenceBatch<B>> {
    vec![
        make_batch::<B>(
            device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 8],
            [2, 4],
        ),
        make_batch::<B>(
            device,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[2, 3, 4, 5, 6, 7, 8, 9],
            [2, 4],
        ),
        make_batch::<B>(
            device,
            &[2, 3, 4, 5, 6, 7, 8, 9],
            &[3, 4, 5, 6, 7, 8, 9, 10],
            [2, 4],
        ),
        make_batch::<B>(
            device,
            &[3, 4, 5, 6, 7, 8, 9, 10],
            &[4, 5, 6, 7, 8, 9, 10, 11],
            [2, 4],
        ),
    ]
}

fn valid_batches<B: BackendTrait>(device: &B::Device) -> Vec<SequenceBatch<B>> {
    vec![make_batch::<B>(
        device,
        &[4, 5, 6, 7, 8, 9, 10, 11],
        &[5, 6, 7, 8, 9, 10, 11, 12],
        [2, 4],
    )]
}

fn probe_loss<B: BackendTrait>(model: &DragonModel<B>) -> f32 {
    let valid_device = burn::tensor::Device::<B>::default();
    let probe = make_batch::<B>(
        &valid_device,
        &[4, 5, 6, 7, 8, 9, 10, 11],
        &[5, 6, 7, 8, 9, 10, 11, 12],
        [2, 4],
    );
    language_model_loss::<B>(model.forward(probe.inputs), probe.targets)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0]
}

fn mean_loss_on_batches<B: BackendTrait>(
    model: &DragonModel<B>,
    batches: impl IntoIterator<Item = SequenceBatch<B>>,
) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for batch in batches {
        let loss = language_model_loss::<B>(model.forward(batch.inputs), batch.targets)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0];
        sum += loss;
        count += 1;
    }
    if count == 0 {
        return f32::NAN;
    }
    sum / count as f32
}

struct AdamwDynamicsEnvironment<'a> {
    run_dir: &'a Path,
    run_name: &'a str,
    training: &'a TrainingHyperparameters,
    model_config: &'a DragonConfig,
    parallel_runtime: &'a ParallelRuntime,
    parallel_config: &'a ParallelConfig,
    device: &'a burn::tensor::Device<OptimizerDynamicsBackend>,
    devices: &'a [burn::tensor::Device<OptimizerDynamicsBackend>],
}

fn build_env<'a>(
    environment: AdamwDynamicsEnvironment<'a>,
    train_batches: Vec<SequenceBatch<OptimizerDynamicsBackend>>,
    valid_batches: Vec<SequenceBatch<OptimizerDynamicsValidBackend>>,
) -> TrainEnvironment<'a, OptimizerDynamicsBackend> {
    let AdamwDynamicsEnvironment {
        run_dir,
        run_name,
        training,
        model_config,
        parallel_runtime,
        parallel_config,
        device,
        devices,
    } = environment;
    let total_steps = training
        .epochs
        .unwrap_or(1)
        .saturating_mul(train_batches.len());
    TrainEnvironment {
        parallel_runtime,
        parallel_config,
        run_dir,
        run_name,
        backend_name: "cpu",
        training,
        resume_checkpoint_epoch: None,
        model_config,
        device,
        devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
        valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: training.epochs.unwrap_or(1),
        total_steps,
        valid_steps: 1,
    }
}

fn build_forward_eggroll_env<'a>(
    run_dir: &'a Path,
    training: &'a TrainingHyperparameters,
    model_config: &'a DragonConfig,
    parallel_runtime: &'a ParallelRuntime,
    device: &'a burn::tensor::Device<OptimizerDynamicsEggrollBackend>,
    train_batches: Vec<SequenceBatch<OptimizerDynamicsEggrollBackend>>,
    valid_batches: Vec<SequenceBatch<OptimizerDynamicsEggrollBackend>>,
) -> ForwardEggrollTrainEnvironment<'a, OptimizerDynamicsEggrollBackend> {
    ForwardEggrollTrainEnvironment {
        parallel_runtime,
        run_dir,
        run_name: "optimizer-dynamics-eggroll",
        backend_name: "cpu",
        training,
        resume_checkpoint_epoch: None,
        model_config,
        device,
        train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
        valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        epochs: training.epochs.unwrap_or(1),
    }
}

fn base_optimizer_config(
    kind: OptimizerDynamicsKind,
    config: &OptimizerDynamicsConfig,
) -> OptimizerConfig {
    let (name, learning_rate, eggroll) = match kind {
        OptimizerDynamicsKind::AdamW => (
            OptimizerKind::Adamw,
            config.adamw_learning_rate,
            burn_eggroll::EggrollConfig::default(),
        ),
        OptimizerDynamicsKind::Eggroll => (
            OptimizerKind::Eggroll,
            config.eggroll_learning_rate,
            burn_eggroll::EggrollConfig {
                sigma: config.eggroll_sigma,
                interval_steps: config.eggroll_interval_steps,
                coefficient_clip: config.eggroll_coefficient_clip,
                fitness_normalization: config.eggroll_fitness_normalization,
                population: burn_eggroll::PopulationConfig {
                    population_size: config.eggroll_population_size,
                    population_chunk_size: config.eggroll_population_size,
                    rank: config.eggroll_rank,
                    seed: config.seed,
                    matrix_noise: config.eggroll_matrix_noise,
                },
                update: burn_eggroll::EggrollUpdateConfig {
                    kind: config.eggroll_update_kind,
                    max_delta_rms: config.eggroll_max_delta_rms,
                    ..burn_eggroll::EggrollUpdateConfig::default()
                },
                ..burn_eggroll::EggrollConfig::default()
            },
        ),
    };
    OptimizerConfig {
        name,
        learning_rate,
        weight_decay: 0.0,
        weight_decay_final: None,
        lr_schedule: None,
        schedule_mode: OptimizerScheduleMode::DragonReference,
        grad_clip_norm: None,
        grad_clip_value: None,
        eggroll,
        eggroll_population_execution: Default::default(),
        eggroll_auto_population: Default::default(),
        predictive_coding: Default::default(),
    }
}

pub fn run_optimizer_dynamics(
    kind: OptimizerDynamicsKind,
    config: &OptimizerDynamicsConfig,
    run_dir: &Path,
) -> Result<OptimizerDynamicsReport> {
    let _guard = dynamics_lock()
        .lock()
        .expect("optimizer dynamics lock poisoned");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime = resolve_parallel_runtime(&parallel_config)?;
    let training = tiny_training_hparams(config);
    let model_config = tiny_model_config();
    let optimizer_cfg = base_optimizer_config(kind, config);
    let start = burn_dragon_time::Instant::now();
    let (
        initial_train_loss,
        final_train_loss,
        initial_loss,
        final_loss,
        train_steps,
        forward_evaluations,
    ) = match kind {
        OptimizerDynamicsKind::AdamW => {
            let device = burn::tensor::Device::<OptimizerDynamicsBackend>::default();
            let valid_device = burn::tensor::Device::<OptimizerDynamicsValidBackend>::default();
            let devices = vec![device];
            OptimizerDynamicsBackend::seed(&device, config.seed);
            let base_model =
                DragonModel::<OptimizerDynamicsBackend>::new(model_config.clone(), &device);
            let initial_train_loss = mean_loss_on_batches(&base_model, train_batches(&device));
            let initial_loss = probe_loss(&base_model.clone().valid());
            let env = build_env(
                AdamwDynamicsEnvironment {
                    run_dir,
                    run_name: "optimizer-dynamics-adamw",
                    training: &training,
                    model_config: &model_config,
                    parallel_runtime: &parallel_runtime,
                    parallel_config: &parallel_config,
                    device: &device,
                    devices: &devices,
                },
                train_batches(&device),
                valid_batches(&valid_device),
            );
            let model = LanguageTrainModel::new(base_model);
            let optimizer = AdamWConfig::new()
                .with_weight_decay(0.0)
                .init::<OptimizerDynamicsBackend, LanguageTrainModel<OptimizerDynamicsBackend>>();
            let trained = train_with_scheduler(&env, model, optimizer, config.adamw_learning_rate)?;
            (
                initial_train_loss,
                mean_loss_on_batches(&trained, train_batches(&device)),
                initial_loss,
                probe_loss(&trained),
                env.total_steps,
                env.total_steps,
            )
        }
        OptimizerDynamicsKind::Eggroll => {
            let device = burn::tensor::Device::<OptimizerDynamicsEggrollBackend>::default();
            OptimizerDynamicsEggrollBackend::seed(&device, config.seed);
            let base_model =
                DragonModel::<OptimizerDynamicsEggrollBackend>::new(model_config.clone(), &device);
            let initial_train_loss = mean_loss_on_batches(&base_model, train_batches(&device));
            let initial_loss = probe_loss(&base_model);
            let train_items = train_batches(&device);
            let total_steps = training
                .epochs
                .unwrap_or(1)
                .saturating_mul(train_items.len());
            let env = build_forward_eggroll_env(
                run_dir,
                &training,
                &model_config,
                &parallel_runtime,
                &device,
                train_items,
                valid_batches(&device),
            );
            let model = LanguageTrainModel::new(base_model);
            let trained = train_with_eggroll_forward_only(&env, &optimizer_cfg, model)?;
            let interval = config.eggroll_interval_steps.max(1);
            let eggroll_steps = (0..total_steps)
                .filter(|step| step.is_multiple_of(interval))
                .count();
            (
                initial_train_loss,
                mean_loss_on_batches(&trained, train_batches(&device)),
                initial_loss,
                probe_loss(&trained),
                total_steps,
                eggroll_steps.saturating_mul(config.eggroll_population_size),
            )
        }
    };
    let elapsed_ms = start.elapsed().as_millis();
    Ok(OptimizerDynamicsReport {
        optimizer: kind,
        seed: config.seed,
        initial_train_loss,
        final_train_loss,
        initial_loss,
        final_loss,
        elapsed_ms,
        train_steps,
        forward_evaluations,
    })
}

pub fn run_manual_adamw_optimizer_dynamics(
    config: &OptimizerDynamicsConfig,
) -> Result<OptimizerDynamicsReport> {
    let _guard = dynamics_lock()
        .lock()
        .expect("optimizer dynamics lock poisoned");
    let device = burn::tensor::Device::<OptimizerDynamicsBackend>::default();
    let training = tiny_training_hparams(config);
    let model_config = tiny_model_config();
    let train_items = train_batches(&device);
    let total_steps = training
        .epochs
        .unwrap_or(1)
        .saturating_mul(train_items.len());

    OptimizerDynamicsBackend::seed(&device, config.seed);
    let base_model = DragonModel::<OptimizerDynamicsBackend>::new(model_config.clone(), &device);
    let initial_train_loss = mean_loss_on_batches(&base_model, train_batches(&device));
    let initial_loss = probe_loss(&base_model.clone().valid());
    let mut model = LanguageTrainModel::new(base_model);
    let optimizer_cfg = base_optimizer_config(OptimizerDynamicsKind::AdamW, config);
    let fresh_model = DragonModel::<OptimizerDynamicsBackend>::new(model_config, &device);
    let mut optimizer = crate::train::resolve_dragon_language_optimizer::<OptimizerDynamicsBackend>(
        &training,
        &optimizer_cfg,
        total_steps,
        fresh_model,
    )?;

    let start = burn_dragon_time::Instant::now();
    for _epoch in 0..training.epochs.unwrap_or(1) {
        for batch in train_items.iter().cloned() {
            let item = burn_train::TrainStep::step(&model, batch);
            let mut accumulator = GradientsAccumulator::new();
            accumulator.accumulate(&model, item.grads);
            let grads = accumulator.grads();
            model = optimizer.step(config.adamw_learning_rate, model, grads);
        }
    }
    let elapsed_ms = start.elapsed().as_millis();
    Ok(OptimizerDynamicsReport {
        optimizer: OptimizerDynamicsKind::AdamW,
        seed: config.seed,
        initial_train_loss,
        final_train_loss: mean_loss_on_batches(&model.model, train_batches(&device)),
        initial_loss,
        final_loss: probe_loss(&model.model.valid()),
        elapsed_ms,
        train_steps: total_steps,
        forward_evaluations: total_steps,
    })
}

pub fn run_optimizer_dynamics_pair(
    config: &OptimizerDynamicsConfig,
    run_root: &Path,
) -> Result<OptimizerDynamicsPairReport> {
    let adamw = run_optimizer_dynamics(
        OptimizerDynamicsKind::AdamW,
        config,
        &run_root.join(format!("seed-{}-adamw", config.seed)),
    )?;
    let eggroll = run_optimizer_dynamics(
        OptimizerDynamicsKind::Eggroll,
        config,
        &run_root.join(format!("seed-{}-eggroll", config.seed)),
    )?;
    Ok(OptimizerDynamicsPairReport { adamw, eggroll })
}

pub fn run_optimizer_dynamics_suite(
    config: &OptimizerDynamicsConfig,
    seeds: &[u64],
    run_root: &Path,
) -> Result<OptimizerDynamicsSuiteReport> {
    let mut pairs = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let mut config = config.clone();
        config.seed = *seed;
        pairs.push(run_optimizer_dynamics_pair(
            &config,
            &run_root.join(format!("seed-{seed}")),
        )?);
    }
    Ok(OptimizerDynamicsSuiteReport { pairs })
}

pub fn run_eggroll_dynamics_sweep(
    candidates: &[EggrollDynamicsCandidate],
    run_root: &Path,
) -> Result<EggrollDynamicsSweepReport> {
    let mut reports = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        reports.push(EggrollDynamicsCandidateReport {
            name: candidate.name.clone(),
            report: run_optimizer_dynamics(
                OptimizerDynamicsKind::Eggroll,
                &candidate.config,
                &run_root.join(&candidate.name),
            )?,
        });
    }
    Ok(EggrollDynamicsSweepReport {
        candidates: reports,
    })
}
