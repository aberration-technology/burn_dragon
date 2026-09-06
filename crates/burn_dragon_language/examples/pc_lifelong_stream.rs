#[cfg(feature = "train")]
use anyhow::{Context, Result, anyhow};
#[cfg(feature = "train")]
use burn::module::{AutodiffModule, Module};
#[cfg(feature = "train")]
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
#[cfg(feature = "train")]
use burn::tensor::backend::{AutodiffBackend, Backend};
#[cfg(feature = "train")]
use burn::tensor::{Tensor, TensorData};
#[cfg(feature = "train")]
use burn_autodiff::Autodiff;
#[cfg(feature = "train")]
use burn_dragon_core::{DragonConfig, DragonModel, RotaryEmbedding, SequenceTrainingExecutor};
#[cfg(feature = "train")]
use burn_dragon_language::train::{
    ContextRecurrenceBatch, ContextRecurrenceSpec, ContextRecurrenceTask,
    ContinualAcquisitionGateMetrics, ContinualAcquisitionThresholds, ContinualTaskAcquisition,
    ContinualTaskEvaluation, StreamingContextSelector, StreamingContextSelectorConfig,
    balanced_context_mask, context_recurrence_batch, context_recurrence_observation_tokens,
    evaluate_continual_acquisition_gate, local_predictive_coding_derivatives,
    local_predictive_coding_derivatives_with_subnetwork_masks,
};
#[cfg(feature = "train")]
use burn_dragon_language::{LocalPredictiveCodingConfig, LocalPredictiveCodingSolver};
#[cfg(feature = "train")]
use burn_ndarray::NdArray;
#[cfg(feature = "train")]
use burn_pc::{
    PredictiveContextBank, PredictiveContextBankConfig, PredictiveContextCandidate,
    PredictiveContextNoveltyGate,
};
#[cfg(feature = "train")]
use serde::Serialize;
#[cfg(feature = "train")]
use std::collections::BTreeMap;
#[cfg(feature = "train")]
use std::path::PathBuf;
#[cfg(feature = "train")]
use std::time::Instant;

#[cfg(feature = "train")]
const TRAIN_SPLIT_SEED: u64 = 0x6c69_6665_5f74_726e;
#[cfg(feature = "train")]
const HOLDOUT_SPLIT_SEED: u64 = 0x6c69_6665_5f76_616c;

#[cfg(feature = "train")]
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum LearningRule {
    Backpropagation,
    FixedPrediction,
    ErrorEquilibrium,
    ReverseGaussSeidel,
    LayerLocalPrediction,
}

#[cfg(feature = "train")]
impl LearningRule {
    fn solver(self) -> Option<LocalPredictiveCodingSolver> {
        match self {
            Self::Backpropagation => None,
            Self::FixedPrediction => Some(LocalPredictiveCodingSolver::FixedPrediction),
            Self::ErrorEquilibrium => Some(LocalPredictiveCodingSolver::ErrorEquilibrium),
            Self::ReverseGaussSeidel => Some(LocalPredictiveCodingSolver::ReverseGaussSeidel),
            Self::LayerLocalPrediction => Some(LocalPredictiveCodingSolver::LayerLocalPrediction),
        }
    }
}

#[cfg(feature = "train")]
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum RoutingMode {
    Dense,
    SelectedSparseSubnetwork,
}

#[cfg(feature = "train")]
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum OptimizerStateScope {
    Shared,
    ContextScoped,
}

#[cfg(feature = "train")]
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ContextSelectorMode {
    /// Causal next-token evidence under each learned context subnetwork. This
    /// is the task-ID-free default and makes no assumption about the stream's
    /// generating family.
    PredictiveEvidence,
    /// Family-aware recurrence descriptor retained as a controlled routing
    /// upper bound for comparison with prior reports.
    RecurrenceDescriptorControl,
}

#[cfg(feature = "train")]
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TrainingTopology {
    DenseShared,
    DenseContextScoped,
    SelectedSparseShared,
    SelectedSparseContextScoped,
}

#[cfg(feature = "train")]
impl TrainingTopology {
    const fn routing_mode(self) -> RoutingMode {
        match self {
            Self::DenseShared | Self::DenseContextScoped => RoutingMode::Dense,
            Self::SelectedSparseShared | Self::SelectedSparseContextScoped => {
                RoutingMode::SelectedSparseSubnetwork
            }
        }
    }

    const fn optimizer_state_scope(self) -> OptimizerStateScope {
        match self {
            Self::DenseShared | Self::SelectedSparseShared => OptimizerStateScope::Shared,
            Self::DenseContextScoped | Self::SelectedSparseContextScoped => {
                OptimizerStateScope::ContextScoped
            }
        }
    }
}

#[cfg(feature = "train")]
#[derive(Debug, Clone)]
struct Args {
    backend: String,
    rules: Vec<LearningRule>,
    topologies: Vec<TrainingTopology>,
    seeds: Vec<u64>,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    latent_total: usize,
    batch_size: usize,
    block_size: usize,
    payload_modulus: usize,
    updates_per_task: usize,
    eval_batches: usize,
    learning_rate: f64,
    pc_learning_rate: f64,
    pc_inference_steps: usize,
    pc_step_size: f32,
    pc_max_grad_norm: Option<f32>,
    pc_prediction_precision: f32,
    pc_energy_diagnostic_every: usize,
    active_fraction: f32,
    selector_mode: ContextSelectorMode,
    context_probe_every_updates: usize,
    context_novelty_confirmations: u64,
    predictive_context: PredictiveContextBankConfig,
    novelty_cosine_threshold: f32,
    centroid_update_rate: f32,
    loss_acquisition_floor_ratio: f64,
    acquisition_tolerance: f64,
    minimum_baseline_loss_reduction: f64,
    minimum_baseline_accuracy_gain: f64,
    output: Option<PathBuf>,
}

#[cfg(feature = "train")]
impl Default for Args {
    fn default() -> Self {
        Self {
            backend: "cpu".to_string(),
            rules: vec![
                LearningRule::Backpropagation,
                LearningRule::FixedPrediction,
                LearningRule::ReverseGaussSeidel,
            ],
            topologies: vec![
                TrainingTopology::DenseShared,
                TrainingTopology::SelectedSparseContextScoped,
            ],
            seeds: vec![17, 29, 43],
            n_layer: 4,
            n_embd: 32,
            n_head: 4,
            latent_total: 512,
            batch_size: 16,
            block_size: 16,
            payload_modulus: 16,
            updates_per_task: 256,
            eval_batches: 4,
            learning_rate: 3.0e-3,
            pc_learning_rate: 3.0e-3,
            pc_inference_steps: 4,
            pc_step_size: 0.05,
            pc_max_grad_norm: Some(1.0),
            pc_prediction_precision: 1.0,
            pc_energy_diagnostic_every: 32,
            active_fraction: 0.25,
            selector_mode: ContextSelectorMode::PredictiveEvidence,
            context_probe_every_updates: 8,
            context_novelty_confirmations: 3,
            predictive_context: PredictiveContextBankConfig {
                max_contexts: 8,
                calibration_update_rate: 0.5,
                novelty_standard_deviations: 3.0,
                ..PredictiveContextBankConfig::default()
            },
            novelty_cosine_threshold: 0.8,
            centroid_update_rate: 0.1,
            loss_acquisition_floor_ratio: 0.9,
            acquisition_tolerance: 0.05,
            minimum_baseline_loss_reduction: 0.5,
            minimum_baseline_accuracy_gain: 0.25,
            output: None,
        }
    }
}

#[cfg(feature = "train")]
#[derive(Debug, Clone, Serialize)]
struct TaskMatrixCell {
    phase: usize,
    task: ContextRecurrenceTask,
    evaluation: ContinualTaskEvaluation,
}

#[cfg(feature = "train")]
#[derive(Debug, Clone, Serialize)]
struct ContextDiscoveryEvent {
    phase: usize,
    task: ContextRecurrenceTask,
    context_index: usize,
    created: bool,
    reserve_loss: Option<f64>,
    reserve_supported_novelty: Option<bool>,
    candidates: Vec<PredictiveContextCandidate>,
}

#[cfg(feature = "train")]
#[derive(Debug, Clone, Serialize)]
struct StreamOutcome {
    seed: u64,
    rule: LearningRule,
    routing_mode: RoutingMode,
    optimizer_state_scope: OptimizerStateScope,
    selector_mode: ContextSelectorMode,
    pre_task: Vec<ContinualTaskEvaluation>,
    task_matrix: Vec<TaskMatrixCell>,
    final_average_accuracy: f64,
    backward_transfer: f64,
    mean_forgetting: f64,
    max_forgetting: f64,
    selector_accuracy: Option<f64>,
    selector_probes: usize,
    selector_committed_probes: usize,
    selector_deferred_probes: usize,
    selector_probe_tokens: usize,
    contexts_created: usize,
    context_discovery_complete: bool,
    context_discovery: Vec<ContextDiscoveryEvent>,
    mean_neuron_mask_overlap: Option<f64>,
    mean_activity_mask_overlap: Option<f64>,
    model_tokens: usize,
    elapsed_seconds: f64,
    model_tokens_per_second: f64,
    local_vjp_calls: usize,
    global_backward_calls: usize,
    energy_diagnostics: usize,
    energy_descent_fraction: Option<f64>,
    max_relative_energy_increase: Option<f64>,
    acquisition_gate: Option<ContinualAcquisitionGateMetrics>,
}

#[cfg(feature = "train")]
#[derive(Debug, Serialize)]
struct MatrixReport {
    schema_version: u32,
    backend: String,
    parameters: usize,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    latent_total: usize,
    recurrence: ContextRecurrenceSpec,
    tasks: Vec<ContextRecurrenceTask>,
    topologies: Vec<TrainingTopology>,
    updates_per_task: usize,
    eval_batches: usize,
    learning_rate: f64,
    pc_learning_rate: f64,
    pc_inference_steps: usize,
    pc_step_size: f32,
    pc_max_grad_norm: Option<f32>,
    pc_prediction_precision: f32,
    pc_energy_diagnostic_every: usize,
    active_fraction: f32,
    selector_mode: ContextSelectorMode,
    context_probe_every_updates: usize,
    context_novelty_confirmations: u64,
    predictive_context: PredictiveContextBankConfig,
    descriptor_control: StreamingContextSelectorConfig,
    loss_acquisition_floor_ratio: f64,
    acquisition_tolerance: f64,
    minimum_baseline_loss_reduction: f64,
    minimum_baseline_accuracy_gain: f64,
    outcomes: Vec<StreamOutcome>,
}

#[cfg(feature = "train")]
fn parse_csv(value: &str) -> Vec<&str> {
    value.split(',').filter(|part| !part.is_empty()).collect()
}

#[cfg(feature = "train")]
fn next_value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    args.next()
        .ok_or_else(|| anyhow!("{name} requires a value"))?
        .parse::<T>()
        .map_err(|error| anyhow!("invalid {name}: {error}"))
}

#[cfg(feature = "train")]
fn parse_args() -> Result<Args> {
    let mut parsed = Args::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => parsed.backend = next_value(&mut args, "--backend")?,
            "--rules" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--rules requires a value"))?;
                parsed.rules = parse_csv(&value)
                    .into_iter()
                    .map(|part| match part {
                        "backpropagation" | "adamw" => Ok(LearningRule::Backpropagation),
                        "fixed_prediction" => Ok(LearningRule::FixedPrediction),
                        "error_equilibrium" | "epc" => Ok(LearningRule::ErrorEquilibrium),
                        "reverse_gauss_seidel" => Ok(LearningRule::ReverseGaussSeidel),
                        "layer_local_prediction" => Ok(LearningRule::LayerLocalPrediction),
                        _ => Err(anyhow!("unsupported learning rule {part:?}")),
                    })
                    .collect::<Result<_>>()?;
            }
            "--topologies" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--topologies requires a value"))?;
                parsed.topologies = parse_csv(&value)
                    .into_iter()
                    .map(|part| match part {
                        "dense_shared" => Ok(TrainingTopology::DenseShared),
                        "dense_context_scoped" => Ok(TrainingTopology::DenseContextScoped),
                        "selected_sparse_shared" => Ok(TrainingTopology::SelectedSparseShared),
                        "selected_sparse_context_scoped" => {
                            Ok(TrainingTopology::SelectedSparseContextScoped)
                        }
                        _ => Err(anyhow!("unsupported training topology {part:?}")),
                    })
                    .collect::<Result<_>>()?;
            }
            "--seeds" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--seeds requires a value"))?;
                parsed.seeds = parse_csv(&value)
                    .into_iter()
                    .map(|part| part.parse().map_err(anyhow::Error::msg))
                    .collect::<Result<_>>()?;
            }
            "--n-layer" => parsed.n_layer = next_value(&mut args, "--n-layer")?,
            "--n-embd" => parsed.n_embd = next_value(&mut args, "--n-embd")?,
            "--n-head" => parsed.n_head = next_value(&mut args, "--n-head")?,
            "--latent-total" => parsed.latent_total = next_value(&mut args, "--latent-total")?,
            "--batch-size" => parsed.batch_size = next_value(&mut args, "--batch-size")?,
            "--block-size" => parsed.block_size = next_value(&mut args, "--block-size")?,
            "--payload-modulus" => {
                parsed.payload_modulus = next_value(&mut args, "--payload-modulus")?
            }
            "--updates-per-task" => {
                parsed.updates_per_task = next_value(&mut args, "--updates-per-task")?
            }
            "--eval-batches" => parsed.eval_batches = next_value(&mut args, "--eval-batches")?,
            "--learning-rate" => parsed.learning_rate = next_value(&mut args, "--learning-rate")?,
            "--pc-learning-rate" => {
                parsed.pc_learning_rate = next_value(&mut args, "--pc-learning-rate")?
            }
            "--pc-inference-steps" => {
                parsed.pc_inference_steps = next_value(&mut args, "--pc-inference-steps")?
            }
            "--pc-step-size" => parsed.pc_step_size = next_value(&mut args, "--pc-step-size")?,
            "--pc-max-grad-norm" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--pc-max-grad-norm requires a value"))?;
                parsed.pc_max_grad_norm = if value == "none" {
                    None
                } else {
                    Some(value.parse().map_err(anyhow::Error::msg)?)
                };
            }
            "--pc-prediction-precision" => {
                parsed.pc_prediction_precision = next_value(&mut args, "--pc-prediction-precision")?
            }
            "--pc-energy-diagnostic-every" => {
                parsed.pc_energy_diagnostic_every =
                    next_value(&mut args, "--pc-energy-diagnostic-every")?
            }
            "--active-fraction" => {
                parsed.active_fraction = next_value(&mut args, "--active-fraction")?
            }
            "--selector" => {
                parsed.selector_mode = match args
                    .next()
                    .ok_or_else(|| anyhow!("--selector requires a value"))?
                    .as_str()
                {
                    "predictive_evidence" => ContextSelectorMode::PredictiveEvidence,
                    "recurrence_descriptor_control" => {
                        ContextSelectorMode::RecurrenceDescriptorControl
                    }
                    other => return Err(anyhow!("unsupported context selector {other:?}")),
                }
            }
            "--context-probe-every-updates" => {
                parsed.context_probe_every_updates =
                    next_value(&mut args, "--context-probe-every-updates")?
            }
            "--context-novelty-confirmations" => {
                parsed.context_novelty_confirmations =
                    next_value(&mut args, "--context-novelty-confirmations")?
            }
            "--context-min-observations" => {
                parsed.predictive_context.minimum_observations =
                    next_value(&mut args, "--context-min-observations")?
            }
            "--context-calibration-rate" => {
                parsed.predictive_context.calibration_update_rate =
                    next_value(&mut args, "--context-calibration-rate")?
            }
            "--context-novelty-z" => {
                parsed.predictive_context.novelty_standard_deviations =
                    next_value(&mut args, "--context-novelty-z")?
            }
            "--context-novelty-margin" => {
                parsed.predictive_context.novelty_absolute_margin =
                    next_value(&mut args, "--context-novelty-margin")?
            }
            "--novelty-cosine-threshold" => {
                parsed.novelty_cosine_threshold =
                    next_value(&mut args, "--novelty-cosine-threshold")?
            }
            "--centroid-update-rate" => {
                parsed.centroid_update_rate = next_value(&mut args, "--centroid-update-rate")?
            }
            "--loss-acquisition-floor-ratio" => {
                parsed.loss_acquisition_floor_ratio =
                    next_value(&mut args, "--loss-acquisition-floor-ratio")?
            }
            "--acquisition-tolerance" => {
                parsed.acquisition_tolerance = next_value(&mut args, "--acquisition-tolerance")?
            }
            "--minimum-baseline-loss-reduction" => {
                parsed.minimum_baseline_loss_reduction =
                    next_value(&mut args, "--minimum-baseline-loss-reduction")?
            }
            "--minimum-baseline-accuracy-gain" => {
                parsed.minimum_baseline_accuracy_gain =
                    next_value(&mut args, "--minimum-baseline-accuracy-gain")?
            }
            "--output" => {
                parsed.output = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("--output requires a value"))?,
                ));
            }
            "--help" | "-h" => {
                println!(
                    "pc_lifelong_stream: --backend cpu|cuda --rules backpropagation,fixed_prediction,error_equilibrium,reverse_gauss_seidel,layer_local_prediction --topologies dense_shared,selected_sparse_context_scoped --selector predictive_evidence|recurrence_descriptor_control --seeds 17,29,43 [model, PC, selector, and output options]"
                );
                std::process::exit(0);
            }
            _ => return Err(anyhow!("unknown argument {arg:?}")),
        }
    }
    if parsed.rules.is_empty()
        || !parsed.rules.contains(&LearningRule::Backpropagation)
        || parsed.topologies.is_empty()
        || parsed.seeds.is_empty()
    {
        return Err(anyhow!(
            "non-empty matrices require a backpropagation acquisition control"
        ));
    }
    if parsed.n_layer == 0
        || parsed.n_embd < 4
        || parsed.n_head == 0
        || parsed.latent_total == 0
        || parsed.context_probe_every_updates == 0
        || parsed.context_novelty_confirmations == 0
        || parsed.updates_per_task == 0
        || parsed.eval_batches == 0
        || !parsed.latent_total.is_multiple_of(parsed.n_embd)
        || !parsed.latent_total.is_multiple_of(parsed.n_head)
    {
        return Err(anyhow!("invalid model or experiment dimensions"));
    }
    if !(0.0..=1.0).contains(&parsed.active_fraction) || parsed.active_fraction == 0.0 {
        return Err(anyhow!("--active-fraction must be in (0, 1]"));
    }
    if parsed.learning_rate <= 0.0
        || parsed.pc_learning_rate <= 0.0
        || parsed.pc_prediction_precision <= 0.0
    {
        return Err(anyhow!("learning rates and PC precision must be > 0"));
    }
    if !(0.0..=1.0).contains(&parsed.loss_acquisition_floor_ratio)
        || parsed.acquisition_tolerance < 0.0
        || parsed.minimum_baseline_loss_reduction < 0.0
        || parsed.minimum_baseline_accuracy_gain < 0.0
        || !parsed.loss_acquisition_floor_ratio.is_finite()
        || !parsed.acquisition_tolerance.is_finite()
        || !parsed.minimum_baseline_loss_reduction.is_finite()
        || !parsed.minimum_baseline_accuracy_gain.is_finite()
    {
        return Err(anyhow!(
            "acquisition thresholds must be finite and non-negative, with --loss-acquisition-floor-ratio in [0, 1]"
        ));
    }
    let recurrence = ContextRecurrenceSpec {
        batch_size: parsed.batch_size,
        block_size: parsed.block_size,
        payload_modulus: parsed.payload_modulus,
    };
    recurrence.validate().map_err(anyhow::Error::msg)?;
    StreamingContextSelectorConfig {
        novelty_cosine_threshold: parsed.novelty_cosine_threshold,
        centroid_update_rate: parsed.centroid_update_rate,
    }
    .validate()
    .map_err(anyhow::Error::msg)?;
    parsed
        .predictive_context
        .validate()
        .map_err(anyhow::Error::msg)?;
    Ok(parsed)
}

#[cfg(feature = "train")]
#[derive(Clone)]
struct ContextMasks<B: Backend> {
    neuron: Option<Tensor<B, 4>>,
    activity: Option<Tensor<B, 4>>,
}

#[cfg(feature = "train")]
struct SparseMaskBank<B: Backend> {
    mode: RoutingMode,
    seed: u64,
    active_fraction: f32,
    neuron_width: usize,
    activity_width: usize,
    masks: Vec<ContextMasks<B>>,
    neuron_values: Vec<Vec<f32>>,
    activity_values: Vec<Vec<f32>>,
}

#[cfg(feature = "train")]
impl<B: Backend> SparseMaskBank<B> {
    fn new(
        mode: RoutingMode,
        seed: u64,
        active_fraction: f32,
        neuron_width: usize,
        activity_width: usize,
    ) -> Self {
        Self {
            mode,
            seed,
            active_fraction,
            neuron_width,
            activity_width,
            masks: Vec::new(),
            neuron_values: Vec::new(),
            activity_values: Vec::new(),
        }
    }

    fn get(&mut self, context: usize, device: &B::Device) -> ContextMasks<B> {
        if self.mode == RoutingMode::Dense {
            return ContextMasks {
                neuron: None,
                activity: None,
            };
        }
        while self.masks.len() <= context {
            let index = self.masks.len();
            let neuron = balanced_context_mask(
                self.seed ^ 0x6e65_7572_6f6e_5f6d,
                index,
                self.neuron_width,
                self.active_fraction,
                &self.neuron_values,
            )
            .expect("sequential neuron context mask");
            let activity = balanced_context_mask(
                self.seed ^ 0x6163_7469_7665_5f6d,
                index,
                self.activity_width,
                self.active_fraction,
                &self.activity_values,
            )
            .expect("sequential activity context mask");
            self.masks.push(ContextMasks {
                neuron: Some(Tensor::from_data(
                    TensorData::new(neuron.clone(), [1, 1, 1, self.neuron_width]),
                    device,
                )),
                activity: Some(Tensor::from_data(
                    TensorData::new(activity.clone(), [1, 1, 1, self.activity_width]),
                    device,
                )),
            });
            self.neuron_values.push(neuron);
            self.activity_values.push(activity);
        }
        self.masks[context].clone()
    }

    fn overlap(&self) -> (Option<f64>, Option<f64>) {
        if self.mode == RoutingMode::Dense {
            return (None, None);
        }
        (
            mean_pairwise_overlap(&self.neuron_values),
            mean_pairwise_overlap(&self.activity_values),
        )
    }
}

#[cfg(feature = "train")]
fn mean_pairwise_overlap(masks: &[Vec<f32>]) -> Option<f64> {
    let mut overlaps = Vec::new();
    for left in 0..masks.len() {
        for right in left + 1..masks.len() {
            let intersection = masks[left]
                .iter()
                .zip(&masks[right])
                .filter(|(left, right)| **left > 0.0 && **right > 0.0)
                .count();
            let active = masks[left].iter().filter(|value| **value > 0.0).count();
            overlaps.push(intersection as f64 / active.max(1) as f64);
        }
    }
    (!overlaps.is_empty()).then(|| overlaps.iter().sum::<f64>() / overlaps.len() as f64)
}

#[cfg(feature = "train")]
enum ContextRouter {
    Predictive {
        bank: PredictiveContextBank,
        novelty_gate: PredictiveContextNoveltyGate,
    },
    DescriptorControl(StreamingContextSelector),
}

#[cfg(feature = "train")]
impl ContextRouter {
    fn new(args: &Args) -> Result<Self> {
        match args.selector_mode {
            ContextSelectorMode::PredictiveEvidence => Ok(Self::Predictive {
                bank: PredictiveContextBank::new(args.predictive_context)
                    .map_err(anyhow::Error::msg)?,
                novelty_gate: PredictiveContextNoveltyGate::new(args.context_novelty_confirmations)
                    .map_err(anyhow::Error::msg)?,
            }),
            ContextSelectorMode::RecurrenceDescriptorControl => Ok(Self::DescriptorControl(
                StreamingContextSelector::new(StreamingContextSelectorConfig {
                    novelty_cosine_threshold: args.novelty_cosine_threshold,
                    centroid_update_rate: args.centroid_update_rate,
                })
                .map_err(anyhow::Error::msg)?,
            )),
        }
    }

    fn known_contexts(&self) -> usize {
        match self {
            Self::Predictive { bank, .. } => bank.known_contexts(),
            Self::DescriptorControl(selector) => selector.known_contexts(),
        }
    }
}

#[cfg(feature = "train")]
#[derive(Debug, Clone)]
struct RoutedContext {
    context_index: usize,
    created: bool,
    novelty_deferred: bool,
    probe_tokens: usize,
    reserve_loss: Option<f64>,
    reserve_supported_novelty: Option<bool>,
    candidates: Vec<PredictiveContextCandidate>,
}

#[cfg(feature = "train")]
fn causal_prefix_loss<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, burn::tensor::Int>,
    spec: ContextRecurrenceSpec,
    masks: ContextMasks<B>,
) -> Tensor<B::InnerBackend, 1>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let observed = context_recurrence_observation_tokens(spec.block_size);
    let probe_time = observed.saturating_sub(1);
    let probe_inputs = inputs
        .clone()
        .slice([0..spec.batch_size, 0..probe_time])
        .inner();
    let probe_targets = inputs.slice([0..spec.batch_size, 1..observed]).inner();
    let plain = model.valid();
    let logits = match (masks.neuron, masks.activity) {
        (Some(neuron), Some(activity)) => plain
            .predictive_coding_forward_with_subnetwork_masks(
                probe_inputs,
                neuron.inner(),
                activity.inner(),
            )
            .expect("validated predictive context masks"),
        (None, None) => plain.forward(probe_inputs),
        _ => unreachable!("complete subnetwork routing requires both masks"),
    };
    plain
        .language_token_losses_from_logits(logits, probe_targets)
        .mean()
        .reshape([1])
}

#[cfg(feature = "train")]
fn read_context_losses<B: Backend>(losses: Vec<Tensor<B, 1>>) -> Result<Vec<f64>> {
    if losses.is_empty() {
        return Ok(Vec::new());
    }
    Tensor::cat(losses, 0)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map(|values| values.into_iter().map(f64::from).collect())
        .map_err(|error| anyhow!("read predictive context losses: {error}"))
}

#[cfg(feature = "train")]
fn route_context<B: AutodiffBackend>(
    router: &mut ContextRouter,
    model: &DragonModel<B>,
    batch: &ContextRecurrenceBatch<B>,
    spec: ContextRecurrenceSpec,
    mask_bank: &mut SparseMaskBank<B>,
    allow_create: bool,
    observe: bool,
) -> Result<RoutedContext>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    match router {
        ContextRouter::DescriptorControl(selector) => {
            let selection = selector
                .select(&batch.stream_descriptor, allow_create)
                .map_err(anyhow::Error::msg)?;
            Ok(RoutedContext {
                context_index: selection.context_index,
                created: selection.created,
                novelty_deferred: false,
                probe_tokens: 0,
                reserve_loss: None,
                reserve_supported_novelty: None,
                candidates: Vec::new(),
            })
        }
        ContextRouter::Predictive { bank, novelty_gate } => {
            let known = bank.known_contexts();
            let device = model.devices()[0].clone();
            let mut loss_tensors = Vec::with_capacity(known);
            for context_index in 0..known {
                loss_tensors.push(causal_prefix_loss(
                    model,
                    batch.inputs.clone(),
                    spec,
                    mask_bank.get(context_index, &device),
                ));
            }
            let mut losses = read_context_losses(loss_tensors)?;
            let mut probe_evaluations = losses.len();
            let mut selection = bank
                .select(&losses, allow_create)
                .map_err(anyhow::Error::msg)?;
            let mut novelty_deferred = false;
            let mut reserve_loss = None;
            let mut reserve_supported_novelty = None;
            if selection.created && known > 0 {
                if selection.replacement.is_none() && known < bank.config().max_contexts {
                    let loss = read_context_losses(vec![causal_prefix_loss(
                        model,
                        batch.inputs.clone(),
                        spec,
                        mask_bank.get(known, &device),
                    )])?[0];
                    let supported = bank
                        .reserve_supports_novelty(&selection, loss)
                        .map_err(anyhow::Error::msg)?;
                    reserve_loss = Some(loss);
                    reserve_supported_novelty = Some(supported);
                    selection.novel_evidence &= supported;
                    losses.push(loss);
                    probe_evaluations += 1;
                }
                let confirmed = observe && novelty_gate.observe(selection.novel_evidence);
                if !confirmed {
                    let fallback = selection
                        .candidates
                        .iter()
                        .min_by(|left, right| left.loss.total_cmp(&right.loss))
                        .expect("known predictive contexts have candidates");
                    selection.context_index = fallback.context_index;
                    selection.created = false;
                    novelty_deferred = selection.novel_evidence;
                }
            } else if selection.created {
                novelty_gate.reset();
            } else if observe {
                novelty_gate.observe(selection.novel_evidence);
            }
            let selected_loss = if selection.created {
                let created = bank.create().map_err(anyhow::Error::msg)?;
                if created != selection.context_index {
                    return Err(anyhow!(
                        "predictive context allocation drifted: selected={} created={created}",
                        selection.context_index
                    ));
                }
                if let Some(loss) = reserve_loss {
                    loss
                } else {
                    let loss = read_context_losses(vec![causal_prefix_loss(
                        model,
                        batch.inputs.clone(),
                        spec,
                        mask_bank.get(created, &device),
                    )])?[0];
                    losses.push(loss);
                    probe_evaluations += 1;
                    loss
                }
            } else {
                losses[selection.context_index]
            };
            if observe && (!selection.novel_evidence || selection.created) {
                bank.observe(selection.context_index, selected_loss)
                    .map_err(anyhow::Error::msg)?;
            }
            let probe_time = context_recurrence_observation_tokens(spec.block_size) - 1;
            Ok(RoutedContext {
                context_index: selection.context_index,
                created: selection.created,
                novelty_deferred,
                probe_tokens: probe_evaluations * spec.batch_size * probe_time,
                reserve_loss,
                reserve_supported_novelty,
                candidates: selection.candidates,
            })
        }
    }
}

#[cfg(feature = "train")]
fn pc_config(
    rule: LearningRule,
    args: &Args,
    diagnostics: bool,
) -> Option<LocalPredictiveCodingConfig> {
    let solver = rule.solver()?;
    let mut config = LocalPredictiveCodingConfig {
        solver,
        sync_diagnostics: diagnostics,
        prediction_precision: args.pc_prediction_precision,
        ..LocalPredictiveCodingConfig::default()
    };
    config.inference.steps = args.pc_inference_steps;
    config.inference.step_size = args.pc_step_size;
    config.inference.max_grad_norm = args.pc_max_grad_norm;
    if matches!(rule, LearningRule::LayerLocalPrediction) {
        config.factor_reduction = burn_dragon_language::PredictiveCodingFactorReduction::Mean;
        config.sync_diagnostics = false;
    }
    Some(config)
}

#[cfg(feature = "train")]
fn forward<B: AutodiffBackend>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, burn::tensor::Int>,
    masks: ContextMasks<B>,
) -> Tensor<B, 3>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    match (masks.neuron, masks.activity) {
        (Some(neuron), Some(activity)) => model
            .predictive_coding_forward_with_subnetwork_masks(inputs, neuron, activity)
            .expect("validated learned sparse masks"),
        (None, None) => model.forward(inputs),
        _ => unreachable!("complete subnetwork routing requires both masks"),
    }
}

#[cfg(feature = "train")]
struct EvaluationContext<'a, B: AutodiffBackend> {
    seed: u64,
    phase: usize,
    spec: ContextRecurrenceSpec,
    eval_batches: usize,
    routing_mode: RoutingMode,
    router: &'a mut ContextRouter,
    expected_contexts: &'a BTreeMap<usize, usize>,
    mask_bank: &'a mut SparseMaskBank<B>,
}

#[cfg(feature = "train")]
fn evaluate<B: AutodiffBackend>(
    model: &DragonModel<B>,
    task: ContextRecurrenceTask,
    context: EvaluationContext<'_, B>,
) -> Result<(ContinualTaskEvaluation, usize, usize, usize)>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let EvaluationContext {
        seed,
        phase,
        spec,
        eval_batches,
        routing_mode,
        router,
        expected_contexts,
        mask_bank,
    } = context;
    let device = model.devices()[0].clone();
    let plain = model.valid();
    let mut loss_sum = 0.0;
    let mut correct_sum = 0.0;
    let mut observations = 0.0;
    let mut selector_correct = 0usize;
    let mut selector_total = 0usize;
    let mut selector_probe_tokens = 0usize;
    for batch_index in 0..eval_batches {
        let batch = context_recurrence_batch::<B>(
            task,
            HOLDOUT_SPLIT_SEED ^ seed,
            (phase * eval_batches + batch_index) as u64,
            spec,
            &device,
        )
        .map_err(anyhow::Error::msg)?;
        let context = match routing_mode {
            RoutingMode::Dense => 0,
            RoutingMode::SelectedSparseSubnetwork => {
                let selected = route_context(router, model, &batch, spec, mask_bank, false, false)?;
                selector_total += 1;
                selector_probe_tokens += selected.probe_tokens;
                selector_correct += usize::from(
                    expected_contexts.get(&task.index()) == Some(&selected.context_index),
                );
                selected.context_index
            }
        };
        let masks = mask_bank.get(context, &device);
        let logits = match (masks.neuron, masks.activity) {
            (Some(neuron), Some(activity)) => plain
                .predictive_coding_forward_with_subnetwork_masks(
                    batch.inputs.inner(),
                    neuron.inner(),
                    activity.inner(),
                )
                .expect("validated evaluation masks"),
            (None, None) => plain.forward(batch.inputs.inner()),
            _ => unreachable!("complete subnetwork routing requires both masks"),
        };
        let targets = batch.targets.inner();
        let loss_mask = batch.loss_mask.inner();
        let losses = plain.language_token_losses_from_logits(logits.clone(), targets.clone());
        let loss = burn_dragon_core::objective::masked_token_mean(losses, Some(loss_mask.clone()));
        let [batch_size, time, _] = logits.shape().dims::<3>();
        let correct = logits
            .argmax(2)
            .reshape([batch_size, time])
            .equal(targets)
            .float()
            * loss_mask.clone().float();
        loss_sum += f64::from(burn_pc::diagnostic_scalar_f32(loss));
        correct_sum += f64::from(burn_pc::diagnostic_scalar_f32(correct.sum().reshape([1])));
        observations += f64::from(burn_pc::diagnostic_scalar_f32(
            loss_mask.float().sum().reshape([1]),
        ));
    }
    Ok((
        ContinualTaskEvaluation {
            loss: loss_sum / eval_batches as f64,
            accuracy: correct_sum / observations.max(1.0),
        },
        selector_correct,
        selector_total,
        selector_probe_tokens,
    ))
}

#[cfg(feature = "train")]
fn summarize_task_matrix(task_matrix: &[TaskMatrixCell]) -> (f64, f64, f64, f64) {
    let tasks = ContextRecurrenceTask::ALL.len();
    let final_phase = tasks - 1;
    let final_values = (0..tasks)
        .map(|task| {
            task_matrix
                .iter()
                .find(|cell| cell.phase == final_phase && cell.task.index() == task)
                .expect("complete final task matrix")
                .evaluation
                .accuracy
        })
        .collect::<Vec<_>>();
    let final_average = final_values.iter().sum::<f64>() / tasks as f64;
    let mut backward_transfer = Vec::new();
    let mut forgetting = Vec::new();
    for (task, final_value) in final_values.iter().copied().take(tasks - 1).enumerate() {
        let learned = task_matrix
            .iter()
            .find(|cell| cell.phase == task && cell.task.index() == task)
            .expect("task acquisition diagonal")
            .evaluation
            .accuracy;
        backward_transfer.push(final_value - learned);
        let best = task_matrix
            .iter()
            .filter(|cell| cell.phase >= task && cell.task.index() == task)
            .map(|cell| cell.evaluation.accuracy)
            .fold(f64::NEG_INFINITY, f64::max);
        forgetting.push((best - final_value).max(0.0));
    }
    let bwt = backward_transfer.iter().sum::<f64>() / backward_transfer.len() as f64;
    let mean_forgetting = forgetting.iter().sum::<f64>() / forgetting.len() as f64;
    let max_forgetting = forgetting.into_iter().fold(0.0, f64::max);
    (final_average, bwt, mean_forgetting, max_forgetting)
}

#[cfg(feature = "train")]
fn run_arm<B, O, F>(
    args: &Args,
    model_config: &DragonConfig,
    seed: u64,
    rule: LearningRule,
    routing_mode: RoutingMode,
    optimizer_state_scope: OptimizerStateScope,
    make_optimizer: &F,
) -> Result<StreamOutcome>
where
    B: AutodiffBackend,
    B::Device: Default + 'static,
    B::FloatTensorPrimitive: 'static,
    O: Optimizer<DragonModel<B>, B>,
    F: Fn() -> O,
{
    let device = B::Device::default();
    B::seed(&device, seed);
    let mut model = DragonModel::<B>::new(model_config.clone(), &device);
    let spec = ContextRecurrenceSpec {
        batch_size: args.batch_size,
        block_size: args.block_size,
        payload_modulus: args.payload_modulus,
    };
    let mut router = ContextRouter::new(args)?;
    let mut expected_contexts = BTreeMap::new();
    let mut mask_bank = SparseMaskBank::<B>::new(
        routing_mode,
        seed,
        args.active_fraction,
        args.latent_total / args.n_head,
        args.n_embd,
    );
    let mut shared_optimizer = make_optimizer();
    let mut context_optimizers = Vec::new();
    let mut pre_task = Vec::new();
    let mut task_matrix = Vec::new();
    let mut context_discovery = Vec::new();
    let mut selector_correct = 0usize;
    let mut selector_total = 0usize;
    let mut selector_committed_total = 0usize;
    let mut selector_deferred = 0usize;
    let mut selector_probe_tokens = 0usize;
    let mut local_vjp_calls = 0usize;
    let mut global_backward_calls = 0usize;
    let mut energy_diagnostics = 0usize;
    let mut energy_descents = 0usize;
    let mut max_relative_energy_increase = 0.0_f64;
    let mut training_elapsed_seconds = 0.0_f64;

    for (phase, task) in ContextRecurrenceTask::ALL.into_iter().enumerate() {
        let context = match routing_mode {
            RoutingMode::Dense => 0,
            RoutingMode::SelectedSparseSubnetwork => {
                let mut final_selection = None;
                let mut bootstrap_contexts = Vec::new();
                for bootstrap_index in 0..args.context_novelty_confirmations {
                    let bootstrap = context_recurrence_batch::<B>(
                        task,
                        TRAIN_SPLIT_SEED ^ seed,
                        bootstrap_index,
                        spec,
                        &device,
                    )
                    .map_err(anyhow::Error::msg)?;
                    let selection = route_context(
                        &mut router,
                        &model,
                        &bootstrap,
                        spec,
                        &mut mask_bank,
                        true,
                        true,
                    )?;
                    selector_total += 1;
                    selector_probe_tokens += selection.probe_tokens;
                    selector_deferred += usize::from(selection.novelty_deferred);
                    bootstrap_contexts.push((selection.context_index, selection.novelty_deferred));
                    let created = selection.created;
                    final_selection = Some(selection);
                    if created {
                        break;
                    }
                }
                let selection = final_selection.expect("at least one context bootstrap probe");
                for (context_index, deferred) in bootstrap_contexts {
                    if !deferred {
                        selector_committed_total += 1;
                        selector_correct += usize::from(context_index == selection.context_index);
                    }
                }
                expected_contexts.insert(task.index(), selection.context_index);
                context_discovery.push(ContextDiscoveryEvent {
                    phase,
                    task,
                    context_index: selection.context_index,
                    created: selection.created,
                    reserve_loss: selection.reserve_loss,
                    reserve_supported_novelty: selection.reserve_supported_novelty,
                    candidates: selection.candidates,
                });
                selection.context_index
            }
        };
        let _ = mask_bank.get(context, &device);
        while context_optimizers.len() <= context {
            context_optimizers.push(make_optimizer());
        }
        let (before, correct, total, probe_tokens) = evaluate(
            &model,
            task,
            EvaluationContext {
                seed,
                phase,
                spec,
                eval_batches: args.eval_batches,
                routing_mode,
                router: &mut router,
                expected_contexts: &expected_contexts,
                mask_bank: &mut mask_bank,
            },
        )?;
        pre_task.push(before);
        selector_correct += correct;
        selector_total += total;
        selector_committed_total += total;
        selector_probe_tokens += probe_tokens;

        let _ = B::sync(&device);
        let phase_started = Instant::now();
        for update in 0..args.updates_per_task {
            let batch = context_recurrence_batch::<B>(
                task,
                TRAIN_SPLIT_SEED ^ seed,
                update as u64,
                spec,
                &device,
            )
            .map_err(anyhow::Error::msg)?;
            let selected_context = match routing_mode {
                RoutingMode::Dense => 0,
                RoutingMode::SelectedSparseSubnetwork => {
                    let probe_due = matches!(&router, ContextRouter::DescriptorControl(_))
                        || update.is_multiple_of(args.context_probe_every_updates);
                    if probe_due {
                        let selection = route_context(
                            &mut router,
                            &model,
                            &batch,
                            spec,
                            &mut mask_bank,
                            true,
                            true,
                        )?;
                        selector_total += 1;
                        selector_probe_tokens += selection.probe_tokens;
                        if selection.novelty_deferred {
                            selector_deferred += 1;
                        } else {
                            selector_committed_total += 1;
                            selector_correct += usize::from(selection.context_index == context);
                        }
                        if selection.created {
                            return Err(anyhow!(
                                "selector created a duplicate context inside task {task:?} at update {update}; reserve_loss={:?} reserve_supported_novelty={:?} candidates={:?}",
                                selection.reserve_loss,
                                selection.reserve_supported_novelty,
                                selection.candidates
                            ));
                        }
                        selection.context_index
                    } else {
                        context
                    }
                }
            };
            let masks = mask_bank.get(selected_context, &device);
            let grads = if rule == LearningRule::Backpropagation {
                let logits = forward(&model, batch.inputs, masks);
                let loss = burn_dragon_core::objective::masked_token_mean(
                    model.language_token_losses_from_logits(logits, batch.targets),
                    Some(batch.loss_mask),
                );
                global_backward_calls += 1;
                GradientsParams::from_grads(loss.backward(), &model)
            } else {
                let diagnostics = args.pc_energy_diagnostic_every > 0
                    && (update.is_multiple_of(args.pc_energy_diagnostic_every)
                        || update + 1 == args.updates_per_task);
                let config = pc_config(rule, args, diagnostics).expect("PC rule");
                let step = match (masks.neuron, masks.activity) {
                    (Some(neuron), Some(activity)) => {
                        local_predictive_coding_derivatives_with_subnetwork_masks(
                            &model,
                            batch.inputs,
                            batch.targets,
                            Some(batch.loss_mask),
                            neuron,
                            activity,
                            &config,
                        )
                    }
                    (None, None) => local_predictive_coding_derivatives(
                        &model,
                        batch.inputs,
                        batch.targets,
                        Some(batch.loss_mask),
                        &config,
                    ),
                    _ => unreachable!("complete subnetwork routing requires both masks"),
                }
                .map_err(anyhow::Error::msg)?;
                local_vjp_calls += step.report.local_vjp_calls;
                global_backward_calls += step.report.global_backward_calls;
                if let (Some(before), Some(after)) =
                    (step.report.energy_before, step.report.energy_after)
                {
                    let tolerance = (before.abs() * 1.0e-6).max(1.0e-8);
                    let relative = (after - before) / before.abs().max(1.0e-12);
                    energy_diagnostics += 1;
                    energy_descents += usize::from(after <= before + tolerance);
                    max_relative_energy_increase =
                        max_relative_energy_increase.max(relative.max(0.0));
                }
                step.grads
            };
            let optimizer = match optimizer_state_scope {
                OptimizerStateScope::Shared => &mut shared_optimizer,
                OptimizerStateScope::ContextScoped => &mut context_optimizers[selected_context],
            };
            let learning_rate = if rule == LearningRule::Backpropagation {
                args.learning_rate
            } else {
                args.pc_learning_rate
            };
            model = optimizer.step(learning_rate, model, grads);
        }
        let _ = B::sync(&device);
        training_elapsed_seconds += phase_started.elapsed().as_secs_f64();

        for seen_task in ContextRecurrenceTask::ALL.into_iter().take(phase + 1) {
            let (evaluation, correct, total, probe_tokens) = evaluate(
                &model,
                seen_task,
                EvaluationContext {
                    seed,
                    phase,
                    spec,
                    eval_batches: args.eval_batches,
                    routing_mode,
                    router: &mut router,
                    expected_contexts: &expected_contexts,
                    mask_bank: &mut mask_bank,
                },
            )?;
            selector_correct += correct;
            selector_total += total;
            selector_committed_total += total;
            selector_probe_tokens += probe_tokens;
            task_matrix.push(TaskMatrixCell {
                phase,
                task: seen_task,
                evaluation,
            });
        }
    }
    let elapsed_seconds = training_elapsed_seconds;
    let model_tokens = ContextRecurrenceTask::ALL.len()
        * args.updates_per_task
        * args.batch_size
        * args.block_size;
    let (final_average_accuracy, backward_transfer, mean_forgetting, max_forgetting) =
        summarize_task_matrix(&task_matrix);
    let (mean_neuron_mask_overlap, mean_activity_mask_overlap) = mask_bank.overlap();
    Ok(StreamOutcome {
        seed,
        rule,
        routing_mode,
        optimizer_state_scope,
        selector_mode: args.selector_mode,
        pre_task,
        task_matrix,
        final_average_accuracy,
        backward_transfer,
        mean_forgetting,
        max_forgetting,
        selector_accuracy: (selector_committed_total > 0)
            .then_some(selector_correct as f64 / selector_committed_total as f64),
        selector_probes: selector_total,
        selector_committed_probes: selector_committed_total,
        selector_deferred_probes: selector_deferred,
        selector_probe_tokens,
        contexts_created: router.known_contexts(),
        context_discovery_complete: routing_mode == RoutingMode::Dense
            || router.known_contexts() == ContextRecurrenceTask::ALL.len(),
        context_discovery,
        mean_neuron_mask_overlap,
        mean_activity_mask_overlap,
        model_tokens,
        elapsed_seconds,
        model_tokens_per_second: model_tokens as f64 / elapsed_seconds.max(f64::EPSILON),
        local_vjp_calls,
        global_backward_calls,
        energy_diagnostics,
        energy_descent_fraction: (energy_diagnostics > 0)
            .then_some(energy_descents as f64 / energy_diagnostics as f64),
        max_relative_energy_increase: (energy_diagnostics > 0)
            .then_some(max_relative_energy_increase),
        acquisition_gate: None,
    })
}

#[cfg(feature = "train")]
fn task_acquisition(outcome: &StreamOutcome, task: usize) -> ContinualTaskAcquisition {
    let after = outcome
        .task_matrix
        .iter()
        .find(|cell| cell.phase == task && cell.task.index() == task)
        .expect("task acquisition diagonal")
        .evaluation;
    ContinualTaskAcquisition {
        loss_reduction: outcome.pre_task[task].loss - after.loss,
        accuracy_gain: after.accuracy - outcome.pre_task[task].accuracy,
    }
}

#[cfg(feature = "train")]
fn apply_acquisition_gates(args: &Args, outcomes: &mut [StreamOutcome]) -> Result<()> {
    let baselines = outcomes
        .iter()
        .filter(|outcome| outcome.rule == LearningRule::Backpropagation)
        .map(|outcome| {
            (
                (
                    outcome.seed,
                    outcome.routing_mode,
                    outcome.optimizer_state_scope,
                ),
                (0..ContextRecurrenceTask::ALL.len())
                    .map(|task| task_acquisition(outcome, task))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for outcome in outcomes {
        let baseline = baselines
            .get(&(
                outcome.seed,
                outcome.routing_mode,
                outcome.optimizer_state_scope,
            ))
            .expect("matched backprop acquisition baseline");
        let candidate = (0..ContextRecurrenceTask::ALL.len())
            .map(|task| task_acquisition(outcome, task))
            .collect::<Vec<_>>();
        outcome.acquisition_gate = Some(
            evaluate_continual_acquisition_gate(
                baseline,
                &candidate,
                ContinualAcquisitionThresholds {
                    loss_floor_ratio: args.loss_acquisition_floor_ratio,
                    accuracy_tolerance: args.acquisition_tolerance,
                    minimum_baseline_loss_reduction: args.minimum_baseline_loss_reduction,
                    minimum_baseline_accuracy_gain: args.minimum_baseline_accuracy_gain,
                },
            )
            .map_err(anyhow::Error::msg)?,
        );
    }
    Ok(())
}

#[cfg(feature = "train")]
fn run<B>(args: &Args) -> Result<MatrixReport>
where
    B: AutodiffBackend,
    B::Device: Default + 'static,
    B::FloatTensorPrimitive: 'static,
{
    let recurrence = ContextRecurrenceSpec {
        batch_size: args.batch_size,
        block_size: args.block_size,
        payload_modulus: args.payload_modulus,
    };
    let mut model_config = DragonConfig {
        n_layer: args.n_layer,
        n_embd: args.n_embd,
        n_head: args.n_head,
        mlp_internal_dim_multiplier: args.latent_total / args.n_embd,
        vocab_size: recurrence.required_vocab_size(),
        dropout: 0.0,
        ..DragonConfig::default()
    };
    model_config.sequence_kernel.executor = SequenceTrainingExecutor::DenseScoreShortContext;
    model_config.fused_kernels.rotary_embedding = RotaryEmbedding::Alibi;
    let device = B::Device::default();
    let probe = DragonModel::<B>::new(model_config.clone(), &device);
    probe
        .predictive_coding_support()
        .map_err(anyhow::Error::msg)?;
    let parameters = probe.num_params();
    drop(probe);
    let make_optimizer = || {
        AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<B, DragonModel<B>>()
    };
    let mut outcomes = Vec::new();
    for &seed in &args.seeds {
        for &topology in &args.topologies {
            let routing_mode = topology.routing_mode();
            let optimizer_state_scope = topology.optimizer_state_scope();
            for &rule in &args.rules {
                eprintln!("seed={seed} topology={topology:?} rule={rule:?}");
                outcomes.push(run_arm::<B, _, _>(
                    args,
                    &model_config,
                    seed,
                    rule,
                    routing_mode,
                    optimizer_state_scope,
                    &make_optimizer,
                )?);
            }
        }
    }
    apply_acquisition_gates(args, &mut outcomes)?;
    Ok(MatrixReport {
        schema_version: 6,
        backend: args.backend.clone(),
        parameters,
        n_layer: args.n_layer,
        n_embd: args.n_embd,
        n_head: args.n_head,
        latent_total: args.latent_total,
        recurrence,
        tasks: ContextRecurrenceTask::ALL.to_vec(),
        topologies: args.topologies.clone(),
        updates_per_task: args.updates_per_task,
        eval_batches: args.eval_batches,
        learning_rate: args.learning_rate,
        pc_learning_rate: args.pc_learning_rate,
        pc_inference_steps: args.pc_inference_steps,
        pc_step_size: args.pc_step_size,
        pc_max_grad_norm: args.pc_max_grad_norm,
        pc_prediction_precision: args.pc_prediction_precision,
        pc_energy_diagnostic_every: args.pc_energy_diagnostic_every,
        active_fraction: args.active_fraction,
        selector_mode: args.selector_mode,
        context_probe_every_updates: args.context_probe_every_updates,
        context_novelty_confirmations: args.context_novelty_confirmations,
        predictive_context: args.predictive_context,
        descriptor_control: StreamingContextSelectorConfig {
            novelty_cosine_threshold: args.novelty_cosine_threshold,
            centroid_update_rate: args.centroid_update_rate,
        },
        loss_acquisition_floor_ratio: args.loss_acquisition_floor_ratio,
        acquisition_tolerance: args.acquisition_tolerance,
        minimum_baseline_loss_reduction: args.minimum_baseline_loss_reduction,
        minimum_baseline_accuracy_gain: args.minimum_baseline_accuracy_gain,
        outcomes,
    })
}

#[cfg(all(feature = "train", feature = "cuda"))]
fn run_cuda(args: &Args) -> Result<MatrixReport> {
    run::<Autodiff<burn_cuda::Cuda<f32>>>(args)
}

#[cfg(all(feature = "train", not(feature = "cuda")))]
fn run_cuda(_args: &Args) -> Result<MatrixReport> {
    Err(anyhow!("pc_lifelong_stream was built without cuda"))
}

#[cfg(feature = "train")]
fn main() -> Result<()> {
    let args = parse_args()?;
    let report = match args.backend.as_str() {
        "cpu" => run::<Autodiff<NdArray<f32>>>(&args),
        "cuda" => run_cuda(&args),
        other => Err(anyhow!("unsupported --backend {other:?}")),
    }
    .context("lifelong PC matrix failed")?;
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(output) = &args.output {
        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create report directory {}", parent.display()))?;
        }
        std::fs::write(output, json)
            .with_context(|| format!("write report {}", output.display()))?;
        println!("wrote {}", output.display());
    } else {
        println!("{json}");
    }
    Ok(())
}

#[cfg(all(test, feature = "train"))]
mod tests {
    use super::*;

    #[test]
    fn error_equilibrium_rule_uses_the_shared_non_equilibrium_contract() {
        let args = Args {
            pc_inference_steps: 1,
            pc_step_size: 0.1,
            pc_prediction_precision: 10.0,
            pc_max_grad_norm: Some(1_000_000.0),
            ..Args::default()
        };
        let config = pc_config(LearningRule::ErrorEquilibrium, &args, true)
            .expect("ePC is a local predictive-coding rule");
        assert_eq!(config.solver, LocalPredictiveCodingSolver::ErrorEquilibrium);
        assert_eq!(config.inference.steps, 1);
        assert_eq!(config.inference.step_size, 0.1);
        assert_eq!(config.inference.max_grad_norm, Some(1_000_000.0));
        assert_eq!(config.prediction_precision, 10.0);
        assert!(config.sync_diagnostics);
        config
            .inference
            .validate("pc_lifelong_stream.local_predictive_coding.inference")
            .expect("ePC lifelong configuration should validate");
    }
}

#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("pc_lifelong_stream requires --features train");
    std::process::exit(2);
}
