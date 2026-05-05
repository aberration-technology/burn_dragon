use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use serde::{Deserialize, Serialize};

use crate::DragonModel;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SelfDistillationKlKind {
    #[default]
    Forward,
    Reverse,
    JensenShannon,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SelfDistillationObjectiveKind {
    #[default]
    NextToken,
    Sdft,
    Sdpo,
    SdftSdpo,
}

impl SelfDistillationObjectiveKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::NextToken => "next_token",
            Self::Sdft => "sdft",
            Self::Sdpo => "sdpo",
            Self::SdftSdpo => "sdft_sdpo",
        }
    }

    pub fn is_next_token(self) -> bool {
        matches!(self, Self::NextToken)
    }
}

pub type TrainingObjectiveKind = SelfDistillationObjectiveKind;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TrainingObjectiveConfig {
    #[default]
    NextToken,
    Sdft(SdftObjectiveConfig),
    Sdpo(SdpoObjectiveConfig),
    SdftSdpo(SdftSdpoObjectiveConfig),
}

impl TrainingObjectiveConfig {
    pub fn kind(&self) -> TrainingObjectiveKind {
        match self {
            Self::NextToken => TrainingObjectiveKind::NextToken,
            Self::Sdft(_) => TrainingObjectiveKind::Sdft,
            Self::Sdpo(_) => TrainingObjectiveKind::Sdpo,
            Self::SdftSdpo(_) => TrainingObjectiveKind::SdftSdpo,
        }
    }

    pub fn label(&self) -> &'static str {
        self.kind().label()
    }

    pub fn is_next_token(&self) -> bool {
        matches!(self, Self::NextToken)
    }

    pub fn ensure_supported(&self, trainer: ObjectiveTrainerKind) -> Result<(), String> {
        validate_training_objective_config(self).map_err(|error| error.to_string())?;
        ensure_objective_supported(self.kind(), trainer)
    }

    pub fn ensure_browser_supported(&self) -> Result<(), String> {
        self.ensure_supported(ObjectiveTrainerKind::Browser)
    }

    pub fn to_window_smoke_objective(&self) -> WindowSelfDistillationSmokeObjective {
        match self {
            Self::NextToken => WindowSelfDistillationSmokeObjective::NextToken,
            Self::Sdft(_) => WindowSelfDistillationSmokeObjective::Sdft,
            Self::Sdpo(config) => {
                WindowSelfDistillationSmokeObjective::Sdpo(config.selected_token_loss_config())
            }
            Self::SdftSdpo(config) => WindowSelfDistillationSmokeObjective::SdftSdpo {
                sdft_weight: config.sdft_weight,
                sdpo_weight: config.sdpo_weight,
                sdpo: config.sdpo.selected_token_loss_config(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TeacherRegularization {
    #[default]
    Ema,
    TrustRegion,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RepromptTruncation {
    Left,
    #[default]
    Right,
    Error,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct SdftObjectiveConfig {
    pub max_completion_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub kl: SelfDistillationKlKind,
    pub generate_from_teacher: bool,
    pub teacher_update_rate: f32,
    pub top_entropy_quantile: Option<f32>,
    pub num_loss_tokens_to_skip: usize,
}

impl Default for SdftObjectiveConfig {
    fn default() -> Self {
        Self {
            max_completion_tokens: 32,
            temperature: 1.0,
            top_k: None,
            kl: SelfDistillationKlKind::Forward,
            generate_from_teacher: false,
            teacher_update_rate: 0.01,
            top_entropy_quantile: None,
            num_loss_tokens_to_skip: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct SdpoObjectiveConfig {
    pub group_size: usize,
    pub max_completion_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub full_logit_distillation: bool,
    pub alpha: f32,
    pub success_reward_threshold: f32,
    pub teacher_regularization: TeacherRegularization,
    pub teacher_update_rate: f32,
    pub distillation_topk: Option<usize>,
    pub distillation_add_tail: bool,
    pub is_clip: Option<f32>,
    pub max_reprompt_len: usize,
    pub reprompt_truncation: RepromptTruncation,
    pub dont_reprompt_on_self_success: bool,
    pub remove_thinking_from_demonstration: bool,
    pub reprompt_template: Option<String>,
    pub solution_template: Option<String>,
    pub feedback_template: Option<String>,
    pub include_environment_feedback: bool,
    pub environment_feedback_only_without_solution: bool,
}

impl SdpoObjectiveConfig {
    pub fn selected_token_loss_config(&self) -> SelectedTokenSdpoLossConfig {
        SelectedTokenSdpoLossConfig {
            kl: SelfDistillationKlKind::from_sdpo_alpha(self.alpha),
            is_clip: self.is_clip,
        }
    }
}

impl Default for SdpoObjectiveConfig {
    fn default() -> Self {
        Self {
            group_size: 2,
            max_completion_tokens: 32,
            temperature: 1.0,
            top_k: None,
            full_logit_distillation: true,
            alpha: 0.5,
            success_reward_threshold: 0.0,
            teacher_regularization: TeacherRegularization::Ema,
            teacher_update_rate: 0.05,
            distillation_topk: None,
            distillation_add_tail: false,
            is_clip: Some(2.0),
            max_reprompt_len: 10_240,
            reprompt_truncation: RepromptTruncation::Right,
            dont_reprompt_on_self_success: false,
            remove_thinking_from_demonstration: false,
            reprompt_template: None,
            solution_template: None,
            feedback_template: None,
            include_environment_feedback: false,
            environment_feedback_only_without_solution: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct SdftSdpoObjectiveConfig {
    pub sdft: SdftObjectiveConfig,
    pub sdpo: SdpoObjectiveConfig,
    pub sdft_weight: f32,
    pub sdpo_weight: f32,
}

impl Default for SdftSdpoObjectiveConfig {
    fn default() -> Self {
        Self {
            sdft: SdftObjectiveConfig::default(),
            sdpo: SdpoObjectiveConfig::default(),
            sdft_weight: 0.5,
            sdpo_weight: 0.5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveTrainerKind {
    SingleDevice,
    Browser,
    Ddp,
    Pipeline,
}

impl ObjectiveTrainerKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::SingleDevice => "single-device",
            Self::Browser => "browser",
            Self::Ddp => "ddp",
            Self::Pipeline => "pipeline",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveSupport {
    FullySupported,
    ConfigAndNumericsOnly,
}

pub fn objective_support(
    objective: SelfDistillationObjectiveKind,
    trainer: ObjectiveTrainerKind,
) -> ObjectiveSupport {
    match (objective, trainer) {
        (SelfDistillationObjectiveKind::NextToken, _) => ObjectiveSupport::FullySupported,
        (
            SelfDistillationObjectiveKind::Sdft
            | SelfDistillationObjectiveKind::Sdpo
            | SelfDistillationObjectiveKind::SdftSdpo,
            ObjectiveTrainerKind::SingleDevice
            | ObjectiveTrainerKind::Ddp
            | ObjectiveTrainerKind::Pipeline,
        ) => ObjectiveSupport::FullySupported,
        (
            SelfDistillationObjectiveKind::Sdft
            | SelfDistillationObjectiveKind::Sdpo
            | SelfDistillationObjectiveKind::SdftSdpo,
            ObjectiveTrainerKind::Browser,
        ) => ObjectiveSupport::ConfigAndNumericsOnly,
    }
}

pub fn ensure_objective_supported(
    objective: SelfDistillationObjectiveKind,
    trainer: ObjectiveTrainerKind,
) -> Result<(), String> {
    if objective_support(objective, trainer) == ObjectiveSupport::FullySupported {
        return Ok(());
    }
    Err(format!(
        "training.objective.type={} is configured, but {} training is only wired for next_token execution; SDFT/SDPO config and numerical kernels are available, but rollout generation, teacher-conditioned reprompts, reward/feedback masks, and EMA teacher updates must be wired before running this objective",
        objective.label(),
        trainer.label()
    ))
}

pub fn validate_training_objective_config(
    objective: &TrainingObjectiveConfig,
) -> anyhow::Result<()> {
    match objective {
        TrainingObjectiveConfig::NextToken => Ok(()),
        TrainingObjectiveConfig::Sdft(config) => {
            validate_sdft_objective_config(config, "training.objective")
        }
        TrainingObjectiveConfig::Sdpo(config) => {
            validate_sdpo_objective_config(config, "training.objective")
        }
        TrainingObjectiveConfig::SdftSdpo(config) => {
            validate_sdft_objective_config(&config.sdft, "training.objective.sdft")?;
            validate_sdpo_objective_config(&config.sdpo, "training.objective.sdpo")?;
            validate_positive_f32(config.sdft_weight, "training.objective.sdft_weight")?;
            validate_positive_f32(config.sdpo_weight, "training.objective.sdpo_weight")?;
            Ok(())
        }
    }
}

pub fn validate_sdft_objective_config(
    config: &SdftObjectiveConfig,
    path: &str,
) -> anyhow::Result<()> {
    if config.max_completion_tokens == 0 {
        return Err(anyhow::anyhow!("{path}.max_completion_tokens must be > 0"));
    }
    validate_positive_f32(config.temperature, &format!("{path}.temperature"))?;
    if let Some(top_k) = config.top_k
        && top_k == 0
    {
        return Err(anyhow::anyhow!("{path}.top_k must be > 0 when set"));
    }
    validate_probability(
        config.teacher_update_rate,
        &format!("{path}.teacher_update_rate"),
    )?;
    if let Some(quantile) = config.top_entropy_quantile {
        validate_probability(quantile, &format!("{path}.top_entropy_quantile"))?;
        return Err(anyhow::anyhow!(
            "{path}.top_entropy_quantile is not wired into rollout masking yet"
        ));
    }
    Ok(())
}

pub fn validate_sdpo_objective_config(
    config: &SdpoObjectiveConfig,
    path: &str,
) -> anyhow::Result<()> {
    if config.group_size == 0 {
        return Err(anyhow::anyhow!("{path}.group_size must be > 0"));
    }
    if config.max_completion_tokens == 0 {
        return Err(anyhow::anyhow!("{path}.max_completion_tokens must be > 0"));
    }
    validate_positive_f32(config.temperature, &format!("{path}.temperature"))?;
    if let Some(top_k) = config.top_k
        && top_k == 0
    {
        return Err(anyhow::anyhow!("{path}.top_k must be > 0 when set"));
    }
    validate_probability(config.alpha, &format!("{path}.alpha"))?;
    if !config.success_reward_threshold.is_finite() {
        return Err(anyhow::anyhow!(
            "{path}.success_reward_threshold must be finite"
        ));
    }
    if config.success_reward_threshold > 0.0 {
        return Err(anyhow::anyhow!(
            "{path}.success_reward_threshold requires an external reward/environment feedback source; token-window SDPO uses batch target demonstrations only"
        ));
    }
    if !config.full_logit_distillation {
        return Err(anyhow::anyhow!(
            "{path}.full_logit_distillation=false is not paper-aligned for the rollout objective; use flat-logit full distillation"
        ));
    }
    if config.teacher_regularization != TeacherRegularization::Ema {
        return Err(anyhow::anyhow!(
            "{path}.teacher_regularization currently supports only \"ema\""
        ));
    }
    validate_probability(
        config.teacher_update_rate,
        &format!("{path}.teacher_update_rate"),
    )?;
    if let Some(topk) = config.distillation_topk
        && topk == 0
    {
        return Err(anyhow::anyhow!(
            "{path}.distillation_topk must be > 0 when set"
        ));
    }
    if config.distillation_topk.is_some() {
        return Err(anyhow::anyhow!(
            "{path}.distillation_topk is not implemented for the rollout objective yet"
        ));
    }
    if config.distillation_add_tail {
        return Err(anyhow::anyhow!(
            "{path}.distillation_add_tail requires distillation_topk support"
        ));
    }
    if let Some(is_clip) = config.is_clip {
        validate_positive_f32(is_clip, &format!("{path}.is_clip"))?;
    }
    if config.max_reprompt_len == 0 {
        return Err(anyhow::anyhow!("{path}.max_reprompt_len must be > 0"));
    }
    validate_optional_nonempty(
        config.reprompt_template.as_ref(),
        &format!("{path}.reprompt_template"),
    )?;
    validate_optional_nonempty(
        config.solution_template.as_ref(),
        &format!("{path}.solution_template"),
    )?;
    validate_optional_nonempty(
        config.feedback_template.as_ref(),
        &format!("{path}.feedback_template"),
    )?;
    if config.reprompt_template.is_some()
        || config.solution_template.is_some()
        || config.feedback_template.is_some()
        || config.include_environment_feedback
        || config.environment_feedback_only_without_solution
        || config.dont_reprompt_on_self_success
        || config.remove_thinking_from_demonstration
    {
        return Err(anyhow::anyhow!(
            "{path} feedback/template SDPO fields require text-level environment feedback plumbing; current language trainer uses tokenized batch target demonstrations"
        ));
    }
    Ok(())
}

fn validate_probability(value: f32, path: &str) -> anyhow::Result<()> {
    if !(0.0..=1.0).contains(&value) {
        return Err(anyhow::anyhow!("{path} must be in [0, 1] (got {value})"));
    }
    Ok(())
}

fn validate_positive_f32(value: f32, path: &str) -> anyhow::Result<()> {
    if value <= 0.0 || !value.is_finite() {
        return Err(anyhow::anyhow!(
            "{path} must be finite and > 0 (got {value})"
        ));
    }
    Ok(())
}

fn validate_optional_nonempty(value: Option<&String>, path: &str) -> anyhow::Result<()> {
    if let Some(value) = value
        && value.trim().is_empty()
    {
        return Err(anyhow::anyhow!("{path} must not be empty when set"));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct SelectedTokenSdpoLossConfig {
    pub kl: SelfDistillationKlKind,
    pub is_clip: Option<f32>,
}

impl Default for SelectedTokenSdpoLossConfig {
    fn default() -> Self {
        Self {
            kl: SelfDistillationKlKind::JensenShannon,
            is_clip: Some(2.0),
        }
    }
}

impl SelfDistillationKlKind {
    pub fn from_sdpo_alpha(alpha: f32) -> Self {
        if alpha <= 0.0 {
            Self::Forward
        } else if alpha >= 1.0 {
            Self::Reverse
        } else {
            Self::JensenShannon
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum WindowSelfDistillationSmokeObjective {
    #[default]
    NextToken,
    Sdft,
    Sdpo(SelectedTokenSdpoLossConfig),
    SdftSdpo {
        sdft_weight: f32,
        sdpo_weight: f32,
        sdpo: SelectedTokenSdpoLossConfig,
    },
}

pub type WindowSelfDistillationObjective = WindowSelfDistillationSmokeObjective;

pub fn log_probs_from_logits<B: Backend>(logits: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, time, vocab] = logits.shape().dims();
    activation::log_softmax(logits.reshape([batch * time, vocab]), 1).reshape([batch, time, vocab])
}

pub fn selected_token_log_probs<B: Backend>(
    log_probs: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let [batch, time, _vocab] = log_probs.shape().dims();
    log_probs
        .gather(2, targets.reshape([batch, time, 1]))
        .reshape([batch, time])
}

pub fn selected_token_log_probs_from_hidden<B: Backend>(
    model: &DragonModel<B>,
    hidden: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    model
        .language_token_losses_from_hidden(hidden, targets)
        .neg()
}

pub fn self_distillation_loss_from_log_probs<B: Backend>(
    student_log_probs: Tensor<B, 3>,
    teacher_log_probs: Tensor<B, 3>,
    mask: Option<Tensor<B, 2, Int>>,
    kind: SelfDistillationKlKind,
) -> Tensor<B, 1> {
    let per_token =
        self_distillation_per_token_from_log_probs(student_log_probs, teacher_log_probs, kind);
    masked_token_mean(per_token, mask)
}

pub fn self_distillation_per_token_from_log_probs<B: Backend>(
    student_log_probs: Tensor<B, 3>,
    teacher_log_probs: Tensor<B, 3>,
    kind: SelfDistillationKlKind,
) -> Tensor<B, 2> {
    match kind {
        SelfDistillationKlKind::Forward => {
            kl_per_token(teacher_log_probs.clone(), student_log_probs.clone())
        }
        SelfDistillationKlKind::Reverse => kl_per_token(student_log_probs, teacher_log_probs),
        SelfDistillationKlKind::JensenShannon => {
            let student_prob = student_log_probs.clone().exp();
            let teacher_prob = teacher_log_probs.clone().exp();
            let mixture_log_probs = (student_prob + teacher_prob)
                .mul_scalar(0.5)
                .clamp_min(1e-12)
                .log();
            let teacher_kl = kl_per_token(teacher_log_probs, mixture_log_probs.clone());
            let student_kl = kl_per_token(student_log_probs, mixture_log_probs);
            (teacher_kl + student_kl).mul_scalar(0.5)
        }
    }
}

pub fn self_distillation_loss_from_logits<B: Backend>(
    student_logits: Tensor<B, 3>,
    teacher_logits: Tensor<B, 3>,
    mask: Option<Tensor<B, 2, Int>>,
    kind: SelfDistillationKlKind,
) -> Tensor<B, 1> {
    self_distillation_loss_from_log_probs(
        log_probs_from_logits(student_logits),
        log_probs_from_logits(teacher_logits),
        mask,
        kind,
    )
}

pub fn sdpo_token_advantage<B: Backend>(
    teacher_token_log_probs: Tensor<B, 2>,
    student_token_log_probs: Tensor<B, 2>,
    mask: Option<Tensor<B, 2, Int>>,
    normalize: bool,
    epsilon: f32,
) -> Tensor<B, 2> {
    let advantage = teacher_token_log_probs - student_token_log_probs;
    let advantage = if let Some(mask) = mask {
        advantage * mask.float()
    } else {
        advantage
    };
    if !normalize {
        return advantage;
    }
    let [batch, time] = advantage.shape().dims();
    let mean = advantage
        .clone()
        .mean_dim(0)
        .mean_dim(1)
        .repeat_dim(0, batch)
        .repeat_dim(1, time);
    let centered = advantage - mean;
    let var = centered
        .clone()
        .powf_scalar(2.0)
        .mean_dim(0)
        .mean_dim(1)
        .repeat_dim(0, batch)
        .repeat_dim(1, time);
    centered / var.add_scalar(epsilon.max(1e-12)).sqrt()
}

pub fn clipped_policy_loss<B: Backend>(
    log_prob_new: Tensor<B, 2>,
    log_prob_old: Tensor<B, 2>,
    advantage: Tensor<B, 2>,
    mask: Option<Tensor<B, 2, Int>>,
    clip_range: Option<f32>,
    weight: f32,
) -> Tensor<B, 1> {
    let weight = weight.max(0.0);
    if weight <= 0.0 {
        return Tensor::<B, 1>::zeros([1], &log_prob_new.device());
    }
    let objective = if let Some(clip) = clip_range.filter(|clip| *clip > 0.0) {
        let log_ratio = (log_prob_new - log_prob_old)
            .clamp_min(-20.0)
            .clamp_max(20.0);
        let ratio = log_ratio.exp();
        let clipped = ratio.clone().clamp_min(1.0 - clip).clamp_max(1.0 + clip);
        let surrogate = ratio * advantage.clone();
        let surrogate_clipped = clipped * advantage;
        let use_clipped = surrogate_clipped.clone().lower_equal(surrogate.clone());
        surrogate.mask_where(use_clipped, surrogate_clipped)
    } else {
        log_prob_new * advantage
    };
    masked_token_mean(objective.mul_scalar(-weight), mask)
}

pub fn selected_token_log_prob_mse_loss<B: Backend>(
    student_token_log_probs: Tensor<B, 2>,
    teacher_token_log_probs: Tensor<B, 2>,
    mask: Option<Tensor<B, 2, Int>>,
) -> Tensor<B, 1> {
    let delta = student_token_log_probs - teacher_token_log_probs.detach();
    masked_token_mean(delta.powf_scalar(2.0), mask)
}

pub fn selected_token_distillation_loss_from_hidden<B: Backend>(
    student_model: &DragonModel<B>,
    teacher_model: &DragonModel<B>,
    student_hidden: Tensor<B, 3>,
    teacher_hidden: Tensor<B, 3>,
    student_targets: Tensor<B, 2, Int>,
    teacher_targets: Tensor<B, 2, Int>,
    mask: Option<Tensor<B, 2, Int>>,
) -> Tensor<B, 1> {
    selected_token_log_prob_mse_loss(
        selected_token_log_probs_from_hidden(student_model, student_hidden, student_targets),
        selected_token_log_probs_from_hidden(teacher_model, teacher_hidden, teacher_targets),
        mask,
    )
}

pub struct SelectedTokenDistillationHiddenBatch<B: Backend> {
    pub student_hidden: Tensor<B, 3>,
    pub teacher_hidden: Tensor<B, 3>,
    pub student_targets: Tensor<B, 2, Int>,
    pub teacher_targets: Tensor<B, 2, Int>,
    pub mask: Option<Tensor<B, 2, Int>>,
}

pub fn selected_token_sdpo_loss_from_hidden<B: Backend>(
    student_model: &DragonModel<B>,
    teacher_model: &DragonModel<B>,
    batch: SelectedTokenDistillationHiddenBatch<B>,
    config: SelectedTokenSdpoLossConfig,
) -> Tensor<B, 1> {
    let new_token_log_probs = selected_token_log_probs_from_hidden(
        student_model,
        batch.student_hidden,
        batch.student_targets,
    );
    let teacher_token_log_probs = selected_token_log_probs_from_hidden(
        teacher_model,
        batch.teacher_hidden,
        batch.teacher_targets,
    )
    .detach();
    let old_token_log_probs = new_token_log_probs.clone().detach();
    let per_token = match config.kl {
        SelfDistillationKlKind::Forward | SelfDistillationKlKind::JensenShannon => {
            selected_token_log_prob_mse_loss(
                new_token_log_probs,
                teacher_token_log_probs,
                batch.mask,
            )
        }
        SelfDistillationKlKind::Reverse => {
            let delta = teacher_token_log_probs - new_token_log_probs;
            masked_token_mean(delta.powf_scalar(2.0), batch.mask)
        }
    };
    let _ = old_token_log_probs;
    per_token
}

pub fn window_smoke_sdft_loss<B: Backend>(
    student_model: &DragonModel<B>,
    teacher_model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    selected_token_distillation_loss_from_hidden(
        student_model,
        teacher_model,
        student_model.forward_hidden(inputs.clone()),
        teacher_model.forward_hidden(inputs),
        targets.clone(),
        targets,
        None,
    )
}

pub fn window_smoke_sdpo_loss<B: Backend>(
    student_model: &DragonModel<B>,
    teacher_model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    config: SelectedTokenSdpoLossConfig,
) -> Tensor<B, 1> {
    selected_token_sdpo_loss_from_hidden(
        student_model,
        teacher_model,
        SelectedTokenDistillationHiddenBatch {
            student_hidden: student_model.forward_hidden(inputs.clone()),
            teacher_hidden: teacher_model.forward_hidden(inputs),
            student_targets: targets.clone(),
            teacher_targets: targets,
            mask: None,
        },
        config,
    )
}

pub fn window_self_distillation_smoke_loss<B: Backend>(
    student_model: &DragonModel<B>,
    teacher_model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    objective: &WindowSelfDistillationSmokeObjective,
) -> Tensor<B, 1> {
    match objective {
        WindowSelfDistillationSmokeObjective::NextToken => {
            let hidden = student_model.forward_hidden(inputs);
            student_model.language_loss_from_hidden(hidden, targets)
        }
        WindowSelfDistillationSmokeObjective::Sdft => {
            window_smoke_sdft_loss(student_model, teacher_model, inputs, targets)
        }
        WindowSelfDistillationSmokeObjective::Sdpo(config) => {
            window_smoke_sdpo_loss(student_model, teacher_model, inputs, targets, *config)
        }
        WindowSelfDistillationSmokeObjective::SdftSdpo {
            sdft_weight,
            sdpo_weight,
            sdpo,
        } => {
            let sdft_weight = sdft_weight.max(0.0);
            let sdpo_weight = sdpo_weight.max(0.0);
            let weight_sum = (sdft_weight + sdpo_weight).max(1.0e-6);
            window_smoke_sdft_loss(
                student_model,
                teacher_model,
                inputs.clone(),
                targets.clone(),
            )
            .mul_scalar(sdft_weight / weight_sum)
                + window_smoke_sdpo_loss(student_model, teacher_model, inputs, targets, *sdpo)
                    .mul_scalar(sdpo_weight / weight_sum)
        }
    }
}

pub fn window_sdft_loss<B: Backend>(
    student_model: &DragonModel<B>,
    teacher_model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    window_smoke_sdft_loss(student_model, teacher_model, inputs, targets)
}

pub fn window_sdpo_loss<B: Backend>(
    student_model: &DragonModel<B>,
    teacher_model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    config: SelectedTokenSdpoLossConfig,
) -> Tensor<B, 1> {
    window_smoke_sdpo_loss(student_model, teacher_model, inputs, targets, config)
}

pub fn window_self_distillation_loss<B: Backend>(
    student_model: &DragonModel<B>,
    teacher_model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    objective: &WindowSelfDistillationObjective,
) -> Tensor<B, 1> {
    window_self_distillation_smoke_loss(student_model, teacher_model, inputs, targets, objective)
}

fn kl_per_token<B: Backend>(
    left_log_probs: Tensor<B, 3>,
    right_log_probs: Tensor<B, 3>,
) -> Tensor<B, 2> {
    let [batch, time, _vocab] = left_log_probs.shape().dims();
    (left_log_probs.clone().exp() * (left_log_probs - right_log_probs))
        .sum_dim(2)
        .reshape([batch, time])
}

pub fn masked_token_mean<B: Backend>(
    values: Tensor<B, 2>,
    mask: Option<Tensor<B, 2, Int>>,
) -> Tensor<B, 1> {
    if let Some(mask) = mask {
        let mask = mask.float();
        return (values * mask.clone())
            .sum()
            .div(mask.sum().clamp_min(1.0))
            .reshape([1]);
    }
    values.mean().reshape([1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_objective_config_accepts_minimal_tagged_sdpo() {
        let objective: TrainingObjectiveConfig =
            serde_json::from_str(r#"{"type":"sdpo"}"#).expect("sdpo objective parses");
        let TrainingObjectiveConfig::Sdpo(config) = objective else {
            panic!("expected sdpo objective");
        };
        assert_eq!(config.alpha, 0.5);
        assert_eq!(config.teacher_regularization, TeacherRegularization::Ema);
        validate_training_objective_config(&TrainingObjectiveConfig::Sdpo(config))
            .expect("default sdpo objective validates");
    }

    #[test]
    fn window_smoke_conversion_uses_shared_sdpo_knobs() {
        let objective = TrainingObjectiveConfig::Sdpo(SdpoObjectiveConfig {
            alpha: 0.0,
            is_clip: Some(1.25),
            ..Default::default()
        });
        let WindowSelfDistillationSmokeObjective::Sdpo(config) =
            objective.to_window_smoke_objective()
        else {
            panic!("expected sdpo smoke objective");
        };
        assert_eq!(config.kl, SelfDistillationKlKind::Forward);
        assert_eq!(config.is_clip, Some(1.25));
    }

    #[test]
    fn composite_window_smoke_conversion_uses_shared_weights() {
        let objective = TrainingObjectiveConfig::SdftSdpo(SdftSdpoObjectiveConfig {
            sdft_weight: 0.25,
            sdpo_weight: 0.75,
            sdpo: SdpoObjectiveConfig {
                alpha: 1.0,
                is_clip: None,
                ..Default::default()
            },
            ..Default::default()
        });
        let WindowSelfDistillationSmokeObjective::SdftSdpo {
            sdft_weight,
            sdpo_weight,
            sdpo,
        } = objective.to_window_smoke_objective()
        else {
            panic!("expected composite smoke objective");
        };
        assert_eq!(sdft_weight, 0.25);
        assert_eq!(sdpo_weight, 0.75);
        assert_eq!(sdpo.kl, SelfDistillationKlKind::Reverse);
        assert_eq!(sdpo.is_clip, None);
    }

    #[test]
    fn browser_support_uses_shared_validation_before_guarding_rollouts() {
        let objective = TrainingObjectiveConfig::Sdft(SdftObjectiveConfig {
            top_k: Some(0),
            ..Default::default()
        });
        let err = objective
            .ensure_browser_supported()
            .expect_err("invalid shared objective should fail before trainer support");
        assert!(
            err.contains("training.objective.top_k"),
            "unexpected error: {err}"
        );

        let objective = TrainingObjectiveConfig::Sdft(Default::default());
        let err = objective
            .ensure_browser_supported()
            .expect_err("browser SDFT should still be guarded");
        assert!(
            err.contains("browser training is only wired for next_token execution"),
            "unexpected error: {err}"
        );
    }
}
