use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::optim::Optimizer;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_core::{DragonConfig, ModelState};
use burn_train::checkpoint::{Checkpointer, FileCheckpointer};
use serde::{Deserialize, Serialize};

use crate::config::PredictiveContextRoutingConfig;
use crate::dataset::scheduler::SequenceBatch;
use crate::train::continual_backprop::LanguageOptimizer;
use crate::train::runtime_checkpoint::StreamingRuntimeStateRecord;
use crate::train::{LanguageTrainModel, balanced_context_mask};

#[derive(Clone)]
pub(crate) struct DragonContextMasks<B: Backend> {
    pub neuron: Tensor<B, 4>,
    pub activity: Tensor<B, 4>,
}

#[derive(Clone)]
pub(crate) struct PredictiveContextValidationRoute<B: Backend> {
    pub identity: burn_pc::PredictiveContextIdentity,
    pub masks: DragonContextMasks<B>,
}

#[derive(Clone)]
pub(crate) struct PredictiveContextValidationRouter<B: Backend> {
    config: PredictiveContextRoutingConfig,
    bank: burn_pc::PredictiveContextBank,
    masks: Vec<DragonContextMasks<B>>,
    device: B::Device,
}

impl<B: Backend> PredictiveContextValidationRouter<B>
where
    B::Device: Clone + 'static,
    B::FloatTensorPrimitive: 'static,
{
    pub(crate) fn known_contexts(&self) -> usize {
        self.bank.known_contexts()
    }

    pub(crate) fn select(
        &self,
        model: &LanguageTrainModel<B>,
        prompt_tokens: &[i64],
    ) -> Result<PredictiveContextValidationRoute<B>> {
        if self.bank.known_contexts() == 0 {
            return Err(anyhow!(
                "predictive context generation requires at least one trained context"
            ));
        }
        if prompt_tokens.len() < 2 {
            let identity = self
                .bank
                .identity(0)
                .ok_or_else(|| anyhow!("predictive context slot zero is missing"))?;
            return Ok(PredictiveContextValidationRoute {
                identity,
                masks: self.masks[0].clone(),
            });
        }

        let probe_time = self
            .config
            .probe_tokens
            .min(prompt_tokens.len().saturating_sub(1))
            .max(1);
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(prompt_tokens[..probe_time].to_vec(), [1, probe_time]),
            &self.device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(prompt_tokens[1..=probe_time].to_vec(), [1, probe_time]),
            &self.device,
        );
        self.select_tensors(model, inputs, targets, None)
    }

    pub(crate) fn select_batch(
        &self,
        model: &LanguageTrainModel<B>,
        batch: &SequenceBatch<B>,
    ) -> Result<PredictiveContextValidationRoute<B>> {
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        let probe_time = self.config.probe_tokens.min(block_size).max(1);
        let inputs =
            LanguageTrainModel::<B>::slice_tokens(batch.inputs.clone(), batch_size, 0, probe_time);
        let targets =
            LanguageTrainModel::<B>::slice_tokens(batch.targets.clone(), batch_size, 0, probe_time);
        let loss_mask = batch
            .loss_mask
            .clone()
            .map(|mask| LanguageTrainModel::<B>::slice_tokens(mask, batch_size, 0, probe_time));
        self.select_tensors(model, inputs, targets, loss_mask)
    }

    fn select_tensors(
        &self,
        model: &LanguageTrainModel<B>,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Result<PredictiveContextValidationRoute<B>> {
        if self.bank.known_contexts() == 0 {
            return Err(anyhow!(
                "predictive context validation requires at least one trained context"
            ));
        }
        let losses = self
            .masks
            .iter()
            .take(self.bank.known_contexts())
            .map(|masks| {
                let logits = model
                    .model
                    .predictive_coding_forward_with_subnetwork_masks(
                        inputs.clone(),
                        masks.neuron.clone(),
                        masks.activity.clone(),
                    )
                    .map_err(anyhow::Error::msg)?;
                Ok(burn_dragon_core::objective::masked_token_mean(
                    model
                        .model
                        .language_token_losses_from_logits(logits, targets.clone()),
                    loss_mask.clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let losses = Tensor::cat(losses, 0)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|error| anyhow!("read predictive context generation losses: {error}"))?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>();
        let selection = self
            .bank
            .select(&losses, false)
            .map_err(anyhow::Error::msg)?;
        let identity = self
            .bank
            .identity(selection.context_index)
            .ok_or_else(|| anyhow!("validation selected a missing predictive context"))?;
        Ok(PredictiveContextValidationRoute {
            identity,
            masks: self.masks[selection.context_index].clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PredictiveContextRoutingDecision {
    pub identity: burn_pc::PredictiveContextIdentity,
    pub created: bool,
    pub replaced: Option<burn_pc::PredictiveContextIdentity>,
    pub novelty_deferred: bool,
    pub probed: bool,
    pub probe_tokens: usize,
    pub selected_loss: Option<f64>,
    pub reserve_loss: Option<f64>,
    pub reserve_supported_novelty: Option<bool>,
    pub candidates: Vec<burn_pc::PredictiveContextCandidate>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct PredictiveContextRoutingCheckpoint {
    pub schema_version: u32,
    pub bank: burn_pc::PredictiveContextBank,
    pub novelty_gate: burn_pc::PredictiveContextNoveltyGate,
    pub current: Option<burn_pc::PredictiveContextIdentity>,
    pub optimizer_generations: Vec<u64>,
}

#[derive(burn::record::Record, Clone, Debug)]
struct PredictiveContextStreamStatesRecord<B: Backend> {
    states: Vec<Option<StreamingRuntimeStateRecord<B>>>,
}

pub(crate) struct PredictiveContextRoutingRuntime<B: AutodiffBackend> {
    config: PredictiveContextRoutingConfig,
    seed: u64,
    n_head: usize,
    latent_per_head: usize,
    n_embd: usize,
    bank: burn_pc::PredictiveContextBank,
    novelty_gate: burn_pc::PredictiveContextNoveltyGate,
    current: Option<burn_pc::PredictiveContextIdentity>,
    masks: Vec<DragonContextMasks<B>>,
    neuron_mask_values: Vec<Vec<f32>>,
    activity_mask_values: Vec<Vec<f32>>,
    stream_states: Vec<Option<ModelState<B>>>,
    optimizers: Vec<LanguageOptimizer<B>>,
    fresh_optimizer: LanguageOptimizer<B>,
    device: B::Device,
    pub checkpoint_path: PathBuf,
}

impl<B: AutodiffBackend> PredictiveContextRoutingRuntime<B>
where
    B::Device: Clone + 'static,
    B::FloatTensorPrimitive: 'static,
{
    pub(crate) fn new(
        config: PredictiveContextRoutingConfig,
        model_config: &DragonConfig,
        seed: u64,
        device: &B::Device,
        optimizer: LanguageOptimizer<B>,
        checkpoint_path: PathBuf,
    ) -> Result<Self> {
        config.bank.validate().map_err(anyhow::Error::msg)?;
        let bank = burn_pc::PredictiveContextBank::new(config.bank).map_err(anyhow::Error::msg)?;
        let novelty_gate = burn_pc::PredictiveContextNoveltyGate::new(config.novelty_confirmations)
            .map_err(anyhow::Error::msg)?;
        Ok(Self {
            seed,
            n_head: model_config.n_head,
            latent_per_head: model_config.latent_total() / model_config.n_head.max(1),
            n_embd: model_config.n_embd,
            config,
            bank,
            novelty_gate,
            current: None,
            masks: Vec::new(),
            neuron_mask_values: Vec::new(),
            activity_mask_values: Vec::new(),
            stream_states: Vec::new(),
            optimizers: Vec::new(),
            fresh_optimizer: optimizer,
            device: device.clone(),
            checkpoint_path,
        })
    }

    pub(crate) fn checkpoint(&self) -> PredictiveContextRoutingCheckpoint {
        PredictiveContextRoutingCheckpoint {
            schema_version: 1,
            bank: self.bank.clone(),
            novelty_gate: self.novelty_gate,
            current: self.current,
            optimizer_generations: self
                .bank
                .lifecycles()
                .iter()
                .map(|lifecycle| lifecycle.generation)
                .collect(),
        }
    }

    pub(crate) fn known_contexts(&self) -> usize {
        self.bank.known_contexts()
    }

    pub(crate) fn current_identity(&self) -> Option<burn_pc::PredictiveContextIdentity> {
        self.current
    }

    pub(crate) fn validation_router(&self) -> PredictiveContextValidationRouter<B::InnerBackend> {
        PredictiveContextValidationRouter {
            config: self.config.clone(),
            bank: self.bank.clone(),
            masks: self
                .masks
                .iter()
                .take(self.bank.known_contexts())
                .map(|masks| DragonContextMasks {
                    neuron: masks.neuron.clone().inner(),
                    activity: masks.activity.clone().inner(),
                })
                .collect(),
            device: self.device.clone(),
        }
    }

    fn build_slot(&self, slot: usize) -> Result<(DragonContextMasks<B>, Vec<f32>, Vec<f32>)> {
        let neuron_values = balanced_context_mask(
            self.seed ^ 0xA11C_E5E1_7A11_0001,
            slot,
            self.n_head.saturating_mul(self.latent_per_head),
            self.config.active_fraction,
            &self.neuron_mask_values,
        )
        .map_err(anyhow::Error::msg)?;
        let activity_values = balanced_context_mask(
            self.seed ^ 0xAC71_0171_5EED_0002,
            slot,
            self.n_embd,
            self.config.active_fraction,
            &self.activity_mask_values,
        )
        .map_err(anyhow::Error::msg)?;
        let neuron = Tensor::<B, 4>::from_data(
            TensorData::new(
                neuron_values.clone(),
                [1, self.n_head, 1, self.latent_per_head],
            ),
            &self.device,
        );
        let activity = Tensor::<B, 4>::from_data(
            TensorData::new(activity_values.clone(), [1, 1, 1, self.n_embd]),
            &self.device,
        );
        Ok((
            DragonContextMasks { neuron, activity },
            neuron_values,
            activity_values,
        ))
    }

    fn ensure_slot(&mut self, context_index: usize) -> Result<()> {
        while self.masks.len() <= context_index {
            let (masks, neuron_values, activity_values) = self.build_slot(self.masks.len())?;
            self.neuron_mask_values.push(neuron_values);
            self.activity_mask_values.push(activity_values);
            self.masks.push(masks);
            self.stream_states.push(None);
            self.optimizers.push(self.fresh_optimizer.clone());
        }
        Ok(())
    }

    fn archive_replaced_optimizer(
        &self,
        identity: burn_pc::PredictiveContextIdentity,
        absolute_step: usize,
    ) -> Result<()> {
        let directory = self.checkpoint_path.join("context-archive").join(format!(
            "slot-{}-generation-{}",
            identity.context_index, identity.generation
        ));
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        FileCheckpointer::new(recorder, &directory, "optimizer")
            .save(
                absolute_step,
                self.optimizers[identity.context_index].to_record(),
            )
            .with_context(|| {
                format!("archive predictive context optimizer {identity:?} at step {absolute_step}")
            })
    }

    fn allocate(
        &mut self,
        selection: &burn_pc::PredictiveContextSelection,
        absolute_step: usize,
    ) -> Result<burn_pc::PredictiveContextAllocation> {
        if let Some(replaced) = selection.replacement {
            self.archive_replaced_optimizer(replaced, absolute_step)?;
        }
        let allocation = self
            .bank
            .allocate_selected(selection)
            .map_err(anyhow::Error::msg)?;
        self.ensure_slot(allocation.identity.context_index)?;
        if allocation.replaced.is_some() {
            self.optimizers[allocation.identity.context_index] = self.fresh_optimizer.clone();
            self.stream_states[allocation.identity.context_index] = None;
        }
        Ok(allocation)
    }

    pub(crate) fn save_checkpoint(&self, run_dir: &Path, epoch: usize) -> Result<()> {
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).with_context(|| {
            format!(
                "create predictive context checkpoint directory {}",
                checkpoint_dir.display()
            )
        })?;
        let checkpoint = self.checkpoint();
        let metadata_path = checkpoint_dir.join(format!("context-routing-{epoch}.json"));
        fs::write(
            &metadata_path,
            serde_json::to_vec_pretty(&checkpoint)
                .context("serialize predictive context checkpoint")?,
        )
        .with_context(|| format!("write {}", metadata_path.display()))?;

        let states = PredictiveContextStreamStatesRecord {
            states: self
                .stream_states
                .iter()
                .map(|state| state.clone().map(StreamingRuntimeStateRecord::from))
                .collect(),
        };
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        FileCheckpointer::new(recorder, &checkpoint_dir, "context-stream-states")
            .save(epoch, states)
            .with_context(|| format!("save predictive context stream states for epoch {epoch}"))?;
        for (context_index, optimizer) in self.optimizers.iter().enumerate() {
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            let prefix = format!("context-optimizer-{context_index}");
            FileCheckpointer::new(recorder, &checkpoint_dir, &prefix)
                .save(epoch, optimizer.to_record())
                .with_context(|| {
                    format!("save predictive context optimizer {context_index} for epoch {epoch}")
                })?;
        }
        Ok(())
    }

    pub(crate) fn restore_checkpoint(
        &mut self,
        run_dir: &Path,
        epoch: usize,
        require_exact: bool,
    ) -> Result<bool> {
        let checkpoint_dir = run_dir.join("checkpoint");
        let metadata_path = checkpoint_dir.join(format!("context-routing-{epoch}.json"));
        if !metadata_path.is_file() {
            if require_exact {
                return Err(anyhow!(
                    "exact resume requires predictive context checkpoint {}",
                    metadata_path.display()
                ));
            }
            return Ok(false);
        }
        let checkpoint: PredictiveContextRoutingCheckpoint = serde_json::from_slice(
            &fs::read(&metadata_path)
                .with_context(|| format!("read {}", metadata_path.display()))?,
        )
        .with_context(|| format!("parse {}", metadata_path.display()))?;
        if checkpoint.schema_version != 1 {
            return Err(anyhow!(
                "unsupported predictive context checkpoint schema {}",
                checkpoint.schema_version
            ));
        }
        if checkpoint.bank.config() != self.config.bank {
            return Err(anyhow!(
                "predictive context checkpoint bank configuration does not match the training contract"
            ));
        }
        self.bank = checkpoint.bank;
        self.novelty_gate = checkpoint.novelty_gate;
        self.current = checkpoint.current;
        for context_index in 0..self.bank.known_contexts() {
            self.ensure_slot(context_index)?;
        }
        let generations = self
            .bank
            .lifecycles()
            .iter()
            .map(|lifecycle| lifecycle.generation)
            .collect::<Vec<_>>();
        if generations != checkpoint.optimizer_generations {
            return Err(anyhow!(
                "predictive context optimizer generations do not match context identities"
            ));
        }
        if let Some(current) = self.current
            && self.bank.identity(current.context_index) != Some(current)
        {
            return Err(anyhow!(
                "predictive context checkpoint has stale current identity {current:?}"
            ));
        }
        for context_index in 0..self.bank.known_contexts() {
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            let prefix = format!("context-optimizer-{context_index}");
            let record = FileCheckpointer::new(recorder, &checkpoint_dir, &prefix)
                .restore(epoch, &self.device)
                .with_context(|| {
                    format!(
                        "restore predictive context optimizer {context_index} for epoch {epoch}"
                    )
                })?;
            self.optimizers[context_index] =
                self.optimizers[context_index].clone().load_record(record);
        }
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let states: PredictiveContextStreamStatesRecord<B> =
            FileCheckpointer::new(recorder, &checkpoint_dir, "context-stream-states")
                .restore(epoch, &self.device)
                .with_context(|| {
                    format!("restore predictive context stream states for epoch {epoch}")
                })?;
        if states.states.len() != self.bank.known_contexts() {
            return Err(anyhow!(
                "predictive context state count {} does not match context count {}",
                states.states.len(),
                self.bank.known_contexts()
            ));
        }
        self.stream_states = states
            .states
            .into_iter()
            .map(|state| state.map(ModelState::from))
            .collect();
        Ok(true)
    }

    fn read_losses(losses: Vec<Tensor<B::InnerBackend, 1>>) -> Result<Vec<f64>> {
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

    fn probe_losses(
        &self,
        model: &LanguageTrainModel<B>,
        batch: &SequenceBatch<B>,
    ) -> Result<Vec<f64>> {
        let losses = self
            .masks
            .iter()
            .take(self.bank.known_contexts())
            .map(|masks| {
                model.predictive_context_probe_loss(
                    batch,
                    masks.neuron.clone(),
                    masks.activity.clone(),
                    self.config.probe_tokens,
                )
            })
            .collect();
        Self::read_losses(losses)
    }

    pub(crate) fn route(
        &mut self,
        model: &LanguageTrainModel<B>,
        batch: &SequenceBatch<B>,
        absolute_step: usize,
    ) -> Result<PredictiveContextRoutingDecision> {
        if self.bank.known_contexts() == 0 {
            let selection = self.bank.select(&[], true).map_err(anyhow::Error::msg)?;
            let allocation = self.allocate(&selection, absolute_step)?;
            let masks = self.masks[allocation.identity.context_index].clone();
            let selected_loss = Self::read_losses(vec![model.predictive_context_probe_loss(
                batch,
                masks.neuron,
                masks.activity,
                self.config.probe_tokens,
            )])?[0];
            self.bank
                .observe(allocation.identity.context_index, selected_loss)
                .map_err(anyhow::Error::msg)?;
            self.current = Some(allocation.identity);
            return Ok(PredictiveContextRoutingDecision {
                identity: allocation.identity,
                created: true,
                replaced: allocation.replaced,
                novelty_deferred: false,
                probed: true,
                probe_tokens: batch.inputs.shape().dims::<2>()[0]
                    * self
                        .config
                        .probe_tokens
                        .min(batch.inputs.shape().dims::<2>()[1]),
                selected_loss: Some(selected_loss),
                reserve_loss: None,
                reserve_supported_novelty: None,
                candidates: Vec::new(),
            });
        }

        let probe_due =
            self.current.is_none() || absolute_step.is_multiple_of(self.config.probe_every_steps);
        if !probe_due {
            let identity = self
                .current
                .ok_or_else(|| anyhow!("predictive context routing has no current context"))?;
            return Ok(PredictiveContextRoutingDecision {
                identity,
                created: false,
                replaced: None,
                novelty_deferred: false,
                probed: false,
                probe_tokens: 0,
                selected_loss: None,
                reserve_loss: None,
                reserve_supported_novelty: None,
                candidates: Vec::new(),
            });
        }

        let mut losses = self.probe_losses(model, batch)?;
        let mut probe_evaluations = losses.len();
        let mut selection = self
            .bank
            .select(&losses, true)
            .map_err(anyhow::Error::msg)?;
        let mut novelty_deferred = false;
        let mut reserve_loss = None;
        let mut reserve_supported_novelty = None;
        if selection.created {
            if selection.replacement.is_none()
                && self.bank.known_contexts() < self.bank.config().max_contexts
            {
                let masks = self.build_slot(self.bank.known_contexts())?.0;
                let loss = Self::read_losses(vec![model.predictive_context_probe_loss(
                    batch,
                    masks.neuron,
                    masks.activity,
                    self.config.probe_tokens,
                )])?[0];
                let supported = self
                    .bank
                    .reserve_supports_novelty(&selection, loss)
                    .map_err(anyhow::Error::msg)?;
                reserve_loss = Some(loss);
                reserve_supported_novelty = Some(supported);
                selection.novel_evidence &= supported;
                losses.push(loss);
                probe_evaluations += 1;
            }
            if !self.novelty_gate.observe(selection.novel_evidence) {
                let fallback = selection
                    .candidates
                    .iter()
                    .min_by(|left, right| left.loss.total_cmp(&right.loss))
                    .ok_or_else(|| anyhow!("novel context selection has no fallback candidate"))?;
                selection.context_index = fallback.context_index;
                selection.created = false;
                selection.replacement = None;
                novelty_deferred = selection.novel_evidence;
            }
        } else {
            self.novelty_gate.observe(selection.novel_evidence);
        }

        let (identity, replaced, selected_loss) = if selection.created {
            let allocation = self.allocate(&selection, absolute_step)?;
            let loss = if let Some(loss) = reserve_loss {
                loss
            } else {
                let masks = self.masks[allocation.identity.context_index].clone();
                let loss = Self::read_losses(vec![model.predictive_context_probe_loss(
                    batch,
                    masks.neuron,
                    masks.activity,
                    self.config.probe_tokens,
                )])?[0];
                if allocation.identity.context_index == losses.len() {
                    losses.push(loss);
                } else {
                    losses[allocation.identity.context_index] = loss;
                }
                probe_evaluations += 1;
                loss
            };
            (allocation.identity, allocation.replaced, loss)
        } else {
            let identity = self
                .bank
                .identity(selection.context_index)
                .ok_or_else(|| anyhow!("selected predictive context disappeared"))?;
            (identity, None, losses[selection.context_index])
        };
        if novelty_deferred {
            self.bank.touch(identity).map_err(anyhow::Error::msg)?;
        } else {
            self.bank
                .observe(identity.context_index, selected_loss)
                .map_err(anyhow::Error::msg)?;
        }
        self.current = Some(identity);
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        Ok(PredictiveContextRoutingDecision {
            identity,
            created: selection.created,
            replaced,
            novelty_deferred,
            probed: true,
            probe_tokens: probe_evaluations * batch_size * self.config.probe_tokens.min(block_size),
            selected_loss: Some(selected_loss),
            reserve_loss,
            reserve_supported_novelty,
            candidates: selection.candidates,
        })
    }

    pub(crate) fn validation_loss(
        &self,
        model: &LanguageTrainModel<B::InnerBackend>,
        batch: SequenceBatch<B::InnerBackend>,
        degeneracy_probe_tokens: usize,
        eos_id: Option<i64>,
    ) -> Result<(
        Tensor<B::InnerBackend, 1>,
        burn_pc::PredictiveContextIdentity,
        Option<crate::train::steps::OutputDegeneracyStats>,
    )> {
        if self.bank.known_contexts() == 0 {
            return Err(anyhow!(
                "predictive context validation requires at least one trained context"
            ));
        }
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        let probe_time = self.config.probe_tokens.min(block_size).max(1);
        let probe_inputs = LanguageTrainModel::<B::InnerBackend>::slice_tokens(
            batch.inputs.clone(),
            batch_size,
            0,
            probe_time,
        );
        let probe_targets = LanguageTrainModel::<B::InnerBackend>::slice_tokens(
            batch.targets.clone(),
            batch_size,
            0,
            probe_time,
        );
        let probe_mask = batch.loss_mask.clone().map(|mask| {
            LanguageTrainModel::<B::InnerBackend>::slice_tokens(mask, batch_size, 0, probe_time)
        });
        let plain = &model.model;
        let losses = self
            .masks
            .iter()
            .take(self.bank.known_contexts())
            .map(|masks| {
                let logits = plain
                    .predictive_coding_forward_with_subnetwork_masks(
                        probe_inputs.clone(),
                        masks.neuron.clone().inner(),
                        masks.activity.clone().inner(),
                    )
                    .expect("validated predictive context masks");
                burn_dragon_core::objective::masked_token_mean(
                    plain.language_token_losses_from_logits(logits, probe_targets.clone()),
                    probe_mask.clone(),
                )
            })
            .collect();
        let losses = Self::read_losses(losses)?;
        let selection = self
            .bank
            .select(&losses, false)
            .map_err(anyhow::Error::msg)?;
        let identity = self
            .bank
            .identity(selection.context_index)
            .ok_or_else(|| anyhow!("validation selected a missing predictive context"))?;
        let masks = &self.masks[selection.context_index];
        let (loss, degeneracy) = model.validation_loss_and_output_degeneracy_with_subnetwork_masks(
            batch,
            masks.neuron.clone().inner(),
            masks.activity.clone().inner(),
            degeneracy_probe_tokens,
            eos_id,
        );
        Ok((loss, identity, degeneracy))
    }

    pub(crate) fn masks(
        &self,
        identity: burn_pc::PredictiveContextIdentity,
    ) -> Result<DragonContextMasks<B>> {
        if self.bank.identity(identity.context_index) != Some(identity) {
            return Err(anyhow!("stale predictive context identity {identity:?}"));
        }
        self.masks
            .get(identity.context_index)
            .cloned()
            .ok_or_else(|| {
                anyhow!(
                    "predictive context mask {} is missing",
                    identity.context_index
                )
            })
    }

    pub(crate) fn take_stream_state(
        &mut self,
        identity: burn_pc::PredictiveContextIdentity,
        reset: bool,
    ) -> Result<Option<ModelState<B>>> {
        if self.bank.identity(identity.context_index) != Some(identity) {
            return Err(anyhow!(
                "stale predictive context state identity {identity:?}"
            ));
        }
        if reset {
            self.stream_states[identity.context_index] = None;
        }
        Ok(self.stream_states[identity.context_index].take())
    }

    pub(crate) fn store_stream_state(
        &mut self,
        identity: burn_pc::PredictiveContextIdentity,
        state: Option<ModelState<B>>,
    ) -> Result<()> {
        if self.bank.identity(identity.context_index) != Some(identity) {
            return Err(anyhow!(
                "stale predictive context state identity {identity:?}"
            ));
        }
        self.stream_states[identity.context_index] = state;
        Ok(())
    }

    pub(crate) fn optimizer_mut(
        &mut self,
        identity: burn_pc::PredictiveContextIdentity,
    ) -> Result<&mut LanguageOptimizer<B>> {
        if self.bank.identity(identity.context_index) != Some(identity) {
            return Err(anyhow!(
                "stale predictive context optimizer identity {identity:?}"
            ));
        }
        self.optimizers
            .get_mut(identity.context_index)
            .ok_or_else(|| {
                anyhow!(
                    "predictive context optimizer {} is missing",
                    identity.context_index
                )
            })
    }
}
