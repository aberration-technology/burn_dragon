use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn_dragon_core::{LayerState, ModelState};
use burn_train::checkpoint::{Checkpointer, FileCheckpointer};

use super::LanguageTrainModel;

const RUNTIME_STATE_PREFIX: &str = "runtime-state";
const TEACHER_MODEL_PREFIX: &str = "teacher-model";
const PC_MANIFEST_PREFIX: &str = "predictive-coding-manifest";
const PC_RUN_MANIFEST: &str = "predictive-coding-program.json";

#[derive(burn::record::Record, Clone, Debug)]
pub(crate) struct LayerRuntimeStateRecord<B: Backend> {
    persist_sequence_state: bool,
    retain_terminal_sequence_state: bool,
    rho: Option<Tensor<B, 4>>,
    rho_norm: Option<Tensor<B, 3>>,
    sequence_aux: Option<Tensor<B, 4>>,
    mamba_angle_state: Option<Tensor<B, 3>>,
    mamba_k_state: Option<Tensor<B, 3>>,
    mamba_v_state: Option<Tensor<B, 3>>,
    slow_rho: Option<Tensor<B, 4>>,
    slow_rho_norm: Option<Tensor<B, 3>>,
    slow_sequence_aux: Option<Tensor<B, 4>>,
    slow_mamba_angle_state: Option<Tensor<B, 3>>,
    slow_mamba_k_state: Option<Tensor<B, 3>>,
    slow_mamba_v_state: Option<Tensor<B, 3>>,
    y_neuron_state: Option<Tensor<B, 3>>,
    hierarchical_slow_hidden: Option<Tensor<B, 4>>,
    clocked_slow_hidden: Option<Tensor<B, 4>>,
    summary_memory_hidden: Option<Tensor<B, 4>>,
}

#[derive(burn::record::Record, Clone, Debug)]
pub(crate) struct StreamingRuntimeStateRecord<B: Backend> {
    position: usize,
    layers: Vec<LayerRuntimeStateRecord<B>>,
}

#[derive(burn::record::Record, Clone, Debug)]
pub(crate) struct DkpFeedbackStateRecord<B: Backend> {
    feedback: Tensor<B, 3>,
    updates: u64,
}

impl<B: Backend> From<super::local_predictive_coding::DkpFeedbackState<B>>
    for DkpFeedbackStateRecord<B>
{
    fn from(state: super::local_predictive_coding::DkpFeedbackState<B>) -> Self {
        Self {
            feedback: state.feedback,
            updates: state.updates,
        }
    }
}

impl<B: Backend> From<DkpFeedbackStateRecord<B>>
    for super::local_predictive_coding::DkpFeedbackState<B>
{
    fn from(record: DkpFeedbackStateRecord<B>) -> Self {
        Self {
            feedback: record.feedback,
            updates: record.updates,
        }
    }
}

impl<B: Backend> From<ModelState<B>> for StreamingRuntimeStateRecord<B> {
    fn from(state: ModelState<B>) -> Self {
        Self {
            position: state.position,
            layers: state
                .layers
                .into_iter()
                .map(|layer| LayerRuntimeStateRecord {
                    persist_sequence_state: layer.persist_sequence_state,
                    retain_terminal_sequence_state: layer.retain_terminal_sequence_state,
                    rho: layer.rho,
                    rho_norm: layer.rho_norm,
                    sequence_aux: layer.sequence_aux,
                    mamba_angle_state: layer.mamba_angle_state,
                    mamba_k_state: layer.mamba_k_state,
                    mamba_v_state: layer.mamba_v_state,
                    slow_rho: layer.slow_rho,
                    slow_rho_norm: layer.slow_rho_norm,
                    slow_sequence_aux: layer.slow_sequence_aux,
                    slow_mamba_angle_state: layer.slow_mamba_angle_state,
                    slow_mamba_k_state: layer.slow_mamba_k_state,
                    slow_mamba_v_state: layer.slow_mamba_v_state,
                    y_neuron_state: layer.y_neuron_state,
                    hierarchical_slow_hidden: layer.hierarchical_slow_hidden,
                    clocked_slow_hidden: layer.clocked_slow_hidden,
                    summary_memory_hidden: layer.summary_memory_hidden,
                })
                .collect(),
        }
    }
}

impl<B: Backend> From<StreamingRuntimeStateRecord<B>> for ModelState<B> {
    fn from(record: StreamingRuntimeStateRecord<B>) -> Self {
        Self {
            position: record.position,
            layers: record
                .layers
                .into_iter()
                .map(|layer| LayerState {
                    persist_sequence_state: layer.persist_sequence_state,
                    retain_terminal_sequence_state: layer.retain_terminal_sequence_state,
                    rho: layer.rho,
                    rho_norm: layer.rho_norm,
                    sequence_aux: layer.sequence_aux,
                    mamba_angle_state: layer.mamba_angle_state,
                    mamba_k_state: layer.mamba_k_state,
                    mamba_v_state: layer.mamba_v_state,
                    slow_rho: layer.slow_rho,
                    slow_rho_norm: layer.slow_rho_norm,
                    slow_sequence_aux: layer.slow_sequence_aux,
                    slow_mamba_angle_state: layer.slow_mamba_angle_state,
                    slow_mamba_k_state: layer.slow_mamba_k_state,
                    slow_mamba_v_state: layer.slow_mamba_v_state,
                    y_neuron_state: layer.y_neuron_state,
                    hierarchical_slow_hidden: layer.hierarchical_slow_hidden,
                    clocked_slow_hidden: layer.clocked_slow_hidden,
                    summary_memory_hidden: layer.summary_memory_hidden,
                    #[cfg(any(feature = "viz", feature = "probe"))]
                    viz: None,
                })
                .collect(),
        }
    }
}

#[derive(burn::record::Record, Clone, Debug)]
pub(crate) struct LanguageRuntimeStateRecord<B: Backend> {
    gradient_scale_step: usize,
    teacher_update_count: Option<usize>,
    streaming_state: Option<StreamingRuntimeStateRecord<B>>,
    dkp_feedback: Option<DkpFeedbackStateRecord<B>>,
}

pub(crate) fn runtime_state_checkpoint_path(run_dir: &Path, epoch: usize) -> PathBuf {
    run_dir
        .join("checkpoint")
        .join(format!("{RUNTIME_STATE_PREFIX}-{epoch}.bin"))
}

pub(crate) fn predictive_coding_manifest_checkpoint_path(run_dir: &Path, epoch: usize) -> PathBuf {
    run_dir
        .join("checkpoint")
        .join(format!("{PC_MANIFEST_PREFIX}-{epoch}.json"))
}

fn save_predictive_coding_manifest(
    run_dir: &Path,
    epoch: usize,
    manifest: &burn_pc::PcCheckpointManifest,
) -> Result<()> {
    manifest.validate()?;
    let path = predictive_coding_manifest_checkpoint_path(run_dir, epoch);
    let temporary = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(manifest).context("serialize PC checkpoint manifest")?;
    fs::write(&temporary, bytes).with_context(|| {
        format!(
            "write temporary PC checkpoint manifest {}",
            temporary.display()
        )
    })?;
    fs::rename(&temporary, &path)
        .with_context(|| format!("publish PC checkpoint manifest {}", path.display()))
}

fn validate_predictive_coding_manifest(
    run_dir: &Path,
    epoch: usize,
    expected: Option<&burn_pc::PcCheckpointManifest>,
    require_exact: bool,
) -> Result<()> {
    let path = predictive_coding_manifest_checkpoint_path(run_dir, epoch);
    let exists = path.is_file();
    match (expected, exists) {
        (Some(_), false) if require_exact => Err(anyhow!(
            "exact predictive-coding resume requires program manifest {}",
            path.display()
        )),
        (Some(_), false) | (None, false) => Ok(()),
        (None, true) if require_exact => Err(anyhow!(
            "exact resume requested a non-PC program but checkpoint {} declares predictive coding",
            path.display()
        )),
        (None, true) => Ok(()),
        (Some(expected), true) => {
            let bytes = fs::read(&path)
                .with_context(|| format!("read PC checkpoint manifest {}", path.display()))?;
            let actual: burn_pc::PcCheckpointManifest = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse PC checkpoint manifest {}", path.display()))?;
            actual.validate_resume(expected).with_context(|| {
                format!(
                    "predictive-coding executable does not match checkpoint {}",
                    path.display()
                )
            })
        }
    }
}

/// Validate the selected checkpoint before Burn restores it, then publish the
/// run-level identity used by observers and peers. The selected checkpoint's
/// sidecar remains authoritative for exact resume.
pub(crate) fn prepare_predictive_coding_checkpoint_contract(
    run_dir: &Path,
    resume_epoch: Option<usize>,
    expected: Option<&burn_pc::PcCheckpointManifest>,
    require_exact: bool,
) -> Result<()> {
    if let Some(epoch) = resume_epoch {
        validate_predictive_coding_manifest(run_dir, epoch, expected, require_exact)?;
    }
    if let Some(expected) = expected {
        expected.validate()?;
        let path = run_dir.join(PC_RUN_MANIFEST);
        let temporary = path.with_extension("json.tmp");
        fs::write(
            &temporary,
            serde_json::to_vec_pretty(expected).context("serialize PC run manifest")?,
        )
        .with_context(|| format!("write temporary PC run manifest {}", temporary.display()))?;
        fs::rename(&temporary, &path)
            .with_context(|| format!("publish PC run manifest {}", path.display()))?;
    }
    Ok(())
}

/// Burn's standard learner owns model checkpoint timing and does not expose a
/// callback carrying Dragon's skipped runtime fields. For non-streaming local
/// PC, synchronize immutable program sidecars after the learner has flushed
/// its asynchronous checkpointers. Streaming PC uses Dragon's event scheduler
/// and writes each sidecar directly in `save_runtime_state_checkpoint`.
pub(crate) fn synchronize_predictive_coding_checkpoint_manifests(
    run_dir: &Path,
    manifest: Option<&burn_pc::PcCheckpointManifest>,
) -> Result<usize> {
    let Some(manifest) = manifest else {
        return Ok(0);
    };
    let checkpoint_dir = run_dir.join("checkpoint");
    if !checkpoint_dir.is_dir() {
        return Ok(0);
    }
    let mut model_epochs = Vec::new();
    for entry in fs::read_dir(&checkpoint_dir)
        .with_context(|| format!("read checkpoint directory {}", checkpoint_dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(epoch) = name
            .strip_prefix("model-")
            .and_then(|name| name.strip_suffix(".bin"))
            .and_then(|value| value.parse::<usize>().ok())
        else {
            continue;
        };
        model_epochs.push(epoch);
    }
    model_epochs.sort_unstable();
    for epoch in &model_epochs {
        save_predictive_coding_manifest(run_dir, *epoch, manifest)?;
    }
    Ok(model_epochs.len())
}

pub(crate) fn save_runtime_state_checkpoint<B>(
    run_dir: &Path,
    epoch: usize,
    model: &LanguageTrainModel<B>,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
{
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let teacher = model.teacher_model_for_checkpoint();
    let record = LanguageRuntimeStateRecord {
        gradient_scale_step: model.gradient_scale_step_for_checkpoint(),
        teacher_update_count: teacher.as_ref().map(|(_, update_count)| *update_count),
        streaming_state: model
            .streaming_state_for_checkpoint()
            .map(StreamingRuntimeStateRecord::from),
        dkp_feedback: model
            .dkp_feedback_for_checkpoint()
            .map(DkpFeedbackStateRecord::from),
    };
    FileCheckpointer::new(recorder, &checkpoint_dir, RUNTIME_STATE_PREFIX)
        .save(epoch, record)
        .with_context(|| {
            format!(
                "failed to save Dragon runtime-state checkpoint {epoch} in {}",
                checkpoint_dir.display()
            )
        })?;
    if let Some((teacher, _)) = teacher {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        FileCheckpointer::new(recorder, &checkpoint_dir, TEACHER_MODEL_PREFIX)
            .save(epoch, teacher.into_record())
            .with_context(|| {
                format!(
                    "failed to save Dragon teacher-model checkpoint {epoch} in {}",
                    checkpoint_dir.display()
                )
            })?;
    }
    if let Some(manifest) = model.predictive_coding_checkpoint_manifest() {
        save_predictive_coding_manifest(run_dir, epoch, &manifest)?;
    }
    Ok(())
}

pub(crate) fn load_runtime_state_checkpoint<B>(
    run_dir: &Path,
    epoch: usize,
    model: &LanguageTrainModel<B>,
    device: &B::Device,
    require_exact: bool,
    external_streaming_state: bool,
) -> Result<bool>
where
    B: AutodiffBackend + Clone + 'static,
{
    let path = runtime_state_checkpoint_path(run_dir, epoch);
    if !path.is_file() {
        if require_exact {
            return Err(anyhow!(
                "exact resume requires runtime-state checkpoint {}",
                path.display()
            ));
        }
        return Ok(false);
    }
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record: LanguageRuntimeStateRecord<B> =
        FileCheckpointer::new(recorder, &checkpoint_dir, RUNTIME_STATE_PREFIX)
            .restore(epoch, device)
            .with_context(|| {
                format!(
                    "failed to restore Dragon runtime-state checkpoint {epoch} from {}",
                    checkpoint_dir.display()
                )
            })?;
    let expected_pc_manifest = model.predictive_coding_checkpoint_manifest();
    validate_predictive_coding_manifest(
        run_dir,
        epoch,
        expected_pc_manifest.as_ref(),
        require_exact,
    )?;
    model.restore_gradient_scale_step_from_checkpoint(record.gradient_scale_step);
    if model.uses_local_pc_feedback_state() && require_exact && record.dkp_feedback.is_none() {
        return Err(anyhow!(
            "exact local-PC resume requires feedback-bank state in {}",
            path.display()
        ));
    }
    if !model.uses_local_pc_feedback_state() && require_exact && record.dkp_feedback.is_some() {
        return Err(anyhow!(
            "runtime-state checkpoint {} contains a local-PC feedback bank but the requested solver does not own one",
            path.display()
        ));
    }
    model.restore_dkp_feedback_from_checkpoint(record.dkp_feedback.map(Into::into));
    match record.streaming_state {
        Some(state) => {
            if !model.tbptt_persist_across_steps {
                return Err(anyhow!(
                    "runtime-state checkpoint {} contains streaming state but training.tbptt_persist_across_steps=false",
                    path.display()
                ));
            }
            model
                .restore_streaming_state_from_checkpoint(state.into())
                .map_err(anyhow::Error::msg)?;
        }
        None if model.tbptt_persist_across_steps && require_exact && !external_streaming_state => {
            return Err(anyhow!(
                "exact resume requires streaming state in {} because training.tbptt_persist_across_steps=true",
                path.display()
            ));
        }
        None => {}
    }
    let expects_teacher = model.teacher_model_for_checkpoint().is_some();
    if require_exact && expects_teacher && record.teacher_update_count.is_none() {
        return Err(anyhow!(
            "exact resume requires teacher metadata in {} because the training contract enables a teacher model",
            path.display()
        ));
    }
    if let Some(update_count) = record.teacher_update_count {
        let teacher_path = run_dir
            .join("checkpoint")
            .join(format!("{TEACHER_MODEL_PREFIX}-{epoch}.bin"));
        if !teacher_path.is_file() {
            return Err(anyhow!(
                "runtime-state checkpoint names teacher state but {} is missing",
                teacher_path.display()
            ));
        }
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let teacher_record =
            FileCheckpointer::new(recorder, run_dir.join("checkpoint"), TEACHER_MODEL_PREFIX)
                .restore(epoch, device)
                .with_context(|| {
                    format!(
                        "failed to restore Dragon teacher-model checkpoint {epoch} from {}",
                        run_dir.join("checkpoint").display()
                    )
                })?;
        let teacher = model.model.clone().load_record(teacher_record);
        model.restore_teacher_model_from_checkpoint(teacher, update_count);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    use burn_dragon_core::{DragonConfig, DragonModel, RotaryEmbedding, SequenceKernelConfig};
    use burn_ndarray::NdArrayDevice;
    use burn_train::checkpoint::Checkpointer;

    use crate::config::{
        LocalPredictiveCodingConfig, LocalPredictiveCodingSolver, TrainingAlgorithm,
    };

    type TestBackend = Autodiff<NdArray<f32>>;

    fn local_pc_model(
        device: &NdArrayDevice,
        local_pc: LocalPredictiveCodingConfig,
    ) -> LanguageTrainModel<TestBackend> {
        let mut config = DragonConfig {
            n_layer: 2,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            dropout: 0.0,
            vocab_size: 16,
            ..DragonConfig::default()
        };
        config.sequence_kernel = SequenceKernelConfig::dense_score_short_context();
        config.fused_kernels.rotary_embedding = RotaryEmbedding::Alibi;
        LanguageTrainModel::new(DragonModel::new(config, device))
            .with_training_algorithm(TrainingAlgorithm::PredictiveCoding)
            .with_local_predictive_coding(local_pc)
    }

    fn dkp_model(device: &NdArrayDevice) -> LanguageTrainModel<TestBackend> {
        local_pc_model(
            device,
            LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::DirectKolenPollack,
                ..LocalPredictiveCodingConfig::default()
            },
        )
    }

    fn residual_adjoint_model(device: &NdArrayDevice) -> LanguageTrainModel<TestBackend> {
        local_pc_model(
            device,
            LocalPredictiveCodingConfig {
                solver: LocalPredictiveCodingSolver::AmortizedAdjoint,
                amortized_adjoint: burn_pc::PcAmortizedAdjointConfig {
                    enabled: true,
                    teacher_warmup_updates: 4,
                    teacher_every_updates: 8,
                    predictor: burn_pc::PcAdjointPredictorKind::ResidualConditioned,
                    ..burn_pc::PcAmortizedAdjointConfig::default()
                },
                ..LocalPredictiveCodingConfig::default()
            },
        )
    }

    fn pc_manifest() -> burn_pc::PcCheckpointManifest {
        burn_pc::PcCheckpointManifest {
            schema_version: burn_pc::PcCheckpointManifest::CURRENT_SCHEMA_VERSION,
            graph_digest: "pc-graph-v1:test".to_string(),
            program_digest: "dragon-pc-program-v1:test".to_string(),
            algorithm: "dragon_local_predictive_coding_v1".to_string(),
            learning_schedule: burn_pc::PcLearningSchedule::Equilibrium,
            execution_contract: burn_pc::PcExecutionContract::strict_local(),
        }
    }

    #[test]
    fn predictive_coding_manifest_resume_is_exact_and_fail_closed() {
        let directory = tempfile::tempdir().expect("temporary checkpoint directory");
        fs::create_dir_all(directory.path().join("checkpoint")).expect("checkpoint directory");
        let expected = pc_manifest();
        save_predictive_coding_manifest(directory.path(), 3, &expected).expect("save manifest");
        validate_predictive_coding_manifest(directory.path(), 3, Some(&expected), true)
            .expect("matching PC manifest");

        let mut mismatch = expected.clone();
        mismatch.program_digest = "dragon-pc-program-v1:changed".to_string();
        assert!(
            validate_predictive_coding_manifest(directory.path(), 3, Some(&mismatch), true)
                .expect_err("changed PC program must not resume")
                .to_string()
                .contains("does not match")
        );
        assert!(
            validate_predictive_coding_manifest(directory.path(), 3, None, true)
                .expect_err("PC checkpoint must not exactly resume as non-PC")
                .to_string()
                .contains("non-PC program")
        );
        assert!(
            validate_predictive_coding_manifest(directory.path(), 4, Some(&expected), true)
                .expect_err("exact PC resume requires its manifest")
                .to_string()
                .contains("requires program manifest")
        );
    }

    #[test]
    fn standard_learner_checkpoints_receive_program_sidecars() {
        let directory = tempfile::tempdir().expect("temporary checkpoint directory");
        let checkpoint_dir = directory.path().join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("checkpoint directory");
        fs::write(checkpoint_dir.join("model-1.bin"), b"model").expect("model one");
        fs::write(checkpoint_dir.join("model-3.bin"), b"model").expect("model three");
        fs::write(checkpoint_dir.join("optim-3.bin"), b"optimizer").expect("optimizer");
        let manifest = pc_manifest();

        prepare_predictive_coding_checkpoint_contract(
            directory.path(),
            None,
            Some(&manifest),
            true,
        )
        .expect("prepare fresh PC run");
        assert!(directory.path().join(PC_RUN_MANIFEST).is_file());
        assert_eq!(
            synchronize_predictive_coding_checkpoint_manifests(directory.path(), Some(&manifest))
                .expect("synchronize manifests"),
            2
        );
        for epoch in [1, 3] {
            assert!(predictive_coding_manifest_checkpoint_path(directory.path(), epoch).is_file());
        }
        prepare_predictive_coding_checkpoint_contract(
            directory.path(),
            Some(3),
            Some(&manifest),
            true,
        )
        .expect("exact standard-learner resume");
    }

    #[test]
    fn runtime_state_record_roundtrip_preserves_recurrent_slots() {
        let device = Default::default();
        let mut state = ModelState::<TestBackend>::new(1);
        state.position = 7;
        let layer = &mut state.layers[0];
        layer.rho = Some(Tensor::from_data(
            TensorData::new(vec![1.0_f32, 2.0], [1, 1, 1, 2]),
            &device,
        ));
        layer.rho_norm = Some(Tensor::from_data(
            TensorData::new(vec![3.0_f32], [1, 1, 1]),
            &device,
        ));
        layer.y_neuron_state = Some(Tensor::from_data(
            TensorData::new(vec![4.0_f32, 5.0], [1, 1, 2]),
            &device,
        ));

        let record = LanguageRuntimeStateRecord {
            gradient_scale_step: 11,
            teacher_update_count: Some(3),
            streaming_state: Some(StreamingRuntimeStateRecord::from(state)),
            dkp_feedback: Some(DkpFeedbackStateRecord {
                feedback: Tensor::from_data(TensorData::new(vec![0.5_f32; 8], [2, 2, 2]), &device),
                updates: 7,
            }),
        };
        let directory = tempfile::tempdir().expect("temporary checkpoint directory");
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        FileCheckpointer::new(recorder, directory.path(), RUNTIME_STATE_PREFIX)
            .save(2, record)
            .expect("save runtime state");
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let restored_record: LanguageRuntimeStateRecord<TestBackend> =
            FileCheckpointer::new(recorder, directory.path(), RUNTIME_STATE_PREFIX)
                .restore(2, &device)
                .expect("restore runtime state");
        assert_eq!(restored_record.gradient_scale_step, 11);
        assert_eq!(restored_record.teacher_update_count, Some(3));
        let dkp = restored_record
            .dkp_feedback
            .clone()
            .expect("DKP feedback record");
        assert_eq!(dkp.updates, 7);
        assert_eq!(dkp.feedback.shape().dims::<3>(), [2, 2, 2]);
        let restored = ModelState::from(
            restored_record
                .streaming_state
                .expect("streaming runtime state"),
        );
        assert_eq!(restored.position, 7);
        assert_eq!(
            restored.layers[0]
                .rho
                .clone()
                .expect("rho")
                .into_data()
                .to_vec::<f32>()
                .expect("rho values"),
            vec![1.0, 2.0]
        );
        assert_eq!(
            restored.layers[0]
                .rho_norm
                .clone()
                .expect("rho norm")
                .into_data()
                .to_vec::<f32>()
                .expect("rho norm values"),
            vec![3.0]
        );
        assert_eq!(
            restored.layers[0]
                .y_neuron_state
                .clone()
                .expect("y state")
                .into_data()
                .to_vec::<f32>()
                .expect("y values"),
            vec![4.0, 5.0]
        );
    }

    #[test]
    fn exact_dkp_resume_restores_learned_feedback_bank() {
        let device = Default::default();
        let source = dkp_model(&device);
        source.restore_dkp_feedback_from_checkpoint(Some(
            super::super::local_predictive_coding::DkpFeedbackState {
                feedback: Tensor::from_data(
                    TensorData::new(vec![0.25_f32; 128], [2, 8, 8]),
                    &device,
                ),
                updates: 19,
            },
        ));
        let directory = tempfile::tempdir().expect("temporary checkpoint directory");
        fs::create_dir_all(directory.path().join("checkpoint")).expect("checkpoint directory");
        save_runtime_state_checkpoint(directory.path(), 3, &source)
            .expect("save exact DKP runtime state");

        let restored = dkp_model(&device);
        assert!(
            load_runtime_state_checkpoint(directory.path(), 3, &restored, &device, true, false)
                .expect("restore exact DKP runtime state")
        );
        let feedback = restored
            .dkp_feedback_for_checkpoint()
            .expect("restored DKP feedback bank");
        assert_eq!(feedback.updates, 19);
        assert_eq!(feedback.feedback.shape().dims::<3>(), [2, 8, 8]);
        let values = feedback
            .feedback
            .into_data()
            .to_vec::<f32>()
            .expect("feedback values");
        assert!(values.iter().all(|value| (*value - 0.25).abs() < 1.0e-6));
    }

    #[test]
    fn exact_residual_adjoint_resume_restores_wide_feedback_bank() {
        let device = Default::default();
        let source = residual_adjoint_model(&device);
        source.restore_dkp_feedback_from_checkpoint(Some(
            super::super::local_predictive_coding::DkpFeedbackState {
                feedback: Tensor::from_data(
                    TensorData::new(vec![0.125_f32; 256], [2, 8, 16]),
                    &device,
                ),
                updates: 73,
            },
        ));
        let directory = tempfile::tempdir().expect("temporary checkpoint directory");
        fs::create_dir_all(directory.path().join("checkpoint")).expect("checkpoint directory");
        save_runtime_state_checkpoint(directory.path(), 5, &source)
            .expect("save exact residual-adjoint runtime state");

        let restored = residual_adjoint_model(&device);
        assert!(
            load_runtime_state_checkpoint(directory.path(), 5, &restored, &device, true, false)
                .expect("restore exact residual-adjoint runtime state")
        );
        let feedback = restored
            .dkp_feedback_for_checkpoint()
            .expect("restored residual-adjoint feedback bank");
        assert_eq!(feedback.updates, 73);
        assert_eq!(feedback.feedback.shape().dims::<3>(), [2, 8, 16]);
        let values = feedback
            .feedback
            .into_data()
            .to_vec::<f32>()
            .expect("feedback values");
        assert!(values.iter().all(|value| (*value - 0.125).abs() < 1.0e-6));
    }
}
