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
}

pub(crate) fn runtime_state_checkpoint_path(run_dir: &Path, epoch: usize) -> PathBuf {
    run_dir
        .join("checkpoint")
        .join(format!("{RUNTIME_STATE_PREFIX}-{epoch}.bin"))
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
    model.restore_gradient_scale_step_from_checkpoint(record.gradient_scale_step);
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
    use burn_train::checkpoint::Checkpointer;

    type TestBackend = Autodiff<NdArray<f32>>;

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
}
