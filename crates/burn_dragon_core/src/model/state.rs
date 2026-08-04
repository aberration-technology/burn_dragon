use std::mem;

use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};

#[derive(Debug, Clone)]
pub struct LayerState<B: Backend> {
    /// Whether the caller carries this state into a later sequence invocation.
    pub persist_sequence_state: bool,
    /// Whether the executor must materialize state at the end of the current invocation.
    pub retain_terminal_sequence_state: bool,
    pub rho: Option<Tensor<B, 4>>,
    pub rho_norm: Option<Tensor<B, 3>>,
    pub sequence_aux: Option<Tensor<B, 4>>,
    pub mamba_angle_state: Option<Tensor<B, 3>>,
    pub mamba_k_state: Option<Tensor<B, 3>>,
    pub mamba_v_state: Option<Tensor<B, 3>>,
    pub slow_rho: Option<Tensor<B, 4>>,
    pub slow_rho_norm: Option<Tensor<B, 3>>,
    pub slow_sequence_aux: Option<Tensor<B, 4>>,
    pub slow_mamba_angle_state: Option<Tensor<B, 3>>,
    pub slow_mamba_k_state: Option<Tensor<B, 3>>,
    pub slow_mamba_v_state: Option<Tensor<B, 3>>,
    pub y_neuron_state: Option<Tensor<B, 3>>,
    pub hierarchical_slow_hidden: Option<Tensor<B, 4>>,
    pub clocked_slow_hidden: Option<Tensor<B, 4>>,
    pub summary_memory_hidden: Option<Tensor<B, 4>>,
    #[cfg(any(feature = "viz", feature = "probe"))]
    pub viz: Option<LayerVizState<B>>,
}

#[derive(Debug, Clone)]
pub struct ModelState<B: Backend> {
    pub layers: Vec<LayerState<B>>,
    pub position: usize,
}

#[cfg(any(feature = "viz", feature = "probe"))]
#[derive(Debug, Clone)]
pub struct LayerVizState<B: Backend> {
    pub x_neuron_last: Tensor<B, 2>,
    pub y_gate_last: Tensor<B, 2>,
    pub y_neuron_last: Tensor<B, 2>,
    pub rho_last: Tensor<B, 2>,
}

impl<B: Backend> ModelState<B> {
    pub fn new(num_layers: usize) -> Self {
        Self::with_sequence_state_policy(num_layers, true, true)
    }

    pub fn new_ephemeral(num_layers: usize) -> Self {
        Self::with_sequence_state_policy(num_layers, false, true)
    }

    pub fn new_stateless(num_layers: usize) -> Self {
        Self::with_sequence_state_policy(num_layers, false, false)
    }

    fn with_sequence_state_policy(
        num_layers: usize,
        persist_sequence_state: bool,
        retain_terminal_sequence_state: bool,
    ) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerState {
                    persist_sequence_state,
                    retain_terminal_sequence_state,
                    rho: None,
                    rho_norm: None,
                    sequence_aux: None,
                    mamba_angle_state: None,
                    mamba_k_state: None,
                    mamba_v_state: None,
                    slow_rho: None,
                    slow_rho_norm: None,
                    slow_sequence_aux: None,
                    slow_mamba_angle_state: None,
                    slow_mamba_k_state: None,
                    slow_mamba_v_state: None,
                    y_neuron_state: None,
                    hierarchical_slow_hidden: None,
                    clocked_slow_hidden: None,
                    summary_memory_hidden: None,
                    #[cfg(any(feature = "viz", feature = "probe"))]
                    viz: None,
                })
                .collect(),
            position: 0,
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.rho = None;
            layer.rho_norm = None;
            layer.sequence_aux = None;
            layer.mamba_angle_state = None;
            layer.mamba_k_state = None;
            layer.mamba_v_state = None;
            layer.slow_rho = None;
            layer.slow_rho_norm = None;
            layer.slow_sequence_aux = None;
            layer.slow_mamba_angle_state = None;
            layer.slow_mamba_k_state = None;
            layer.slow_mamba_v_state = None;
            layer.y_neuron_state = None;
            layer.hierarchical_slow_hidden = None;
            layer.clocked_slow_hidden = None;
            layer.summary_memory_hidden = None;
        }
        self.position = 0;
    }

    pub fn len(&self) -> usize {
        self.position
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn trim(&mut self, max_len: usize) {
        let _ = max_len;
    }

    pub fn detach_in_place(&mut self) {
        for layer in &mut self.layers {
            layer.rho = layer.rho.take().map(|tensor| tensor.detach());
            layer.rho_norm = layer.rho_norm.take().map(|tensor| tensor.detach());
            layer.sequence_aux = layer.sequence_aux.take().map(|tensor| tensor.detach());
            layer.mamba_angle_state = layer.mamba_angle_state.take().map(|tensor| tensor.detach());
            layer.mamba_k_state = layer.mamba_k_state.take().map(|tensor| tensor.detach());
            layer.mamba_v_state = layer.mamba_v_state.take().map(|tensor| tensor.detach());
            layer.slow_rho = layer.slow_rho.take().map(|tensor| tensor.detach());
            layer.slow_rho_norm = layer.slow_rho_norm.take().map(|tensor| tensor.detach());
            layer.slow_sequence_aux = layer.slow_sequence_aux.take().map(|tensor| tensor.detach());
            layer.slow_mamba_angle_state = layer
                .slow_mamba_angle_state
                .take()
                .map(|tensor| tensor.detach());
            layer.slow_mamba_k_state = layer
                .slow_mamba_k_state
                .take()
                .map(|tensor| tensor.detach());
            layer.slow_mamba_v_state = layer
                .slow_mamba_v_state
                .take()
                .map(|tensor| tensor.detach());
            layer.y_neuron_state = layer.y_neuron_state.take().map(|tensor| tensor.detach());
            layer.hierarchical_slow_hidden = layer
                .hierarchical_slow_hidden
                .take()
                .map(|tensor| tensor.detach());
            layer.clocked_slow_hidden = layer
                .clocked_slow_hidden
                .take()
                .map(|tensor| tensor.detach());
            layer.summary_memory_hidden = layer
                .summary_memory_hidden
                .take()
                .map(|tensor| tensor.detach());
        }
    }

    pub fn detached_clone(&self) -> Self {
        let mut detached = self.clone();
        detached.detach_in_place();
        detached
    }

    /// Repeat every recurrent-state batch row by `copies` while preserving stream position.
    ///
    /// This is primarily useful after encoding one shared prefix: downstream branches can carry
    /// the exact same prefix state without recomputing the prefix for every continuation.
    pub fn repeat_batch(&self, copies: usize) -> Self {
        assert!(copies > 0, "model-state batch copies must be non-zero");
        Self {
            layers: self
                .layers
                .iter()
                .map(|layer| LayerState {
                    persist_sequence_state: layer.persist_sequence_state,
                    retain_terminal_sequence_state: layer.retain_terminal_sequence_state,
                    rho: repeat_optional_batch(&layer.rho, copies),
                    rho_norm: repeat_optional_batch(&layer.rho_norm, copies),
                    sequence_aux: repeat_optional_batch(&layer.sequence_aux, copies),
                    mamba_angle_state: repeat_optional_batch(&layer.mamba_angle_state, copies),
                    mamba_k_state: repeat_optional_batch(&layer.mamba_k_state, copies),
                    mamba_v_state: repeat_optional_batch(&layer.mamba_v_state, copies),
                    slow_rho: repeat_optional_batch(&layer.slow_rho, copies),
                    slow_rho_norm: repeat_optional_batch(&layer.slow_rho_norm, copies),
                    slow_sequence_aux: repeat_optional_batch(&layer.slow_sequence_aux, copies),
                    slow_mamba_angle_state: repeat_optional_batch(
                        &layer.slow_mamba_angle_state,
                        copies,
                    ),
                    slow_mamba_k_state: repeat_optional_batch(&layer.slow_mamba_k_state, copies),
                    slow_mamba_v_state: repeat_optional_batch(&layer.slow_mamba_v_state, copies),
                    y_neuron_state: repeat_optional_batch(&layer.y_neuron_state, copies),
                    hierarchical_slow_hidden: repeat_optional_batch(
                        &layer.hierarchical_slow_hidden,
                        copies,
                    ),
                    clocked_slow_hidden: repeat_optional_batch(&layer.clocked_slow_hidden, copies),
                    summary_memory_hidden: repeat_optional_batch(
                        &layer.summary_memory_hidden,
                        copies,
                    ),
                    #[cfg(any(feature = "viz", feature = "probe"))]
                    viz: layer.viz.as_ref().map(|viz| LayerVizState {
                        x_neuron_last: viz.x_neuron_last.clone().repeat_dim(0, copies),
                        y_gate_last: viz.y_gate_last.clone().repeat_dim(0, copies),
                        y_neuron_last: viz.y_neuron_last.clone().repeat_dim(0, copies),
                        rho_last: viz.rho_last.clone().repeat_dim(0, copies),
                    }),
                })
                .collect(),
            position: self.position,
        }
    }

    /// Select recurrent-state batch rows while preserving their shared stream position.
    ///
    /// Repeated indices are supported, allowing one encoded prefix row to fan out into several
    /// independently evaluated continuations without re-encoding the prefix.
    pub fn select_batch(&self, indices: Tensor<B, 1, Int>) -> Self {
        Self {
            layers: self
                .layers
                .iter()
                .map(|layer| LayerState {
                    persist_sequence_state: layer.persist_sequence_state,
                    retain_terminal_sequence_state: layer.retain_terminal_sequence_state,
                    rho: select_optional_batch(&layer.rho, &indices),
                    rho_norm: select_optional_batch(&layer.rho_norm, &indices),
                    sequence_aux: select_optional_batch(&layer.sequence_aux, &indices),
                    mamba_angle_state: select_optional_batch(&layer.mamba_angle_state, &indices),
                    mamba_k_state: select_optional_batch(&layer.mamba_k_state, &indices),
                    mamba_v_state: select_optional_batch(&layer.mamba_v_state, &indices),
                    slow_rho: select_optional_batch(&layer.slow_rho, &indices),
                    slow_rho_norm: select_optional_batch(&layer.slow_rho_norm, &indices),
                    slow_sequence_aux: select_optional_batch(&layer.slow_sequence_aux, &indices),
                    slow_mamba_angle_state: select_optional_batch(
                        &layer.slow_mamba_angle_state,
                        &indices,
                    ),
                    slow_mamba_k_state: select_optional_batch(&layer.slow_mamba_k_state, &indices),
                    slow_mamba_v_state: select_optional_batch(&layer.slow_mamba_v_state, &indices),
                    y_neuron_state: select_optional_batch(&layer.y_neuron_state, &indices),
                    hierarchical_slow_hidden: select_optional_batch(
                        &layer.hierarchical_slow_hidden,
                        &indices,
                    ),
                    clocked_slow_hidden: select_optional_batch(
                        &layer.clocked_slow_hidden,
                        &indices,
                    ),
                    summary_memory_hidden: select_optional_batch(
                        &layer.summary_memory_hidden,
                        &indices,
                    ),
                    #[cfg(any(feature = "viz", feature = "probe"))]
                    viz: layer.viz.as_ref().map(|viz| LayerVizState {
                        x_neuron_last: viz.x_neuron_last.clone().select(0, indices.clone()),
                        y_gate_last: viz.y_gate_last.clone().select(0, indices.clone()),
                        y_neuron_last: viz.y_neuron_last.clone().select(0, indices.clone()),
                        rho_last: viz.rho_last.clone().select(0, indices.clone()),
                    }),
                })
                .collect(),
            position: self.position,
        }
    }

    #[cfg(any(feature = "viz", feature = "probe"))]
    pub fn take_viz(&mut self) -> Vec<Option<LayerVizState<B>>> {
        self.layers
            .iter_mut()
            .map(|layer| layer.viz.take())
            .collect()
    }

    #[cfg(any(feature = "viz", feature = "probe"))]
    pub fn clear_viz(&mut self) {
        for layer in &mut self.layers {
            layer.viz = None;
        }
    }
}

fn repeat_optional_batch<B: Backend, const D: usize>(
    tensor: &Option<Tensor<B, D>>,
    copies: usize,
) -> Option<Tensor<B, D>> {
    tensor
        .as_ref()
        .map(|tensor| tensor.clone().repeat_dim(0, copies))
}

fn select_optional_batch<B: Backend, const D: usize>(
    tensor: &Option<Tensor<B, D>>,
    indices: &Tensor<B, 1, Int>,
) -> Option<Tensor<B, D>> {
    tensor
        .as_ref()
        .map(|tensor| tensor.clone().select(0, indices.clone()))
}

impl<B: AutodiffBackend> ModelState<B> {
    pub fn inner_cloned(&self) -> ModelState<B::InnerBackend> {
        ModelState {
            layers: self
                .layers
                .iter()
                .map(|layer| LayerState {
                    persist_sequence_state: layer.persist_sequence_state,
                    retain_terminal_sequence_state: layer.retain_terminal_sequence_state,
                    rho: layer.rho.clone().map(Tensor::inner),
                    rho_norm: layer.rho_norm.clone().map(Tensor::inner),
                    sequence_aux: layer.sequence_aux.clone().map(Tensor::inner),
                    mamba_angle_state: layer.mamba_angle_state.clone().map(Tensor::inner),
                    mamba_k_state: layer.mamba_k_state.clone().map(Tensor::inner),
                    mamba_v_state: layer.mamba_v_state.clone().map(Tensor::inner),
                    slow_rho: layer.slow_rho.clone().map(Tensor::inner),
                    slow_rho_norm: layer.slow_rho_norm.clone().map(Tensor::inner),
                    slow_sequence_aux: layer.slow_sequence_aux.clone().map(Tensor::inner),
                    slow_mamba_angle_state: layer.slow_mamba_angle_state.clone().map(Tensor::inner),
                    slow_mamba_k_state: layer.slow_mamba_k_state.clone().map(Tensor::inner),
                    slow_mamba_v_state: layer.slow_mamba_v_state.clone().map(Tensor::inner),
                    y_neuron_state: layer.y_neuron_state.clone().map(Tensor::inner),
                    hierarchical_slow_hidden: layer
                        .hierarchical_slow_hidden
                        .clone()
                        .map(Tensor::inner),
                    clocked_slow_hidden: layer.clocked_slow_hidden.clone().map(Tensor::inner),
                    summary_memory_hidden: layer.summary_memory_hidden.clone().map(Tensor::inner),
                    #[cfg(any(feature = "viz", feature = "probe"))]
                    viz: layer.viz.clone().map(|viz| LayerVizState {
                        x_neuron_last: viz.x_neuron_last.inner(),
                        y_gate_last: viz.y_gate_last.inner(),
                        y_neuron_last: viz.y_neuron_last.inner(),
                        rho_last: viz.rho_last.inner(),
                    }),
                })
                .collect(),
            position: self.position,
        }
    }

    pub fn from_inner_cloned(state: ModelState<B::InnerBackend>) -> Self {
        ModelState {
            layers: state
                .layers
                .into_iter()
                .map(|layer| LayerState {
                    persist_sequence_state: layer.persist_sequence_state,
                    retain_terminal_sequence_state: layer.retain_terminal_sequence_state,
                    rho: layer.rho.map(Tensor::from_inner),
                    rho_norm: layer.rho_norm.map(Tensor::from_inner),
                    sequence_aux: layer.sequence_aux.map(Tensor::from_inner),
                    mamba_angle_state: layer.mamba_angle_state.map(Tensor::from_inner),
                    mamba_k_state: layer.mamba_k_state.map(Tensor::from_inner),
                    mamba_v_state: layer.mamba_v_state.map(Tensor::from_inner),
                    slow_rho: layer.slow_rho.map(Tensor::from_inner),
                    slow_rho_norm: layer.slow_rho_norm.map(Tensor::from_inner),
                    slow_sequence_aux: layer.slow_sequence_aux.map(Tensor::from_inner),
                    slow_mamba_angle_state: layer.slow_mamba_angle_state.map(Tensor::from_inner),
                    slow_mamba_k_state: layer.slow_mamba_k_state.map(Tensor::from_inner),
                    slow_mamba_v_state: layer.slow_mamba_v_state.map(Tensor::from_inner),
                    y_neuron_state: layer.y_neuron_state.map(Tensor::from_inner),
                    hierarchical_slow_hidden: layer
                        .hierarchical_slow_hidden
                        .map(Tensor::from_inner),
                    clocked_slow_hidden: layer.clocked_slow_hidden.map(Tensor::from_inner),
                    summary_memory_hidden: layer.summary_memory_hidden.map(Tensor::from_inner),
                    #[cfg(any(feature = "viz", feature = "probe"))]
                    viz: layer.viz.map(|viz| LayerVizState {
                        x_neuron_last: Tensor::from_inner(viz.x_neuron_last),
                        y_gate_last: Tensor::from_inner(viz.y_gate_last),
                        y_neuron_last: Tensor::from_inner(viz.y_neuron_last),
                        rho_last: Tensor::from_inner(viz.rho_last),
                    }),
                })
                .collect(),
            position: state.position,
        }
    }
}

impl<B: Backend> LayerState<B> {
    pub fn swap_fast_slow_sequence_state(&mut self) {
        mem::swap(&mut self.rho, &mut self.slow_rho);
        mem::swap(&mut self.rho_norm, &mut self.slow_rho_norm);
        mem::swap(&mut self.sequence_aux, &mut self.slow_sequence_aux);
        mem::swap(
            &mut self.mamba_angle_state,
            &mut self.slow_mamba_angle_state,
        );
        mem::swap(&mut self.mamba_k_state, &mut self.slow_mamba_k_state);
        mem::swap(&mut self.mamba_v_state, &mut self.slow_mamba_v_state);
    }
}

#[cfg(any(feature = "viz", feature = "probe"))]
impl<B: Backend> LayerState<B> {
    pub fn take_viz(&mut self) -> Option<LayerVizState<B>> {
        self.viz.take()
    }
}
