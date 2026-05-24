use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::model::state::LayerState;

#[derive(Debug, Clone)]
pub struct Mamba3State<B: Backend> {
    pub ssm: Tensor<B, 4>,
    pub angle: Tensor<B, 3>,
    pub k: Tensor<B, 3>,
    pub v: Tensor<B, 3>,
}

pub fn gated_deltanet2_state<B: Backend>(
    layer_state: &LayerState<B>,
    batch: usize,
    heads: usize,
    latent: usize,
    dense_dim: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    match layer_state.rho.as_ref() {
        Some(state) if state.shape().dims::<4>() == [batch, heads, latent, dense_dim] => {
            state.clone()
        }
        _ => Tensor::<B, 4>::zeros([batch, heads, latent, dense_dim], device),
    }
}

pub fn write_gated_deltanet2_state<B: Backend>(
    layer_state: &mut LayerState<B>,
    state: Tensor<B, 4>,
) {
    layer_state.rho = Some(state);
    layer_state.rho_norm = None;
    layer_state.sequence_aux = None;
    layer_state.mamba_angle_state = None;
    layer_state.mamba_k_state = None;
    layer_state.mamba_v_state = None;
}

pub fn mamba3_state<B: Backend>(
    layer_state: &LayerState<B>,
    batch: usize,
    nheads: usize,
    headdim: usize,
    d_state: usize,
    angle_dim: usize,
    device: &B::Device,
) -> Mamba3State<B> {
    let ssm = match layer_state.rho.as_ref() {
        Some(state) if state.shape().dims::<4>() == [batch, nheads, headdim, d_state] => {
            state.clone()
        }
        _ => Tensor::<B, 4>::zeros([batch, nheads, headdim, d_state], device),
    };
    let angle = match layer_state.mamba_angle_state.as_ref() {
        Some(state) if state.shape().dims::<3>() == [batch, nheads, angle_dim] => state.clone(),
        _ => Tensor::<B, 3>::zeros([batch, nheads, angle_dim], device),
    };
    let k = match layer_state.mamba_k_state.as_ref() {
        Some(state) if state.shape().dims::<3>() == [batch, nheads, d_state] => state.clone(),
        _ => Tensor::<B, 3>::zeros([batch, nheads, d_state], device),
    };
    let v = match layer_state.mamba_v_state.as_ref() {
        Some(state) if state.shape().dims::<3>() == [batch, nheads, headdim] => state.clone(),
        _ => Tensor::<B, 3>::zeros([batch, nheads, headdim], device),
    };
    Mamba3State { ssm, angle, k, v }
}

pub fn write_mamba3_state<B: Backend>(
    layer_state: &mut LayerState<B>,
    ssm: Tensor<B, 4>,
    angle: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
) {
    layer_state.rho = Some(ssm);
    layer_state.rho_norm = None;
    layer_state.sequence_aux = None;
    layer_state.mamba_angle_state = Some(angle);
    layer_state.mamba_k_state = Some(k);
    layer_state.mamba_v_state = Some(v);
}
