use burn::nn::Dropout;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_dragon_kernel::api::projection::LowrankGradInputExecutor;
#[cfg(any(feature = "benchmark", feature = "train", feature = "cuda"))]
use std::any::Any;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[cfg(any(feature = "benchmark", feature = "train", feature = "cuda"))]
use burn_cubecl::cubecl::Runtime;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
#[cfg(feature = "cuda")]
use burn_cuda::CudaDevice;
#[cfg(any(feature = "benchmark", feature = "train"))]
use burn_wgpu::{WgpuDevice, WgpuRuntime};

use crate::kernel::{BlockPattern1d, relu_lowrank};

#[derive(Debug)]
pub struct LowRankResidualOutput<B: Backend> {
    pub next: Tensor<B, 4>,
    pub attention_readout: Option<Tensor<B, 4>>,
    pub residual_delta: Option<Tensor<B, 4>>,
    pub x_neuron: Tensor<B, 4>,
    pub y_gate: Tensor<B, 4>,
    pub y_neuron: Tensor<B, 4>,
}

struct LowRankResidualInternal<B: Backend> {
    next: Tensor<B, 4>,
    attention_readout: Option<Tensor<B, 4>>,
    residual_delta: Option<Tensor<B, 4>>,
    x_neuron: Option<Tensor<B, 4>>,
    y_gate: Option<Tensor<B, 4>>,
    y_neuron: Option<Tensor<B, 4>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LowRankResidualStepMode {
    native_projection_relu_fused: bool,
    keep_aux: bool,
    keep_metric_aux: bool,
}

impl LowRankResidualStepMode {
    const fn full_output() -> Self {
        Self {
            native_projection_relu_fused: false,
            keep_aux: true,
            keep_metric_aux: false,
        }
    }
    const fn full_output_relu_native() -> Self {
        Self {
            native_projection_relu_fused: true,
            ..Self::full_output()
        }
    }
    #[cfg(any(feature = "probe", test))]
    const fn with_metrics() -> Self {
        Self {
            keep_metric_aux: true,
            ..Self::full_output()
        }
    }
    const fn next_only() -> Self {
        Self {
            native_projection_relu_fused: false,
            keep_aux: false,
            keep_metric_aux: false,
        }
    }
    const fn next_only_relu_native() -> Self {
        Self {
            native_projection_relu_fused: true,
            ..Self::next_only()
        }
    }
}

struct LowRankResidualStepConfig<'a, B: Backend> {
    encoder: Tensor<B, 4>,
    encoder_v: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
    dropout: &'a Dropout,
    use_fused_x: bool,
    use_fused_y: bool,
    x_relu_threshold: f32,
    y_relu_threshold: f32,
    apply_threshold: bool,
    latent_pattern: &'a BlockPattern1d,
    lowrank_grad_input_executor: LowrankGradInputExecutor,
    sparse_mask: Option<Tensor<B, 4>>,
    mode: LowRankResidualStepMode,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LowRankResidualProfileSnapshot {
    pub calls: u64,
    pub total_ns: u128,
    pub x_projection_ns: u128,
    pub x_post_quant_ns: u128,
    pub attention_norm_ns: u128,
    pub attention_mixer_ns: u128,
    pub attention_post_norm_ns: u128,
    pub y_projection_ns: u128,
    pub y_post_quant_ns: u128,
    pub y_neuron_ns: u128,
    pub decoder_tail_ns: u128,
    pub mlp_norm_ns: u128,
    pub residual_combine_ns: u128,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowRankResidualMemoryStageSnapshot {
    pub reserved_bytes: u64,
    pub in_use_bytes: u64,
    pub tracked_tensor_bytes: u64,
}

impl LowRankResidualMemoryStageSnapshot {
    fn should_replace(self, observed: Self) -> bool {
        (
            observed.in_use_bytes,
            observed.reserved_bytes,
            observed.tracked_tensor_bytes,
        ) > (
            self.in_use_bytes,
            self.reserved_bytes,
            self.tracked_tensor_bytes,
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LowRankResidualMemoryProfileSnapshot {
    pub calls: u64,
    pub after_attention_norm: LowRankResidualMemoryStageSnapshot,
    pub after_y_projection: LowRankResidualMemoryStageSnapshot,
    pub after_y_post_quant: LowRankResidualMemoryStageSnapshot,
    pub after_y_neuron: LowRankResidualMemoryStageSnapshot,
    pub after_decoder_tail: LowRankResidualMemoryStageSnapshot,
    pub after_mlp_norm: LowRankResidualMemoryStageSnapshot,
}

static LOWRANK_RESIDUAL_PROFILE: OnceLock<Mutex<LowRankResidualProfileSnapshot>> = OnceLock::new();
static LOWRANK_RESIDUAL_MEMORY_PROFILE: OnceLock<Mutex<LowRankResidualMemoryProfileSnapshot>> =
    OnceLock::new();
static LOWRANK_RESIDUAL_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static LOWRANK_RESIDUAL_MEMORY_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static LOWRANK_RESIDUAL_MEMORY_PROFILE_SYNC_ENABLED: OnceLock<bool> = OnceLock::new();
static LEGACY_FLAT_DECODER_TAIL_ENABLED: OnceLock<bool> = OnceLock::new();

fn lowrank_residual_profile_enabled() -> bool {
    *LOWRANK_RESIDUAL_PROFILE_ENABLED
        .get_or_init(|| std::env::var_os("DragonModel_STAGE_PROFILE").is_some())
}

fn lowrank_residual_memory_profile_enabled() -> bool {
    *LOWRANK_RESIDUAL_MEMORY_PROFILE_ENABLED
        .get_or_init(|| std::env::var_os("DragonModel_STAGE_PROFILE_MEMORY").is_some())
}

fn lowrank_residual_memory_profile_sync_enabled() -> bool {
    *LOWRANK_RESIDUAL_MEMORY_PROFILE_SYNC_ENABLED
        .get_or_init(|| std::env::var_os("DragonModel_STAGE_PROFILE_MEMORY_SYNC").is_some())
}

fn legacy_flat_decoder_tail_enabled() -> bool {
    *LEGACY_FLAT_DECODER_TAIL_ENABLED
        .get_or_init(|| std::env::var_os("BURN_DRAGON_LEGACY_FLAT_DECODER_TAIL").is_some())
}

fn lowrank_residual_profile_state() -> &'static Mutex<LowRankResidualProfileSnapshot> {
    LOWRANK_RESIDUAL_PROFILE.get_or_init(|| Mutex::new(LowRankResidualProfileSnapshot::default()))
}

fn lowrank_residual_memory_profile_state() -> &'static Mutex<LowRankResidualMemoryProfileSnapshot> {
    LOWRANK_RESIDUAL_MEMORY_PROFILE
        .get_or_init(|| Mutex::new(LowRankResidualMemoryProfileSnapshot::default()))
}

pub fn lowrank_residual_profile_reset() {
    if let Ok(mut state) = lowrank_residual_profile_state().lock() {
        *state = LowRankResidualProfileSnapshot::default();
    }
}

pub fn lowrank_residual_memory_profile_reset() {
    if let Ok(mut state) = lowrank_residual_memory_profile_state().lock() {
        *state = LowRankResidualMemoryProfileSnapshot::default();
    }
}

pub fn lowrank_residual_profile_snapshot() -> LowRankResidualProfileSnapshot {
    lowrank_residual_profile_state()
        .lock()
        .map(|state| *state)
        .unwrap_or_default()
}

pub fn lowrank_residual_memory_profile_snapshot() -> LowRankResidualMemoryProfileSnapshot {
    lowrank_residual_memory_profile_state()
        .lock()
        .map(|state| *state)
        .unwrap_or_default()
}

fn lowrank_residual_profile_record(observed: LowRankResidualProfileSnapshot) {
    if let Ok(mut state) = lowrank_residual_profile_state().lock() {
        state.calls = state.calls.saturating_add(observed.calls);
        state.total_ns = state.total_ns.saturating_add(observed.total_ns);
        state.x_projection_ns = state
            .x_projection_ns
            .saturating_add(observed.x_projection_ns);
        state.x_post_quant_ns = state
            .x_post_quant_ns
            .saturating_add(observed.x_post_quant_ns);
        state.attention_norm_ns = state
            .attention_norm_ns
            .saturating_add(observed.attention_norm_ns);
        state.attention_mixer_ns = state
            .attention_mixer_ns
            .saturating_add(observed.attention_mixer_ns);
        state.attention_post_norm_ns = state
            .attention_post_norm_ns
            .saturating_add(observed.attention_post_norm_ns);
        state.y_projection_ns = state
            .y_projection_ns
            .saturating_add(observed.y_projection_ns);
        state.y_post_quant_ns = state
            .y_post_quant_ns
            .saturating_add(observed.y_post_quant_ns);
        state.y_neuron_ns = state.y_neuron_ns.saturating_add(observed.y_neuron_ns);
        state.decoder_tail_ns = state
            .decoder_tail_ns
            .saturating_add(observed.decoder_tail_ns);
        state.mlp_norm_ns = state.mlp_norm_ns.saturating_add(observed.mlp_norm_ns);
        state.residual_combine_ns = state
            .residual_combine_ns
            .saturating_add(observed.residual_combine_ns);
    }
}

fn lowrank_residual_memory_usage<B: Backend>(device: &B::Device) -> Option<(u64, u64)>
where
    B::Device: 'static,
{
    if lowrank_residual_memory_profile_sync_enabled() {
        let _ = B::sync(device);
    }

    #[cfg(feature = "cuda")]
    if let Some(cuda_device) = (device as &dyn Any).downcast_ref::<CudaDevice>() {
        let usage = <CudaRuntime as Runtime>::client(cuda_device)
            .memory_usage()
            .expect("cuda memory usage");
        return Some((usage.bytes_reserved, usage.bytes_in_use));
    }

    #[cfg(any(feature = "benchmark", feature = "train"))]
    if let Some(wgpu_device) = (device as &dyn Any).downcast_ref::<WgpuDevice>() {
        let usage = <WgpuRuntime as Runtime>::client(wgpu_device)
            .memory_usage()
            .expect("wgpu memory usage");
        return Some((usage.bytes_reserved, usage.bytes_in_use));
    }

    None
}

fn tensor_bytes<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> u64 {
    tensor.shape().num_elements() as u64 * core::mem::size_of::<B::FloatElem>() as u64
}

fn lowrank_residual_memory_record_stage<B: Backend>(
    stage: fn(&mut LowRankResidualMemoryProfileSnapshot) -> &mut LowRankResidualMemoryStageSnapshot,
    device: &B::Device,
    tracked_tensor_bytes: u64,
) where
    B::Device: 'static,
{
    if let Some((reserved_bytes, in_use_bytes)) = lowrank_residual_memory_usage::<B>(device) {
        let observed = LowRankResidualMemoryStageSnapshot {
            reserved_bytes,
            in_use_bytes,
            tracked_tensor_bytes,
        };
        if let Ok(mut profile) = lowrank_residual_memory_profile_state().lock() {
            let slot = stage(&mut profile);
            if slot.should_replace(observed) {
                *slot = observed;
            }
        }
    }
}

fn decode_y_neuron_tail_flat<B: Backend>(
    y_neuron: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
) -> Tensor<B, 4> {
    let [batch, heads, time, latent] = y_neuron.shape().dims::<4>();
    let dim = decoder.shape().dims::<2>()[1];
    y_neuron
        .swap_dims(1, 2)
        .reshape([batch * time, heads * latent])
        .matmul(decoder)
        .reshape([batch, 1, time, dim])
}

fn decode_y_neuron_tail_headwise<B: Backend>(
    y_neuron: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
) -> Tensor<B, 4> {
    let [batch, heads, time, latent] = y_neuron.shape().dims::<4>();
    let dim = decoder.shape().dims::<2>()[1];
    if heads == 1 {
        return decode_y_neuron_tail_flat(y_neuron, decoder);
    }
    let decoder_by_head = decoder.reshape([heads, latent, dim]);
    let mixed_by_head = y_neuron
        .swap_dims(0, 1)
        .reshape([heads, batch * time, latent]);
    mixed_by_head
        .matmul(decoder_by_head)
        .sum_dim(0)
        .reshape([batch, 1, time, dim])
}

fn decode_y_neuron_tail<B: Backend>(y_neuron: Tensor<B, 4>, decoder: Tensor<B, 2>) -> Tensor<B, 4> {
    if legacy_flat_decoder_tail_enabled() {
        decode_y_neuron_tail_flat(y_neuron, decoder)
    } else {
        decode_y_neuron_tail_headwise(y_neuron, decoder)
    }
}

fn lowrank_residual_step_impl<B, FAttn, FNorm, FAct>(
    current: Tensor<B, 4>,
    config: LowRankResidualStepConfig<'_, B>,
    mut attention: FAttn,
    apply_latent: FAct,
    apply_norm: FNorm,
) -> LowRankResidualInternal<B>
where
    B: Backend,
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
    FAttn: FnMut(Tensor<B, 4>, Tensor<B, 4>) -> Tensor<B, 4>,
    FNorm: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
    FAct: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
{
    let LowRankResidualStepConfig {
        encoder,
        encoder_v,
        decoder,
        dropout,
        use_fused_x,
        use_fused_y,
        x_relu_threshold,
        y_relu_threshold,
        apply_threshold,
        latent_pattern,
        lowrank_grad_input_executor,
        sparse_mask,
        mode,
    } = config;
    let LowRankResidualStepMode {
        keep_aux,
        keep_metric_aux,
        ..
    } = mode;
    let prof_enabled = lowrank_residual_profile_enabled();
    let memory_prof_enabled = lowrank_residual_memory_profile_enabled();
    let total_start = prof_enabled.then(Instant::now);
    let mut x_projection_ns = 0;
    let x_post_quant_ns = 0;
    let mut attention_mixer_ns = 0;
    let mut attention_post_norm_ns = 0;
    let mut y_projection_ns = 0;
    let y_post_quant_ns = 0;
    let mut y_neuron_ns = 0;
    let mut decoder_tail_ns = 0;
    let mut mlp_norm_ns = 0;
    let mut residual_combine_ns = 0;

    let use_fused_any = use_fused_x || use_fused_y;
    let x_grad_input_executor = match lowrank_grad_input_executor {
        LowrankGradInputExecutor::KernelTiled => LowrankGradInputExecutor::AlignedMatmul,
        other => other,
    };
    let y_grad_input_executor = lowrank_grad_input_executor;
    let sparse_mask = if use_fused_any && latent_pattern.is_sparse() {
        sparse_mask.or_else(|| {
            let latent = encoder.shape().dims::<4>()[3];
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        })
    } else {
        None
    };

    let x_projection_start = prof_enabled.then(Instant::now);
    let x_neuron = if use_fused_x {
        relu_lowrank::fused_forward_with_executor(
            current.clone(),
            encoder,
            None,
            if apply_threshold {
                x_relu_threshold
            } else {
                0.0
            },
            latent_pattern,
            sparse_mask.clone(),
            x_grad_input_executor,
        )
    } else {
        let mut x_latent = current.clone().matmul(encoder);
        if apply_threshold && x_relu_threshold != 0.0 {
            x_latent = x_latent.sub_scalar(x_relu_threshold);
        }
        apply_latent(x_latent)
    };
    if let Some(start) = x_projection_start {
        x_projection_ns = start.elapsed().as_nanos();
    }

    let attention_mixer_start = prof_enabled.then(Instant::now);
    let attn = attention(x_neuron.clone(), current.clone());
    if let Some(start) = attention_mixer_start {
        attention_mixer_ns = start.elapsed().as_nanos();
    }
    let attention_post_norm_start = prof_enabled.then(Instant::now);
    let attn = apply_norm(attn);
    if let Some(start) = attention_post_norm_start {
        attention_post_norm_ns = start.elapsed().as_nanos();
    }
    let attention_norm_ns = attention_mixer_ns.saturating_add(attention_post_norm_ns);
    let attn_out = if keep_metric_aux {
        Some(attn.clone())
    } else {
        None
    };
    let attn_bytes = tensor_bytes(&attn);
    if memory_prof_enabled {
        lowrank_residual_memory_record_stage::<B>(
            |profile| &mut profile.after_attention_norm,
            &current.device(),
            tensor_bytes(&current) + tensor_bytes(&x_neuron) + attn_bytes,
        );
    }

    let y_projection_start = prof_enabled.then(Instant::now);
    let y_gate = if use_fused_y {
        relu_lowrank::fused_forward_with_executor(
            attn.clone(),
            encoder_v,
            None,
            if apply_threshold {
                y_relu_threshold
            } else {
                0.0
            },
            latent_pattern,
            sparse_mask,
            y_grad_input_executor,
        )
    } else {
        let mut y_latent = attn.clone().matmul(encoder_v);
        if apply_threshold && y_relu_threshold != 0.0 {
            y_latent = y_latent.sub_scalar(y_relu_threshold);
        }
        apply_latent(y_latent)
    };
    if let Some(start) = y_projection_start {
        y_projection_ns = start.elapsed().as_nanos();
    }
    if memory_prof_enabled {
        lowrank_residual_memory_record_stage::<B>(
            |profile| &mut profile.after_y_projection,
            &current.device(),
            tensor_bytes(&current) + tensor_bytes(&x_neuron) + attn_bytes + tensor_bytes(&y_gate),
        );
        lowrank_residual_memory_record_stage::<B>(
            |profile| &mut profile.after_y_post_quant,
            &current.device(),
            tensor_bytes(&current) + tensor_bytes(&x_neuron) + attn_bytes + tensor_bytes(&y_gate),
        );
    }

    let y_neuron_start = prof_enabled.then(Instant::now);
    let (y_neuron, x_neuron_out, y_gate_out) = if keep_aux {
        let y_neuron = dropout.forward(x_neuron.clone() * y_gate.clone());
        (y_neuron, Some(x_neuron), Some(y_gate))
    } else {
        let y_neuron = dropout.forward(x_neuron * y_gate);
        (y_neuron, None, None)
    };
    if let Some(start) = y_neuron_start {
        y_neuron_ns = start.elapsed().as_nanos();
    }
    let y_neuron_bytes = tensor_bytes(&y_neuron);
    if memory_prof_enabled {
        lowrank_residual_memory_record_stage::<B>(
            |profile| &mut profile.after_y_neuron,
            &current.device(),
            tensor_bytes(&current) + attn_bytes + y_neuron_bytes,
        );
    }
    let y_neuron_out = keep_aux.then(|| y_neuron.clone());

    let decoder_tail_start = prof_enabled.then(Instant::now);
    let mlp_out = decode_y_neuron_tail(y_neuron, decoder);
    if let Some(start) = decoder_tail_start {
        decoder_tail_ns = start.elapsed().as_nanos();
    }
    if memory_prof_enabled {
        lowrank_residual_memory_record_stage::<B>(
            |profile| &mut profile.after_decoder_tail,
            &current.device(),
            tensor_bytes(&current) + y_neuron_bytes + tensor_bytes(&mlp_out),
        );
    }

    let mlp_norm_start = prof_enabled.then(Instant::now);
    let mlp_out = apply_norm(mlp_out);
    if let Some(start) = mlp_norm_start {
        mlp_norm_ns = start.elapsed().as_nanos();
    }
    if memory_prof_enabled {
        lowrank_residual_memory_record_stage::<B>(
            |profile| &mut profile.after_mlp_norm,
            &current.device(),
            tensor_bytes(&current) + y_neuron_bytes + tensor_bytes(&mlp_out),
        );
    }
    let residual_delta_out = if keep_metric_aux {
        Some(mlp_out.clone())
    } else {
        None
    };
    let residual_combine_start = prof_enabled.then(Instant::now);
    let next = apply_norm(current + mlp_out);
    if let Some(start) = residual_combine_start {
        residual_combine_ns = start.elapsed().as_nanos();
    }

    if let Some(start) = total_start {
        lowrank_residual_profile_record(LowRankResidualProfileSnapshot {
            calls: 1,
            total_ns: start.elapsed().as_nanos(),
            x_projection_ns,
            x_post_quant_ns,
            attention_norm_ns,
            attention_mixer_ns,
            attention_post_norm_ns,
            y_projection_ns,
            y_post_quant_ns,
            y_neuron_ns,
            decoder_tail_ns,
            mlp_norm_ns,
            residual_combine_ns,
        });
    }
    if memory_prof_enabled {
        if let Ok(mut profile) = lowrank_residual_memory_profile_state().lock() {
            profile.calls = profile.calls.saturating_add(1);
        }
    }

    LowRankResidualInternal {
        next,
        attention_readout: attn_out,
        residual_delta: residual_delta_out,
        x_neuron: x_neuron_out,
        y_gate: y_gate_out,
        y_neuron: y_neuron_out,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn lowrank_residual_step<B, FAttn, FNorm, FAct>(
    current: Tensor<B, 4>,
    encoder: Tensor<B, 4>,
    encoder_v: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
    dropout: &Dropout,
    use_fused_x: bool,
    use_fused_y: bool,
    relu_threshold: f32,
    apply_threshold: bool,
    latent_pattern: &BlockPattern1d,
    lowrank_grad_input_executor: LowrankGradInputExecutor,
    sparse_mask: Option<Tensor<B, 4>>,
    attention: FAttn,
    apply_latent: FAct,
    apply_norm: FNorm,
) -> LowRankResidualOutput<B>
where
    B: Backend,
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
    FAttn: FnMut(Tensor<B, 4>, Tensor<B, 4>) -> Tensor<B, 4>,
    FNorm: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
    FAct: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
{
    let output = lowrank_residual_step_impl(
        current,
        LowRankResidualStepConfig {
            encoder,
            encoder_v,
            decoder,
            dropout,
            use_fused_x,
            use_fused_y,
            x_relu_threshold: relu_threshold,
            y_relu_threshold: relu_threshold,
            apply_threshold,
            latent_pattern,
            lowrank_grad_input_executor,
            sparse_mask,
            mode: LowRankResidualStepMode::full_output(),
        },
        attention,
        apply_latent,
        apply_norm,
    );
    LowRankResidualOutput {
        next: output.next,
        attention_readout: output.attention_readout,
        residual_delta: output.residual_delta,
        x_neuron: output.x_neuron.expect("x_neuron for full residual output"),
        y_gate: output.y_gate.expect("y_gate for full residual output"),
        y_neuron: output.y_neuron.expect("y_neuron for full residual output"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn lowrank_residual_step_branch_thresholds_relu_native<B, FAttn, FNorm, FAct>(
    current: Tensor<B, 4>,
    encoder: Tensor<B, 4>,
    encoder_v: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
    dropout: &Dropout,
    use_fused_x: bool,
    use_fused_y: bool,
    x_relu_threshold: f32,
    y_relu_threshold: f32,
    apply_threshold: bool,
    latent_pattern: &BlockPattern1d,
    lowrank_grad_input_executor: LowrankGradInputExecutor,
    sparse_mask: Option<Tensor<B, 4>>,
    attention: FAttn,
    apply_latent: FAct,
    apply_norm: FNorm,
) -> LowRankResidualOutput<B>
where
    B: Backend,
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
    FAttn: FnMut(Tensor<B, 4>, Tensor<B, 4>) -> Tensor<B, 4>,
    FNorm: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
    FAct: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
{
    let output = lowrank_residual_step_impl(
        current,
        LowRankResidualStepConfig {
            encoder,
            encoder_v,
            decoder,
            dropout,
            use_fused_x,
            use_fused_y,
            x_relu_threshold,
            y_relu_threshold,
            apply_threshold,
            latent_pattern,
            lowrank_grad_input_executor,
            sparse_mask,
            mode: LowRankResidualStepMode::full_output_relu_native(),
        },
        attention,
        apply_latent,
        apply_norm,
    );
    LowRankResidualOutput {
        next: output.next,
        attention_readout: output.attention_readout,
        residual_delta: output.residual_delta,
        x_neuron: output.x_neuron.expect("x_neuron for full residual output"),
        y_gate: output.y_gate.expect("y_gate for full residual output"),
        y_neuron: output.y_neuron.expect("y_neuron for full residual output"),
    }
}

#[cfg(any(feature = "probe", test))]
#[allow(clippy::too_many_arguments)]
pub fn lowrank_residual_step_with_metrics_branch_thresholds<B, FAttn, FNorm, FAct>(
    current: Tensor<B, 4>,
    encoder: Tensor<B, 4>,
    encoder_v: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
    dropout: &Dropout,
    use_fused_x: bool,
    use_fused_y: bool,
    x_relu_threshold: f32,
    y_relu_threshold: f32,
    apply_threshold: bool,
    latent_pattern: &BlockPattern1d,
    lowrank_grad_input_executor: LowrankGradInputExecutor,
    sparse_mask: Option<Tensor<B, 4>>,
    attention: FAttn,
    apply_latent: FAct,
    apply_norm: FNorm,
) -> LowRankResidualOutput<B>
where
    B: Backend,
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
    FAttn: FnMut(Tensor<B, 4>, Tensor<B, 4>) -> Tensor<B, 4>,
    FNorm: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
    FAct: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
{
    let output = lowrank_residual_step_impl(
        current,
        LowRankResidualStepConfig {
            encoder,
            encoder_v,
            decoder,
            dropout,
            use_fused_x,
            use_fused_y,
            x_relu_threshold,
            y_relu_threshold,
            apply_threshold,
            latent_pattern,
            lowrank_grad_input_executor,
            sparse_mask,
            mode: LowRankResidualStepMode::with_metrics(),
        },
        attention,
        apply_latent,
        apply_norm,
    );
    LowRankResidualOutput {
        next: output.next,
        attention_readout: output.attention_readout,
        residual_delta: output.residual_delta,
        x_neuron: output.x_neuron.expect("x_neuron for full residual output"),
        y_gate: output.y_gate.expect("y_gate for full residual output"),
        y_neuron: output.y_neuron.expect("y_neuron for full residual output"),
    }
}

#[cfg(any(feature = "probe", test))]
#[allow(clippy::too_many_arguments, dead_code)]
pub fn lowrank_residual_step_with_metrics<B, FAttn, FNorm, FAct>(
    current: Tensor<B, 4>,
    encoder: Tensor<B, 4>,
    encoder_v: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
    dropout: &Dropout,
    use_fused_x: bool,
    use_fused_y: bool,
    relu_threshold: f32,
    apply_threshold: bool,
    latent_pattern: &BlockPattern1d,
    lowrank_grad_input_executor: LowrankGradInputExecutor,
    sparse_mask: Option<Tensor<B, 4>>,
    attention: FAttn,
    apply_latent: FAct,
    apply_norm: FNorm,
) -> LowRankResidualOutput<B>
where
    B: Backend,
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
    FAttn: FnMut(Tensor<B, 4>, Tensor<B, 4>) -> Tensor<B, 4>,
    FNorm: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
    FAct: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
{
    lowrank_residual_step_with_metrics_branch_thresholds(
        current,
        encoder,
        encoder_v,
        decoder,
        dropout,
        use_fused_x,
        use_fused_y,
        relu_threshold,
        relu_threshold,
        apply_threshold,
        latent_pattern,
        lowrank_grad_input_executor,
        sparse_mask,
        attention,
        apply_latent,
        apply_norm,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn lowrank_residual_step_next_branch_thresholds<B, FAttn, FNorm, FAct>(
    current: Tensor<B, 4>,
    encoder: Tensor<B, 4>,
    encoder_v: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
    dropout: &Dropout,
    use_fused_x: bool,
    use_fused_y: bool,
    x_relu_threshold: f32,
    y_relu_threshold: f32,
    apply_threshold: bool,
    latent_pattern: &BlockPattern1d,
    lowrank_grad_input_executor: LowrankGradInputExecutor,
    sparse_mask: Option<Tensor<B, 4>>,
    attention: FAttn,
    apply_latent: FAct,
    apply_norm: FNorm,
) -> Tensor<B, 4>
where
    B: Backend,
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
    FAttn: FnMut(Tensor<B, 4>, Tensor<B, 4>) -> Tensor<B, 4>,
    FNorm: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
    FAct: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
{
    lowrank_residual_step_impl(
        current,
        LowRankResidualStepConfig {
            encoder,
            encoder_v,
            decoder,
            dropout,
            use_fused_x,
            use_fused_y,
            x_relu_threshold,
            y_relu_threshold,
            apply_threshold,
            latent_pattern,
            lowrank_grad_input_executor,
            sparse_mask,
            mode: LowRankResidualStepMode::next_only(),
        },
        attention,
        apply_latent,
        apply_norm,
    )
    .next
}

#[allow(clippy::too_many_arguments)]
pub fn lowrank_residual_step_next_branch_thresholds_relu_native<B, FAttn, FNorm, FAct>(
    current: Tensor<B, 4>,
    encoder: Tensor<B, 4>,
    encoder_v: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
    dropout: &Dropout,
    use_fused_x: bool,
    use_fused_y: bool,
    x_relu_threshold: f32,
    y_relu_threshold: f32,
    apply_threshold: bool,
    latent_pattern: &BlockPattern1d,
    lowrank_grad_input_executor: LowrankGradInputExecutor,
    sparse_mask: Option<Tensor<B, 4>>,
    attention: FAttn,
    apply_latent: FAct,
    apply_norm: FNorm,
) -> Tensor<B, 4>
where
    B: Backend,
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
    FAttn: FnMut(Tensor<B, 4>, Tensor<B, 4>) -> Tensor<B, 4>,
    FNorm: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
    FAct: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
{
    lowrank_residual_step_impl(
        current,
        LowRankResidualStepConfig {
            encoder,
            encoder_v,
            decoder,
            dropout,
            use_fused_x,
            use_fused_y,
            x_relu_threshold,
            y_relu_threshold,
            apply_threshold,
            latent_pattern,
            lowrank_grad_input_executor,
            sparse_mask,
            mode: LowRankResidualStepMode::next_only_relu_native(),
        },
        attention,
        apply_latent,
        apply_norm,
    )
    .next
}

#[allow(clippy::too_many_arguments, dead_code)]
pub fn lowrank_residual_step_next<B, FAttn, FNorm, FAct>(
    current: Tensor<B, 4>,
    encoder: Tensor<B, 4>,
    encoder_v: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
    dropout: &Dropout,
    use_fused_x: bool,
    use_fused_y: bool,
    relu_threshold: f32,
    apply_threshold: bool,
    latent_pattern: &BlockPattern1d,
    lowrank_grad_input_executor: LowrankGradInputExecutor,
    sparse_mask: Option<Tensor<B, 4>>,
    attention: FAttn,
    apply_latent: FAct,
    apply_norm: FNorm,
) -> Tensor<B, 4>
where
    B: Backend,
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
    FAttn: FnMut(Tensor<B, 4>, Tensor<B, 4>) -> Tensor<B, 4>,
    FNorm: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
    FAct: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
{
    lowrank_residual_step_next_branch_thresholds(
        current,
        encoder,
        encoder_v,
        decoder,
        dropout,
        use_fused_x,
        use_fused_y,
        relu_threshold,
        relu_threshold,
        apply_threshold,
        latent_pattern,
        lowrank_grad_input_executor,
        sparse_mask,
        attention,
        apply_latent,
        apply_norm,
    )
}
