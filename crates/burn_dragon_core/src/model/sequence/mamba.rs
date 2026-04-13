use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Tensor, TensorData};
use burn_dragon_kernel::kernels::sequence::mamba3::forward::{
    Mamba3TensorizedState, tensorized_mamba3_forward,
};
use serde::{Deserialize, Serialize};

use super::config::SequenceMemorySystem;

fn default_mamba_d_state() -> usize {
    16
}

fn default_mamba_d_conv() -> usize {
    4
}

fn default_mamba_expand() -> usize {
    2
}

fn default_mamba_dt_min() -> f32 {
    1.0e-3
}

fn default_mamba_dt_max() -> f32 {
    1.0e-1
}

fn default_mamba_dt_scale() -> f32 {
    1.0
}

fn default_mamba_headdim() -> usize {
    128
}

fn default_mamba_ngroups() -> usize {
    1
}

fn default_mamba_a_init_min() -> f32 {
    1.0
}

fn default_mamba_a_init_max() -> f32 {
    16.0
}

fn default_mamba_norm_eps() -> f32 {
    1.0e-5
}

fn default_mamba_rope_fraction() -> f32 {
    0.5
}

fn default_mamba_dt_init_floor() -> f32 {
    1.0e-4
}

fn default_mamba_a_floor() -> f32 {
    1.0e-4
}

fn default_mamba_chunk_size() -> usize {
    64
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct MambaSequenceConfig {
    #[serde(default = "default_mamba_d_state")]
    pub d_state: usize,
    #[serde(default = "default_mamba_d_conv")]
    pub d_conv: usize,
    #[serde(default = "default_mamba_expand")]
    pub expand: usize,
    #[serde(default)]
    pub dt_rank: Option<usize>,
    #[serde(default = "default_mamba_dt_min")]
    pub dt_min: f32,
    #[serde(default = "default_mamba_dt_max")]
    pub dt_max: f32,
    #[serde(default = "default_mamba_dt_scale")]
    pub dt_scale: f32,
    #[serde(default = "default_true")]
    pub conv_bias: bool,
    #[serde(default = "default_true")]
    pub use_fast_path: bool,
    #[serde(default = "default_mamba_headdim")]
    pub headdim: usize,
    #[serde(default = "default_mamba_ngroups")]
    pub ngroups: usize,
    #[serde(default = "default_mamba_a_init_min")]
    pub a_init_min: f32,
    #[serde(default = "default_mamba_a_init_max")]
    pub a_init_max: f32,
    #[serde(default = "default_mamba_norm_eps")]
    pub norm_eps: f32,
    #[serde(default = "default_mamba_rope_fraction")]
    pub rope_fraction: f32,
    #[serde(default = "default_mamba_dt_init_floor")]
    pub dt_init_floor: f32,
    #[serde(default = "default_mamba_a_floor")]
    pub a_floor: f32,
    #[serde(default = "default_mamba_chunk_size")]
    pub chunk_size: usize,
    #[serde(default)]
    pub is_outproj_norm: bool,
    #[serde(default)]
    pub is_mimo: bool,
    #[serde(default = "default_mamba_ngroups")]
    pub mimo_rank: usize,
}

impl Default for MambaSequenceConfig {
    fn default() -> Self {
        Self {
            d_state: default_mamba_d_state(),
            d_conv: default_mamba_d_conv(),
            expand: default_mamba_expand(),
            dt_rank: None,
            dt_min: default_mamba_dt_min(),
            dt_max: default_mamba_dt_max(),
            dt_scale: default_mamba_dt_scale(),
            conv_bias: default_true(),
            use_fast_path: default_true(),
            headdim: default_mamba_headdim(),
            ngroups: default_mamba_ngroups(),
            a_init_min: default_mamba_a_init_min(),
            a_init_max: default_mamba_a_init_max(),
            norm_eps: default_mamba_norm_eps(),
            rope_fraction: default_mamba_rope_fraction(),
            dt_init_floor: default_mamba_dt_init_floor(),
            a_floor: default_mamba_a_floor(),
            chunk_size: default_mamba_chunk_size(),
            is_outproj_norm: false,
            is_mimo: false,
            mimo_rank: default_mamba_ngroups(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedMambaSequenceConfig {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub dt_rank: usize,
    pub dt_min: f32,
    pub dt_max: f32,
    pub dt_scale: f32,
    pub conv_bias: bool,
    pub use_fast_path: bool,
    pub headdim: usize,
    pub ngroups: usize,
    pub nheads: usize,
    pub a_init_min: f32,
    pub a_init_max: f32,
    pub norm_eps: f32,
    pub rope_fraction: f32,
    pub dt_init_floor: f32,
    pub a_floor: f32,
    pub chunk_size: usize,
    pub is_outproj_norm: bool,
    pub is_mimo: bool,
    pub mimo_rank: usize,
    pub num_rope_angles: usize,
}

impl ResolvedMambaSequenceConfig {
    pub fn mamba3_in_proj_dim(self) -> usize {
        2 * self.d_inner
            + 2 * self.ngroups * self.mimo_rank * self.d_state
            + 3 * self.nheads
            + self.num_rope_angles
    }
}

impl MambaSequenceConfig {
    pub fn validate(
        &self,
        memory_system: SequenceMemorySystem,
        d_model: usize,
    ) -> Result<(), String> {
        if self.d_state == 0 {
            return Err("d_state must be positive".to_string());
        }
        if self.d_conv == 0 {
            return Err("d_conv must be positive".to_string());
        }
        if self.expand == 0 {
            return Err("expand must be positive".to_string());
        }
        if self.dt_min <= 0.0 || !self.dt_min.is_finite() {
            return Err("dt_min must be finite and positive".to_string());
        }
        if self.dt_max < self.dt_min || !self.dt_max.is_finite() {
            return Err("dt_max must be finite and >= dt_min".to_string());
        }
        if self.dt_scale <= 0.0 || !self.dt_scale.is_finite() {
            return Err("dt_scale must be finite and positive".to_string());
        }
        let d_inner = d_model.max(1) * self.expand.max(1);
        if matches!(memory_system, SequenceMemorySystem::Mamba3StateSpaceDuality) {
            if self.headdim == 0 {
                return Err(format!("headdim must be positive for {memory_system:?}"));
            }
            if d_inner % self.headdim != 0 {
                return Err(format!(
                    "{memory_system:?} requires d_inner divisible by headdim (got d_inner={d_inner} headdim={})",
                    self.headdim
                ));
            }
            let nheads = d_inner / self.headdim;
            if self.ngroups == 0 {
                return Err(format!("ngroups must be positive for {memory_system:?}"));
            }
            if nheads % self.ngroups != 0 {
                return Err(format!(
                    "{memory_system:?} requires nheads divisible by ngroups (got nheads={nheads} ngroups={})",
                    self.ngroups
                ));
            }
            if self.norm_eps <= 0.0 || !self.norm_eps.is_finite() {
                return Err("norm_eps must be finite and positive".to_string());
            }
        }
        if matches!(memory_system, SequenceMemorySystem::Mamba3StateSpaceDuality) {
            if (self.rope_fraction - 0.5).abs() > 1.0e-6
                && (self.rope_fraction - 1.0).abs() > 1.0e-6
            {
                return Err(
                    "mamba3_state_space_duality currently supports rope_fraction = 0.5 or 1.0"
                        .to_string(),
                );
            }
            if self.dt_init_floor <= 0.0 || !self.dt_init_floor.is_finite() {
                return Err("dt_init_floor must be finite and positive".to_string());
            }
            if self.a_floor <= 0.0 || !self.a_floor.is_finite() {
                return Err("a_floor must be finite and positive".to_string());
            }
            if self.chunk_size == 0 {
                return Err(
                    "chunk_size must be positive for mamba3_state_space_duality".to_string()
                );
            }
            if self.is_mimo {
                return Err("mamba3_state_space_duality MIMO is not implemented in burn_dragon yet; set model.mamba.is_mimo = false".to_string());
            }
            let split_tensor_size = ((self.d_state as f32) * self.rope_fraction).floor() as usize;
            let split_tensor_size = split_tensor_size - (split_tensor_size % 2);
            if split_tensor_size < 2 {
                return Err("mamba3_state_space_duality requires at least one rotary pair in d_state * rope_fraction".to_string());
            }
        }
        Ok(())
    }

    pub fn resolve(
        &self,
        d_model: usize,
        memory_system: SequenceMemorySystem,
    ) -> ResolvedMambaSequenceConfig {
        self.validate(memory_system, d_model)
            .unwrap_or_else(|message| panic!("{message}"));
        let d_model = d_model.max(1);
        let d_state = self.d_state.max(1);
        let d_conv = self.d_conv.max(1);
        let expand = self.expand.max(1);
        let d_inner = d_model * expand;
        let dt_rank = self.dt_rank.unwrap_or_else(|| d_model.div_ceil(16)).max(1);
        let headdim = self.headdim.max(1);
        let nheads = if matches!(memory_system, SequenceMemorySystem::Mamba3StateSpaceDuality) {
            d_inner / headdim
        } else {
            0
        };
        let split_tensor_size = ((d_state as f32) * self.rope_fraction).floor() as usize;
        let split_tensor_size = split_tensor_size - (split_tensor_size % 2);
        let num_rope_angles =
            if matches!(memory_system, SequenceMemorySystem::Mamba3StateSpaceDuality) {
                (split_tensor_size / 2).max(1)
            } else {
                0
            };
        ResolvedMambaSequenceConfig {
            d_model,
            d_inner,
            d_state,
            d_conv,
            dt_rank,
            dt_min: self.dt_min.max(1.0e-6),
            dt_max: self.dt_max.max(self.dt_min.max(1.0e-6)),
            dt_scale: self.dt_scale.max(1.0e-6),
            conv_bias: self.conv_bias,
            use_fast_path: self.use_fast_path,
            headdim,
            ngroups: self.ngroups.max(1),
            nheads,
            a_init_min: self.a_init_min.max(1.0e-6),
            a_init_max: self.a_init_max.max(self.a_init_min.max(1.0e-6)),
            norm_eps: self.norm_eps.max(1.0e-8),
            rope_fraction: self.rope_fraction,
            dt_init_floor: self.dt_init_floor.max(1.0e-6),
            a_floor: self.a_floor.max(1.0e-6),
            chunk_size: self.chunk_size.max(1),
            is_outproj_norm: self.is_outproj_norm,
            is_mimo: self.is_mimo,
            mimo_rank: self.mimo_rank.max(1),
            num_rope_angles,
        }
    }
}

#[derive(Module, Debug)]
pub struct Mamba3SequenceParameters<B: Backend> {
    d_model: usize,
    d_inner: usize,
    d_state: usize,
    headdim: usize,
    ngroups: usize,
    nheads: usize,
    norm_eps: f32,
    num_rope_angles: usize,
    a_floor: f32,
    chunk_size: usize,
    in_proj: Param<Tensor<B, 2>>,
    dt_bias: Param<Tensor<B, 1>>,
    b_bias: Param<Tensor<B, 2>>,
    c_bias: Param<Tensor<B, 2>>,
    b_norm_weight: Param<Tensor<B, 1>>,
    c_norm_weight: Param<Tensor<B, 1>>,
    d_skip: Param<Tensor<B, 1>>,
    out_proj: Param<Tensor<B, 2>>,
}

impl<B: Backend> Mamba3SequenceParameters<B> {
    pub fn new(config: ResolvedMambaSequenceConfig, device: &B::Device) -> Self {
        let in_std = (1.0 / config.d_model.max(1) as f32).sqrt();
        let out_std = (1.0 / config.d_inner.max(1) as f32).sqrt();
        let log_dt_min = config.dt_min.ln();
        let log_dt_max = config.dt_max.ln();
        let dt_sample = Tensor::<B, 1>::random(
            [config.nheads],
            TensorDistribution::Uniform(log_dt_min as f64, log_dt_max as f64),
            device,
        )
        .exp()
        .clamp_min(config.dt_init_floor);
        let dt_bias_values = dt_sample
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("mamba3 dt bias init")
            .into_iter()
            .map(|dt| dt + (-(-dt).exp_m1()).ln())
            .collect::<Vec<_>>();
        let in_proj = Param::from_tensor(Tensor::<B, 2>::random(
            [config.d_model, config.mamba3_in_proj_dim()],
            TensorDistribution::Normal(0.0, in_std as f64),
            device,
        ));
        let dt_bias = Param::from_tensor(Tensor::<B, 1>::from_data(
            TensorData::new(dt_bias_values, [config.nheads]),
            device,
        ));
        let b_bias = Param::from_tensor(Tensor::<B, 2>::ones(
            [config.nheads, config.d_state],
            device,
        ));
        let c_bias = Param::from_tensor(Tensor::<B, 2>::ones(
            [config.nheads, config.d_state],
            device,
        ));
        let b_norm_weight = Param::from_tensor(Tensor::<B, 1>::ones([config.d_state], device));
        let c_norm_weight = Param::from_tensor(Tensor::<B, 1>::ones([config.d_state], device));
        let d_skip = Param::from_tensor(Tensor::<B, 1>::ones([config.nheads], device));
        let out_proj = Param::from_tensor(Tensor::<B, 2>::random(
            [config.d_inner, config.d_model],
            TensorDistribution::Normal(0.0, out_std as f64),
            device,
        ));
        Self {
            d_model: config.d_model,
            d_inner: config.d_inner,
            d_state: config.d_state,
            headdim: config.headdim,
            ngroups: config.ngroups,
            nheads: config.nheads,
            norm_eps: config.norm_eps,
            num_rope_angles: config.num_rope_angles,
            a_floor: config.a_floor,
            chunk_size: config.chunk_size,
            in_proj,
            dt_bias,
            b_bias,
            c_bias,
            b_norm_weight,
            c_norm_weight,
            d_skip,
            out_proj,
        }
    }

    pub fn config(&self) -> ResolvedMambaSequenceConfig {
        ResolvedMambaSequenceConfig {
            d_model: self.d_model,
            d_inner: self.d_inner,
            d_state: self.d_state,
            d_conv: default_mamba_d_conv(),
            dt_rank: self.d_model.div_ceil(16),
            dt_min: default_mamba_dt_min(),
            dt_max: default_mamba_dt_max(),
            dt_scale: default_mamba_dt_scale(),
            conv_bias: false,
            use_fast_path: false,
            headdim: self.headdim,
            ngroups: self.ngroups,
            nheads: self.nheads,
            a_init_min: default_mamba_a_init_min(),
            a_init_max: default_mamba_a_init_max(),
            norm_eps: self.norm_eps,
            rope_fraction: default_mamba_rope_fraction(),
            dt_init_floor: default_mamba_dt_init_floor(),
            a_floor: self.a_floor,
            chunk_size: self.chunk_size,
            is_outproj_norm: false,
            is_mimo: false,
            mimo_rank: 1,
            num_rope_angles: self.num_rope_angles,
        }
    }

    pub fn in_proj_tensor(&self) -> Tensor<B, 2> {
        self.in_proj.val()
    }

    pub fn dt_bias_tensor(&self) -> Tensor<B, 1> {
        self.dt_bias.val()
    }

    pub fn b_bias_tensor(&self) -> Tensor<B, 2> {
        self.b_bias.val()
    }

    pub fn c_bias_tensor(&self) -> Tensor<B, 2> {
        self.c_bias.val()
    }

    pub fn b_norm_weight_tensor(&self) -> Tensor<B, 1> {
        self.b_norm_weight.val()
    }

    pub fn c_norm_weight_tensor(&self) -> Tensor<B, 1> {
        self.c_norm_weight.val()
    }

    pub fn d_skip_tensor(&self) -> Tensor<B, 1> {
        self.d_skip.val()
    }

    pub fn out_proj_tensor(&self) -> Tensor<B, 2> {
        self.out_proj.val()
    }

    pub fn blended_with(&self, fresh: &Self, alpha: f32) -> Self {
        Self {
            d_model: self.d_model,
            d_inner: self.d_inner,
            d_state: self.d_state,
            headdim: self.headdim,
            ngroups: self.ngroups,
            nheads: self.nheads,
            norm_eps: self.norm_eps,
            num_rope_angles: self.num_rope_angles,
            a_floor: self.a_floor,
            chunk_size: self.chunk_size,
            in_proj: Param::from_tensor(MambaSequenceParameters::<B>::blend_param(
                self.in_proj.val(),
                fresh.in_proj.val(),
                alpha,
            )),
            dt_bias: Param::from_tensor(MambaSequenceParameters::<B>::blend_param(
                self.dt_bias.val(),
                fresh.dt_bias.val(),
                alpha,
            )),
            b_bias: Param::from_tensor(MambaSequenceParameters::<B>::blend_param(
                self.b_bias.val(),
                fresh.b_bias.val(),
                alpha,
            )),
            c_bias: Param::from_tensor(MambaSequenceParameters::<B>::blend_param(
                self.c_bias.val(),
                fresh.c_bias.val(),
                alpha,
            )),
            b_norm_weight: Param::from_tensor(MambaSequenceParameters::<B>::blend_param(
                self.b_norm_weight.val(),
                fresh.b_norm_weight.val(),
                alpha,
            )),
            c_norm_weight: Param::from_tensor(MambaSequenceParameters::<B>::blend_param(
                self.c_norm_weight.val(),
                fresh.c_norm_weight.val(),
                alpha,
            )),
            d_skip: Param::from_tensor(MambaSequenceParameters::<B>::blend_param(
                self.d_skip.val(),
                fresh.d_skip.val(),
                alpha,
            )),
            out_proj: Param::from_tensor(MambaSequenceParameters::<B>::blend_param(
                self.out_proj.val(),
                fresh.out_proj.val(),
                alpha,
            )),
        }
    }

    pub fn matched_fresh_rms(&self, fresh: &Self) -> Self {
        Self {
            d_model: self.d_model,
            d_inner: self.d_inner,
            d_state: self.d_state,
            headdim: self.headdim,
            ngroups: self.ngroups,
            nheads: self.nheads,
            norm_eps: self.norm_eps,
            num_rope_angles: self.num_rope_angles,
            a_floor: self.a_floor,
            chunk_size: self.chunk_size,
            in_proj: Param::from_tensor(MambaSequenceParameters::<B>::match_fresh_rms(
                self.in_proj.val(),
                fresh.in_proj.val(),
            )),
            dt_bias: Param::from_tensor(MambaSequenceParameters::<B>::match_fresh_rms(
                self.dt_bias.val(),
                fresh.dt_bias.val(),
            )),
            b_bias: Param::from_tensor(MambaSequenceParameters::<B>::match_fresh_rms(
                self.b_bias.val(),
                fresh.b_bias.val(),
            )),
            c_bias: Param::from_tensor(MambaSequenceParameters::<B>::match_fresh_rms(
                self.c_bias.val(),
                fresh.c_bias.val(),
            )),
            b_norm_weight: Param::from_tensor(MambaSequenceParameters::<B>::match_fresh_rms(
                self.b_norm_weight.val(),
                fresh.b_norm_weight.val(),
            )),
            c_norm_weight: Param::from_tensor(MambaSequenceParameters::<B>::match_fresh_rms(
                self.c_norm_weight.val(),
                fresh.c_norm_weight.val(),
            )),
            d_skip: Param::from_tensor(MambaSequenceParameters::<B>::match_fresh_rms(
                self.d_skip.val(),
                fresh.d_skip.val(),
            )),
            out_proj: Param::from_tensor(MambaSequenceParameters::<B>::match_fresh_rms(
                self.out_proj.val(),
                fresh.out_proj.val(),
            )),
        }
    }
}

#[derive(Module, Debug)]
pub struct MambaSequenceParameters<B: Backend> {
    mamba3: Mamba3SequenceParameters<B>,
}

impl<B: Backend> MambaSequenceParameters<B> {
    fn param_rms<const D: usize>(tensor: Tensor<B, D>) -> f32 {
        let values = tensor
            .powf_scalar(2.0)
            .mean()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("mamba rms scalar");
        values.first().copied().unwrap_or(0.0).sqrt()
    }

    fn blend_param<const D: usize>(
        source: Tensor<B, D>,
        fresh: Tensor<B, D>,
        alpha: f32,
    ) -> Tensor<B, D> {
        let alpha = alpha.clamp(0.0, 1.0);
        (fresh.mul_scalar(1.0 - alpha) + source.mul_scalar(alpha)).detach()
    }

    fn match_fresh_rms<const D: usize>(source: Tensor<B, D>, fresh: Tensor<B, D>) -> Tensor<B, D> {
        let source_rms = Self::param_rms(source.clone());
        let fresh_rms = Self::param_rms(fresh);
        if source_rms <= 1.0e-8 || !source_rms.is_finite() || !fresh_rms.is_finite() {
            return source;
        }
        source.mul_scalar(fresh_rms / source_rms).detach()
    }

    pub fn new(
        config: ResolvedMambaSequenceConfig,
        memory_system: SequenceMemorySystem,
        device: &B::Device,
    ) -> Self {
        match memory_system {
            SequenceMemorySystem::Mamba3StateSpaceDuality => Self {
                mamba3: Mamba3SequenceParameters::new(config, device),
            },
            other => panic!("unsupported memory system {other:?} for mamba params"),
        }
    }

    pub fn mamba3(&self) -> &Mamba3SequenceParameters<B> {
        &self.mamba3
    }

    pub fn blended_with(&self, fresh: &Self, alpha: f32) -> Self {
        Self {
            mamba3: self.mamba3.blended_with(&fresh.mamba3, alpha),
        }
    }

    pub fn matched_fresh_rms(&self, fresh: &Self) -> Self {
        Self {
            mamba3: self.mamba3.matched_fresh_rms(&fresh.mamba3),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MambaReferenceState<B: Backend> {
    pub ssm: Tensor<B, 4>,
    pub angle: Option<Tensor<B, 3>>,
    pub k: Option<Tensor<B, 3>>,
    pub v: Option<Tensor<B, 3>>,
}

fn mamba3_reference<B: Backend>(
    hidden_states: Tensor<B, 4>,
    params: &Mamba3SequenceParameters<B>,
    state: Option<MambaReferenceState<B>>,
) -> (Tensor<B, 4>, MambaReferenceState<B>) {
    let config = params.config();
    let tensorized = tensorized_mamba3_forward(
        hidden_states,
        config.d_inner,
        config.d_state,
        config.headdim,
        config.ngroups,
        config.num_rope_angles,
        config.norm_eps,
        config.a_floor,
        config.chunk_size,
        params.in_proj_tensor(),
        params.dt_bias_tensor(),
        params.b_bias_tensor(),
        params.c_bias_tensor(),
        params.b_norm_weight_tensor(),
        params.c_norm_weight_tensor(),
        params.d_skip_tensor(),
        params.out_proj_tensor(),
        state.map(|state| Mamba3TensorizedState {
            ssm: state.ssm,
            angle: state.angle.expect("mamba3 reference requires angle state"),
            k: state.k.expect("mamba3 reference requires k state"),
            v: state.v.expect("mamba3 reference requires v state"),
        }),
    );
    (
        tensorized.context,
        MambaReferenceState {
            ssm: tensorized.state.ssm,
            angle: Some(tensorized.state.angle),
            k: Some(tensorized.state.k),
            v: Some(tensorized.state.v),
        },
    )
}

pub fn mamba_reference<B: Backend>(
    hidden_states: Tensor<B, 4>,
    params: &MambaSequenceParameters<B>,
    state: Option<MambaReferenceState<B>>,
) -> (Tensor<B, 4>, MambaReferenceState<B>) {
    mamba3_reference(hidden_states, params.mamba3(), state)
}
