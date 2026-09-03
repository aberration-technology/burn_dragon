use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor, Param, ParamId,
};
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DragonNormKind {
    #[default]
    LayerNorm,
    #[serde(alias = "rmsnorm")]
    RmsNorm,
    #[serde(alias = "dyt")]
    DynamicTanh,
    Derf,
}

impl core::fmt::Display for DragonNormKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<B: Backend> Module<B> for DragonNormKind {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for DragonNormKind {
    type InnerModule = DragonNormKind;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for DragonNormKind {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for DragonNormKind {}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct DragonNormConfig {
    #[serde(default)]
    pub kind: DragonNormKind,
    #[serde(default = "default_norm_epsilon")]
    pub eps: f32,
    #[serde(default)]
    pub alpha_init: Option<f32>,
    #[serde(default)]
    pub shift_init: Option<f32>,
}

impl Default for DragonNormConfig {
    fn default() -> Self {
        Self {
            kind: DragonNormKind::default(),
            eps: default_norm_epsilon(),
            alpha_init: None,
            shift_init: None,
        }
    }
}

const fn default_norm_epsilon() -> f32 {
    1e-5
}

const fn default_dyt_alpha_init() -> f32 {
    0.5
}

const fn default_derf_alpha_init() -> f32 {
    0.886_226_95
}

const fn default_norm_shift_init() -> f32 {
    0.0
}

impl DragonNormConfig {
    pub fn resolved_alpha_init(&self) -> f32 {
        self.alpha_init.unwrap_or(match self.kind {
            DragonNormKind::LayerNorm | DragonNormKind::RmsNorm => 1.0,
            DragonNormKind::DynamicTanh => default_dyt_alpha_init(),
            DragonNormKind::Derf => default_derf_alpha_init(),
        })
    }

    pub fn resolved_shift_init(&self) -> f32 {
        self.shift_init.unwrap_or(default_norm_shift_init())
    }
}

impl<B: Backend> Module<B> for DragonNormConfig {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for DragonNormConfig {
    type InnerModule = DragonNormConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for DragonNormConfig {
    fn content(&self, content: Content) -> Option<Content> {
        let summary = format!(
            "kind={}, eps={}, alpha_init={}, shift_init={}",
            self.kind,
            self.eps,
            self.resolved_alpha_init(),
            self.resolved_shift_init()
        );
        content
            .set_top_level_type("DragonNormConfig")
            .add_formatted(&summary)
            .optional()
    }
}

impl ModuleDisplay for DragonNormConfig {}

#[derive(Module, Debug)]
pub struct DragonNorm<B: Backend> {
    kind: DragonNormKind,
    #[module(skip)]
    eps: f32,
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
    alpha: Param<Tensor<B, 1>>,
    shift: Param<Tensor<B, 1>>,
}

pub(crate) struct DragonNormVjp<B: Backend, const D: usize> {
    pub grad_input: Tensor<B, D>,
    pub grad_gamma: Tensor<B, 1>,
    pub grad_beta: Tensor<B, 1>,
    pub grad_alpha: Tensor<B, 1>,
    pub grad_shift: Tensor<B, 1>,
}

impl<B: Backend> DragonNorm<B> {
    pub(crate) fn parameter_ids(&self) -> (ParamId, ParamId, ParamId, ParamId) {
        (self.gamma.id, self.beta.id, self.alpha.id, self.shift.id)
    }

    fn param_rms<const D: usize>(tensor: Tensor<B, D>) -> f32 {
        let values = tensor
            .powf_scalar(2.0)
            .mean()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("dragon norm rms scalar");
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

    pub fn new(config: &DragonNormConfig, width: usize, device: &B::Device) -> Self {
        let width = width.max(1);
        let alpha_init = config.resolved_alpha_init();
        let shift_init = config.resolved_shift_init();
        Self {
            kind: config.kind,
            eps: config.eps.max(1e-8),
            gamma: Param::from_tensor(Tensor::<B, 1>::ones([width], device)),
            beta: Param::from_tensor(Tensor::<B, 1>::zeros([width], device)),
            alpha: Param::from_tensor(Tensor::<B, 1>::ones([1], device).mul_scalar(alpha_init)),
            shift: Param::from_tensor(Tensor::<B, 1>::ones([1], device).mul_scalar(shift_init)),
        }
    }

    pub fn blended_with(&self, fresh: &Self, alpha: f32) -> Self {
        Self {
            kind: self.kind,
            eps: self.eps,
            gamma: Param::from_tensor(Self::blend_param(
                self.gamma.val(),
                fresh.gamma.val(),
                alpha,
            )),
            beta: Param::from_tensor(Self::blend_param(self.beta.val(), fresh.beta.val(), alpha)),
            alpha: Param::from_tensor(Self::blend_param(
                self.alpha.val(),
                fresh.alpha.val(),
                alpha,
            )),
            shift: Param::from_tensor(Self::blend_param(
                self.shift.val(),
                fresh.shift.val(),
                alpha,
            )),
        }
    }

    pub(crate) fn value_clone(&self) -> Self {
        Self {
            kind: self.kind,
            eps: self.eps,
            gamma: Param::from_tensor(self.gamma.val()),
            beta: Param::from_tensor(self.beta.val()),
            alpha: Param::from_tensor(self.alpha.val()),
            shift: Param::from_tensor(self.shift.val()),
        }
    }

    pub fn matched_fresh_rms(&self, fresh: &Self) -> Self {
        Self {
            kind: self.kind,
            eps: self.eps,
            gamma: Param::from_tensor(Self::match_fresh_rms(self.gamma.val(), fresh.gamma.val())),
            beta: Param::from_tensor(Self::match_fresh_rms(self.beta.val(), fresh.beta.val())),
            alpha: Param::from_tensor(Self::match_fresh_rms(self.alpha.val(), fresh.alpha.val())),
            shift: Param::from_tensor(Self::match_fresh_rms(self.shift.val(), fresh.shift.val())),
        }
    }

    pub fn kind(&self) -> DragonNormKind {
        self.kind
    }

    pub fn forward<const D: usize>(&self, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let gamma = self.param_view::<D>(self.gamma.val());
        let beta = self.param_view::<D>(self.beta.val());

        let output = match self.kind {
            DragonNormKind::LayerNorm => {
                let (var, mean) = tensor.clone().var_mean_bias(D - 1);
                tensor.sub(mean).div(var.add_scalar(self.eps).sqrt())
            }
            DragonNormKind::RmsNorm => {
                let (var, mean) = tensor.clone().var_mean_bias(D - 1);
                let rms = var.add(mean.powf_scalar(2.0)).add_scalar(self.eps).sqrt();
                tensor.div(rms)
            }
            DragonNormKind::DynamicTanh => {
                let alpha = self.scalar_param_view::<D>(self.alpha.val());
                tensor.mul(alpha).tanh()
            }
            DragonNormKind::Derf => {
                let alpha = self.scalar_param_view::<D>(self.alpha.val());
                let shift = self.scalar_param_view::<D>(self.shift.val());
                tensor.mul(alpha).add(shift).erf()
            }
        };

        output.mul(gamma).add(beta)
    }

    /// Plain-backend VJP for only the normalization input.
    pub(crate) fn vjp_input<const D: usize>(
        &self,
        input: Tensor<B, D>,
        grad_output: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let width = input.shape().dims::<D>()[D - 1].max(1);
        let gamma = self.param_view::<D>(self.gamma.val());
        let grad = grad_output.clone() * gamma;

        match self.kind {
            DragonNormKind::LayerNorm => {
                let (variance, mean) = input.clone().var_mean_bias(D - 1);
                let inverse_std = variance.add_scalar(self.eps).sqrt().recip();
                let centered = input - mean;
                let normalized = centered * inverse_std.clone();
                let grad_sum = grad.clone().sum_dim(D - 1);
                let grad_normalized_sum = (grad.clone() * normalized.clone()).sum_dim(D - 1);
                (grad.mul_scalar(width as f32) - grad_sum - normalized * grad_normalized_sum)
                    * inverse_std.mul_scalar(1.0 / width as f32)
            }
            DragonNormKind::RmsNorm => {
                let mean_square = input.clone().square().mean_dim(D - 1);
                let inverse_rms = mean_square.add_scalar(self.eps).sqrt().recip();
                let projected = (grad.clone() * input.clone()).mean_dim(D - 1);
                grad * inverse_rms.clone() - input * projected * inverse_rms.powf_scalar(3.0)
            }
            DragonNormKind::DynamicTanh => {
                let alpha = self.scalar_param_view::<D>(self.alpha.val());
                let activated = (input.clone() * alpha.clone()).tanh();
                let activation_derivative =
                    activated.clone().square().mul_scalar(-1.0).add_scalar(1.0);
                grad * alpha * activation_derivative
            }
            DragonNormKind::Derf => {
                let alpha = self.scalar_param_view::<D>(self.alpha.val());
                let shift = self.scalar_param_view::<D>(self.shift.val());
                let preactivation = input.clone() * alpha.clone() + shift;
                let erf_derivative = preactivation
                    .square()
                    .mul_scalar(-1.0)
                    .exp()
                    .mul_scalar(2.0 / std::f32::consts::PI.sqrt());
                let grad_preactivation = grad * erf_derivative;
                grad_preactivation * alpha
            }
        }
    }

    /// Plain-backend VJP for the input and every normalization parameter.
    ///
    /// The four parameter tensors are returned in `(gamma, beta, alpha, shift)`
    /// order. Parameters unused by the selected normalization kind receive an
    /// exact zero derivative, preserving one stable optimizer/checkpoint schema.
    pub(crate) fn vjp_with_parameters<const D: usize>(
        &self,
        input: Tensor<B, D>,
        grad_output: Tensor<B, D>,
    ) -> DragonNormVjp<B, D> {
        let width = input.shape().dims::<D>()[D - 1].max(1);
        let gamma = self.param_view::<D>(self.gamma.val());
        let grad = grad_output.clone() * gamma;
        let reduce_width = |tensor: Tensor<B, D>| {
            let rows = tensor
                .shape()
                .dims::<D>()
                .into_iter()
                .take(D - 1)
                .product::<usize>()
                .max(1);
            tensor.reshape([rows, width]).sum_dim(0).reshape([width])
        };
        let grad_beta = reduce_width(grad_output.clone());
        let zero_alpha = self.alpha.val().zeros_like();
        let zero_shift = self.shift.val().zeros_like();

        match self.kind {
            DragonNormKind::LayerNorm => {
                let (variance, mean) = input.clone().var_mean_bias(D - 1);
                let inverse_std = variance.add_scalar(self.eps).sqrt().recip();
                let normalized = (input - mean) * inverse_std.clone();
                let grad_sum = grad.clone().sum_dim(D - 1);
                let grad_normalized_sum = (grad.clone() * normalized.clone()).sum_dim(D - 1);
                let grad_input = (grad.mul_scalar(width as f32)
                    - grad_sum
                    - normalized.clone() * grad_normalized_sum)
                    * inverse_std.mul_scalar(1.0 / width as f32);
                DragonNormVjp {
                    grad_input,
                    grad_gamma: reduce_width(grad_output * normalized),
                    grad_beta,
                    grad_alpha: zero_alpha,
                    grad_shift: zero_shift,
                }
            }
            DragonNormKind::RmsNorm => {
                let mean_square = input.clone().square().mean_dim(D - 1);
                let inverse_rms = mean_square.add_scalar(self.eps).sqrt().recip();
                let projected = (grad.clone() * input.clone()).mean_dim(D - 1);
                let grad_input = grad * inverse_rms.clone()
                    - input.clone() * projected * inverse_rms.clone().powf_scalar(3.0);
                DragonNormVjp {
                    grad_input,
                    grad_gamma: reduce_width(grad_output * input * inverse_rms),
                    grad_beta,
                    grad_alpha: zero_alpha,
                    grad_shift: zero_shift,
                }
            }
            DragonNormKind::DynamicTanh => {
                let alpha = self.scalar_param_view::<D>(self.alpha.val());
                let activated = (input.clone() * alpha).tanh();
                let derivative = activated.clone().square().mul_scalar(-1.0).add_scalar(1.0);
                DragonNormVjp {
                    grad_input: grad.clone()
                        * self.scalar_param_view::<D>(self.alpha.val())
                        * derivative.clone(),
                    grad_gamma: reduce_width(grad_output * activated),
                    grad_beta,
                    grad_alpha: (grad * derivative * input).sum().reshape([1]),
                    grad_shift: zero_shift,
                }
            }
            DragonNormKind::Derf => {
                let alpha = self.scalar_param_view::<D>(self.alpha.val());
                let shift = self.scalar_param_view::<D>(self.shift.val());
                let preactivation = input.clone() * alpha.clone() + shift;
                let activated = preactivation.clone().erf();
                let grad_preactivation = grad
                    * preactivation
                        .square()
                        .mul_scalar(-1.0)
                        .exp()
                        .mul_scalar(2.0 / std::f32::consts::PI.sqrt());
                DragonNormVjp {
                    grad_input: grad_preactivation.clone() * alpha,
                    grad_gamma: reduce_width(grad_output * activated),
                    grad_beta,
                    grad_alpha: (grad_preactivation.clone() * input).sum().reshape([1]),
                    grad_shift: grad_preactivation.sum().reshape([1]),
                }
            }
        }
    }

    fn param_view<const D: usize>(&self, param: Tensor<B, 1>) -> Tensor<B, D> {
        let [width] = param.shape().dims::<1>();
        let mut shape = [1; D];
        shape[D - 1] = width;
        param.reshape(shape)
    }

    fn scalar_param_view<const D: usize>(&self, param: Tensor<B, 1>) -> Tensor<B, D> {
        let shape = [1; D];
        param.reshape(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::optim::GradientsParams;
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;
    type AutodiffBackend = Autodiff<Backend>;

    fn device() -> burn::tensor::Device<Backend> {
        burn::tensor::Device::<Backend>::default()
    }

    #[test]
    fn layer_norm_zero_centers_rows() {
        let device = device();
        let norm = DragonNorm::<Backend>::new(
            &DragonNormConfig {
                kind: DragonNormKind::LayerNorm,
                ..Default::default()
            },
            2,
            &device,
        );
        let x = Tensor::<Backend, 2>::from_data(TensorData::new(vec![1.0, 3.0], [1, 2]), &device);
        let y = norm.forward(x);
        let data = y.into_data().to_vec::<f32>().expect("f32 data");
        assert!(
            (data[0] + 1.0).abs() < 1e-4,
            "expected approx -1, got {}",
            data[0]
        );
        assert!(
            (data[1] - 1.0).abs() < 1e-4,
            "expected approx 1, got {}",
            data[1]
        );
    }

    #[test]
    fn rms_norm_preserves_direction_without_mean_centering() {
        let device = device();
        let norm = DragonNorm::<Backend>::new(
            &DragonNormConfig {
                kind: DragonNormKind::RmsNorm,
                ..Default::default()
            },
            2,
            &device,
        );
        let x = Tensor::<Backend, 2>::from_data(TensorData::new(vec![3.0, 3.0], [1, 2]), &device);
        let y = norm.forward(x);
        let data = y.into_data().to_vec::<f32>().expect("f32 data");
        assert!(
            (data[0] - 1.0).abs() < 1e-4,
            "expected approx 1, got {}",
            data[0]
        );
        assert!(
            (data[1] - 1.0).abs() < 1e-4,
            "expected approx 1, got {}",
            data[1]
        );
    }

    #[test]
    fn dynamic_tanh_is_bounded() {
        let device = device();
        let norm = DragonNorm::<Backend>::new(
            &DragonNormConfig {
                kind: DragonNormKind::DynamicTanh,
                ..Default::default()
            },
            2,
            &device,
        );
        let x =
            Tensor::<Backend, 2>::from_data(TensorData::new(vec![-10.0, 10.0], [1, 2]), &device);
        let y = norm.forward(x);
        let data = y.into_data().to_vec::<f32>().expect("f32 data");
        assert!(
            data[0] > -1.01 && data[0] < -0.9,
            "expected bounded negative output, got {}",
            data[0]
        );
        assert!(
            data[1] < 1.01 && data[1] > 0.9,
            "expected bounded positive output, got {}",
            data[1]
        );
    }

    #[test]
    fn derf_is_bounded() {
        let device = device();
        let norm = DragonNorm::<Backend>::new(
            &DragonNormConfig {
                kind: DragonNormKind::Derf,
                ..Default::default()
            },
            2,
            &device,
        );
        let x =
            Tensor::<Backend, 2>::from_data(TensorData::new(vec![-10.0, 10.0], [1, 2]), &device);
        let y = norm.forward(x);
        let data = y.into_data().to_vec::<f32>().expect("f32 data");
        assert!(
            data[0] > -1.01 && data[0] < -0.9,
            "expected bounded negative output, got {}",
            data[0]
        );
        assert!(
            data[1] < 1.01 && data[1] > 0.9,
            "expected bounded positive output, got {}",
            data[1]
        );
    }

    #[test]
    fn dyt_and_derf_use_scalar_alpha_and_shift_parameters() {
        let device = device();
        let dyt = DragonNorm::<Backend>::new(
            &DragonNormConfig {
                kind: DragonNormKind::DynamicTanh,
                ..Default::default()
            },
            8,
            &device,
        );
        let derf = DragonNorm::<Backend>::new(
            &DragonNormConfig {
                kind: DragonNormKind::Derf,
                ..Default::default()
            },
            8,
            &device,
        );

        assert_eq!(dyt.alpha.val().shape().dims::<1>(), [1]);
        assert_eq!(dyt.shift.val().shape().dims::<1>(), [1]);
        assert_eq!(derf.alpha.val().shape().dims::<1>(), [1]);
        assert_eq!(derf.shift.val().shape().dims::<1>(), [1]);
    }

    #[test]
    fn norm_kind_specific_defaults_resolve_expected_alpha() {
        let dyt = DragonNormConfig {
            kind: DragonNormKind::DynamicTanh,
            ..Default::default()
        };
        let derf = DragonNormConfig {
            kind: DragonNormKind::Derf,
            ..Default::default()
        };

        assert!((dyt.resolved_alpha_init() - 0.5).abs() < 1e-6);
        assert!((derf.resolved_alpha_init() - 0.886_226_95).abs() < 1e-6);
    }

    fn max_abs_diff<const D: usize>(left: Tensor<Backend, D>, right: Tensor<Backend, D>) -> f32 {
        (left - right)
            .abs()
            .max()
            .into_data()
            .to_vec::<f32>()
            .expect("f32 difference")[0]
    }

    #[test]
    fn plain_vjp_matches_autodiff_for_every_norm_kind() {
        let device = burn::tensor::Device::<AutodiffBackend>::default();
        for kind in [
            DragonNormKind::LayerNorm,
            DragonNormKind::RmsNorm,
            DragonNormKind::DynamicTanh,
            DragonNormKind::Derf,
        ] {
            let norm = DragonNorm::<AutodiffBackend>::new(
                &DragonNormConfig {
                    kind,
                    ..Default::default()
                },
                3,
                &device,
            );
            let ids = norm.parameter_ids();
            let input = Tensor::<AutodiffBackend, 2>::from_data(
                TensorData::new(vec![-0.8, 0.3, 1.2, 0.5, -1.1, 0.7], [2, 3]),
                &device,
            )
            .require_grad();
            let grad_output = Tensor::<AutodiffBackend, 2>::from_data(
                TensorData::new(vec![0.2, -0.4, 0.7, -0.3, 0.6, 0.1], [2, 3]),
                &device,
            );
            let mut raw_grads = (norm.forward(input.clone()) * grad_output.clone())
                .sum()
                .backward();
            let grad_input = input
                .grad_remove(&mut raw_grads)
                .expect("normalization input gradient");
            let parameter_grads = GradientsParams::from_grads(raw_grads, &norm);

            let plain = norm.valid();
            let vjp = plain.vjp_with_parameters(input.detach().inner(), grad_output.inner());
            let gamma_grad = parameter_grads
                .get::<Backend, 1>(ids.0)
                .expect("gamma gradient");
            let beta_grad = parameter_grads
                .get::<Backend, 1>(ids.1)
                .expect("beta gradient");

            assert!(
                max_abs_diff(grad_input, vjp.grad_input) < 2.0e-4,
                "{kind:?} input VJP mismatch"
            );
            assert!(
                max_abs_diff(gamma_grad, vjp.grad_gamma) < 2.0e-4,
                "{kind:?} gamma VJP mismatch"
            );
            assert!(
                max_abs_diff(beta_grad, vjp.grad_beta) < 2.0e-4,
                "{kind:?} beta VJP mismatch"
            );

            match kind {
                DragonNormKind::LayerNorm | DragonNormKind::RmsNorm => {
                    assert_eq!(vjp.grad_alpha.abs().sum().into_scalar(), 0.0);
                    assert_eq!(vjp.grad_shift.abs().sum().into_scalar(), 0.0);
                }
                DragonNormKind::DynamicTanh => {
                    let alpha_grad = parameter_grads
                        .get::<Backend, 1>(ids.2)
                        .expect("alpha gradient");
                    assert!(max_abs_diff(alpha_grad, vjp.grad_alpha) < 2.0e-4);
                    assert_eq!(vjp.grad_shift.abs().sum().into_scalar(), 0.0);
                }
                DragonNormKind::Derf => {
                    let alpha_grad = parameter_grads
                        .get::<Backend, 1>(ids.2)
                        .expect("alpha gradient");
                    let shift_grad = parameter_grads
                        .get::<Backend, 1>(ids.3)
                        .expect("shift gradient");
                    assert!(max_abs_diff(alpha_grad, vjp.grad_alpha) < 2.0e-4);
                    assert!(max_abs_diff(shift_grad, vjp.grad_shift) < 2.0e-4);
                }
            }
        }
    }
}
