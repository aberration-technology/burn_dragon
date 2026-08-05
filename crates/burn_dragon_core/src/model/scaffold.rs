use burn::module::{Module, Param, ParamId};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_eggroll::{
    ScaffoldAdapterManifest, ScaffoldArtifactSizeReport, ScaffoldTensorSpec, adapter_a_tensor,
    scaffold_tensor,
};
use serde::{Deserialize, Serialize};

use super::{DragonConfig, DragonRandomScaffoldConfig, HierarchicalDragonSharing};
use crate::model::dragon::SharedLowrankWeights;
use crate::model::init::{DragonInitializer, DragonProjectionRole};

const ENCODER_PATH: &str = "shared.encoder";
const ENCODER_V_PATH: &str = "shared.encoder_v";
const DECODER_PATH: &str = "shared.decoder";
const SLOW_ENCODER_PATH: &str = "shared.slow_encoder";
const SLOW_ENCODER_V_PATH: &str = "shared.slow_encoder_v";
const SLOW_DECODER_PATH: &str = "shared.slow_decoder";

#[derive(Module, Debug)]
pub(crate) struct HeadwiseScaffoldAdapter<B: Backend> {
    pub(super) a: Param<Tensor<B, 3>>,
    pub(super) b: Param<Tensor<B, 3>>,
    pub(super) gain: Param<Tensor<B, 1>>,
    scale: f32,
}

impl<B: Backend> HeadwiseScaffoldAdapter<B> {
    fn new(
        path: &str,
        heads: usize,
        input: usize,
        output: usize,
        config: &DragonRandomScaffoldConfig,
        device: &B::Device,
    ) -> Self {
        let a_path = format!("{path}.adapter_a");
        let a = adapter_a_tensor::<B, 3>(
            config.seed,
            &a_path,
            [heads, input, config.rank],
            input,
            device,
        )
        .unwrap_or_else(|error| panic!("failed to initialize {a_path}: {error}"));
        let b = Tensor::<B, 3>::zeros([heads, config.rank, output], device);
        let gain = Tensor::<B, 1>::ones([1], device).set_require_grad(config.trainable_gain);
        Self {
            a: Param::from_tensor(a),
            b: Param::from_tensor(b),
            gain: Param::from_tensor(gain),
            scale: config.adapter_spec().scale(),
        }
    }

    fn effective(&self, scaffold: Tensor<B, 3>) -> Tensor<B, 3> {
        let base = scaffold.detach() * self.gain.val().reshape([1, 1, 1]);
        base + self.a.val().matmul(self.b.val()).mul_scalar(self.scale)
    }

    fn trainable_ids(&self, include_gain: bool) -> Vec<ParamId> {
        let mut ids = vec![self.a.id, self.b.id];
        if include_gain {
            ids.push(self.gain.id);
        }
        ids
    }

    fn gain_id(&self) -> ParamId {
        self.gain.id
    }

    fn trainable_elements(&self, include_gain: bool) -> usize {
        self.a.val().shape().num_elements()
            + self.b.val().shape().num_elements()
            + usize::from(include_gain)
    }
}

#[derive(Module, Debug)]
pub(crate) struct MatrixScaffoldAdapter<B: Backend> {
    pub(super) a: Param<Tensor<B, 2>>,
    pub(super) b: Param<Tensor<B, 2>>,
    pub(super) gain: Param<Tensor<B, 1>>,
    scale: f32,
}

impl<B: Backend> MatrixScaffoldAdapter<B> {
    fn new(
        path: &str,
        input: usize,
        output: usize,
        config: &DragonRandomScaffoldConfig,
        device: &B::Device,
    ) -> Self {
        let a_path = format!("{path}.adapter_a");
        let a = adapter_a_tensor::<B, 2>(config.seed, &a_path, [input, config.rank], input, device)
            .unwrap_or_else(|error| panic!("failed to initialize {a_path}: {error}"));
        let b = Tensor::<B, 2>::zeros([config.rank, output], device);
        let gain = Tensor::<B, 1>::ones([1], device).set_require_grad(config.trainable_gain);
        Self {
            a: Param::from_tensor(a),
            b: Param::from_tensor(b),
            gain: Param::from_tensor(gain),
            scale: config.adapter_spec().scale(),
        }
    }

    fn effective(&self, scaffold: Tensor<B, 2>) -> Tensor<B, 2> {
        let base = scaffold.detach() * self.gain.val().reshape([1, 1]);
        base + self.a.val().matmul(self.b.val()).mul_scalar(self.scale)
    }

    fn trainable_ids(&self, include_gain: bool) -> Vec<ParamId> {
        let mut ids = vec![self.a.id, self.b.id];
        if include_gain {
            ids.push(self.gain.id);
        }
        ids
    }

    fn gain_id(&self) -> ParamId {
        self.gain.id
    }

    fn trainable_elements(&self, include_gain: bool) -> usize {
        self.a.val().shape().num_elements()
            + self.b.val().shape().num_elements()
            + usize::from(include_gain)
    }
}

#[derive(Module, Debug)]
pub(crate) struct DragonScaffoldAdapterSet<B: Backend> {
    pub(super) encoder: HeadwiseScaffoldAdapter<B>,
    pub(super) encoder_v: HeadwiseScaffoldAdapter<B>,
    pub(super) decoder: MatrixScaffoldAdapter<B>,
}

impl<B: Backend> DragonScaffoldAdapterSet<B> {
    fn new(
        prefix: ScaffoldPrefix,
        config: &DragonRandomScaffoldConfig,
        heads: usize,
        embd: usize,
        latent_per_head: usize,
        device: &B::Device,
    ) -> Self {
        let latent_total = heads * latent_per_head;
        Self {
            encoder: HeadwiseScaffoldAdapter::new(
                prefix.encoder(),
                heads,
                embd,
                latent_per_head,
                config,
                device,
            ),
            encoder_v: HeadwiseScaffoldAdapter::new(
                prefix.encoder_v(),
                heads,
                embd,
                latent_per_head,
                config,
                device,
            ),
            decoder: MatrixScaffoldAdapter::new(
                prefix.decoder(),
                latent_total,
                embd,
                config,
                device,
            ),
        }
    }

    fn effective(&self, scaffold: SharedLowrankWeights<B>) -> SharedLowrankWeights<B> {
        SharedLowrankWeights {
            encoder: self.encoder.effective(scaffold.encoder),
            encoder_v: self.encoder_v.effective(scaffold.encoder_v),
            decoder: self.decoder.effective(scaffold.decoder),
        }
    }

    fn trainable_ids(&self, include_gain: bool) -> Vec<ParamId> {
        self.encoder
            .trainable_ids(include_gain)
            .into_iter()
            .chain(self.encoder_v.trainable_ids(include_gain))
            .chain(self.decoder.trainable_ids(include_gain))
            .collect()
    }

    fn encoder_ids(&self, include_gain: bool) -> Vec<ParamId> {
        self.encoder
            .trainable_ids(include_gain)
            .into_iter()
            .chain(self.encoder_v.trainable_ids(include_gain))
            .collect()
    }

    fn decoder_ids(&self, include_gain: bool) -> Vec<ParamId> {
        self.decoder.trainable_ids(include_gain)
    }

    fn gain_ids(&self) -> Vec<ParamId> {
        vec![
            self.encoder.gain_id(),
            self.encoder_v.gain_id(),
            self.decoder.gain_id(),
        ]
    }

    fn trainable_elements(&self, include_gain: bool) -> usize {
        self.encoder.trainable_elements(include_gain)
            + self.encoder_v.trainable_elements(include_gain)
            + self.decoder.trainable_elements(include_gain)
    }
}

#[derive(Clone, Copy)]
enum ScaffoldPrefix {
    Fast,
    Slow,
}

impl ScaffoldPrefix {
    fn encoder(self) -> &'static str {
        match self {
            Self::Fast => ENCODER_PATH,
            Self::Slow => SLOW_ENCODER_PATH,
        }
    }

    fn encoder_v(self) -> &'static str {
        match self {
            Self::Fast => ENCODER_V_PATH,
            Self::Slow => SLOW_ENCODER_V_PATH,
        }
    }

    fn decoder(self) -> &'static str {
        match self {
            Self::Fast => DECODER_PATH,
            Self::Slow => SLOW_DECODER_PATH,
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct DragonRandomScaffoldAdapters<B: Backend> {
    pub(super) fast: DragonScaffoldAdapterSet<B>,
    slow: Option<DragonScaffoldAdapterSet<B>>,
}

impl<B: Backend> DragonRandomScaffoldAdapters<B> {
    pub(crate) fn new(
        model: &DragonConfig,
        config: &DragonRandomScaffoldConfig,
        device: &B::Device,
    ) -> Self {
        let slow = (model.hierarchical_dragon.enabled
            && model.hierarchical_dragon.weight_sharing == HierarchicalDragonSharing::Split)
            .then(|| {
                DragonScaffoldAdapterSet::new(
                    ScaffoldPrefix::Slow,
                    config,
                    model.n_head,
                    model.n_embd,
                    model.latent_per_head(),
                    device,
                )
            });
        Self {
            fast: DragonScaffoldAdapterSet::new(
                ScaffoldPrefix::Fast,
                config,
                model.n_head,
                model.n_embd,
                model.latent_per_head(),
                device,
            ),
            slow,
        }
    }

    pub(crate) fn effective_fast(
        &self,
        scaffold: SharedLowrankWeights<B>,
    ) -> SharedLowrankWeights<B> {
        self.fast.effective(scaffold)
    }

    pub(crate) fn effective_slow(
        &self,
        scaffold: SharedLowrankWeights<B>,
    ) -> SharedLowrankWeights<B> {
        self.slow
            .as_ref()
            .expect("split slow scaffold adapter missing")
            .effective(scaffold)
    }

    pub(crate) fn trainable_ids(&self, include_gain: bool) -> Vec<ParamId> {
        self.fast
            .trainable_ids(include_gain)
            .into_iter()
            .chain(
                self.slow
                    .as_ref()
                    .into_iter()
                    .flat_map(|slow| slow.trainable_ids(include_gain)),
            )
            .collect()
    }

    pub(crate) fn encoder_ids(&self, include_gain: bool) -> Vec<ParamId> {
        self.fast
            .encoder_ids(include_gain)
            .into_iter()
            .chain(
                self.slow
                    .as_ref()
                    .into_iter()
                    .flat_map(|slow| slow.encoder_ids(include_gain)),
            )
            .collect()
    }

    pub(crate) fn decoder_ids(&self, include_gain: bool) -> Vec<ParamId> {
        self.fast
            .decoder_ids(include_gain)
            .into_iter()
            .chain(
                self.slow
                    .as_ref()
                    .into_iter()
                    .flat_map(|slow| slow.decoder_ids(include_gain)),
            )
            .collect()
    }

    pub(crate) fn gain_ids(&self) -> Vec<ParamId> {
        self.fast
            .gain_ids()
            .into_iter()
            .chain(
                self.slow
                    .as_ref()
                    .into_iter()
                    .flat_map(DragonScaffoldAdapterSet::gain_ids),
            )
            .collect()
    }

    pub(crate) fn trainable_elements(&self, include_gain: bool) -> usize {
        self.fast.trainable_elements(include_gain)
            + self
                .slow
                .as_ref()
                .map(|slow| slow.trainable_elements(include_gain))
                .unwrap_or(0)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct DragonRandomScaffoldReport {
    pub manifest: ScaffoldAdapterManifest,
    pub full_projection_elements: usize,
    pub trainable_adapter_elements: usize,
    pub frozen_scaffold_elements: usize,
    pub fp32_size: ScaffoldArtifactSizeReport,
}

pub(crate) fn scaffold_tensor_specs(config: &DragonConfig) -> Vec<ScaffoldTensorSpec> {
    let latent = config.latent_per_head();
    let latent_total = config.latent_total();
    let residual_depth = config.n_layer.max(1) * config.rollout_fast_steps_per_slow_step.max(1);
    let initializer = DragonInitializer::new(&config.initialization);
    let encoder_std = initializer.projection_standard_deviation(
        DragonProjectionRole::Encoder,
        config.n_embd,
        latent,
        residual_depth,
    );
    let encoder_v_std = initializer.projection_standard_deviation(
        DragonProjectionRole::EncoderValue,
        config.n_embd,
        latent,
        residual_depth,
    );
    let decoder_std = initializer.projection_standard_deviation(
        DragonProjectionRole::Decoder,
        latent_total,
        config.n_embd,
        residual_depth,
    );
    let mut tensors = vec![
        ScaffoldTensorSpec::new(ENCODER_PATH, [config.n_head, config.n_embd, latent])
            .with_standard_deviation(encoder_std),
        ScaffoldTensorSpec::new(ENCODER_V_PATH, [config.n_head, config.n_embd, latent])
            .with_standard_deviation(encoder_v_std),
        ScaffoldTensorSpec::new(DECODER_PATH, [latent_total, config.n_embd])
            .with_standard_deviation(decoder_std),
    ];
    if config.hierarchical_dragon.enabled
        && config.hierarchical_dragon.weight_sharing == HierarchicalDragonSharing::Split
    {
        tensors.extend([
            ScaffoldTensorSpec::new(SLOW_ENCODER_PATH, [config.n_head, config.n_embd, latent])
                .with_standard_deviation(encoder_std),
            ScaffoldTensorSpec::new(SLOW_ENCODER_V_PATH, [config.n_head, config.n_embd, latent])
                .with_standard_deviation(encoder_v_std),
            ScaffoldTensorSpec::new(SLOW_DECODER_PATH, [latent_total, config.n_embd])
                .with_standard_deviation(decoder_std),
        ]);
    }
    tensors
}

pub fn build_dragon_random_scaffold_manifest(config: &DragonConfig) -> ScaffoldAdapterManifest {
    ScaffoldAdapterManifest::new(
        config.random_scaffold.scaffold_spec(),
        config.random_scaffold.adapter_spec(),
        scaffold_tensor_specs(config),
    )
}

pub(crate) fn initialize_scaffold_3d<B: Backend>(
    config: &DragonConfig,
    path: &str,
    shape: [usize; 3],
    device: &B::Device,
) -> Tensor<B, 3> {
    let tensor = scaffold_tensor_specs(config)
        .into_iter()
        .find(|tensor| tensor.path == path)
        .unwrap_or_else(|| panic!("missing scaffold tensor specification for {path}"));
    scaffold_tensor(
        &config.random_scaffold.scaffold_spec(),
        &tensor,
        shape,
        device,
    )
    .unwrap_or_else(|error| panic!("failed to initialize scaffold {path}: {error}"))
    .set_require_grad(false)
}

pub(crate) fn initialize_scaffold_2d<B: Backend>(
    config: &DragonConfig,
    path: &str,
    shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2> {
    let tensor = scaffold_tensor_specs(config)
        .into_iter()
        .find(|tensor| tensor.path == path)
        .unwrap_or_else(|| panic!("missing scaffold tensor specification for {path}"));
    scaffold_tensor(
        &config.random_scaffold.scaffold_spec(),
        &tensor,
        shape,
        device,
    )
    .unwrap_or_else(|error| panic!("failed to initialize scaffold {path}: {error}"))
    .set_require_grad(false)
}

pub(crate) fn fast_scaffold_paths() -> (&'static str, &'static str, &'static str) {
    (ENCODER_PATH, ENCODER_V_PATH, DECODER_PATH)
}

pub(crate) fn slow_scaffold_paths() -> (&'static str, &'static str, &'static str) {
    (SLOW_ENCODER_PATH, SLOW_ENCODER_V_PATH, SLOW_DECODER_PATH)
}

pub(crate) fn build_report<B: Backend>(
    config: &DragonConfig,
    adapters: &DragonRandomScaffoldAdapters<B>,
) -> DragonRandomScaffoldReport {
    let manifest = build_dragon_random_scaffold_manifest(config);
    let fp32_size = manifest
        .size_report(std::mem::size_of::<f32>())
        .expect("validated Dragon random scaffold manifest");
    DragonRandomScaffoldReport {
        full_projection_elements: fp32_size.scaffold_elements,
        trainable_adapter_elements: adapters
            .trainable_elements(config.random_scaffold.trainable_gain),
        frozen_scaffold_elements: fp32_size.scaffold_elements,
        fp32_size,
        manifest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Module;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn config(seed: u64) -> DragonConfig {
        let mut config = DragonConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            ..DragonConfig::default()
        };
        config.random_scaffold.enabled = true;
        config.random_scaffold.seed = seed;
        config.random_scaffold.rank = 2;
        config
    }

    fn values<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
        tensor.to_data().into_vec::<f32>().expect("tensor values")
    }

    #[test]
    fn zero_b_adapter_preserves_scaffold_function() {
        let device = Default::default();
        let config = config(7);
        let adapters = DragonRandomScaffoldAdapters::<TestBackend>::new(
            &config,
            &config.random_scaffold,
            &device,
        );
        let [heads, embd, latent] = [config.n_head, config.n_embd, config.latent_per_head()];
        let (encoder_path, encoder_v_path, decoder_path) = fast_scaffold_paths();
        let scaffold = SharedLowrankWeights {
            encoder: initialize_scaffold_3d(&config, encoder_path, [heads, embd, latent], &device),
            encoder_v: initialize_scaffold_3d(
                &config,
                encoder_v_path,
                [heads, embd, latent],
                &device,
            ),
            decoder: initialize_scaffold_2d(&config, decoder_path, [heads * latent, embd], &device),
        };
        let expected = scaffold.clone();
        let effective = adapters.effective_fast(scaffold);
        assert_eq!(values(effective.encoder), values(expected.encoder));
        assert_eq!(values(effective.encoder_v), values(expected.encoder_v));
        assert_eq!(values(effective.decoder), values(expected.decoder));
    }

    #[test]
    fn adapter_manifest_and_parameter_count_match_module() {
        let device = Default::default();
        let config = config(11);
        let adapters = DragonRandomScaffoldAdapters::<TestBackend>::new(
            &config,
            &config.random_scaffold,
            &device,
        );
        let report = build_report(&config, &adapters);
        assert_eq!(
            report.trainable_adapter_elements,
            report.fp32_size.adapter_elements + report.fp32_size.gain_elements
        );
        assert!(report.fp32_size.byte_reduction_fraction() > 0.0);
        assert_eq!(
            adapters
                .trainable_ids(config.random_scaffold.trainable_gain)
                .len(),
            9
        );
    }

    #[test]
    fn manifest_uses_dragon_projection_scale_and_rank_stabilized_adapter_scale() {
        let device = Default::default();
        let mut config = config(11);
        config.n_layer = 4;
        config.random_scaffold.rank = 4;
        config.random_scaffold.scaling = burn_eggroll::LowRankScaling::RankStabilized;
        let adapters = DragonRandomScaffoldAdapters::<TestBackend>::new(
            &config,
            &config.random_scaffold,
            &device,
        );
        let report = build_report(&config, &adapters);
        assert!(
            report
                .manifest
                .tensors
                .iter()
                .all(|tensor| (tensor.standard_deviation - 0.01).abs() < 1.0e-7)
        );
        assert!((adapters.fast.encoder.scale - 8.0).abs() < f32::EPSILON);
    }

    #[test]
    fn scaffold_seed_changes_base_but_not_shapes() {
        let device = Default::default();
        let first = config(13);
        let second = config(17);
        let path = fast_scaffold_paths().0;
        let shape = [first.n_head, first.n_embd, first.latent_per_head()];
        let a = initialize_scaffold_3d(&first, path, shape, &device);
        let b = initialize_scaffold_3d(&second, path, shape, &device);
        assert_ne!(values(a), values(b));
    }

    #[test]
    fn scaffold_params_are_frozen_and_adapter_params_are_module_state() {
        let device = Default::default();
        let config = config(19);
        let scaffold = initialize_scaffold_3d::<TestBackend>(
            &config,
            fast_scaffold_paths().0,
            [config.n_head, config.n_embd, config.latent_per_head()],
            &device,
        );
        assert!(!scaffold.is_require_grad());

        let adapters = DragonRandomScaffoldAdapters::<TestBackend>::new(
            &config,
            &config.random_scaffold,
            &device,
        );
        let mut count = 0usize;
        struct Counter<'a>(&'a mut usize);
        impl burn::module::ModuleVisitor<TestBackend> for Counter<'_> {
            fn visit_float<const D: usize>(&mut self, _param: &Param<Tensor<TestBackend, D>>) {
                *self.0 += 1;
            }
        }
        adapters.visit(&mut Counter(&mut count));
        assert_eq!(count, 9);
    }
}
