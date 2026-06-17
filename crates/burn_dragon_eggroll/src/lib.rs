#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow};
use burn::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_eggroll::{
    AntitheticSign, BackendTensorOptimizerState, EggrollConfig, EggrollMetrics,
    MatrixNoiseCoefficient, MatrixNoiseMode, MatrixNoiseSpec, accumulate_gaussian_gradient_tensor,
    accumulate_low_rank_gradient_matrix_stack_from_coefficients_with_mode, eggroll_metrics,
    normalize_fitness, perturb_gaussian_tensor, perturb_matrix_stack_with_mode, tensor_update,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct MatrixParamInfo {
    pub path: String,
    pub param_id: u64,
    pub rank: usize,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct MatrixParamCatalog {
    pub params: Vec<MatrixParamInfo>,
}

impl MatrixParamCatalog {
    pub fn len(&self) -> usize {
        self.params.len()
    }

    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    pub fn parameter_elements(&self) -> usize {
        self.params
            .iter()
            .map(|param| param.shape.iter().product::<usize>())
            .sum()
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct AntitheticFitness {
    pub pair_index: u64,
    pub plus: f32,
    pub minus: f32,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct PairGradientCoefficient {
    pub pair_index: u64,
    pub coefficient: f32,
}

#[derive(Debug, Clone, Default)]
pub struct EggrollModuleOptimizerState<B: Backend> {
    params: BTreeMap<u64, BackendTensorOptimizerState<B, 1>>,
}

impl<B: Backend> EggrollModuleOptimizerState<B> {
    pub fn new() -> Self {
        Self {
            params: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.params.len()
    }

    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

pub fn collect_matrix_params<B, M>(module: &M) -> MatrixParamCatalog
where
    B: Backend,
    M: Module<B>,
{
    let mut visitor = MatrixParamCatalogVisitor::default();
    module.visit(&mut visitor);
    MatrixParamCatalog {
        params: visitor.params,
    }
}

pub fn perturb_module<B, M>(
    module: M,
    config: &EggrollConfig,
    generation: u64,
    pair_index: u64,
    sign: AntitheticSign,
) -> M
where
    B: Backend,
    M: Module<B>,
{
    perturb_module_with_allowed_param_ids(module, config, generation, pair_index, sign, None)
}

pub fn perturb_module_with_allowed_param_ids<B, M>(
    module: M,
    config: &EggrollConfig,
    generation: u64,
    pair_index: u64,
    sign: AntitheticSign,
    allowed_param_ids: Option<&BTreeSet<u64>>,
) -> M
where
    B: Backend,
    M: Module<B>,
{
    let mut mapper = EggrollPerturbMapper {
        seed: config.population.seed,
        generation,
        pair_index,
        rank: config.population.rank,
        matrix_noise: config.population.matrix_noise,
        sigma: config.effective_sigma(generation),
        sign,
        allowed_param_ids,
        perturbed_params: 0,
    };
    module.map(&mut mapper)
}

pub fn pair_gradient_coefficients(
    config: &EggrollConfig,
    generation: u64,
    fitness: &[AntitheticFitness],
) -> Result<Vec<PairGradientCoefficient>> {
    config.validate()?;
    if fitness.is_empty() {
        return Err(anyhow!("eggroll fitness population must not be empty"));
    }
    let mut population = Vec::with_capacity(fitness.len() * 2);
    for item in fitness {
        if !item.plus.is_finite() || !item.minus.is_finite() {
            return Err(anyhow!(
                "eggroll fitness values must be finite for pair {}",
                item.pair_index
            ));
        }
        population.push(item.plus);
        population.push(item.minus);
    }
    let normalized = normalize_fitness(&population, config.fitness_normalization);
    let sigma = config.effective_sigma(generation);
    Ok(fitness
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            let plus = normalized[idx * 2];
            let minus = normalized[idx * 2 + 1];
            let mut coefficient =
                burn_eggroll::antithetic_fitness_gradient_weight(plus, minus, sigma, fitness.len());
            if let Some(clip) = config.coefficient_clip {
                coefficient = coefficient.clamp(-clip, clip);
            }
            PairGradientCoefficient {
                pair_index: item.pair_index,
                coefficient,
            }
        })
        .collect())
}

pub fn apply_antithetic_update<B, M>(
    module: M,
    config: &EggrollConfig,
    generation: u64,
    fitness: &[AntitheticFitness],
    state: &mut EggrollModuleOptimizerState<B>,
) -> Result<(M, EggrollMetrics)>
where
    B: Backend,
    M: Module<B>,
{
    apply_antithetic_update_with_allowed_param_ids(module, config, generation, fitness, state, None)
}

pub fn apply_antithetic_update_with_allowed_param_ids<B, M>(
    module: M,
    config: &EggrollConfig,
    generation: u64,
    fitness: &[AntitheticFitness],
    state: &mut EggrollModuleOptimizerState<B>,
    allowed_param_ids: Option<&BTreeSet<u64>>,
) -> Result<(M, EggrollMetrics)>
where
    B: Backend,
    M: Module<B>,
{
    let coefficients = pair_gradient_coefficients(config, generation, fitness)?;
    let mut population = Vec::with_capacity(fitness.len() * 2);
    for item in fitness {
        population.push(item.plus);
        population.push(item.minus);
    }
    let metrics = eggroll_metrics(
        generation,
        population.len(),
        config.population.rank,
        config.effective_sigma(generation),
        &population,
        &coefficients
            .iter()
            .map(|coefficient| coefficient.coefficient)
            .collect::<Vec<_>>(),
        config.coefficient_clip,
    );
    let mut mapper = EggrollUpdateMapper {
        seed: config.population.seed,
        generation,
        rank: config.population.rank,
        matrix_noise: config.population.matrix_noise,
        coefficients: &coefficients,
        update: &config.update,
        state,
        allowed_param_ids,
        updated_params: 0,
    };
    let module = module.map(&mut mapper);
    Ok((module, metrics))
}

pub fn apply_antithetic_update_to_tensor_with_coefficients<B, const D: usize>(
    tensor: Tensor<B, D>,
    param_id: u64,
    config: &EggrollConfig,
    generation: u64,
    coefficients: &[PairGradientCoefficient],
    state: &mut EggrollModuleOptimizerState<B>,
) -> Tensor<B, D>
where
    B: Backend,
{
    let require_grad = tensor.is_require_grad();
    if D == 0 || coefficients.is_empty() {
        return tensor.detach().set_require_grad(require_grad);
    }
    let shape: [usize; D] = tensor.shape().dims();
    let device = tensor.device();
    let gradient = if D < 2 {
        coefficients
            .iter()
            .fold(None, |accumulated, coefficient| {
                let spec = MatrixNoiseSpec::new(
                    config.population.seed,
                    param_id,
                    generation,
                    coefficient.pair_index,
                    config.population.rank,
                );
                Some(accumulate_gaussian_gradient_tensor(
                    accumulated,
                    coefficient.coefficient,
                    shape,
                    spec,
                    &device,
                ))
            })
            .expect("non-empty coefficients should produce a gradient")
    } else {
        let coefficients = matrix_noise_coefficients(coefficients);
        accumulate_low_rank_gradient_matrix_stack_from_coefficients_with_mode(
            shape,
            config.population.seed,
            param_id,
            generation,
            config.population.rank,
            &coefficients,
            config.population.matrix_noise,
            &device,
        )
    };
    let elements = shape.iter().product::<usize>();
    tensor_update(
        tensor.reshape([elements]),
        gradient.reshape([elements]),
        state.params.entry(param_id).or_default(),
        &config.update,
    )
    .reshape(shape)
    .detach()
    .set_require_grad(require_grad)
}

fn matrix_noise_coefficients(
    coefficients: &[PairGradientCoefficient],
) -> Vec<MatrixNoiseCoefficient> {
    coefficients
        .iter()
        .map(|coefficient| {
            MatrixNoiseCoefficient::new(coefficient.pair_index, coefficient.coefficient)
        })
        .collect()
}

#[derive(Default)]
struct MatrixParamCatalogVisitor {
    params: Vec<MatrixParamInfo>,
}

impl<B: Backend> ModuleVisitor<B> for MatrixParamCatalogVisitor {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.push_param("", param.id, param.val().shape().dims::<D>());
    }

    fn visit_float_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        tensor: &Tensor<B, D>,
    ) {
        self.push_param(&path.join("."), id, tensor.shape().dims::<D>());
    }
}

impl MatrixParamCatalogVisitor {
    fn push_param<const D: usize>(&mut self, path: &str, id: ParamId, shape: [usize; D]) {
        if D < 2 {
            return;
        }
        if self.params.iter().any(|param| param.param_id == id.val()) {
            return;
        }
        self.params.push(MatrixParamInfo {
            path: path.to_string(),
            param_id: id.val(),
            rank: D,
            shape: shape.to_vec(),
        });
    }
}

struct EggrollPerturbMapper<'a> {
    seed: u64,
    generation: u64,
    pair_index: u64,
    rank: usize,
    matrix_noise: MatrixNoiseMode,
    sigma: f32,
    sign: AntitheticSign,
    allowed_param_ids: Option<&'a BTreeSet<u64>>,
    perturbed_params: usize,
}

impl<B: Backend> ModuleMapper<B> for EggrollPerturbMapper<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let require_grad = tensor.is_require_grad();
        if D == 0
            || self
                .allowed_param_ids
                .is_some_and(|allowed| !allowed.contains(&id.val()))
        {
            return Param::from_mapped_value(id, tensor, mapper);
        }
        let spec = MatrixNoiseSpec::new(
            self.seed,
            id.val(),
            self.generation,
            self.pair_index,
            self.rank,
        );
        let tensor = if D < 2 {
            perturb_gaussian_tensor(tensor, self.sigma, self.sign, spec)
        } else {
            perturb_matrix_stack_with_mode(tensor, self.sigma, self.sign, spec, self.matrix_noise)
        }
        .detach()
        .set_require_grad(require_grad);
        self.perturbed_params = self.perturbed_params.saturating_add(1);
        Param::from_mapped_value(id, tensor, mapper)
    }
}

struct EggrollUpdateMapper<'a, B: Backend> {
    seed: u64,
    generation: u64,
    rank: usize,
    matrix_noise: MatrixNoiseMode,
    coefficients: &'a [PairGradientCoefficient],
    update: &'a burn_eggroll::EggrollUpdateConfig,
    state: &'a mut EggrollModuleOptimizerState<B>,
    allowed_param_ids: Option<&'a BTreeSet<u64>>,
    updated_params: usize,
}

impl<B: Backend> ModuleMapper<B> for EggrollUpdateMapper<'_, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let require_grad = tensor.is_require_grad();
        if D == 0
            || self.coefficients.is_empty()
            || self
                .allowed_param_ids
                .is_some_and(|allowed| !allowed.contains(&id.val()))
        {
            return Param::from_mapped_value(id, tensor, mapper);
        }
        let shape: [usize; D] = tensor.shape().dims();
        let device = tensor.device();
        let gradient = if D < 2 {
            self.coefficients
                .iter()
                .fold(None, |accumulated, coefficient| {
                    let spec = MatrixNoiseSpec::new(
                        self.seed,
                        id.val(),
                        self.generation,
                        coefficient.pair_index,
                        self.rank,
                    );
                    Some(accumulate_gaussian_gradient_tensor(
                        accumulated,
                        coefficient.coefficient,
                        shape,
                        spec,
                        &device,
                    ))
                })
                .expect("non-empty eggroll coefficients should produce a gradient")
        } else {
            let coefficients = matrix_noise_coefficients(self.coefficients);
            accumulate_low_rank_gradient_matrix_stack_from_coefficients_with_mode(
                shape,
                self.seed,
                id.val(),
                self.generation,
                self.rank,
                &coefficients,
                self.matrix_noise,
                &device,
            )
        };
        let elements = shape.iter().product::<usize>();
        let param_state = self.state.params.entry(id.val()).or_default();
        let tensor = tensor_update(
            tensor.reshape([elements]),
            gradient.reshape([elements]),
            param_state,
            self.update,
        )
        .reshape(shape)
        .detach()
        .set_require_grad(require_grad);
        self.updated_params = self.updated_params.saturating_add(1);
        Param::from_mapped_value(id, tensor, mapper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Param;
    use burn::tensor::{Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[derive(Module, Debug)]
    struct ToyModule<B: Backend> {
        weight: Param<Tensor<B, 2>>,
        bias: Param<Tensor<B, 1>>,
        stack: Param<Tensor<B, 3>>,
    }

    fn device() -> burn::tensor::Device<TestBackend> {
        Default::default()
    }

    fn toy_module() -> ToyModule<TestBackend> {
        let device = device();
        ToyModule {
            weight: Param::from_data(TensorData::new(vec![0.0; 12], [3, 4]), &device),
            bias: Param::from_data(TensorData::new(vec![1.0; 4], [4]), &device),
            stack: Param::from_data(TensorData::new(vec![0.0; 24], [2, 3, 4]), &device),
        }
    }

    fn tensor_values<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
        tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("tensor values")
    }

    fn max_abs_diff(lhs: Vec<f32>, rhs: Vec<f32>) -> f32 {
        assert_eq!(lhs.len(), rhs.len(), "tensor length mismatch");
        lhs.into_iter()
            .zip(rhs)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max)
    }

    fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert_eq!(lhs.len(), rhs.len(), "tensor length mismatch");
        lhs.iter()
            .zip(rhs.iter())
            .map(|(left, right)| left * right)
            .sum()
    }

    #[test]
    fn catalog_collects_matrix_stack_parameters_only() {
        let module = toy_module();
        let catalog = collect_matrix_params::<TestBackend, _>(&module);
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog.parameter_elements(), 36);
        assert!(catalog.params.iter().any(|param| param.shape == [3, 4]));
        assert!(catalog.params.iter().any(|param| param.shape == [2, 3, 4]));
    }

    #[test]
    fn perturbation_is_antithetic_around_base_module() {
        let config = EggrollConfig {
            sigma: 0.01,
            ..EggrollConfig::default()
        };
        let base = toy_module();
        let plus =
            perturb_module::<TestBackend, _>(base.clone(), &config, 3, 7, AntitheticSign::Plus);
        let minus =
            perturb_module::<TestBackend, _>(base.clone(), &config, 3, 7, AntitheticSign::Minus);
        let base_values = tensor_values(base.weight.val());
        let plus_values = tensor_values(plus.weight.val());
        let minus_values = tensor_values(minus.weight.val());
        for ((base, plus), minus) in base_values
            .iter()
            .zip(plus_values.iter())
            .zip(minus_values.iter())
        {
            assert!((*plus - *base).abs() > 0.0);
            assert!((*plus + *minus - 2.0 * *base).abs() < 1.0e-5);
        }
    }

    #[test]
    fn perturbation_respects_matrix_noise_mode() {
        let raw_config = EggrollConfig {
            sigma: 0.01,
            population: burn_eggroll::PopulationConfig {
                rank: 2,
                matrix_noise: MatrixNoiseMode::Raw,
                ..burn_eggroll::PopulationConfig::default()
            },
            ..EggrollConfig::default()
        };
        let spectral_config = EggrollConfig {
            population: burn_eggroll::PopulationConfig {
                matrix_noise: MatrixNoiseMode::OrthogonalSpectral,
                ..raw_config.population.clone()
            },
            ..raw_config.clone()
        };
        let base = toy_module();
        let raw =
            perturb_module::<TestBackend, _>(base.clone(), &raw_config, 3, 7, AntitheticSign::Plus);
        let spectral =
            perturb_module::<TestBackend, _>(base, &spectral_config, 3, 7, AntitheticSign::Plus);
        assert_ne!(
            tensor_values(raw.weight.val()),
            tensor_values(spectral.weight.val())
        );
        assert_ne!(
            tensor_values(raw.stack.val()),
            tensor_values(spectral.stack.val())
        );
    }

    #[test]
    fn scoped_perturbation_only_changes_allowed_parameters() {
        let config = EggrollConfig {
            sigma: 0.01,
            ..EggrollConfig::default()
        };
        let base = toy_module();
        let mut allowed = BTreeSet::new();
        allowed.insert(base.weight.id.val());
        let updated = perturb_module_with_allowed_param_ids::<TestBackend, _>(
            base.clone(),
            &config,
            3,
            7,
            AntitheticSign::Plus,
            Some(&allowed),
        );

        assert_ne!(
            tensor_values(base.weight.val()),
            tensor_values(updated.weight.val())
        );
        assert_eq!(
            tensor_values(base.bias.val()),
            tensor_values(updated.bias.val())
        );
        assert_eq!(
            tensor_values(base.stack.val()),
            tensor_values(updated.stack.val())
        );
    }

    #[test]
    fn update_changes_float_params() {
        let config = EggrollConfig {
            sigma: 0.01,
            ..EggrollConfig::default()
        };
        let base = toy_module();
        let base_weight = tensor_values(base.weight.val());
        let base_bias = tensor_values(base.bias.val());
        let mut state = EggrollModuleOptimizerState::new();
        let fitness = [AntitheticFitness {
            pair_index: 0,
            plus: 1.0,
            minus: 0.0,
        }];
        let (updated, _metrics) =
            apply_antithetic_update(base, &config, 0, &fitness, &mut state).expect("update");
        let updated_weight = tensor_values(updated.weight.val());
        let updated_bias = tensor_values(updated.bias.val());
        assert_ne!(updated_weight, base_weight);
        assert_ne!(updated_bias, base_bias);
        assert_eq!(state.len(), 3);
    }

    #[test]
    fn fitness_update_moves_toward_higher_fitness_perturbation() {
        let config = EggrollConfig {
            sigma: 0.01,
            population: burn_eggroll::PopulationConfig {
                rank: 2,
                ..burn_eggroll::PopulationConfig::default()
            },
            ..EggrollConfig::default()
        };
        let base = toy_module();
        let plus =
            perturb_module::<TestBackend, _>(base.clone(), &config, 0, 0, AntitheticSign::Plus);
        let base_weight = tensor_values(base.weight.val());
        let plus_delta = tensor_values(plus.weight.val())
            .into_iter()
            .zip(base_weight.iter())
            .map(|(plus, base)| plus - base)
            .collect::<Vec<_>>();
        let fitness = [AntitheticFitness {
            pair_index: 0,
            plus: 1.0,
            minus: 0.0,
        }];
        let mut state = EggrollModuleOptimizerState::new();
        let (updated, _metrics) =
            apply_antithetic_update(base, &config, 0, &fitness, &mut state).expect("update");
        let update_delta = tensor_values(updated.weight.val())
            .into_iter()
            .zip(base_weight.iter())
            .map(|(updated, base)| updated - base)
            .collect::<Vec<_>>();

        assert!(
            dot(&update_delta, &plus_delta) > 0.0,
            "higher plus fitness should move parameters toward the plus perturbation"
        );
    }

    #[test]
    fn scoped_update_only_changes_allowed_parameters() {
        let config = EggrollConfig {
            sigma: 0.01,
            ..EggrollConfig::default()
        };
        let base = toy_module();
        let base_weight = tensor_values(base.weight.val());
        let base_bias = tensor_values(base.bias.val());
        let base_stack = tensor_values(base.stack.val());
        let mut allowed = BTreeSet::new();
        allowed.insert(base.stack.id.val());
        let mut state = EggrollModuleOptimizerState::new();
        let fitness = [AntitheticFitness {
            pair_index: 0,
            plus: 0.0,
            minus: 1.0,
        }];
        let (updated, _metrics) = apply_antithetic_update_with_allowed_param_ids(
            base,
            &config,
            0,
            &fitness,
            &mut state,
            Some(&allowed),
        )
        .expect("update");
        assert_eq!(tensor_values(updated.weight.val()), base_weight);
        assert_eq!(tensor_values(updated.bias.val()), base_bias);
        assert_ne!(tensor_values(updated.stack.val()), base_stack);
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn tensor_update_helper_matches_scoped_module_update() {
        let config = EggrollConfig {
            sigma: 0.01,
            ..EggrollConfig::default()
        };
        let base = toy_module();
        let fitness = [AntitheticFitness {
            pair_index: 0,
            plus: 0.0,
            minus: 1.0,
        }];
        let mut allowed = BTreeSet::new();
        allowed.insert(base.stack.id.val());
        let mut module_state = EggrollModuleOptimizerState::new();
        let (updated, _metrics) = apply_antithetic_update_with_allowed_param_ids(
            base.clone(),
            &config,
            0,
            &fitness,
            &mut module_state,
            Some(&allowed),
        )
        .expect("module update");

        let coefficients = pair_gradient_coefficients(&config, 0, &fitness).expect("coefficients");
        let mut tensor_state = EggrollModuleOptimizerState::new();
        let tensor = apply_antithetic_update_to_tensor_with_coefficients(
            base.stack.val(),
            base.stack.id.val(),
            &config,
            0,
            &coefficients,
            &mut tensor_state,
        );

        let diff = max_abs_diff(tensor_values(updated.stack.val()), tensor_values(tensor));
        assert!(diff <= 1.0e-6, "tensor update drifted by {diff}");
    }

    #[test]
    fn update_uses_effective_sigma_for_generation() {
        let config = EggrollConfig {
            sigma: 0.01,
            coefficient_clip: None,
            sigma_decay: 0.5,
            ..EggrollConfig::default()
        };
        let fitness = [AntitheticFitness {
            pair_index: 0,
            plus: 1.0,
            minus: 0.0,
        }];
        let gen0 = pair_gradient_coefficients(&config, 0, &fitness).expect("gen0 coefficients");
        let gen2 = pair_gradient_coefficients(&config, 2, &fitness).expect("gen2 coefficients");
        assert!((gen2[0].coefficient / gen0[0].coefficient - 4.0).abs() < 1.0e-5);

        let base = toy_module();
        let mut state = EggrollModuleOptimizerState::new();
        let (_updated, metrics) =
            apply_antithetic_update(base, &config, 2, &fitness, &mut state).expect("update");
        assert!((metrics.sigma - 0.0025).abs() < 1.0e-8);
    }

    #[test]
    fn coefficients_are_clipped_when_configured() {
        let config = EggrollConfig {
            sigma: 0.001,
            coefficient_clip: Some(3.0),
            fitness_normalization: burn_eggroll::FitnessNormalization::Center,
            ..EggrollConfig::default()
        };
        let fitness = [AntitheticFitness {
            pair_index: 0,
            plus: 1000.0,
            minus: -1000.0,
        }];
        let coefficients = pair_gradient_coefficients(&config, 0, &fitness).expect("coefficients");
        assert_eq!(coefficients.len(), 1);
        assert_eq!(coefficients[0].coefficient, -3.0);
    }
}
