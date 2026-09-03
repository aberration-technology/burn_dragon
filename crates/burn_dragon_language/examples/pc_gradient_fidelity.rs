#[cfg(feature = "train")]
use anyhow::{Context, Result, anyhow};
#[cfg(feature = "train")]
use burn::module::Module;
#[cfg(feature = "train")]
use burn::tensor::backend::{AutodiffBackend, Backend};
#[cfg(feature = "train")]
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(feature = "train")]
use burn_autodiff::Autodiff;
#[cfg(feature = "train")]
use burn_dragon_core::{
    DragonConfig, DragonInitializationKind, DragonModel, RotaryEmbedding, SequenceTrainingExecutor,
};
#[cfg(feature = "train")]
use burn_dragon_language::train::{
    LocalPredictiveCodingGradientFidelityReport, local_predictive_coding_gradient_fidelity,
};
#[cfg(feature = "train")]
use burn_dragon_language::{LocalPredictiveCodingConfig, LocalPredictiveCodingSolver};
#[cfg(feature = "train")]
use burn_ndarray::NdArray;
#[cfg(feature = "train")]
use serde::Serialize;

#[cfg(feature = "train")]
#[derive(Debug, Clone)]
struct Args {
    backend: String,
    solver: LocalPredictiveCodingSolver,
    parameterization: burn_pc::PcParameterizationKind,
    shared_reuse_reduction: burn_pc::PcSharedReuseReduction,
    initialization: DragonInitializationKind,
    seed: u64,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    latent_total: usize,
    vocab_size: usize,
    batch_size: usize,
    block_size: usize,
    inference_steps: Vec<usize>,
    step_sizes: Vec<f32>,
    dual_step_sizes: Vec<f32>,
    penalties: Vec<f32>,
    prediction_precisions: Vec<f32>,
    max_grad_norm: Option<f32>,
    mask_period: usize,
}

#[cfg(feature = "train")]
impl Default for Args {
    fn default() -> Self {
        Self {
            backend: "cpu".to_string(),
            solver: LocalPredictiveCodingSolver::SynchronousEquilibrium,
            parameterization: burn_pc::PcParameterizationKind::Standard,
            shared_reuse_reduction: burn_pc::PcSharedReuseReduction::RootMeanSquare,
            initialization: DragonInitializationKind::SimpleNormal,
            seed: 20260804,
            n_layer: 4,
            n_embd: 96,
            n_head: 4,
            latent_total: 3072,
            vocab_size: 272,
            batch_size: 32,
            block_size: 128,
            inference_steps: vec![1, 2, 4, 8],
            step_sizes: vec![0.01, 0.05, 0.1],
            dual_step_sizes: vec![0.1],
            penalties: vec![1.0],
            prediction_precisions: vec![1.0],
            max_grad_norm: Some(1.0),
            mask_period: 5,
        }
    }
}

#[cfg(feature = "train")]
#[derive(Debug, Serialize)]
struct FidelityArm {
    solver: LocalPredictiveCodingSolver,
    inference_steps: usize,
    step_size: Option<f32>,
    dual_step_size: Option<f32>,
    penalty: Option<f32>,
    prediction_precision: Option<f32>,
    max_grad_norm: Option<f32>,
    report: LocalPredictiveCodingGradientFidelityReport,
}

#[cfg(feature = "train")]
#[derive(Debug, Serialize)]
struct FidelityMatrix {
    schema_version: u32,
    backend: String,
    solver: LocalPredictiveCodingSolver,
    parameterization: burn_pc::PcParameterizationKind,
    shared_reuse_reduction: burn_pc::PcSharedReuseReduction,
    initialization: DragonInitializationKind,
    seed: u64,
    parameters: usize,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    latent_total: usize,
    vocab_size: usize,
    batch_size: usize,
    block_size: usize,
    mask_period: usize,
    arms: Vec<FidelityArm>,
}

#[cfg(feature = "train")]
struct DiagnosticBatch<B: Backend> {
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
}

#[cfg(feature = "train")]
fn parse_value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    args.next()
        .ok_or_else(|| anyhow!("{name} requires a value"))?
        .parse::<T>()
        .map_err(|error| anyhow!("invalid {name}: {error}"))
}

#[cfg(feature = "train")]
fn parse_csv<T: std::str::FromStr>(value: String, name: &str) -> Result<Vec<T>>
where
    T::Err: std::fmt::Display,
{
    let values = value
        .split(',')
        .map(|part| {
            part.parse::<T>()
                .map_err(|error| anyhow!("invalid {name} value {part:?}: {error}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        return Err(anyhow!("{name} must contain at least one value"));
    }
    Ok(values)
}

#[cfg(feature = "train")]
fn parse_args() -> Result<Args> {
    let mut parsed = Args::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => parsed.backend = parse_value(&mut args, "--backend")?,
            "--solver" => {
                parsed.solver = match parse_value::<String>(&mut args, "--solver")?.as_str() {
                    "synchronous_equilibrium" => {
                        LocalPredictiveCodingSolver::SynchronousEquilibrium
                    }
                    "reverse_gauss_seidel" => LocalPredictiveCodingSolver::ReverseGaussSeidel,
                    "augmented_lagrangian" => LocalPredictiveCodingSolver::AugmentedLagrangian,
                    "error_equilibrium" => LocalPredictiveCodingSolver::ErrorEquilibrium,
                    "fixed_prediction" => LocalPredictiveCodingSolver::FixedPrediction,
                    "layer_local_prediction" => LocalPredictiveCodingSolver::LayerLocalPrediction,
                    value => {
                        return Err(anyhow!(
                            "unsupported --solver {value}; expected synchronous_equilibrium, reverse_gauss_seidel, augmented_lagrangian, error_equilibrium, fixed_prediction, or layer_local_prediction"
                        ));
                    }
                }
            }
            "--parameterization" => {
                parsed.parameterization =
                    match parse_value::<String>(&mut args, "--parameterization")?.as_str() {
                        "standard" => burn_pc::PcParameterizationKind::Standard,
                        "mu_pc" => burn_pc::PcParameterizationKind::MuPc,
                        value => {
                            return Err(anyhow!(
                                "unsupported --parameterization {value}; expected standard or mu_pc"
                            ));
                        }
                    }
            }
            "--shared-reuse-reduction" => {
                parsed.shared_reuse_reduction = match parse_value::<String>(
                    &mut args,
                    "--shared-reuse-reduction",
                )?
                .as_str()
                {
                    "sum" => burn_pc::PcSharedReuseReduction::Sum,
                    "mean" => burn_pc::PcSharedReuseReduction::Mean,
                    "root_mean_square" | "rms" => burn_pc::PcSharedReuseReduction::RootMeanSquare,
                    value => {
                        return Err(anyhow!(
                            "unsupported --shared-reuse-reduction {value}; expected sum, mean, or root_mean_square"
                        ));
                    }
                }
            }
            "--initialization" => {
                parsed.initialization = match parse_value::<String>(&mut args, "--initialization")?
                    .as_str()
                {
                    "simple_normal" => DragonInitializationKind::SimpleNormal,
                    "near_critical" => DragonInitializationKind::NearCritical,
                    "he_glorot" => DragonInitializationKind::HeGlorot,
                    "headwise_semi_orthogonal" => DragonInitializationKind::HeadwiseSemiOrthogonal,
                    value => {
                        return Err(anyhow!(
                            "unsupported --initialization {value}; expected simple_normal, near_critical, he_glorot, or headwise_semi_orthogonal"
                        ));
                    }
                }
            }
            "--seed" => parsed.seed = parse_value(&mut args, "--seed")?,
            "--n-layer" => parsed.n_layer = parse_value(&mut args, "--n-layer")?,
            "--n-embd" => parsed.n_embd = parse_value(&mut args, "--n-embd")?,
            "--n-head" => parsed.n_head = parse_value(&mut args, "--n-head")?,
            "--latent-total" => parsed.latent_total = parse_value(&mut args, "--latent-total")?,
            "--vocab-size" => parsed.vocab_size = parse_value(&mut args, "--vocab-size")?,
            "--batch-size" => parsed.batch_size = parse_value(&mut args, "--batch-size")?,
            "--block-size" => parsed.block_size = parse_value(&mut args, "--block-size")?,
            "--inference-steps" => {
                parsed.inference_steps = parse_csv(
                    args.next()
                        .ok_or_else(|| anyhow!("--inference-steps requires a value"))?,
                    "--inference-steps",
                )?;
            }
            "--step-sizes" => {
                parsed.step_sizes = parse_csv(
                    args.next()
                        .ok_or_else(|| anyhow!("--step-sizes requires a value"))?,
                    "--step-sizes",
                )?;
            }
            "--dual-step-sizes" => {
                parsed.dual_step_sizes = parse_csv(
                    args.next()
                        .ok_or_else(|| anyhow!("--dual-step-sizes requires a value"))?,
                    "--dual-step-sizes",
                )?;
            }
            "--penalties" => {
                parsed.penalties = parse_csv(
                    args.next()
                        .ok_or_else(|| anyhow!("--penalties requires a value"))?,
                    "--penalties",
                )?;
            }
            "--prediction-precisions" => {
                parsed.prediction_precisions = parse_csv(
                    args.next()
                        .ok_or_else(|| anyhow!("--prediction-precisions requires a value"))?,
                    "--prediction-precisions",
                )?;
            }
            "--max-grad-norm" => {
                let value: String = parse_value(&mut args, "--max-grad-norm")?;
                parsed.max_grad_norm = if value == "none" {
                    None
                } else {
                    Some(
                        value
                            .parse::<f32>()
                            .map_err(|error| anyhow!("invalid --max-grad-norm: {error}"))?,
                    )
                };
            }
            "--mask-period" => parsed.mask_period = parse_value(&mut args, "--mask-period")?,
            "--help" | "-h" => {
                println!(
                    "usage: cargo run -p burn_dragon_language --release --example pc_gradient_fidelity --features train[,cuda] -- --backend <cpu|wgpu|cuda> [--solver <synchronous_equilibrium|reverse_gauss_seidel|augmented_lagrangian|error_equilibrium|fixed_prediction|layer_local_prediction>] [--parameterization <standard|mu_pc>] [--shared-reuse-reduction <sum|mean|root_mean_square>] [--initialization <simple_normal|near_critical|he_glorot|headwise_semi_orthogonal>] [--seed N] [--n-layer N] [--n-embd N] [--n-head N] [--latent-total N] [--vocab-size N] [--batch-size N] [--block-size N] [--inference-steps 1,2,4,8] [--step-sizes 0.01,0.05,0.1] [--dual-step-sizes 0.03,0.1] [--penalties 0.3,1,3] [--prediction-precisions 0.3,1,3] [--max-grad-norm <N|none>] [--mask-period N]"
                );
                std::process::exit(0);
            }
            other => return Err(anyhow!("unknown argument {other}")),
        }
    }

    if parsed.n_layer == 0
        || parsed.n_embd == 0
        || parsed.n_head == 0
        || parsed.latent_total == 0
        || parsed.vocab_size < 2
        || parsed.batch_size == 0
        || parsed.block_size == 0
    {
        return Err(anyhow!("all model and batch dimensions must be positive"));
    }
    if !parsed.latent_total.is_multiple_of(parsed.n_embd)
        || !parsed.latent_total.is_multiple_of(parsed.n_head)
    {
        return Err(anyhow!(
            "--latent-total must be divisible by both --n-embd and --n-head"
        ));
    }
    Ok(parsed)
}

#[cfg(feature = "train")]
#[derive(Debug, Clone, Copy, Default)]
struct FidelitySetting {
    inference_steps: Option<usize>,
    step_size: Option<f32>,
    prediction_precision: Option<f32>,
    dual_step_size: Option<f32>,
    penalty: Option<f32>,
}

#[cfg(feature = "train")]
fn deterministic_batch<B: Backend>(args: &Args, device: &B::Device) -> DiagnosticBatch<B> {
    let elements = args.batch_size * args.block_size;
    let mut state = args.seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut stream = Vec::with_capacity(elements + args.batch_size);
    for index in 0..(elements + args.batch_size) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let arithmetic = (index as u64)
            .wrapping_mul(17)
            .wrapping_add((index / args.block_size) as u64 * 31);
        stream.push(((state ^ arithmetic) % args.vocab_size as u64) as i64);
    }

    let mut inputs = Vec::with_capacity(elements);
    let mut targets = Vec::with_capacity(elements);
    for batch in 0..args.batch_size {
        let start = batch * (args.block_size + 1);
        inputs.extend_from_slice(&stream[start..start + args.block_size]);
        targets.extend_from_slice(&stream[start + 1..start + args.block_size + 1]);
    }
    let mask = (args.mask_period > 0).then(|| {
        let values = (0..elements)
            .map(|index| i64::from(!index.is_multiple_of(args.mask_period)))
            .collect::<Vec<_>>();
        Tensor::from_data(
            TensorData::new(values, [args.batch_size, args.block_size]),
            device,
        )
    });
    DiagnosticBatch {
        inputs: Tensor::from_data(
            TensorData::new(inputs, [args.batch_size, args.block_size]),
            device,
        ),
        targets: Tensor::from_data(
            TensorData::new(targets, [args.batch_size, args.block_size]),
            device,
        ),
        loss_mask: mask,
    }
}

#[cfg(feature = "train")]
fn run<B>(args: &Args) -> Result<FidelityMatrix>
where
    B: AutodiffBackend,
    B::Device: Default + 'static,
    B::FloatTensorPrimitive: 'static,
{
    let device = B::Device::default();
    B::seed(&device, args.seed);
    let mut model_config = DragonConfig {
        n_layer: args.n_layer,
        n_embd: args.n_embd,
        n_head: args.n_head,
        mlp_internal_dim_multiplier: args.latent_total / args.n_embd,
        vocab_size: args.vocab_size,
        dropout: 0.0,
        ..DragonConfig::default()
    };
    model_config.sequence_kernel.executor = SequenceTrainingExecutor::DenseScoreShortContext;
    model_config.fused_kernels.rotary_embedding = RotaryEmbedding::Alibi;
    model_config.initialization.kind = args.initialization;
    let model = DragonModel::<B>::new(model_config, &device);
    model
        .predictive_coding_support()
        .map_err(anyhow::Error::msg)?;
    let parameters = model.num_params();
    let batch = deterministic_batch::<B>(args, &device);
    if matches!(
        args.solver,
        LocalPredictiveCodingSolver::DirectKolenPollack
            | LocalPredictiveCodingSolver::AmortizedAdjoint
    ) {
        return Err(anyhow!(
            "feedback-bank solvers require the stateful LanguageTrainModel experiment runner"
        ));
    }

    let settings = match args.solver {
        LocalPredictiveCodingSolver::SynchronousEquilibrium
        | LocalPredictiveCodingSolver::ReverseGaussSeidel
        | LocalPredictiveCodingSolver::ErrorEquilibrium => args
            .inference_steps
            .iter()
            .flat_map(|&steps| {
                args.step_sizes.iter().flat_map(move |&step_size| {
                    args.prediction_precisions
                        .iter()
                        .map(move |&precision| FidelitySetting {
                            inference_steps: Some(steps),
                            step_size: Some(step_size),
                            prediction_precision: Some(precision),
                            ..FidelitySetting::default()
                        })
                })
            })
            .collect::<Vec<_>>(),
        LocalPredictiveCodingSolver::AugmentedLagrangian => args
            .inference_steps
            .iter()
            .flat_map(|&steps| {
                args.step_sizes.iter().flat_map(move |&step_size| {
                    args.dual_step_sizes
                        .iter()
                        .flat_map(move |&dual_step_size| {
                            args.penalties.iter().map(move |&penalty| FidelitySetting {
                                inference_steps: Some(steps),
                                step_size: Some(step_size),
                                prediction_precision: Some(1.0),
                                dual_step_size: Some(dual_step_size),
                                penalty: Some(penalty),
                            })
                        })
                })
            })
            .collect::<Vec<_>>(),
        LocalPredictiveCodingSolver::FixedPrediction
        | LocalPredictiveCodingSolver::LayerLocalPrediction
        | LocalPredictiveCodingSolver::DirectKolenPollack
        | LocalPredictiveCodingSolver::AmortizedAdjoint
        | LocalPredictiveCodingSolver::FirstOrderAdjoint => {
            vec![FidelitySetting::default()]
        }
    };
    let mut arms = Vec::with_capacity(settings.len());
    for setting in settings {
        let mut config = LocalPredictiveCodingConfig {
            solver: args.solver,
            parameterization: args.parameterization,
            shared_reuse_reduction: args.shared_reuse_reduction,
            ..LocalPredictiveCodingConfig::default()
        };
        if matches!(
            args.solver,
            LocalPredictiveCodingSolver::LayerLocalPrediction
        ) {
            config.factor_reduction = burn_dragon_language::PredictiveCodingFactorReduction::Mean;
        }
        if let (Some(steps), Some(step_size), Some(prediction_precision)) = (
            setting.inference_steps,
            setting.step_size,
            setting.prediction_precision,
        ) {
            if matches!(
                args.solver,
                LocalPredictiveCodingSolver::AugmentedLagrangian
            ) {
                config.augmented_lagrangian.steps = steps;
                config.augmented_lagrangian.primal_step_size = step_size;
                config.augmented_lagrangian.dual_step_size = setting
                    .dual_step_size
                    .expect("PC-ALM setting has a dual rate");
                config.augmented_lagrangian.penalty =
                    setting.penalty.expect("PC-ALM setting has a penalty");
            } else {
                config.inference.steps = steps;
                config.inference.step_size = step_size;
                config.prediction_precision = prediction_precision;
            }
        }
        let applied_max_grad_norm = setting.inference_steps.and(args.max_grad_norm);
        if matches!(
            args.solver,
            LocalPredictiveCodingSolver::AugmentedLagrangian
        ) {
            config.augmented_lagrangian.max_primal_grad_norm = applied_max_grad_norm;
        } else {
            config.inference.max_grad_norm = applied_max_grad_norm;
        }
        config.sync_diagnostics = !matches!(
            args.solver,
            LocalPredictiveCodingSolver::LayerLocalPrediction
                | LocalPredictiveCodingSolver::FirstOrderAdjoint
        );
        let report = local_predictive_coding_gradient_fidelity(
            &model,
            batch.inputs.clone(),
            batch.targets.clone(),
            batch.loss_mask.clone(),
            &config,
        )
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("PC fidelity arm solver={:?}", args.solver))?;
        arms.push(FidelityArm {
            solver: args.solver,
            inference_steps: report.pc_step.inference_steps,
            step_size: setting.step_size,
            dual_step_size: setting.dual_step_size,
            penalty: setting.penalty,
            prediction_precision: setting.prediction_precision,
            max_grad_norm: applied_max_grad_norm,
            report,
        });
    }

    Ok(FidelityMatrix {
        schema_version: 7,
        backend: args.backend.clone(),
        solver: args.solver,
        parameterization: args.parameterization,
        shared_reuse_reduction: args.shared_reuse_reduction,
        initialization: args.initialization,
        seed: args.seed,
        parameters,
        n_layer: args.n_layer,
        n_embd: args.n_embd,
        n_head: args.n_head,
        latent_total: args.latent_total,
        vocab_size: args.vocab_size,
        batch_size: args.batch_size,
        block_size: args.block_size,
        mask_period: args.mask_period,
        arms,
    })
}

#[cfg(all(feature = "train", feature = "cuda"))]
fn run_cuda(args: &Args) -> Result<FidelityMatrix> {
    run::<Autodiff<burn_cuda::Cuda<f32>>>(args)
}

#[cfg(all(feature = "train", not(feature = "cuda")))]
fn run_cuda(_args: &Args) -> Result<FidelityMatrix> {
    Err(anyhow!(
        "pc_gradient_fidelity was built without the cuda feature"
    ))
}

#[cfg(feature = "train")]
fn main() -> Result<()> {
    let args = parse_args()?;
    let report = match args.backend.as_str() {
        "cpu" => run::<Autodiff<NdArray<f32>>>(&args)?,
        "wgpu" => run::<Autodiff<burn_wgpu::Wgpu<f32>>>(&args)?,
        "cuda" => run_cuda(&args)?,
        other => return Err(anyhow!("unsupported backend {other}")),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report).context("serialize fidelity matrix")?
    );
    Ok(())
}

#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("pc_gradient_fidelity requires the train feature");
    std::process::exit(2);
}
