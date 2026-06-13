use crate::config::AutoBatchSizeConfig;
use crate::train::prelude::*;
use burn_dragon_train::train::runtime::{
    DeviceMemoryUsage, cleanup_device_memory, device_memory_usage_safe,
};

#[derive(Debug, Clone, Serialize)]
pub struct StartupAutotuneProbe {
    pub batch_size: usize,
    pub reserved_mb: Option<f64>,
    pub in_use_mb: Option<f64>,
    pub host_baseline_mb: Option<f64>,
    pub host_used_mb: Option<f64>,
    pub host_available_mb: Option<f64>,
    pub host_delta_mb: Option<f64>,
    pub projected_host_used_mb: Option<f64>,
    pub fit_target: bool,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StartupAutotuneReport {
    pub backend_name: String,
    pub config_source: String,
    pub target_device_memory_mb: usize,
    pub resolved_memory_cap_mb: Option<f64>,
    pub system_memory_total_mb: Option<f64>,
    pub max_system_memory_fraction: f32,
    pub probe_safety_margin: f32,
    pub target_effective_batch_size: Option<usize>,
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    pub max_probe_batch_size: Option<usize>,
    pub probe_steps: usize,
    pub resolved_batch_size: usize,
    pub resolved_batch_size_verified: bool,
    pub selection_basis: String,
    pub resolved_gradient_accumulation_steps: usize,
    pub resolved_effective_batch_size: usize,
    pub probes: Vec<StartupAutotuneProbe>,
}

#[derive(Debug, Clone)]
struct BatchAutotuneSettings {
    config_source: &'static str,
    target_device_memory_mb: usize,
    min_batch_size: usize,
    max_batch_size: usize,
    max_probe_batch_size: Option<usize>,
    probe_steps: usize,
    binary_search: bool,
    max_system_memory_fraction: f32,
    probe_safety_margin: f32,
}

pub fn resolve_startup_batch_size<B>(
    config: &TrainingConfig,
    dataset: &Arc<Dataset>,
    backend_name: &str,
    device: &B::Device,
) -> Result<Option<StartupAutotuneReport>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone + 'static,
{
    let Some(autotune) = resolve_batch_autotune_settings(config, backend_name) else {
        return Ok(None);
    };

    let min_batch_size = autotune.min_batch_size;
    let max_batch_size = autotune.max_batch_size.max(min_batch_size);
    let target_effective_batch_size = config
        .training
        .target_effective_batch_size
        .filter(|value| *value > 0);
    let training_kernel_block_size =
        crate::train::utils::effective_training_kernel_block_size(&config.training);

    let tokenizer = dataset.tokenizer();
    let mut model_config = build_model_config_with_tokenizer(
        &config.model,
        training_kernel_block_size,
        tokenizer.as_ref(),
    )?;
    apply_wgpu_fused_core_override(
        &mut model_config,
        backend_name,
        WgpuFusedCoreOverride {
            recurrent: config.wgpu.training.fused_core_recurrent,
            rollout: config.wgpu.training.fused_core_rollout,
        },
    );
    let (effective_sequence_kernel, _, _) =
        crate::train::backend::resolve_effective_training_sequence_kernel(
            model_config.sequence_kernel,
            config.training.sequence_kernel_override,
            backend_name,
            training_kernel_block_size,
        );
    model_config.sequence_kernel = effective_sequence_kernel;
    let summary_event_token_ids = model_config.summary_memory.write_trigger_token_ids.clone();

    let mut probes = Vec::new();
    let system_memory_total_bytes = system_memory_snapshot().map(|snapshot| snapshot.total_bytes);
    let device_target_bytes = (autotune.target_device_memory_mb > 0)
        .then(|| (autotune.target_device_memory_mb as u64) * 1024 * 1024);
    let host_cap_bytes = system_memory_total_bytes.map(|bytes| {
        ((bytes as f64)
            * autotune
                .max_system_memory_fraction
                .min(0.9)
                .max(f32::EPSILON) as f64) as u64
    });
    let target_bytes = resolve_memory_cap_bytes(
        autotune.target_device_memory_mb,
        autotune.max_system_memory_fraction,
        system_memory_total_bytes,
    );
    let (mut low, mut high) = (min_batch_size, max_batch_size);
    let mut best_fit = None;
    let mut best_fit_verified = false;

    for candidate in startup_candidate_sequence(min_batch_size, max_batch_size) {
        let maybe_skip = probe_memory_cap_skip(
            candidate,
            &probes,
            device_target_bytes,
            host_cap_bytes,
            autotune.probe_safety_margin,
        );
        if let Some(skip) = maybe_skip {
            probes.push(skip);
            high = candidate;
            break;
        }
        if candidate_exceeds_probe_cap(candidate, autotune.max_probe_batch_size) {
            let Some(predicted) = predicted_fit_probe(
                candidate,
                &probes,
                device_target_bytes,
                host_cap_bytes,
                autotune.probe_safety_margin,
            ) else {
                probes.push(probe_cap_skip_probe(candidate, system_memory_snapshot()));
                high = candidate;
                break;
            };
            probes.push(predicted);
            best_fit = Some(candidate);
            best_fit_verified = false;
            low = candidate;
            if candidate == max_batch_size {
                break;
            }
            continue;
        }
        let probe = probe_batch_size::<B>(ProbeBatchRequest {
            dataset,
            model_config: &model_config,
            block_size: config.training.block_size,
            tbptt_chunk_size: config.training.tbptt_chunk_size,
            batch_size: candidate,
            probe_steps: autotune.probe_steps,
            device_target_bytes,
            host_cap_bytes,
            summary_event_token_ids: summary_event_token_ids.as_deref(),
            device,
        });
        let fit_target = probe.fit_target;
        probes.push(probe);
        if fit_target {
            best_fit = Some(candidate);
            best_fit_verified = true;
            low = candidate;
            if candidate == max_batch_size {
                break;
            }
        } else {
            high = candidate;
            break;
        }
    }

    if autotune.binary_search && best_fit.is_some() && high > low + 1 {
        while high > low + 1 {
            let candidate = low + ((high - low) / 2);
            let maybe_skip = probe_memory_cap_skip(
                candidate,
                &probes,
                device_target_bytes,
                host_cap_bytes,
                autotune.probe_safety_margin,
            );
            if let Some(skip) = maybe_skip {
                probes.push(skip);
                high = candidate;
                continue;
            }
            if candidate_exceeds_probe_cap(candidate, autotune.max_probe_batch_size) {
                let Some(predicted) = predicted_fit_probe(
                    candidate,
                    &probes,
                    device_target_bytes,
                    host_cap_bytes,
                    autotune.probe_safety_margin,
                ) else {
                    probes.push(probe_cap_skip_probe(candidate, system_memory_snapshot()));
                    high = candidate;
                    continue;
                };
                probes.push(predicted);
                best_fit = Some(candidate);
                best_fit_verified = false;
                low = candidate;
                continue;
            }
            let probe = probe_batch_size::<B>(ProbeBatchRequest {
                dataset,
                model_config: &model_config,
                block_size: config.training.block_size,
                tbptt_chunk_size: config.training.tbptt_chunk_size,
                batch_size: candidate,
                probe_steps: autotune.probe_steps,
                device_target_bytes,
                host_cap_bytes,
                summary_event_token_ids: summary_event_token_ids.as_deref(),
                device,
            });
            let fit_target = probe.fit_target;
            probes.push(probe);
            if fit_target {
                best_fit = Some(candidate);
                best_fit_verified = true;
                low = candidate;
            } else {
                high = candidate;
            }
        }
    }

    let Some(resolved_batch_size) = best_fit else {
        return Err(anyhow!(
            "startup autotune could not find a safe batch size between {} and {} under target {} MiB; probes={}",
            min_batch_size,
            max_batch_size,
            autotune.target_device_memory_mb,
            format_probe_summary(&probes)
        ));
    };

    info!(
        "startup autotune: resolved batch_size={} grad_accumulation_steps={} effective_batch_size={} (target={} MiB, probed {} candidates)",
        resolved_batch_size,
        resolve_gradient_accumulation_steps(
            resolved_batch_size,
            config.training.gradient_accumulation_steps,
            target_effective_batch_size,
        ),
        resolved_batch_size.saturating_mul(resolve_gradient_accumulation_steps(
            resolved_batch_size,
            config.training.gradient_accumulation_steps,
            target_effective_batch_size,
        )),
        autotune.target_device_memory_mb,
        probes.len(),
    );

    let resolved_gradient_accumulation_steps = resolve_gradient_accumulation_steps(
        resolved_batch_size,
        config.training.gradient_accumulation_steps,
        target_effective_batch_size,
    );
    let resolved_effective_batch_size =
        resolved_batch_size.saturating_mul(resolved_gradient_accumulation_steps);

    Ok(Some(StartupAutotuneReport {
        backend_name: backend_name.to_string(),
        config_source: autotune.config_source.to_string(),
        target_device_memory_mb: autotune.target_device_memory_mb,
        resolved_memory_cap_mb: target_bytes.map(burn_dragon_train::train::runtime::bytes_to_mb),
        system_memory_total_mb: system_memory_total_bytes
            .map(burn_dragon_train::train::runtime::bytes_to_mb),
        max_system_memory_fraction: autotune.max_system_memory_fraction,
        probe_safety_margin: autotune.probe_safety_margin,
        target_effective_batch_size,
        min_batch_size,
        max_batch_size,
        max_probe_batch_size: autotune.max_probe_batch_size,
        probe_steps: autotune.probe_steps,
        resolved_batch_size,
        resolved_batch_size_verified: best_fit_verified,
        selection_basis: if best_fit_verified {
            "verified".to_string()
        } else {
            "predicted".to_string()
        },
        resolved_gradient_accumulation_steps,
        resolved_effective_batch_size,
        probes,
    }))
}

fn resolve_batch_autotune_settings(
    config: &TrainingConfig,
    backend_name: &str,
) -> Option<BatchAutotuneSettings> {
    if config.training.auto_batch_size.enabled {
        let auto = &config.training.auto_batch_size;
        let min_batch_size = auto.min_batch_size.max(1);
        return Some(BatchAutotuneSettings {
            config_source: "training.auto_batch_size",
            target_device_memory_mb: auto.target_device_memory_mb,
            min_batch_size,
            max_batch_size: auto
                .max_batch_size
                .unwrap_or(config.training.batch_size)
                .max(min_batch_size),
            max_probe_batch_size: auto.max_probe_batch_size,
            probe_steps: auto.probe_steps.max(1),
            binary_search: auto.binary_search,
            max_system_memory_fraction: auto.max_system_memory_fraction.min(0.9),
            probe_safety_margin: auto.probe_safety_margin.max(1.0),
        });
    }

    let autotune = &config.wgpu.training.startup_autotune;
    if !autotune.enabled || !backend_name.starts_with("wgpu") {
        return None;
    }
    let min_batch_size = autotune.min_batch_size.max(1);
    Some(BatchAutotuneSettings {
        config_source: "wgpu.training.startup_autotune",
        target_device_memory_mb: autotune.target_device_memory_mb,
        min_batch_size,
        max_batch_size: autotune
            .max_batch_size
            .unwrap_or(config.training.batch_size)
            .max(min_batch_size),
        max_probe_batch_size: None,
        probe_steps: autotune.probe_steps.max(1),
        binary_search: autotune.binary_search,
        max_system_memory_fraction: 0.85,
        probe_safety_margin: 1.15,
    })
}

pub fn resolve_scaled_auto_batch_size(
    config: &AutoBatchSizeConfig,
    current_batch_size: usize,
    old_capacity_units: usize,
    new_capacity_units: usize,
) -> usize {
    let min_batch_size = config.min_batch_size.max(1);
    let max_batch_size = config
        .max_batch_size
        .unwrap_or(current_batch_size.max(min_batch_size))
        .max(min_batch_size);
    let current_batch_size = current_batch_size.max(min_batch_size).min(max_batch_size);
    if !config.enabled
        || !config.recompute_on_neuron_scale
        || old_capacity_units == 0
        || new_capacity_units <= old_capacity_units
    {
        return current_batch_size;
    }

    let exponent = config.scale_memory_exponent.max(0.0) as f64;
    let ratio = old_capacity_units as f64 / new_capacity_units as f64;
    let scaled = ((current_batch_size as f64) * ratio.powf(exponent)).floor() as usize;
    scaled.max(min_batch_size).min(max_batch_size)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SystemMemorySnapshot {
    total_bytes: u64,
    available_bytes: u64,
}

impl SystemMemorySnapshot {
    fn used_bytes(self) -> u64 {
        self.total_bytes.saturating_sub(self.available_bytes)
    }

    fn used_mb(self) -> f64 {
        burn_dragon_train::train::runtime::bytes_to_mb(self.used_bytes())
    }

    fn available_mb(self) -> f64 {
        burn_dragon_train::train::runtime::bytes_to_mb(self.available_bytes)
    }
}

fn system_memory_snapshot() -> Option<SystemMemorySnapshot> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = None;
    let mut available_kb = None;
    let mut free_kb = None;
    for line in content.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let value = rest
            .split_whitespace()
            .next()
            .and_then(|value| value.parse::<u64>().ok());
        match key {
            "MemTotal" => total_kb = value,
            "MemAvailable" => available_kb = value,
            "MemFree" => free_kb = value,
            _ => {}
        }
    }
    let total_bytes = total_kb?.saturating_mul(1024);
    let available_bytes = available_kb.or(free_kb)?.saturating_mul(1024);
    Some(SystemMemorySnapshot {
        total_bytes,
        available_bytes,
    })
}

fn resolve_memory_cap_bytes(
    target_device_memory_mb: usize,
    max_system_memory_fraction: f32,
    system_memory_total_bytes: Option<u64>,
) -> Option<u64> {
    let host_cap = system_memory_total_bytes.map(|bytes| {
        ((bytes as f64) * max_system_memory_fraction.min(0.9).max(f32::EPSILON) as f64) as u64
    });
    let target_cap =
        (target_device_memory_mb > 0).then(|| (target_device_memory_mb as u64) * 1024 * 1024);
    match (target_cap, host_cap) {
        (Some(target), Some(host)) => Some(target.min(host)),
        (Some(target), None) => Some(target),
        (None, Some(host)) => Some(host),
        (None, None) => None,
    }
}

fn predicted_over_memory_cap(
    candidate: usize,
    probes: &[StartupAutotuneProbe],
    memory_cap_bytes: Option<u64>,
    safety_margin: f32,
) -> Option<StartupAutotuneProbe> {
    let memory_cap_bytes = memory_cap_bytes?;
    let predicted_bytes = predict_candidate_device_bytes(candidate, probes)?;
    if predicted_bytes * safety_margin.max(1.0) as f64 <= memory_cap_bytes as f64 {
        return None;
    }

    Some(predicted_probe(
        candidate,
        Some(predicted_bytes),
        None,
        None,
        None,
        "predicted_over_memory_cap",
    ))
}

fn probe_memory_cap_skip(
    candidate: usize,
    probes: &[StartupAutotuneProbe],
    device_target_bytes: Option<u64>,
    host_cap_bytes: Option<u64>,
    safety_margin: f32,
) -> Option<StartupAutotuneProbe> {
    probe_memory_cap_skip_with_snapshot(
        candidate,
        probes,
        device_target_bytes,
        host_cap_bytes,
        safety_margin,
        system_memory_snapshot(),
    )
}

fn probe_memory_cap_skip_with_snapshot(
    candidate: usize,
    probes: &[StartupAutotuneProbe],
    device_target_bytes: Option<u64>,
    host_cap_bytes: Option<u64>,
    safety_margin: f32,
    snapshot: Option<SystemMemorySnapshot>,
) -> Option<StartupAutotuneProbe> {
    if let Some(skip) =
        predicted_over_memory_cap(candidate, probes, device_target_bytes, safety_margin)
    {
        return Some(skip);
    }

    let Some(host_cap_bytes) = host_cap_bytes else {
        return None;
    };
    let Some(snapshot) = snapshot else {
        return None;
    };
    if snapshot.used_bytes() >= host_cap_bytes {
        return Some(StartupAutotuneProbe {
            batch_size: candidate,
            reserved_mb: None,
            in_use_mb: None,
            host_baseline_mb: Some(snapshot.used_mb()),
            host_used_mb: Some(snapshot.used_mb()),
            host_available_mb: Some(snapshot.available_mb()),
            host_delta_mb: Some(0.0),
            projected_host_used_mb: Some(snapshot.used_mb()),
            fit_target: false,
            status: "host_memory_cap_reached".to_string(),
        });
    }

    let predicted_host_delta_bytes = predict_candidate_host_delta_bytes(candidate, probes)?;
    let projected_host_bytes = snapshot.used_bytes().saturating_add(
        (predicted_host_delta_bytes * safety_margin.max(1.0) as f64).max(0.0) as u64,
    );
    if projected_host_bytes <= host_cap_bytes {
        return None;
    }

    Some(predicted_probe(
        candidate,
        predict_candidate_device_bytes(candidate, probes),
        Some(snapshot),
        Some(predicted_host_delta_bytes),
        Some(projected_host_bytes),
        "predicted_host_memory_cap",
    ))
}

fn candidate_exceeds_probe_cap(candidate: usize, max_probe_batch_size: Option<usize>) -> bool {
    max_probe_batch_size.is_some_and(|max_probe_batch_size| candidate > max_probe_batch_size.max(1))
}

fn predicted_fit_probe(
    candidate: usize,
    probes: &[StartupAutotuneProbe],
    device_target_bytes: Option<u64>,
    host_cap_bytes: Option<u64>,
    safety_margin: f32,
) -> Option<StartupAutotuneProbe> {
    predicted_fit_probe_with_snapshot(
        candidate,
        probes,
        device_target_bytes,
        host_cap_bytes,
        safety_margin,
        system_memory_snapshot(),
    )
}

fn predicted_fit_probe_with_snapshot(
    candidate: usize,
    probes: &[StartupAutotuneProbe],
    device_target_bytes: Option<u64>,
    host_cap_bytes: Option<u64>,
    safety_margin: f32,
    snapshot: Option<SystemMemorySnapshot>,
) -> Option<StartupAutotuneProbe> {
    let predicted_device_bytes = predict_candidate_device_bytes(candidate, probes);
    if let Some(device_target_bytes) = device_target_bytes {
        let predicted_device_bytes = predicted_device_bytes?;
        if predicted_device_bytes * safety_margin.max(1.0) as f64 > device_target_bytes as f64 {
            return None;
        }
    }

    let predicted_host_delta_bytes = predict_candidate_host_delta_bytes(candidate, probes);
    let mut projected_host_bytes = None;
    if let (Some(host_cap_bytes), Some(snapshot)) = (host_cap_bytes, snapshot) {
        if snapshot.used_bytes() >= host_cap_bytes {
            return None;
        }
        let predicted_host_delta_bytes = predicted_host_delta_bytes?;
        let projected = snapshot.used_bytes().saturating_add(
            (predicted_host_delta_bytes * safety_margin.max(1.0) as f64).max(0.0) as u64,
        );
        if projected > host_cap_bytes {
            return None;
        }
        projected_host_bytes = Some(projected);
    }

    if predicted_device_bytes.is_none() && predicted_host_delta_bytes.is_none() {
        return None;
    }
    let mut probe = predicted_probe(
        candidate,
        predicted_device_bytes,
        snapshot,
        predicted_host_delta_bytes,
        projected_host_bytes,
        "predicted_fit",
    );
    probe.fit_target = true;
    Some(probe)
}

fn probe_cap_skip_probe(
    candidate: usize,
    snapshot: Option<SystemMemorySnapshot>,
) -> StartupAutotuneProbe {
    StartupAutotuneProbe {
        batch_size: candidate,
        reserved_mb: None,
        in_use_mb: None,
        host_baseline_mb: snapshot.map(SystemMemorySnapshot::used_mb),
        host_used_mb: snapshot.map(SystemMemorySnapshot::used_mb),
        host_available_mb: snapshot.map(SystemMemorySnapshot::available_mb),
        host_delta_mb: Some(0.0),
        projected_host_used_mb: snapshot.map(SystemMemorySnapshot::used_mb),
        fit_target: false,
        status: "probe_cap_no_prediction".to_string(),
    }
}

fn predict_candidate_device_bytes(
    candidate: usize,
    probes: &[StartupAutotuneProbe],
) -> Option<f64> {
    predict_candidate_bytes_from_points(
        candidate,
        probes
            .iter()
            .filter(|probe| measured_probe_can_predict(probe))
            .filter_map(|probe| {
                if probe.batch_size >= candidate {
                    return None;
                }
                let usage_mb = match (probe.reserved_mb, probe.in_use_mb) {
                    (Some(reserved), Some(in_use)) => reserved.max(in_use),
                    (Some(reserved), None) => reserved,
                    (None, Some(in_use)) => in_use,
                    (None, None) => return None,
                };
                Some((probe.batch_size, usage_mb * 1024.0 * 1024.0))
            })
            .collect(),
    )
}

fn predict_candidate_host_delta_bytes(
    candidate: usize,
    probes: &[StartupAutotuneProbe],
) -> Option<f64> {
    predict_candidate_bytes_from_points(
        candidate,
        probes
            .iter()
            .filter(|probe| measured_probe_can_predict(probe))
            .filter_map(|probe| {
                if probe.batch_size >= candidate {
                    return None;
                }
                Some((probe.batch_size, probe.host_delta_mb? * 1024.0 * 1024.0))
            })
            .collect(),
    )
}

fn measured_probe_can_predict(probe: &StartupAutotuneProbe) -> bool {
    !probe.status.starts_with("predicted_")
        && probe.status != "host_memory_cap_reached"
        && probe.status != "probe_cap_no_prediction"
        && probe.status != "probe_failed"
}

fn predict_candidate_bytes_from_points(
    candidate: usize,
    mut measured: Vec<(usize, f64)>,
) -> Option<f64> {
    measured.sort_by_key(|(batch_size, _)| *batch_size);
    let [(prev_batch, prev_bytes), (last_batch, last_bytes)] = match measured.as_slice() {
        [.., prev, last] => [*prev, *last],
        _ => return None,
    };
    if candidate <= last_batch || last_batch <= prev_batch {
        return None;
    }
    let slope = ((last_bytes - prev_bytes) / (last_batch - prev_batch) as f64).max(0.0);
    Some(last_bytes + slope * (candidate - last_batch) as f64)
}

fn predicted_probe(
    candidate: usize,
    predicted_device_bytes: Option<f64>,
    snapshot: Option<SystemMemorySnapshot>,
    predicted_host_delta_bytes: Option<f64>,
    projected_host_bytes: Option<u64>,
    status: &str,
) -> StartupAutotuneProbe {
    StartupAutotuneProbe {
        batch_size: candidate,
        reserved_mb: predicted_device_bytes
            .map(|bytes| burn_dragon_train::train::runtime::bytes_to_mb(bytes.max(0.0) as u64)),
        in_use_mb: None,
        host_baseline_mb: snapshot.map(SystemMemorySnapshot::used_mb),
        host_used_mb: snapshot.map(SystemMemorySnapshot::used_mb),
        host_available_mb: snapshot.map(SystemMemorySnapshot::available_mb),
        host_delta_mb: predicted_host_delta_bytes
            .map(|bytes| burn_dragon_train::train::runtime::bytes_to_mb(bytes.max(0.0) as u64)),
        projected_host_used_mb: projected_host_bytes
            .map(burn_dragon_train::train::runtime::bytes_to_mb),
        fit_target: false,
        status: status.to_string(),
    }
}

fn probe_batch_size<B>(request: ProbeBatchRequest<'_, B>) -> StartupAutotuneProbe
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone + 'static,
{
    let ProbeBatchRequest {
        dataset,
        model_config,
        block_size,
        tbptt_chunk_size,
        batch_size,
        probe_steps,
        device_target_bytes,
        host_cap_bytes,
        summary_event_token_ids,
        device,
    } = request;
    cleanup_device_memory::<B>(device, true);
    let baseline_host_usage = system_memory_snapshot();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let model = LanguageTrainModel::new(DragonModel::<B>::new(model_config.clone(), device))
            .with_tbptt_chunk_size(tbptt_chunk_size);
        let mut peak_usage: Option<DeviceMemoryUsage> = None;
        let mut peak_host_usage: Option<SystemMemorySnapshot> = baseline_host_usage;

        for _ in 0..probe_steps {
            let batch = sample_batch_with_shape::<B, _>(
                &**dataset,
                DatasetSplit::Train,
                batch_size,
                block_size,
                summary_event_token_ids,
                0,
                device,
            );
            let output = burn_train::TrainStep::step(&model, batch);
            drop(output);
            let _ = B::sync(device);
            if let Some(usage) = device_memory_usage_safe::<B>(device) {
                peak_usage = Some(match peak_usage {
                    Some(current)
                        if current.reserved_bytes.max(current.in_use_bytes)
                            >= usage.reserved_bytes.max(usage.in_use_bytes) =>
                    {
                        current
                    }
                    _ => usage,
                });
            }
            if let Some(snapshot) = system_memory_snapshot() {
                peak_host_usage = Some(match peak_host_usage {
                    Some(current) if current.used_bytes() >= snapshot.used_bytes() => current,
                    _ => snapshot,
                });
            }
        }

        drop(model);
        cleanup_device_memory::<B>(device, true);
        (peak_usage, peak_host_usage)
    }));

    match result {
        Ok((peak_usage, peak_host_usage)) => {
            let device_fits = match device_target_bytes {
                Some(device_target_bytes) => peak_usage
                    .map(|usage| {
                        usage.reserved_bytes.max(usage.in_use_bytes) <= device_target_bytes
                    })
                    .unwrap_or(false),
                None => true,
            };
            let host_fits = match host_cap_bytes {
                Some(host_cap_bytes) => peak_host_usage
                    .map(|snapshot| snapshot.used_bytes() <= host_cap_bytes)
                    .unwrap_or(false),
                None => true,
            };
            let fit_target = device_fits && host_fits;
            StartupAutotuneProbe {
                batch_size,
                reserved_mb: peak_usage.map(DeviceMemoryUsage::reserved_mb),
                in_use_mb: peak_usage.map(DeviceMemoryUsage::in_use_mb),
                host_baseline_mb: baseline_host_usage.map(SystemMemorySnapshot::used_mb),
                host_used_mb: peak_host_usage.map(SystemMemorySnapshot::used_mb),
                host_available_mb: peak_host_usage.map(SystemMemorySnapshot::available_mb),
                host_delta_mb: host_delta_mb(baseline_host_usage, peak_host_usage),
                projected_host_used_mb: None,
                fit_target,
                status: if fit_target {
                    "fit".to_string()
                } else if !host_fits {
                    "over_host_target".to_string()
                } else {
                    "over_target".to_string()
                },
            }
        }
        Err(_) => {
            cleanup_device_memory::<B>(device, true);
            let failed_host_usage = system_memory_snapshot();
            StartupAutotuneProbe {
                batch_size,
                reserved_mb: None,
                in_use_mb: None,
                host_baseline_mb: baseline_host_usage.map(SystemMemorySnapshot::used_mb),
                host_used_mb: failed_host_usage.map(SystemMemorySnapshot::used_mb),
                host_available_mb: failed_host_usage.map(SystemMemorySnapshot::available_mb),
                host_delta_mb: host_delta_mb(baseline_host_usage, failed_host_usage),
                projected_host_used_mb: None,
                fit_target: false,
                status: "probe_failed".to_string(),
            }
        }
    }
}

fn host_delta_mb(
    baseline: Option<SystemMemorySnapshot>,
    peak: Option<SystemMemorySnapshot>,
) -> Option<f64> {
    let baseline = baseline?;
    let peak = peak?;
    Some(burn_dragon_train::train::runtime::bytes_to_mb(
        peak.used_bytes().saturating_sub(baseline.used_bytes()),
    ))
}

struct ProbeBatchRequest<'a, B: AutodiffBackend> {
    dataset: &'a Arc<Dataset>,
    model_config: &'a DragonConfig,
    block_size: usize,
    tbptt_chunk_size: Option<usize>,
    batch_size: usize,
    probe_steps: usize,
    device_target_bytes: Option<u64>,
    host_cap_bytes: Option<u64>,
    summary_event_token_ids: Option<&'a [u32]>,
    device: &'a B::Device,
}

fn startup_candidate_sequence(min_batch_size: usize, max_batch_size: usize) -> Vec<usize> {
    let mut candidates = Vec::new();
    let mut current = min_batch_size.max(1);
    candidates.push(current);

    while current < max_batch_size {
        let next = current.saturating_mul(2).min(max_batch_size);
        if next == current {
            break;
        }
        candidates.push(next);
        current = next;
    }

    candidates
}

pub fn resolve_gradient_accumulation_steps(
    resolved_batch_size: usize,
    configured_gradient_accumulation_steps: usize,
    target_effective_batch_size: Option<usize>,
) -> usize {
    match target_effective_batch_size {
        Some(target_effective_batch_size) => target_effective_batch_size
            .max(resolved_batch_size.max(1))
            .div_ceil(resolved_batch_size.max(1))
            .max(1),
        None => configured_gradient_accumulation_steps.max(1),
    }
}

fn format_probe_summary(probes: &[StartupAutotuneProbe]) -> String {
    probes
        .iter()
        .map(|probe| match (probe.reserved_mb, probe.in_use_mb) {
            (Some(reserved), Some(in_use)) => format!(
                "bs{}:{}:{reserved:.1}/{in_use:.1}MiB",
                probe.batch_size, probe.status
            ),
            (Some(reserved), None) => {
                format!("bs{}:{}:{reserved:.1}MiB", probe.batch_size, probe.status)
            }
            _ => format!("bs{}:{}", probe.batch_size, probe.status),
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::{
        StartupAutotuneProbe, SystemMemorySnapshot, candidate_exceeds_probe_cap,
        predicted_fit_probe, predicted_fit_probe_with_snapshot, predicted_over_memory_cap,
        probe_cap_skip_probe, probe_memory_cap_skip_with_snapshot,
        resolve_gradient_accumulation_steps, resolve_memory_cap_bytes,
        resolve_scaled_auto_batch_size, startup_candidate_sequence,
    };
    use crate::config::AutoBatchSizeConfig;

    fn fit_probe(batch_size: usize, reserved_mb: f64) -> StartupAutotuneProbe {
        fit_probe_with_host_delta(batch_size, reserved_mb, reserved_mb)
    }

    fn fit_probe_with_host_delta(
        batch_size: usize,
        reserved_mb: f64,
        host_delta_mb: f64,
    ) -> StartupAutotuneProbe {
        StartupAutotuneProbe {
            batch_size,
            reserved_mb: Some(reserved_mb),
            in_use_mb: None,
            host_baseline_mb: Some(50.0),
            host_used_mb: None,
            host_available_mb: None,
            host_delta_mb: Some(host_delta_mb),
            projected_host_used_mb: None,
            fit_target: true,
            status: "fit".to_string(),
        }
    }

    #[test]
    fn startup_candidate_sequence_doubles_and_caps_at_max() {
        assert_eq!(startup_candidate_sequence(4, 4), vec![4]);
        assert_eq!(startup_candidate_sequence(4, 20), vec![4, 8, 16, 20]);
        assert_eq!(startup_candidate_sequence(3, 24), vec![3, 6, 12, 24]);
    }

    #[test]
    fn resolve_gradient_accumulation_steps_ceil_divides_to_target_effective_batch() {
        assert_eq!(resolve_gradient_accumulation_steps(64, 1, None), 1);
        assert_eq!(resolve_gradient_accumulation_steps(64, 3, None), 3);
        assert_eq!(resolve_gradient_accumulation_steps(64, 1, Some(64)), 1);
        assert_eq!(resolve_gradient_accumulation_steps(21, 1, Some(64)), 4);
        assert_eq!(resolve_gradient_accumulation_steps(16, 1, Some(128)), 8);
    }

    #[test]
    fn scaled_auto_batch_reduces_with_capacity_growth_and_clamps() {
        let config = AutoBatchSizeConfig {
            enabled: true,
            min_batch_size: 4,
            max_batch_size: Some(32),
            recompute_on_neuron_scale: true,
            scale_memory_exponent: 1.0,
            ..Default::default()
        };
        assert_eq!(resolve_scaled_auto_batch_size(&config, 32, 1024, 2048), 16);
        assert_eq!(resolve_scaled_auto_batch_size(&config, 8, 1024, 8192), 4);
    }

    #[test]
    fn scaled_auto_batch_can_disable_scale_recompute() {
        let config = AutoBatchSizeConfig {
            enabled: true,
            max_batch_size: Some(32),
            recompute_on_neuron_scale: false,
            ..Default::default()
        };
        assert_eq!(resolve_scaled_auto_batch_size(&config, 32, 1024, 4096), 32);
    }

    #[test]
    fn auto_batch_defaults_to_ninety_percent_host_cap() {
        let config = AutoBatchSizeConfig::default();

        assert_eq!(config.max_system_memory_fraction, 0.9);
    }

    #[test]
    fn memory_cap_uses_smaller_target_and_host_fraction() {
        let gib = 1024 * 1024 * 1024;
        let near = |actual: Option<u64>, expected: u64| {
            let actual = actual.expect("cap");
            let delta = actual.abs_diff(expected);
            assert!(
                delta <= 4096,
                "actual={actual} expected={expected} delta={delta}"
            );
        };
        assert_eq!(
            resolve_memory_cap_bytes(90 * 1024, 0.9, Some(121 * gib)),
            Some(90 * gib)
        );
        near(resolve_memory_cap_bytes(0, 0.9, Some(100 * gib)), 90 * gib);
        near(
            resolve_memory_cap_bytes(96 * 1024, 0.5, Some(100 * gib)),
            50 * gib,
        );
    }

    #[test]
    fn predicted_memory_cap_skips_dangerous_next_probe() {
        let probes = vec![fit_probe(1, 10.0), fit_probe(2, 20.0), fit_probe(4, 56.0)];
        let cap = Some(90 * 1024 * 1024);
        assert!(predicted_over_memory_cap(5, &probes, cap, 1.15).is_none());
        let skip =
            predicted_over_memory_cap(8, &probes, cap, 1.15).expect("batch 8 should be skipped");
        assert_eq!(skip.status, "predicted_over_memory_cap");
        assert_eq!(skip.batch_size, 8);
    }

    #[test]
    fn predicted_memory_cap_ignores_prior_synthetic_skips() {
        let probes = vec![
            fit_probe(4, 37_250.0),
            StartupAutotuneProbe {
                batch_size: 8,
                reserved_mb: Some(70_898.0),
                in_use_mb: None,
                host_baseline_mb: Some(50.0),
                host_used_mb: None,
                host_available_mb: None,
                host_delta_mb: Some(70_898.0),
                projected_host_used_mb: None,
                fit_target: false,
                status: "predicted_over_memory_cap".to_string(),
            },
            fit_probe(6, 61_254.0),
        ];
        let skip = predicted_over_memory_cap(7, &probes, Some(70_000 * 1024 * 1024), 1.15)
            .expect("batch 7 should be skipped using real batch 4/6 probes");
        assert_eq!(skip.status, "predicted_over_memory_cap");
        assert_eq!(skip.batch_size, 7);
    }

    #[test]
    fn host_memory_cap_predictor_skips_when_system_ram_projection_is_too_high() {
        let probes = vec![fit_probe(1, 10.0), fit_probe(2, 20.0), fit_probe(4, 56.0)];
        let snapshot = SystemMemorySnapshot {
            total_bytes: 200 * 1024 * 1024,
            available_bytes: 150 * 1024 * 1024,
        };
        let skip = probe_memory_cap_skip_with_snapshot(
            8,
            &probes,
            None,
            Some(120 * 1024 * 1024),
            1.15,
            Some(snapshot),
        )
        .expect("projected host usage should skip before probing");
        assert_eq!(skip.status, "predicted_host_memory_cap");
        assert_eq!(skip.batch_size, 8);
        assert!(skip.host_delta_mb.unwrap() > 120.0);
        assert!(skip.projected_host_used_mb.unwrap() > 120.0);
    }

    #[test]
    fn host_memory_cap_predictor_uses_host_delta_not_reserved_device_bytes() {
        let probes = vec![
            fit_probe_with_host_delta(1, 10_000.0, 5.0),
            fit_probe_with_host_delta(2, 20_000.0, 10.0),
            fit_probe_with_host_delta(4, 56_000.0, 20.0),
        ];
        let snapshot = SystemMemorySnapshot {
            total_bytes: 200 * 1024 * 1024,
            available_bytes: 150 * 1024 * 1024,
        };

        let skip = probe_memory_cap_skip_with_snapshot(
            8,
            &probes,
            None,
            Some(120 * 1024 * 1024),
            1.15,
            Some(snapshot),
        );

        assert!(
            skip.is_none(),
            "large CUDA reserved-memory predictions must not force a host-RAM skip"
        );
    }

    #[test]
    fn probe_cap_uses_prediction_without_executing_large_probe() {
        let probes = vec![fit_probe(1, 10.0), fit_probe(2, 20.0), fit_probe(4, 56.0)];
        let snapshot = SystemMemorySnapshot {
            total_bytes: 200 * 1024 * 1024,
            available_bytes: 150 * 1024 * 1024,
        };

        assert!(candidate_exceeds_probe_cap(8, Some(4)));
        let predicted = predicted_fit_probe(5, &probes, None, None, 1.15).expect("predicted fit");

        assert_eq!(predicted.batch_size, 5);
        assert_eq!(predicted.status, "predicted_fit");
        assert!(predicted.fit_target);
        let skip = probe_cap_skip_probe(8, Some(snapshot));
        assert_eq!(skip.status, "probe_cap_no_prediction");
        assert!(!skip.fit_target);
    }

    #[test]
    fn probe_cap_prediction_checks_host_delta_against_host_cap() {
        let probes = vec![
            fit_probe_with_host_delta(1, 10_000.0, 5.0),
            fit_probe_with_host_delta(2, 20_000.0, 10.0),
            fit_probe_with_host_delta(4, 56_000.0, 20.0),
        ];

        let snapshot = SystemMemorySnapshot {
            total_bytes: 200 * 1024 * 1024,
            available_bytes: 150 * 1024 * 1024,
        };
        let predicted = predicted_fit_probe_with_snapshot(
            8,
            &probes,
            None,
            Some(120 * 1024 * 1024),
            1.15,
            Some(snapshot),
        )
        .expect("host delta projection fits the host cap");

        assert_eq!(predicted.status, "predicted_fit");
        assert!(predicted.fit_target);
        assert_eq!(predicted.host_delta_mb, Some(40.0));
    }
}
