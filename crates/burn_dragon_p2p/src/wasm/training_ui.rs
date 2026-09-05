use dioxus::prelude::*;

use super::training::DragonBrowserTrainingResult;
use super::training_progress::DragonBrowserTrainingUiState;

pub(super) fn browser_training_panel(
    state: &DragonBrowserTrainingUiState,
    result: Option<&DragonBrowserTrainingResult>,
) -> Element {
    let phase = state.progress.phase;
    let current_window = state
        .current_window
        .map(|window| window.to_string())
        .unwrap_or_else(|| "-".into());
    let progress_is_queued =
        phase == super::training_progress::DragonBrowserTrainingPhase::SubmittingTraining;
    let batch_progress = if state.progress.planned_batches > 0 {
        format!(
            "{} {} / {} max",
            state.progress.submitted_batches,
            if progress_is_queued {
                "queued"
            } else {
                "complete"
            },
            state.progress.planned_batches,
        )
    } else {
        "-".into()
    };
    let token_progress = if state.progress.submitted_tokens > 0 {
        state.progress.submitted_tokens.to_string()
    } else {
        "-".into()
    };
    let completed_window = if state.completed_windows > 0 {
        state.completed_windows.to_string()
    } else {
        "-".into()
    };
    let backend = result.map_or_else(|| "-".into(), |result| result.backend.clone());
    let model_size = result.map_or_else(
        || "-".into(),
        |result| parameter_count_label(result.model_parameters),
    );
    let train_loss = result.map_or_else(
        || "-".into(),
        |result| {
            if result.train_loss_observed {
                format!("{:.4}", result.train_loss_mean)
            } else {
                "not sampled".into()
            }
        },
    );
    let eval_loss = result
        .and_then(|result| result.eval_loss)
        .map_or_else(|| "-".into(), |loss| format!("{loss:.4}"));
    let throughput = result
        .and_then(|result| result.tokens_per_second)
        .map_or_else(|| "-".into(), |rate| format!("{rate:.1} tok/s"));
    let training_duty = result
        .and_then(|result| result.training_duty_percent)
        .map_or_else(|| "-".into(), |duty| format!("{duty:.0}%"));
    let learning_rate = result.map_or_else(
        || "-".into(),
        |result| format!("{:.5e}", result.learning_rate),
    );
    let timing = result.map_or_else(
        || "-".into(),
        |result| {
            format!(
                "{} total / {} submit / {} sync",
                duration_label(result.total_time_ms),
                duration_label(result.phase_timings.training_submission_ms),
                duration_label(result.phase_timings.loss_synchronization_ms),
            )
        },
    );
    let contribution = result.map_or_else(
        || "-".into(),
        |result| match result.live_participant.as_ref() {
            Some(live) if live.artifact_published && live.update_announced => {
                "update announced".into()
            }
            Some(live) if live.receipt_submission_accepted => "telemetry accepted".into(),
            Some(live) if live.receipt_submission_deferred => {
                format!("{} telemetry item(s) pending", live.pending_receipt_count)
            }
            Some(_) => "not accepted".into(),
            None => "local only".into(),
        },
    );
    let detail = state.failure.clone().unwrap_or_else(|| match phase {
        super::training_progress::DragonBrowserTrainingPhase::MeasuringAdapter => {
            "Measuring completed GPU work before filling the signed window".into()
        }
        super::training_progress::DragonBrowserTrainingPhase::SynchronizingLoss => {
            "GPU work submitted; waiting for measured completion".into()
        }
        _ if state.completed_windows > 0 => {
            "Last result remains pinned while the next signed window runs".into()
        }
        _ => "No completed training window yet".into(),
    });
    let completed_windows = state.completed_windows.to_string();
    let phase_slug = phase.slug();
    let progress_max = state.progress.planned_batches.max(1).to_string();
    let progress_value = state
        .progress
        .submitted_batches
        .min(state.progress.planned_batches.max(1))
        .to_string();
    let token_progress_label = if progress_is_queued {
        "tokens queued"
    } else {
        "tokens complete"
    };

    rsx! {
        section {
            class: "panel compact-panel dragon-training-panel",
            "data-training-phase": phase_slug,
            "data-training-completed-windows": completed_windows,
            header { class: "dragon-training-header",
                div {
                    div { class: "eyebrow", "local" }
                    h2 { class: "browser-focus-title", "browser training" }
                }
                div {
                    class: "dragon-training-phase",
                    role: "status",
                    "aria-live": "polite",
                    "aria-atomic": "true",
                    "{phase.label()}"
                }
            }
            div { class: "dragon-training-progress" ,
                div { class: "dragon-training-progress-item",
                    span { "current window" }
                    strong { "{current_window}" }
                }
                div { class: "dragon-training-progress-item",
                    span { "batches" }
                    strong { "{batch_progress}" }
                }
                div { class: "dragon-training-progress-item",
                    span { "{token_progress_label}" }
                    strong { "{token_progress}" }
                }
            }
            progress {
                class: "dragon-training-batch-progress",
                max: progress_max,
                value: progress_value,
                "aria-label": "submitted training batches",
            }
            div { class: "dragon-training-last-window",
                div { class: "keyvalue-row",
                    span { "last window" }
                    strong { "{completed_window}" }
                }
                div { class: "keyvalue-row",
                    span { "backend" }
                    strong { "{backend}" }
                }
                div { class: "keyvalue-row",
                    span { "model" }
                    strong { "{model_size}" }
                }
                div { class: "keyvalue-row",
                    span { "train loss" }
                    strong { "{train_loss}" }
                }
                div { class: "keyvalue-row",
                    span { "eval loss" }
                    strong { "{eval_loss}" }
                }
                div { class: "keyvalue-row",
                    span { "throughput" }
                    strong { "{throughput}" }
                }
                div { class: "keyvalue-row",
                    span { "training duty" }
                    strong { "{training_duty}" }
                }
                div { class: "keyvalue-row",
                    span { "learning rate" }
                    strong { "{learning_rate}" }
                }
                div { class: "keyvalue-row",
                    span { "timing" }
                    strong { "{timing}" }
                }
                div { class: "keyvalue-row",
                    span { "network result" }
                    strong { "{contribution}" }
                }
            }
            p { class: "dragon-training-detail", "{detail}" }
        }
    }
}

fn duration_label(milliseconds: u64) -> String {
    if milliseconds >= 1_000 {
        format!("{:.1}s", milliseconds as f64 / 1_000.0)
    } else {
        format!("{milliseconds}ms")
    }
}

fn parameter_count_label(parameters: usize) -> String {
    if parameters >= 1_000_000 {
        format!("{:.1}M params", parameters as f64 / 1_000_000.0)
    } else if parameters >= 1_000 {
        format!("{:.1}K params", parameters as f64 / 1_000.0)
    } else {
        format!("{parameters} params")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_labels_are_compact_and_stable() {
        assert_eq!(duration_label(875), "875ms");
        assert_eq!(duration_label(1_250), "1.2s");
        assert_eq!(parameter_count_label(27_145_000), "27.1M params");
    }
}
