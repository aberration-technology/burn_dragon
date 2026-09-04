export function browserConfigTrainingConfig(browserConfig) {
  if (!browserConfig || typeof browserConfig !== "object") {
    return null;
  }
  const nested = browserConfig.config?.training;
  if (nested && typeof nested === "object") {
    return nested;
  }
  return null;
}

export function validateBrowserCanaryTrainingPolicy({
  expectTraining,
  useProductionTrainingProfile,
  minAcceptedReceipts,
}) {
  if (expectTraining && useProductionTrainingProfile && minAcceptedReceipts < 1) {
    throw new Error("canonical training canary requires at least one accepted browser receipt");
  }
  if (expectTraining && !useProductionTrainingProfile && minAcceptedReceipts !== 0) {
    throw new Error("local WebGPU training smoke cannot require canonical browser receipts");
  }
}

export function applyBrowserCanaryProfile(
  browserConfig,
  {
    expectTraining = false,
    expectCheckpointSync = false,
    useProductionTrainingProfile = false,
  } = {},
) {
  if (!browserConfig || typeof browserConfig !== "object") {
    return browserConfig;
  }
  if (expectTraining && expectCheckpointSync) {
    throw new Error("browser canary cannot train and verify checkpoint sync in the same lane");
  }

  const profiled = JSON.parse(JSON.stringify(browserConfig));
  const training = browserConfigTrainingConfig(profiled);
  if (!training) {
    return profiled;
  }

  if (expectCheckpointSync) {
    return profiled;
  }

  if (!expectTraining) {
    if (training.live_participant && typeof training.live_participant === "object") {
      training.live_participant.publish_canonical_update = false;
      training.live_participant.load_active_head_artifact = false;
    }
    return profiled;
  }

  if (useProductionTrainingProfile) {
    if (training.live_participant && typeof training.live_participant === "object") {
      training.live_participant.publish_canonical_update = false;
    }
    return profiled;
  }

  training.max_train_batches = 1;
  training.max_eval_batches = 0;
  training.block_size = Math.min(Number(training.block_size ?? 32) || 32, 32);
  if (training.model_config && typeof training.model_config === "object") {
    training.model_config.n_embd = 16;
    training.model_config.n_head = 1;
    training.model_config.n_layer = 1;
    training.model_config.n_expert = 1;
    training.model_config.mlp_internal_dim_multiplier = 2;
    if (training.model_config.mhc && typeof training.model_config.mhc === "object") {
      training.model_config.mhc.enabled = false;
    }
    if (
      training.model_config.attention_residual &&
      typeof training.model_config.attention_residual === "object"
    ) {
      training.model_config.attention_residual.enabled = false;
    }
    if (
      training.model_config.block_attention_residual &&
      typeof training.model_config.block_attention_residual === "object"
    ) {
      training.model_config.block_attention_residual.enabled = false;
    }
    if (training.model_config.fused_kernels && typeof training.model_config.fused_kernels === "object") {
      training.model_config.fused_kernels.enabled = false;
    }
  }
  // The bounded profile validates local WebGPU execution only. It must not pose
  // as a participant in an authority-signed canonical revision.
  training.live_participant = null;
  return profiled;
}
