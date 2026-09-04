import assert from "node:assert/strict";
import test from "node:test";

import {
  applyBrowserCanaryProfile,
  browserConfigTrainingConfig,
  validateBrowserCanaryTrainingPolicy,
} from "./live-browser-canary-profile.mjs";

function browserConfigFixture() {
  return {
    config: {
      network: {
        seed_node_urls: ["dns4/example.test/tcp/443/wss"],
      },
      training: {
        block_size: 256,
        max_train_batches: 8,
        max_eval_batches: 4,
        model_config: {
          n_embd: 512,
          n_head: 8,
          n_layer: 12,
          n_expert: 4,
          mlp_internal_dim_multiplier: 8,
          mhc: { enabled: true },
          attention_residual: { enabled: true },
          block_attention_residual: { enabled: true },
          fused_kernels: { enabled: true },
        },
        live_participant: {
          publish_canonical_update: true,
          load_active_head_artifact: true,
        },
      },
    },
  };
}

test("connect-only profile prevents artifact traffic without mutating production config", () => {
  const source = browserConfigFixture();
  const profiled = applyBrowserCanaryProfile(source);
  const training = browserConfigTrainingConfig(profiled);

  assert.notEqual(profiled, source);
  assert.equal(training.live_participant.publish_canonical_update, false);
  assert.equal(training.live_participant.load_active_head_artifact, false);
  assert.equal(training.model_config.n_embd, 512);
  assert.equal(source.config.training.live_participant.publish_canonical_update, true);
  assert.equal(source.config.training.live_participant.load_active_head_artifact, true);
});

test("checkpoint profile preserves production artifact loading policy", () => {
  const source = browserConfigFixture();
  const profiled = applyBrowserCanaryProfile(source, {
    expectCheckpointSync: true,
  });

  assert.notEqual(profiled, source);
  assert.deepEqual(profiled, source);
});

test("lightweight training profile is bounded and detached from canonical participation", () => {
  const source = browserConfigFixture();
  const profiled = applyBrowserCanaryProfile(source, {
    expectTraining: true,
  });
  const training = browserConfigTrainingConfig(profiled);

  assert.equal(training.block_size, 32);
  assert.equal(training.max_train_batches, 1);
  assert.equal(training.max_eval_batches, 0);
  assert.equal(training.model_config.n_embd, 16);
  assert.equal(training.model_config.n_head, 1);
  assert.equal(training.model_config.n_layer, 1);
  assert.equal(training.model_config.n_expert, 1);
  assert.equal(training.model_config.mlp_internal_dim_multiplier, 2);
  assert.equal(training.model_config.mhc.enabled, false);
  assert.equal(training.model_config.attention_residual.enabled, false);
  assert.equal(training.model_config.block_attention_residual.enabled, false);
  assert.equal(training.model_config.fused_kernels.enabled, false);
  assert.equal(training.live_participant, null);
});

test("production training profile preserves model and head loading while preventing publication", () => {
  const source = browserConfigFixture();
  const expected = browserConfigFixture();
  expected.config.training.live_participant.publish_canonical_update = false;
  const profiled = applyBrowserCanaryProfile(source, {
    expectTraining: true,
    useProductionTrainingProfile: true,
  });
  const training = browserConfigTrainingConfig(profiled);

  assert.deepEqual(profiled, expected);
  assert.equal(training.live_participant.publish_canonical_update, false);
  assert.equal(training.live_participant.load_active_head_artifact, true);
  assert.equal(source.config.training.live_participant.publish_canonical_update, true);
});

test("combined training and checkpoint expectations are rejected", () => {
  assert.throws(
    () =>
      applyBrowserCanaryProfile(browserConfigFixture(), {
        expectTraining: true,
        expectCheckpointSync: true,
      }),
    /cannot train and verify checkpoint sync/,
  );
});

test("training receipt policy separates local smoke from canonical participation", () => {
  assert.doesNotThrow(() =>
    validateBrowserCanaryTrainingPolicy({
      expectTraining: true,
      useProductionTrainingProfile: false,
      minAcceptedReceipts: 0,
    }),
  );
  assert.doesNotThrow(() =>
    validateBrowserCanaryTrainingPolicy({
      expectTraining: true,
      useProductionTrainingProfile: true,
      minAcceptedReceipts: 2,
    }),
  );
  assert.throws(
    () =>
      validateBrowserCanaryTrainingPolicy({
        expectTraining: true,
        useProductionTrainingProfile: false,
        minAcceptedReceipts: 1,
      }),
    /local WebGPU training smoke cannot require canonical browser receipts/,
  );
  assert.throws(
    () =>
      validateBrowserCanaryTrainingPolicy({
        expectTraining: true,
        useProductionTrainingProfile: true,
        minAcceptedReceipts: 0,
      }),
    /canonical training canary requires at least one accepted browser receipt/,
  );
});
