#![allow(clippy::let_unit_value)]
use super::*;
use burn::tensor::Int;
use burn::tensor::backend::AutodiffBackend;
use burn_cubecl::cubecl::prelude::Tensor;

const BACKWARD_META_LEN: usize = 5;
const BACKWARD_WORKGROUP_SIZE_X: u32 = 64;
const BACKWARD_WGPU_WORKGROUP_SIZE: usize = BACKWARD_WORKGROUP_SIZE_X as usize;
const BACKWARD_WGPU_EMBD_TILE: usize = 16;
const BACKWARD_WGPU_LATENT_TILE: usize = 16;
#[cfg(feature = "cuda")]
const STREAMING_BACKWARD_META_LEN: usize = 9;
#[cfg(feature = "cuda")]
const STREAMING_BACKWARD_WORKGROUP_SIZE_X: u32 = 128;
#[cfg(feature = "cuda")]
const STREAMING_BACKWARD_WORKGROUP_SIZE: usize = STREAMING_BACKWARD_WORKGROUP_SIZE_X as usize;
#[cfg(feature = "cuda")]
const STREAMING_QUERY_WORKGROUP_SIZE_X: u32 = 32;
#[cfg(feature = "cuda")]
const STREAMING_QUERY_WORKGROUP_SIZE: usize = STREAMING_QUERY_WORKGROUP_SIZE_X as usize;

/// Plain-backend input VJP for the fused recurrent linear-attention kernel.
///
/// The decay schedule is treated as fixed because Dragon's local predictive-
/// coding factors do not optimize ALiBi slopes. `grad_initial_rho` is returned
/// only when the caller supplied an incoming recurrent state.
#[derive(Debug, Clone)]
pub struct RecurrentAttentionInputVjp<B: BackendTrait> {
    pub grad_query: BurnTensor<B, 4>,
    pub grad_value: BurnTensor<B, 4>,
    pub grad_initial_rho: Option<BurnTensor<B, 4>>,
}

#[cube]
fn reduce_partials_wgpu(
    partials: &mut SharedMemory<f32>,
    lane: usize,
    #[comptime] workgroup_size: usize,
) {
    if comptime!(workgroup_size >= 128usize) {
        if lane < 64usize {
            partials[lane] = partials[lane] + partials[lane + 64usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 64usize) {
        if lane < 32usize {
            partials[lane] = partials[lane] + partials[lane + 32usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 32usize) {
        if lane < 16usize {
            partials[lane] = partials[lane] + partials[lane + 16usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 16usize) {
        if lane < 8usize {
            partials[lane] = partials[lane] + partials[lane + 8usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 8usize) {
        if lane < 4usize {
            partials[lane] = partials[lane] + partials[lane + 4usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 4usize) {
        if lane < 2usize {
            partials[lane] = partials[lane] + partials[lane + 2usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 2usize) {
        if lane < 1usize {
            partials[lane] = partials[lane] + partials[lane + 1usize];
        }
        sync_cube();
    }
}

/// Exact history-free query and incoming-state VJP for recurrent attention.
///
/// The context contribution to `dL/dq_t` depends only on the forward state
/// entering token `t`, so it can be accumulated in a forward scan. The state
/// transition contribution depends only on the reverse adjoint, so it can be
/// accumulated in a reverse scan. Keeping both scans in one workgroup avoids
/// materializing the full `[batch, head, time, latent, embedding]` history.
#[cfg(feature = "cuda")]
#[cube(launch)]
fn recurrent_attention_streaming_grad_query_cuda_kernel(
    query: &Tensor<f32>,
    value: &Tensor<f32>,
    initial_rho: &Tensor<f32>,
    grad_output: &Tensor<f32>,
    grad_terminal_rho: &Tensor<f32>,
    decay: &Tensor<f32>,
    grad_query_partial: &mut Tensor<f32>,
    grad_initial_rho: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let value_heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;
    let embd_groups = u32::cast_from(params[6]) as usize;
    let has_terminal_rho = params[8] > f32::cast_from(0u32);

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let packed = CUBE_POS_X as usize;
    let l = packed / embd_groups;
    let embd_group = packed % embd_groups;
    let lane = UNIT_POS_X as usize;
    let e = embd_group * STREAMING_QUERY_WORKGROUP_SIZE + lane;
    if b >= batch || h >= heads || l >= latent {
        terminate!();
    }

    let active_e = e < embd;
    let mut value_head = h;
    if value_heads == 1usize {
        value_head = 0usize;
    }
    let decay_value = decay[h * decay.stride(0)];
    let rho_index = b * initial_rho.stride(0)
        + h * initial_rho.stride(1)
        + l * initial_rho.stride(2)
        + e * initial_rho.stride(3);
    let mut forward_state = if active_e {
        initial_rho[rho_index]
    } else {
        f32::cast_from(0u32)
    };
    // Context path: c_t[e] = sum_l rho_t[l,e] * q_t[l].
    let mut t = 0usize;
    while t < time {
        let query_index =
            b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
        let q = query[query_index];
        let value_index = b * value.stride(0)
            + value_head * value.stride(1)
            + t * value.stride(2)
            + e * value.stride(3);
        let grad_output_index = b * grad_output.stride(0)
            + h * grad_output.stride(1)
            + t * grad_output.stride(2)
            + e * grad_output.stride(3);
        let value_t = if active_e {
            value[value_index]
        } else {
            f32::cast_from(0u32)
        };
        let grad_out = if active_e {
            grad_output[grad_output_index]
        } else {
            f32::cast_from(0u32)
        };
        let partial = plane_sum(forward_state * grad_out);
        if lane == 0usize {
            let output_index = b * grad_query_partial.stride(0)
                + h * grad_query_partial.stride(1)
                + embd_group * grad_query_partial.stride(2)
                + t * grad_query_partial.stride(3)
                + l * grad_query_partial.stride(4);
            grad_query_partial[output_index] = partial;
        }
        if active_e {
            forward_state = (forward_state + q * value_t) * decay_value;
        }
        t += 1usize;
    }

    // Transition path: rho_{t+1} = d * (rho_t + q_t outer v_t).
    let terminal_index = b * grad_terminal_rho.stride(0)
        + h * grad_terminal_rho.stride(1)
        + l * grad_terminal_rho.stride(2)
        + e * grad_terminal_rho.stride(3);
    let mut reverse_state = if active_e && has_terminal_rho {
        grad_terminal_rho[terminal_index] * decay_value
    } else {
        f32::cast_from(0u32)
    };
    let mut tau = 0usize;
    while tau < time {
        let reverse_t = time - 1usize - tau;
        let query_index = b * query.stride(0)
            + h * query.stride(1)
            + reverse_t * query.stride(2)
            + l * query.stride(3);
        let q = query[query_index];
        let value_index = b * value.stride(0)
            + value_head * value.stride(1)
            + reverse_t * value.stride(2)
            + e * value.stride(3);
        let grad_output_index = b * grad_output.stride(0)
            + h * grad_output.stride(1)
            + reverse_t * grad_output.stride(2)
            + e * grad_output.stride(3);
        let value_t = if active_e {
            value[value_index]
        } else {
            f32::cast_from(0u32)
        };
        let grad_out = if active_e {
            grad_output[grad_output_index]
        } else {
            f32::cast_from(0u32)
        };
        let partial = plane_sum(reverse_state * value_t);
        if lane == 0usize {
            let output_index = b * grad_query_partial.stride(0)
                + h * grad_query_partial.stride(1)
                + embd_group * grad_query_partial.stride(2)
                + reverse_t * grad_query_partial.stride(3)
                + l * grad_query_partial.stride(4);
            grad_query_partial[output_index] = grad_query_partial[output_index] + partial;
        }
        if active_e {
            reverse_state = (reverse_state + q * grad_out) * decay_value;
        }
        tau += 1usize;
    }

    if active_e {
        let grad_initial_index = ((b * heads + h) * latent + l) * embd + e;
        grad_initial_rho[grad_initial_index] = reverse_state / decay_value;
    }
}

/// Exact history-free value VJP. Each workgroup owns one latent tile and
/// emits a compact partial; a final device reduction combines latent tiles.
#[cfg(feature = "cuda")]
#[cube(launch)]
fn recurrent_attention_streaming_grad_value_cuda_kernel(
    query: &Tensor<f32>,
    grad_output: &Tensor<f32>,
    grad_terminal_rho: &Tensor<f32>,
    decay: &Tensor<f32>,
    grad_value_partial: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;
    let latent_groups = u32::cast_from(params[7]) as usize;
    let has_terminal_rho = params[8] > f32::cast_from(0u32);

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let packed = CUBE_POS_X as usize;
    let e = packed / latent_groups;
    let latent_group = packed % latent_groups;
    let lane = UNIT_POS_X as usize;
    let l = latent_group * STREAMING_BACKWARD_WORKGROUP_SIZE + lane;
    if b >= batch || h >= heads || e >= embd {
        terminate!();
    }

    let active_l = l < latent;
    let decay_value = decay[h * decay.stride(0)];
    let terminal_index = b * grad_terminal_rho.stride(0)
        + h * grad_terminal_rho.stride(1)
        + l * grad_terminal_rho.stride(2)
        + e * grad_terminal_rho.stride(3);
    let mut reverse_state = if active_l && has_terminal_rho {
        grad_terminal_rho[terminal_index] * decay_value
    } else {
        f32::cast_from(0u32)
    };
    let mut partials = SharedMemory::<f32>::new_aligned(STREAMING_BACKWARD_WORKGROUP_SIZE, 1usize);

    let mut tau = 0usize;
    while tau < time {
        let reverse_t = time - 1usize - tau;
        let query_index = b * query.stride(0)
            + h * query.stride(1)
            + reverse_t * query.stride(2)
            + l * query.stride(3);
        let grad_output_index = b * grad_output.stride(0)
            + h * grad_output.stride(1)
            + reverse_t * grad_output.stride(2)
            + e * grad_output.stride(3);
        let q = if active_l {
            query[query_index]
        } else {
            f32::cast_from(0u32)
        };
        let grad_out = if active_l {
            grad_output[grad_output_index]
        } else {
            f32::cast_from(0u32)
        };
        partials[lane] = reverse_state * q;
        sync_cube();
        reduce_partials_wgpu(&mut partials, lane, STREAMING_BACKWARD_WORKGROUP_SIZE);
        if lane == 0usize {
            let output_index = b * grad_value_partial.stride(0)
                + h * grad_value_partial.stride(1)
                + latent_group * grad_value_partial.stride(2)
                + reverse_t * grad_value_partial.stride(3)
                + e * grad_value_partial.stride(4);
            grad_value_partial[output_index] = partials[0usize];
        }
        sync_cube();
        if active_l {
            reverse_state = (reverse_state + q * grad_out) * decay_value;
        }
        tau += 1usize;
    }
}

#[cube(launch)]
fn recurrent_attention_grad_query_decay_grouped_wgpu_kernel(
    forward_state: &Tensor<f32>,
    reverse_state_rev: &Tensor<f32>,
    grad_output: &Tensor<f32>,
    value: &Tensor<f32>,
    query: &Tensor<f32>,
    decay: &Tensor<f32>,
    grad_query: &mut Tensor<f32>,
    grad_decay_grouped: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] embd_tile: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / time.max(1);
    let t = z % time.max(1);
    let h = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let latent_group = CUBE_POS_X as usize;
    let l = latent_group * BACKWARD_WGPU_WORKGROUP_SIZE + lane;
    if b >= batch || h >= heads || t >= time {
        terminate!();
    }

    let active_l = l < latent;
    let tau = time - 1usize - t;
    let decay_value = decay[h * decay.stride(0)];
    let eps = f32::cast_from(1.0e-8f32);
    let neg_eps = f32::cast_from(-1.0e-8f32);
    let safe_decay = if decay_value < eps && decay_value > neg_eps {
        eps
    } else {
        decay_value
    };

    let q = if active_l {
        let query_index =
            b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
        query[query_index]
    } else {
        f32::cast_from(0u32)
    };

    let mut grad_output_tile = SharedMemory::<f32>::new_aligned(embd_tile, 1usize);
    let mut value_tile = SharedMemory::<f32>::new_aligned(embd_tile, 1usize);
    let mut decay_partials = SharedMemory::<f32>::new_aligned(BACKWARD_WGPU_WORKGROUP_SIZE, 1usize);

    let mut grad_query_acc = f32::cast_from(0u32);
    let mut grad_decay_acc = f32::cast_from(0u32);
    let mut embd_base = 0usize;
    while embd_base < embd {
        let mut load_offset = lane;
        while load_offset < embd_tile {
            let e = embd_base + load_offset;
            if e < embd {
                let grad_output_index = b * grad_output.stride(0)
                    + h * grad_output.stride(1)
                    + t * grad_output.stride(2)
                    + e * grad_output.stride(3);
                let value_index = b * value.stride(0)
                    + h * value.stride(1)
                    + t * value.stride(2)
                    + e * value.stride(3);
                grad_output_tile[load_offset] = grad_output[grad_output_index];
                value_tile[load_offset] = value[value_index];
            } else {
                grad_output_tile[load_offset] = f32::cast_from(0u32);
                value_tile[load_offset] = f32::cast_from(0u32);
            }
            load_offset += BACKWARD_WGPU_WORKGROUP_SIZE;
        }
        sync_cube();

        if active_l {
            let mut tile_offset = 0usize;
            while tile_offset < embd_tile {
                let e = embd_base + tile_offset;
                if e < embd {
                    let forward_index = b * forward_state.stride(0)
                        + h * forward_state.stride(1)
                        + t * forward_state.stride(2)
                        + l * forward_state.stride(3)
                        + e * forward_state.stride(4);
                    let reverse_index = b * reverse_state_rev.stride(0)
                        + h * reverse_state_rev.stride(1)
                        + tau * reverse_state_rev.stride(2)
                        + l * reverse_state_rev.stride(3)
                        + e * reverse_state_rev.stride(4);
                    let forward = forward_state[forward_index];
                    let reverse = reverse_state_rev[reverse_index];
                    let grad_out = grad_output_tile[tile_offset];
                    let value_at_t = value_tile[tile_offset];
                    grad_query_acc += forward * grad_out + reverse * value_at_t;
                    grad_decay_acc += (reverse / safe_decay) * (forward + q * value_at_t);
                }
                tile_offset += 1usize;
            }
        }

        sync_cube();
        embd_base += embd_tile;
    }

    if active_l {
        let out_index = b * grad_query.stride(0)
            + h * grad_query.stride(1)
            + t * grad_query.stride(2)
            + l * grad_query.stride(3);
        grad_query[out_index] = grad_query_acc;
    }

    decay_partials[lane] = if active_l {
        grad_decay_acc
    } else {
        f32::cast_from(0u32)
    };
    sync_cube();
    reduce_partials_wgpu(&mut decay_partials, lane, BACKWARD_WGPU_WORKGROUP_SIZE);
    if lane == 0usize {
        let out_index = b * grad_decay_grouped.stride(0)
            + h * grad_decay_grouped.stride(1)
            + t * grad_decay_grouped.stride(2)
            + latent_group * grad_decay_grouped.stride(3);
        grad_decay_grouped[out_index] = decay_partials[0usize];
    }
}

#[cube(launch)]
fn recurrent_attention_grad_value_tiled_wgpu_kernel(
    reverse_state_rev: &Tensor<f32>,
    query: &Tensor<f32>,
    grad_value: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] latent_tile: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / time.max(1);
    let t = z % time.max(1);
    let h = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || e >= embd {
        terminate!();
    }

    let tau = time - 1usize - t;
    let mut query_tile = SharedMemory::<f32>::new_aligned(latent_tile, 1usize);
    let mut grad = f32::cast_from(0u32);
    let mut latent_base = 0usize;
    while latent_base < latent {
        let mut load_offset = lane;
        while load_offset < latent_tile {
            let l = latent_base + load_offset;
            if l < latent {
                let query_index = b * query.stride(0)
                    + h * query.stride(1)
                    + t * query.stride(2)
                    + l * query.stride(3);
                query_tile[load_offset] = query[query_index];
            } else {
                query_tile[load_offset] = f32::cast_from(0u32);
            }
            load_offset += BACKWARD_WGPU_WORKGROUP_SIZE;
        }
        sync_cube();

        let mut tile_offset = 0usize;
        while tile_offset < latent_tile {
            let l = latent_base + tile_offset;
            if l < latent {
                let reverse_index = b * reverse_state_rev.stride(0)
                    + h * reverse_state_rev.stride(1)
                    + tau * reverse_state_rev.stride(2)
                    + l * reverse_state_rev.stride(3)
                    + e * reverse_state_rev.stride(4);
                grad += reverse_state_rev[reverse_index] * query_tile[tile_offset];
            }
            tile_offset += 1usize;
        }

        sync_cube();
        latent_base += latent_tile;
    }

    let out_index = b * grad_value.stride(0)
        + h * grad_value.stride(1)
        + t * grad_value.stride(2)
        + e * grad_value.stride(3);
    grad_value[out_index] = grad;
}

#[cube(launch)]
fn recurrent_attention_grad_value_reduced_tiled_wgpu_kernel(
    reverse_state_rev: &Tensor<f32>,
    query: &Tensor<f32>,
    grad_value: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] latent_tile: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;

    let b = CUBE_POS_Z as usize;
    let t = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || t >= time || e >= embd {
        terminate!();
    }

    let tau = time - 1usize - t;
    let mut query_tile = SharedMemory::<f32>::new_aligned(latent_tile, 1usize);
    let mut grad = f32::cast_from(0u32);

    let mut h = 0usize;
    while h < heads {
        let mut latent_base = 0usize;
        while latent_base < latent {
            let mut load_offset = lane;
            while load_offset < latent_tile {
                let l = latent_base + load_offset;
                if l < latent {
                    let query_index = b * query.stride(0)
                        + h * query.stride(1)
                        + t * query.stride(2)
                        + l * query.stride(3);
                    query_tile[load_offset] = query[query_index];
                } else {
                    query_tile[load_offset] = f32::cast_from(0u32);
                }
                load_offset += BACKWARD_WGPU_WORKGROUP_SIZE;
            }
            sync_cube();

            let mut tile_offset = 0usize;
            while tile_offset < latent_tile {
                let l = latent_base + tile_offset;
                if l < latent {
                    let reverse_index = b * reverse_state_rev.stride(0)
                        + h * reverse_state_rev.stride(1)
                        + tau * reverse_state_rev.stride(2)
                        + l * reverse_state_rev.stride(3)
                        + e * reverse_state_rev.stride(4);
                    grad += reverse_state_rev[reverse_index] * query_tile[tile_offset];
                }
                tile_offset += 1usize;
            }

            sync_cube();
            latent_base += latent_tile;
        }
        h += 1usize;
    }

    let out_index = b * grad_value.stride(0) + t * grad_value.stride(2) + e * grad_value.stride(3);
    grad_value[out_index] = grad;
}

#[cube(launch)]
fn recurrent_attention_grad_query_kernel(
    forward_state: &Tensor<f32>,
    reverse_state_rev: &Tensor<f32>,
    grad_output: &Tensor<f32>,
    value: &Tensor<f32>,
    grad_query: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / time.max(1);
    let t = z % time.max(1);
    let h = CUBE_POS_Y as usize;
    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || l >= latent {
        terminate!();
    }

    let tau = time - 1usize - t;
    let mut grad = f32::cast_from(0u32);
    let mut e = 0usize;
    while e < embd {
        let forward_index = b * forward_state.stride(0)
            + h * forward_state.stride(1)
            + t * forward_state.stride(2)
            + l * forward_state.stride(3)
            + e * forward_state.stride(4);
        let reverse_index = b * reverse_state_rev.stride(0)
            + h * reverse_state_rev.stride(1)
            + tau * reverse_state_rev.stride(2)
            + l * reverse_state_rev.stride(3)
            + e * reverse_state_rev.stride(4);
        let grad_output_index = b * grad_output.stride(0)
            + h * grad_output.stride(1)
            + t * grad_output.stride(2)
            + e * grad_output.stride(3);
        let value_index =
            b * value.stride(0) + h * value.stride(1) + t * value.stride(2) + e * value.stride(3);
        grad += forward_state[forward_index] * grad_output[grad_output_index]
            + reverse_state_rev[reverse_index] * value[value_index];
        e += 1usize;
    }

    let out_index = b * grad_query.stride(0)
        + h * grad_query.stride(1)
        + t * grad_query.stride(2)
        + l * grad_query.stride(3);
    grad_query[out_index] = grad;
}

#[cube(launch)]
fn recurrent_attention_grad_value_kernel(
    reverse_state_rev: &Tensor<f32>,
    query: &Tensor<f32>,
    grad_value: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / time.max(1);
    let t = z % time.max(1);
    let h = CUBE_POS_Y as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || e >= embd {
        terminate!();
    }

    let tau = time - 1usize - t;
    let mut grad = f32::cast_from(0u32);
    let mut l = 0usize;
    while l < latent {
        let reverse_index = b * reverse_state_rev.stride(0)
            + h * reverse_state_rev.stride(1)
            + tau * reverse_state_rev.stride(2)
            + l * reverse_state_rev.stride(3)
            + e * reverse_state_rev.stride(4);
        let query_index =
            b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
        grad += reverse_state_rev[reverse_index] * query[query_index];
        l += 1usize;
    }

    let out_index = b * grad_value.stride(0)
        + h * grad_value.stride(1)
        + t * grad_value.stride(2)
        + e * grad_value.stride(3);
    grad_value[out_index] = grad;
}

#[cube(launch)]
fn recurrent_attention_grad_decay_kernel(
    forward_state: &Tensor<f32>,
    reverse_state_rev: &Tensor<f32>,
    query: &Tensor<f32>,
    value: &Tensor<f32>,
    decay: &Tensor<f32>,
    grad_decay_partial: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let t = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time {
        terminate!();
    }

    let tau = time - 1usize - t;
    let decay_value = decay[h * decay.stride(0)];
    let eps = f32::cast_from(1.0e-8f32);
    let neg_eps = f32::cast_from(-1.0e-8f32);
    let safe_decay = if decay_value < eps && decay_value > neg_eps {
        eps
    } else {
        decay_value
    };

    let mut grad = f32::cast_from(0u32);
    let mut l = 0usize;
    while l < latent {
        let query_index =
            b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
        let q = query[query_index];
        let mut e = 0usize;
        while e < embd {
            let forward_index = b * forward_state.stride(0)
                + h * forward_state.stride(1)
                + t * forward_state.stride(2)
                + l * forward_state.stride(3)
                + e * forward_state.stride(4);
            let reverse_index = b * reverse_state_rev.stride(0)
                + h * reverse_state_rev.stride(1)
                + tau * reverse_state_rev.stride(2)
                + l * reverse_state_rev.stride(3)
                + e * reverse_state_rev.stride(4);
            let value_index = b * value.stride(0)
                + h * value.stride(1)
                + t * value.stride(2)
                + e * value.stride(3);
            grad += (reverse_state_rev[reverse_index] / safe_decay)
                * (forward_state[forward_index] + q * value[value_index]);
            e += 1usize;
        }
        l += 1usize;
    }

    let out_index = b * grad_decay_partial.stride(0)
        + h * grad_decay_partial.stride(1)
        + t * grad_decay_partial.stride(2);
    grad_decay_partial[out_index] = grad;
}

#[cfg(feature = "cuda")]
fn streaming_backward_params_tensor<B: BackendTrait>(
    batch: usize,
    heads: usize,
    value_heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
    embd_groups: usize,
    latent_groups: usize,
    has_terminal_rho: bool,
    device: &B::Device,
) -> BurnTensor<B, 1> {
    BurnTensor::<B, 1>::from_data(
        TensorData::new(
            vec![
                batch as f32,
                heads as f32,
                value_heads as f32,
                time as f32,
                latent as f32,
                embd as f32,
                embd_groups as f32,
                latent_groups as f32,
                u8::from(has_terminal_rho) as f32,
            ],
            [STREAMING_BACKWARD_META_LEN],
        ),
        device,
    )
}

#[cfg(feature = "cuda")]
fn recurrent_attention_streaming_grad_query_cuda_runtime(
    query: CubeTensor<CudaRuntime>,
    value: CubeTensor<CudaRuntime>,
    initial_rho: CubeTensor<CudaRuntime>,
    grad_output: CubeTensor<CudaRuntime>,
    grad_terminal_rho: CubeTensor<CudaRuntime>,
    decay: CubeTensor<CudaRuntime>,
    params: CubeTensor<CudaRuntime>,
    embd_groups: usize,
) -> (CubeTensor<CudaRuntime>, CubeTensor<CudaRuntime>) {
    let [batch, heads, time, latent] = query.meta.shape.dims::<4>();
    let embd = grad_output.meta.shape.dims::<4>()[3];
    let client = query.client.clone();
    let device = query.device.clone();
    let grad_query_partial = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, embd_groups, time, latent]),
    );
    let grad_initial_rho = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device,
        Shape::new([batch * heads * latent * embd]),
    );
    let cube_dim = CubeDim::new_1d(STREAMING_QUERY_WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static((latent * embd_groups) as u32, heads as u32, batch as u32);
    let _ = recurrent_attention_streaming_grad_query_cuda_kernel::launch::<CudaRuntime>(
        &client,
        cube_count,
        cube_dim,
        into_contiguous(query).into_tensor_arg(),
        into_contiguous(value).into_tensor_arg(),
        into_contiguous(initial_rho).into_tensor_arg(),
        into_contiguous(grad_output).into_tensor_arg(),
        into_contiguous(grad_terminal_rho).into_tensor_arg(),
        into_contiguous(decay).into_tensor_arg(),
        grad_query_partial.clone().into_tensor_arg(),
        grad_initial_rho.clone().into_tensor_arg(),
        into_contiguous(params).into_tensor_arg(),
    );
    (grad_query_partial, grad_initial_rho)
}

#[cfg(feature = "cuda")]
fn recurrent_attention_streaming_grad_value_cuda_runtime(
    query: CubeTensor<CudaRuntime>,
    grad_output: CubeTensor<CudaRuntime>,
    grad_terminal_rho: CubeTensor<CudaRuntime>,
    decay: CubeTensor<CudaRuntime>,
    params: CubeTensor<CudaRuntime>,
    latent_groups: usize,
) -> CubeTensor<CudaRuntime> {
    let [batch, heads, time, _] = query.meta.shape.dims::<4>();
    let embd = grad_output.meta.shape.dims::<4>()[3];
    let client = query.client.clone();
    let device = query.device.clone();
    let grad_value_partial = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, latent_groups, time, embd]),
    );
    let cube_dim = CubeDim::new_1d(STREAMING_BACKWARD_WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static((embd * latent_groups) as u32, heads as u32, batch as u32);
    let _ = recurrent_attention_streaming_grad_value_cuda_kernel::launch::<CudaRuntime>(
        &client,
        cube_count,
        cube_dim,
        into_contiguous(query).into_tensor_arg(),
        into_contiguous(grad_output).into_tensor_arg(),
        into_contiguous(grad_terminal_rho).into_tensor_arg(),
        into_contiguous(decay).into_tensor_arg(),
        grad_value_partial.clone().into_tensor_arg(),
        into_contiguous(params).into_tensor_arg(),
    );
    grad_value_partial
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn finish_streaming_recurrent_attention_input_vjp<B: BackendTrait>(
    grad_query_partial: BurnTensor<B, 5>,
    grad_value_partial: BurnTensor<B, 5>,
    grad_initial_rho_flat: BurnTensor<B, 1>,
    batch: usize,
    heads: usize,
    value_heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
    embd_groups: usize,
    latent_groups: usize,
    had_initial_rho: bool,
) -> RecurrentAttentionInputVjp<B> {
    let grad_query = if embd_groups == 1 {
        grad_query_partial.reshape([batch, heads, time, latent])
    } else {
        grad_query_partial
            .sum_dim(2)
            .reshape([batch, heads, time, latent])
    };
    let grad_value_heads = if latent_groups == 1 {
        grad_value_partial.reshape([batch, heads, time, embd])
    } else {
        grad_value_partial
            .sum_dim(2)
            .reshape([batch, heads, time, embd])
    };
    let grad_value = if value_heads == heads {
        grad_value_heads
    } else {
        grad_value_heads.sum_dim(1).reshape([batch, 1, time, embd])
    };
    let grad_initial_rho =
        had_initial_rho.then(|| grad_initial_rho_flat.reshape([batch, heads, latent, embd]));

    RecurrentAttentionInputVjp {
        grad_query,
        grad_value,
        grad_initial_rho,
    }
}

#[cfg(feature = "cuda")]
fn try_streaming_recurrent_attention_input_vjp_cuda<B: BackendTrait>(
    grad_output: BurnTensor<B, 4>,
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    initial_rho: BurnTensor<B, 4>,
    decay: BurnTensor<B, 1>,
    grad_terminal_rho: Option<BurnTensor<B, 4>>,
    had_initial_rho: bool,
) -> Option<RecurrentAttentionInputVjp<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let value_heads = value.shape().dims::<4>()[1];
    let embd = grad_output.shape().dims::<4>()[3];
    let embd_groups = embd.div_ceil(STREAMING_QUERY_WORKGROUP_SIZE);
    let latent_groups = latent.div_ceil(STREAMING_BACKWARD_WORKGROUP_SIZE);
    let has_terminal_rho = grad_terminal_rho.is_some();
    let terminal = grad_terminal_rho.unwrap_or_else(|| initial_rho.clone());
    let params = streaming_backward_params_tensor::<B>(
        batch,
        heads,
        value_heads,
        time,
        latent,
        embd,
        embd_groups,
        latent_groups,
        has_terminal_rho,
        &query.device(),
    );

    let query_cube =
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(query.clone().into_primitive().tensor())?;
    let value_cube =
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(value.into_primitive().tensor())?;
    let initial_cube =
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(initial_rho.into_primitive().tensor())?;
    let grad_output_cube = try_cast_primitive::<B, CubeTensor<CudaRuntime>>(
        grad_output.clone().into_primitive().tensor(),
    )?;
    let terminal_cube =
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(terminal.into_primitive().tensor())?;
    let decay_cube =
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(decay.into_primitive().tensor())?;
    let params_cube =
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(params.into_primitive().tensor())?;
    if query_cube.dtype != DType::F32
        || value_cube.dtype != DType::F32
        || initial_cube.dtype != DType::F32
        || grad_output_cube.dtype != DType::F32
        || terminal_cube.dtype != DType::F32
        || decay_cube.dtype != DType::F32
        || params_cube.dtype != DType::F32
    {
        return None;
    }

    let (grad_query_partial, grad_initial_rho) =
        recurrent_attention_streaming_grad_query_cuda_runtime(
            query_cube.clone(),
            value_cube,
            initial_cube,
            grad_output_cube.clone(),
            terminal_cube.clone(),
            decay_cube.clone(),
            params_cube.clone(),
            embd_groups,
        );
    let grad_value_partial = recurrent_attention_streaming_grad_value_cuda_runtime(
        query_cube,
        grad_output_cube,
        terminal_cube,
        decay_cube,
        params_cube,
        latent_groups,
    );

    let grad_query_partial = BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(
        try_cast_backend::<B, _>(grad_query_partial)?,
    ));
    let grad_value_partial = BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(
        try_cast_backend::<B, _>(grad_value_partial)?,
    ));
    let grad_initial_rho_flat = BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(
        try_cast_backend::<B, _>(grad_initial_rho)?,
    ));

    Some(finish_streaming_recurrent_attention_input_vjp(
        grad_query_partial,
        grad_value_partial,
        grad_initial_rho_flat,
        batch,
        heads,
        value_heads,
        time,
        latent,
        embd,
        embd_groups,
        latent_groups,
        had_initial_rho,
    ))
}

#[cfg(feature = "cuda")]
fn try_streaming_recurrent_attention_input_vjp_cuda_fusion<B, BT>(
    grad_output: BurnTensor<B, 4>,
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    initial_rho: BurnTensor<B, 4>,
    decay: BurnTensor<B, 1>,
    grad_terminal_rho: Option<BurnTensor<B, 4>>,
    had_initial_rho: bool,
) -> Option<RecurrentAttentionInputVjp<B>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<CudaRuntime>>>() {
        return None;
    }

    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let value_heads = value.shape().dims::<4>()[1];
    let embd = grad_output.shape().dims::<4>()[3];
    let embd_groups = embd.div_ceil(STREAMING_QUERY_WORKGROUP_SIZE);
    let latent_groups = latent.div_ceil(STREAMING_BACKWARD_WORKGROUP_SIZE);
    let has_terminal_rho = grad_terminal_rho.is_some();
    let terminal = grad_terminal_rho.unwrap_or_else(|| initial_rho.clone());
    let params = streaming_backward_params_tensor::<B>(
        batch,
        heads,
        value_heads,
        time,
        latent,
        embd,
        embd_groups,
        latent_groups,
        has_terminal_rho,
        &query.device(),
    );

    let query_primitive = query.clone().into_primitive().tensor();
    let fusion_query: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(query_primitive)?;
    let fusion_client = fusion_query.client.clone();
    let query_cube =
        fusion_client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(fusion_query);
    if query_cube.dtype != DType::F32 {
        return None;
    }
    let value_cube = resolve_fusion_tensor_runtime::<B, BT, CudaRuntime, 4>(&value)?;
    let initial_cube = resolve_fusion_tensor_runtime::<B, BT, CudaRuntime, 4>(&initial_rho)?;
    let grad_output_cube = resolve_fusion_tensor_runtime::<B, BT, CudaRuntime, 4>(&grad_output)?;
    let terminal_cube = resolve_fusion_tensor_runtime::<B, BT, CudaRuntime, 4>(&terminal)?;
    let decay_cube = resolve_fusion_tensor_runtime::<B, BT, CudaRuntime, 1>(&decay)?;
    let params_cube = resolve_fusion_tensor_runtime::<B, BT, CudaRuntime, 1>(&params)?;

    let (grad_query_partial, grad_initial_rho) =
        recurrent_attention_streaming_grad_query_cuda_runtime(
            query_cube.clone(),
            value_cube,
            initial_cube,
            grad_output_cube.clone(),
            terminal_cube.clone(),
            decay_cube.clone(),
            params_cube.clone(),
            embd_groups,
        );
    let grad_value_partial = recurrent_attention_streaming_grad_value_cuda_runtime(
        query_cube,
        grad_output_cube,
        terminal_cube,
        decay_cube,
        params_cube,
        latent_groups,
    );

    let grad_query_partial = register_fusion_float_tensor(&fusion_client, grad_query_partial);
    let grad_value_partial = register_fusion_float_tensor(&fusion_client, grad_value_partial);
    let grad_initial_rho = register_fusion_float_tensor(&fusion_client, grad_initial_rho);
    let grad_query_partial = BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(
        try_cast_backend::<B, _>(grad_query_partial)?,
    ));
    let grad_value_partial = BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(
        try_cast_backend::<B, _>(grad_value_partial)?,
    ));
    let grad_initial_rho_flat = BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(
        try_cast_backend::<B, _>(grad_initial_rho)?,
    ));

    Some(finish_streaming_recurrent_attention_input_vjp(
        grad_query_partial,
        grad_value_partial,
        grad_initial_rho_flat,
        batch,
        heads,
        value_heads,
        time,
        latent,
        embd,
        embd_groups,
        latent_groups,
        had_initial_rho,
    ))
}

fn backward_params_tensor<B: BackendTrait>(
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
    device: &B::Device,
) -> BurnTensor<B, 1> {
    BurnTensor::<B, 1>::from_data(
        TensorData::new(
            vec![
                batch as f32,
                heads as f32,
                time as f32,
                latent as f32,
                embd as f32,
            ],
            [BACKWARD_META_LEN],
        ),
        device,
    )
}

fn reverse_time_indices<B: BackendTrait>(time: usize, device: &B::Device) -> BurnTensor<B, 1, Int> {
    BurnTensor::<B, 1, Int>::from_data(
        TensorData::new((0..time as i64).rev().collect::<Vec<_>>(), [time]),
        device,
    )
}

fn reverse_time_tensor4<B: BackendTrait>(tensor: BurnTensor<B, 4>) -> BurnTensor<B, 4> {
    let time = tensor.shape().dims::<4>()[2];
    let device = tensor.device();
    tensor.select(2, reverse_time_indices::<B>(time, &device))
}

fn recurrent_meta_tensor<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value_heads: usize,
    embd: usize,
) -> BurnTensor<B, 1> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    CompiledRecurrentAttentionPlan::new(
        batch,
        heads,
        value_heads,
        time,
        latent,
        embd,
        &query.device(),
    )
    .meta()
}

fn recurrent_attention_grad_query_runtime<R: CubeRuntime>(
    forward_state: CubeTensor<R>,
    reverse_state_rev: CubeTensor<R>,
    grad_output: CubeTensor<R>,
    value: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let [batch, heads, time, latent, _] = forward_state.meta.shape.dims::<5>();
    let client = forward_state.client.clone();
    let device = forward_state.device.clone();
    let grad_query = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent]),
    );
    let cube_dim = CubeDim::new_1d(BACKWARD_WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(latent as u32, BACKWARD_WORKGROUP_SIZE_X),
        heads as u32,
        (batch * time) as u32,
    );
    let _ = recurrent_attention_grad_query_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        into_contiguous(forward_state).into_tensor_arg(),
        into_contiguous(reverse_state_rev).into_tensor_arg(),
        into_contiguous(grad_output).into_tensor_arg(),
        into_contiguous(value).into_tensor_arg(),
        grad_query.clone().into_tensor_arg(),
        into_contiguous(params).into_tensor_arg(),
    );
    grad_query
}

fn recurrent_attention_grad_query_decay_grouped_wgpu_runtime(
    forward_state: CubeTensor<WgpuRuntime>,
    reverse_state_rev: CubeTensor<WgpuRuntime>,
    grad_output: CubeTensor<WgpuRuntime>,
    value: CubeTensor<WgpuRuntime>,
    query: CubeTensor<WgpuRuntime>,
    decay: CubeTensor<WgpuRuntime>,
    params: CubeTensor<WgpuRuntime>,
) -> (CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>) {
    let [batch, heads, time, latent, _] = forward_state.meta.shape.dims::<5>();
    let client = forward_state.client.clone();
    let device = forward_state.device.clone();
    let grad_query = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, latent]),
    );
    let latent_groups = latent.div_ceil(BACKWARD_WGPU_WORKGROUP_SIZE);
    let grad_decay_grouped = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent_groups]),
    );
    let cube_dim = CubeDim::new_1d(BACKWARD_WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(latent_groups as u32, heads as u32, (batch * time) as u32);
    let _ = recurrent_attention_grad_query_decay_grouped_wgpu_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count,
        cube_dim,
        into_contiguous(forward_state).into_tensor_arg(),
        into_contiguous(reverse_state_rev).into_tensor_arg(),
        into_contiguous(grad_output).into_tensor_arg(),
        into_contiguous(value).into_tensor_arg(),
        into_contiguous(query).into_tensor_arg(),
        into_contiguous(decay).into_tensor_arg(),
        grad_query.clone().into_tensor_arg(),
        grad_decay_grouped.clone().into_tensor_arg(),
        into_contiguous(params).into_tensor_arg(),
        BACKWARD_WGPU_EMBD_TILE,
    );
    (grad_query, grad_decay_grouped)
}

#[cfg(feature = "cuda")]
fn recurrent_attention_grad_value_runtime<R: CubeRuntime>(
    reverse_state_rev: CubeTensor<R>,
    query: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let [batch, heads, time, _, embd] = reverse_state_rev.meta.shape.dims::<5>();
    let client = reverse_state_rev.client.clone();
    let device = reverse_state_rev.device.clone();
    let grad_value = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, embd]),
    );
    let cube_dim = CubeDim::new_1d(BACKWARD_WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(embd as u32, BACKWARD_WORKGROUP_SIZE_X),
        heads as u32,
        (batch * time) as u32,
    );
    let _ = recurrent_attention_grad_value_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        into_contiguous(reverse_state_rev).into_tensor_arg(),
        into_contiguous(query).into_tensor_arg(),
        grad_value.clone().into_tensor_arg(),
        into_contiguous(params).into_tensor_arg(),
    );
    grad_value
}

fn recurrent_attention_grad_value_tiled_wgpu_runtime(
    reverse_state_rev: CubeTensor<WgpuRuntime>,
    query: CubeTensor<WgpuRuntime>,
    params: CubeTensor<WgpuRuntime>,
) -> CubeTensor<WgpuRuntime> {
    let [batch, heads, time, _, embd] = reverse_state_rev.meta.shape.dims::<5>();
    let client = reverse_state_rev.client.clone();
    let device = reverse_state_rev.device.clone();
    let grad_value = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, embd]),
    );
    let cube_dim = CubeDim::new_1d(BACKWARD_WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(embd as u32, BACKWARD_WORKGROUP_SIZE_X),
        heads as u32,
        (batch * time) as u32,
    );
    let _ = recurrent_attention_grad_value_tiled_wgpu_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count,
        cube_dim,
        into_contiguous(reverse_state_rev).into_tensor_arg(),
        into_contiguous(query).into_tensor_arg(),
        grad_value.clone().into_tensor_arg(),
        into_contiguous(params).into_tensor_arg(),
        BACKWARD_WGPU_LATENT_TILE,
    );
    grad_value
}

fn recurrent_attention_grad_value_reduced_tiled_wgpu_runtime(
    reverse_state_rev: CubeTensor<WgpuRuntime>,
    query: CubeTensor<WgpuRuntime>,
    params: CubeTensor<WgpuRuntime>,
) -> CubeTensor<WgpuRuntime> {
    let [batch, _heads, time, _, embd] = reverse_state_rev.meta.shape.dims::<5>();
    let client = reverse_state_rev.client.clone();
    let device = reverse_state_rev.device.clone();
    let grad_value = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device,
        Shape::new([batch, 1, time, embd]),
    );
    let cube_dim = CubeDim::new_1d(BACKWARD_WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(embd as u32, BACKWARD_WORKGROUP_SIZE_X),
        time as u32,
        batch as u32,
    );
    let _ = recurrent_attention_grad_value_reduced_tiled_wgpu_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count,
        cube_dim,
        into_contiguous(reverse_state_rev).into_tensor_arg(),
        into_contiguous(query).into_tensor_arg(),
        grad_value.clone().into_tensor_arg(),
        into_contiguous(params).into_tensor_arg(),
        BACKWARD_WGPU_LATENT_TILE,
    );
    grad_value
}

fn recurrent_attention_grad_decay_runtime<R: CubeRuntime>(
    forward_state: CubeTensor<R>,
    reverse_state_rev: CubeTensor<R>,
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    decay: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let [batch, heads, time, _, _] = forward_state.meta.shape.dims::<5>();
    let client = forward_state.client.clone();
    let device = forward_state.device.clone();
    let grad_decay_partial =
        empty_device::<R, f32>(client.clone(), device, Shape::new([batch, heads, time]));
    let cube_dim = CubeDim::new_1d(BACKWARD_WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(time as u32, BACKWARD_WORKGROUP_SIZE_X),
        heads as u32,
        batch as u32,
    );
    let _ = recurrent_attention_grad_decay_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        into_contiguous(forward_state).into_tensor_arg(),
        into_contiguous(reverse_state_rev).into_tensor_arg(),
        into_contiguous(query).into_tensor_arg(),
        into_contiguous(value).into_tensor_arg(),
        into_contiguous(decay).into_tensor_arg(),
        grad_decay_partial.clone().into_tensor_arg(),
        into_contiguous(params).into_tensor_arg(),
    );
    grad_decay_partial
}

fn recurrent_attention_reverse_state_history_with_terminal<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    grad_output: BurnTensor<B, 4>,
    decay: BurnTensor<B, 1>,
    grad_terminal_rho: Option<BurnTensor<B, 4>>,
) -> Option<(BurnTensor<B, 5>, BurnTensor<B, 4>)>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, _time, latent] = query.shape().dims::<4>();
    let embd = grad_output.shape().dims::<4>()[3];
    let query_rev = reverse_time_tensor4(query);
    let grad_output_rev = reverse_time_tensor4(grad_output);
    // The recurrent kernel decays after each state write. Seeding the reverse
    // recurrence with d * dL/drho_T makes its state before the last token equal
    // the exact cotangent consumed by q_{T-1} and v_{T-1}.
    let reverse_rho = match grad_terminal_rho {
        Some(grad) => {
            if grad.shape().dims::<4>() != [batch, heads, latent, embd] {
                return None;
            }
            grad * decay.clone().reshape([1, heads, 1, 1])
        }
        None => BurnTensor::<B, 4>::zeros([batch, heads, latent, embd], &query_rev.device()),
    };
    let meta = recurrent_meta_tensor(&query_rev, heads, embd);

    let captured = try_direct_path_runtime_with_state_history::<B, WgpuRuntime>(
        &query_rev,
        &grad_output_rev,
        &reverse_rho,
        &decay,
        &meta,
    )
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_direct_path_runtime_with_state_history::<B, CudaRuntime>(
                &query_rev,
                &grad_output_rev,
                &reverse_rho,
                &decay,
                &meta,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })?;

    Some((captured.state_history, captured.rho))
}

pub(super) fn recurrent_attention_reverse_state_history<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    grad_output: BurnTensor<B, 4>,
    decay: BurnTensor<B, 1>,
) -> Option<(BurnTensor<B, 5>, BurnTensor<B, 4>)>
where
    B::FloatTensorPrimitive: 'static,
{
    recurrent_attention_reverse_state_history_with_terminal(query, grad_output, decay, None)
}

/// Evaluate the exact VJP of the production recurrent linear-attention
/// executor without constructing an autodiff graph.
///
/// The forward state history is rematerialized one factor at a time. This
/// keeps local-PC memory bounded by one recurrent layer instead of retaining a
/// global history for every shared-weight layer.
pub fn try_fused_recurrent_attention_input_vjp<B: BackendTrait>(
    grad_output: BurnTensor<B, 4>,
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    initial_rho: Option<BurnTensor<B, 4>>,
    decay: BurnTensor<B, 1>,
    grad_terminal_rho: Option<BurnTensor<B, 4>>,
) -> Option<RecurrentAttentionInputVjp<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let [value_batch, value_heads, value_time, embd] = value.shape().dims::<4>();
    if batch == 0
        || heads == 0
        || time == 0
        || latent == 0
        || embd == 0
        || value_batch != batch
        || value_time != time
        || (value_heads != 1 && value_heads != heads)
        || grad_output.shape().dims::<4>() != [batch, heads, time, embd]
        || decay.shape().dims::<1>() != [heads]
    {
        return None;
    }

    let had_initial_rho = initial_rho.is_some();
    let rho = match initial_rho {
        Some(rho) if rho.shape().dims::<4>() == [batch, heads, latent, embd] => rho,
        Some(_) => return None,
        None => BurnTensor::<B, 4>::zeros([batch, heads, latent, embd], &query.device()),
    };
    if grad_terminal_rho
        .as_ref()
        .is_some_and(|grad| grad.shape().dims::<4>() != [batch, heads, latent, embd])
    {
        return None;
    }
    recurrent_vjp_route_profile_record_attempt();
    #[cfg(feature = "cuda")]
    if let Some(vjp) = try_streaming_recurrent_attention_input_vjp_cuda(
        grad_output.clone(),
        query.clone(),
        value.clone(),
        rho.clone(),
        decay.clone(),
        grad_terminal_rho.clone(),
        had_initial_rho,
    ) {
        recurrent_vjp_route_profile_record_route(RecurrentVjpRouteKind::CudaStreamingDirect);
        return Some(vjp);
    }
    #[cfg(feature = "cuda")]
    if let Some(vjp) = try_streaming_recurrent_attention_input_vjp_cuda_fusion::<B, u8>(
        grad_output.clone(),
        query.clone(),
        value.clone(),
        rho.clone(),
        decay.clone(),
        grad_terminal_rho.clone(),
        had_initial_rho,
    ) {
        recurrent_vjp_route_profile_record_route(RecurrentVjpRouteKind::CudaStreamingFusion);
        return Some(vjp);
    }
    let meta = recurrent_meta_tensor(&query, value_heads, embd);
    let captured = try_direct_path_runtime_with_state_history::<B, WgpuRuntime>(
        &query, &value, &rho, &decay, &meta,
    )
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_direct_path_runtime_with_state_history::<B, CudaRuntime>(
                &query, &value, &rho, &decay, &meta,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })?;

    let value_per_head = if value_heads == heads {
        value.clone()
    } else {
        value.clone().repeat_dim(1, heads)
    };
    let (reverse_state_rev, reverse_final_rho) =
        recurrent_attention_reverse_state_history_with_terminal(
            query.clone(),
            grad_output.clone(),
            decay.clone(),
            grad_terminal_rho,
        )?;
    let params = backward_params_tensor(batch, heads, time, latent, embd, &query.device());
    let grad_query = try_direct_recurrent_grad_query::<B>(
        captured.state_history,
        reverse_state_rev.clone(),
        grad_output,
        value_per_head,
        params.clone(),
    )?;
    let grad_value_heads =
        try_direct_recurrent_grad_value::<B>(reverse_state_rev, query, value_heads, params)?;
    let grad_value = if value_heads == heads {
        grad_value_heads
    } else {
        grad_value_heads.sum_dim(1).reshape([batch, 1, time, embd])
    };
    let grad_initial_rho = had_initial_rho
        .then(|| reverse_final_rho.div(decay.add_scalar(1.0e-8).reshape([1, heads, 1, 1])));

    let result = RecurrentAttentionInputVjp {
        grad_query,
        grad_value,
        grad_initial_rho,
    };
    recurrent_vjp_route_profile_record_route(RecurrentVjpRouteKind::HistoryFallback);
    Some(result)
}

pub(super) fn recurrent_attention_backward_impl<B: BackendTrait>(
    ops: Ops<RecurrentAttentionBackwardState<B::FloatTensorPrimitive>, 4>,
    grads: &mut Gradients,
) where
    B::FloatTensorPrimitive: 'static,
{
    let grad_output =
        BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(grads.consume::<B>(&ops.node)));
    let RecurrentAttentionBackwardState {
        query,
        value,
        rho,
        decay,
        state_history,
    } = ops.state;
    let parents = ops.parents;

    let query = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(query));
    let value = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(value));
    let rho = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(rho));
    let decay = BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(decay));
    let state_history = BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(state_history));

    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let value_heads = value.shape().dims::<4>()[1];
    let embd = value.shape().dims::<4>()[3];
    let value_per_head = if value_heads == heads {
        value.clone()
    } else {
        value.clone().repeat_dim(1, heads)
    };

    let (reverse_state_rev, reverse_final_rho) = recurrent_attention_reverse_state_history(
        query.clone(),
        grad_output.clone(),
        decay.clone(),
    )
    .expect("recurrent custom backward reverse state history");
    let params = backward_params_tensor(batch, heads, time, latent, embd, &query.device());

    let fused_query_decay = if parents[0].is_some() || parents[3].is_some() {
        try_direct_recurrent_grad_query_decay_grouped::<B>(
            state_history.clone(),
            reverse_state_rev.clone(),
            grad_output.clone(),
            value_per_head.clone(),
            query.clone(),
            decay.clone(),
            params.clone(),
        )
    } else {
        None
    };

    if let Some(parent) = &parents[0] {
        let grad_query = if let Some((grad_query, _)) = fused_query_decay.clone() {
            grad_query
        } else {
            try_direct_recurrent_grad_query::<B>(
                state_history.clone(),
                reverse_state_rev.clone(),
                grad_output.clone(),
                value_per_head.clone(),
                params.clone(),
            )
            .expect("recurrent grad_query runtime")
        };
        grads.register::<B>(parent.id, grad_query.into_primitive().tensor());
    }

    if let Some(parent) = &parents[1] {
        let grad_value_heads = try_direct_recurrent_grad_value::<B>(
            reverse_state_rev.clone(),
            query.clone(),
            value_heads,
            params.clone(),
        )
        .expect("recurrent grad_value runtime");
        let grad_value = if value_heads == heads {
            grad_value_heads
        } else {
            grad_value_heads.sum_dim(1).reshape([batch, 1, time, embd])
        };
        grads.register::<B>(parent.id, grad_value.into_primitive().tensor());
    }

    if let Some(parent) = &parents[2] {
        let decay_safe = decay.clone().add_scalar(1.0e-8).reshape([1, heads, 1, 1]);
        let grad_rho = reverse_final_rho.div(decay_safe);
        let _ = rho;
        grads.register::<B>(parent.id, grad_rho.into_primitive().tensor());
    }

    if let Some(parent) = &parents[3] {
        let grad_decay_partial = if let Some((_, grad_decay_grouped)) = fused_query_decay {
            grad_decay_grouped.sum_dim(3).reshape([batch, heads, time])
        } else {
            try_direct_recurrent_grad_decay::<B>(
                state_history,
                reverse_state_rev,
                query,
                value_per_head,
                decay.clone(),
                params,
            )
            .expect("recurrent grad_decay runtime")
        };
        let grad_decay = grad_decay_partial.sum_dim(2).sum_dim(0);
        grads.register::<B>(parent.id, grad_decay.into_primitive().tensor());
    }
}

impl Backward<WgpuCubeBackend, 4> for FusedRecurrentAttentionBackward<WgpuCubeBackend> {
    type State = RecurrentAttentionBackwardState<CubeTensor<WgpuRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 4>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        recurrent_attention_backward_impl::<WgpuCubeBackend>(ops, grads);
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 4> for FusedRecurrentAttentionBackward<CudaCubeBackend> {
    type State = RecurrentAttentionBackwardState<CubeTensor<CudaRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 4>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        recurrent_attention_backward_impl::<CudaCubeBackend>(ops, grads);
    }
}

fn try_direct_recurrent_grad_query<B: BackendTrait>(
    forward_state: BurnTensor<B, 5>,
    reverse_state_rev: BurnTensor<B, 5>,
    grad_output: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    params: BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_forward = forward_state.into_primitive().tensor();
    let prim_reverse = reverse_state_rev.into_primitive().tensor();
    let prim_grad = grad_output.into_primitive().tensor();
    let prim_value = value.into_primitive().tensor();
    let prim_params = params.into_primitive().tensor();
    if let Some(forward_state) =
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(prim_forward.clone())
    {
        let reverse_state_rev = try_cast_primitive::<B, _>(prim_reverse.clone())?;
        let grad_output = try_cast_primitive::<B, _>(prim_grad.clone())?;
        let value = try_cast_primitive::<B, _>(prim_value.clone())?;
        let params = try_cast_primitive::<B, _>(prim_params.clone())?;
        let tensor = recurrent_attention_grad_query_runtime::<WgpuRuntime>(
            forward_state,
            reverse_state_rev,
            grad_output,
            value,
            params,
        );
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(tensor)?,
        )));
    }

    #[cfg(feature = "cuda")]
    if let Some(forward_state) = try_cast_primitive::<B, CubeTensor<CudaRuntime>>(prim_forward) {
        let reverse_state_rev = try_cast_primitive::<B, _>(prim_reverse)?;
        let grad_output = try_cast_primitive::<B, _>(prim_grad)?;
        let value = try_cast_primitive::<B, _>(prim_value)?;
        let params = try_cast_primitive::<B, _>(prim_params)?;
        let tensor = recurrent_attention_grad_query_runtime::<CudaRuntime>(
            forward_state,
            reverse_state_rev,
            grad_output,
            value,
            params,
        );
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(tensor)?,
        )));
    }

    None
}

fn try_direct_recurrent_grad_query_decay_grouped<B: BackendTrait>(
    forward_state: BurnTensor<B, 5>,
    reverse_state_rev: BurnTensor<B, 5>,
    grad_output: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    query: BurnTensor<B, 4>,
    decay: BurnTensor<B, 1>,
    params: BurnTensor<B, 1>,
) -> Option<(BurnTensor<B, 4>, BurnTensor<B, 4>)>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_forward = forward_state.into_primitive().tensor();
    let prim_reverse = reverse_state_rev.into_primitive().tensor();
    let prim_grad = grad_output.into_primitive().tensor();
    let prim_value = value.into_primitive().tensor();
    let prim_query = query.into_primitive().tensor();
    let prim_decay = decay.into_primitive().tensor();
    let prim_params = params.into_primitive().tensor();
    let forward_state = try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(prim_forward)?;
    let reverse_state_rev = try_cast_primitive::<B, _>(prim_reverse)?;
    let grad_output = try_cast_primitive::<B, _>(prim_grad)?;
    let value = try_cast_primitive::<B, _>(prim_value)?;
    let query = try_cast_primitive::<B, _>(prim_query)?;
    let decay = try_cast_primitive::<B, _>(prim_decay)?;
    let params = try_cast_primitive::<B, _>(prim_params)?;
    let (grad_query, grad_decay_grouped) =
        recurrent_attention_grad_query_decay_grouped_wgpu_runtime(
            forward_state,
            reverse_state_rev,
            grad_output,
            value,
            query,
            decay,
            params,
        );
    Some((
        BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
            grad_query,
        )?)),
        BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
            grad_decay_grouped,
        )?)),
    ))
}

fn try_direct_recurrent_grad_value<B: BackendTrait>(
    reverse_state_rev: BurnTensor<B, 5>,
    query: BurnTensor<B, 4>,
    value_heads: usize,
    params: BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_reverse = reverse_state_rev.into_primitive().tensor();
    let prim_query = query.into_primitive().tensor();
    let prim_params = params.into_primitive().tensor();
    if let Some(reverse_state_rev) =
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(prim_reverse.clone())
    {
        let query = try_cast_primitive::<B, _>(prim_query.clone())?;
        let params = try_cast_primitive::<B, _>(prim_params.clone())?;
        let tensor = if value_heads == 1 {
            recurrent_attention_grad_value_reduced_tiled_wgpu_runtime(
                reverse_state_rev,
                query,
                params,
            )
        } else {
            recurrent_attention_grad_value_tiled_wgpu_runtime(reverse_state_rev, query, params)
        };
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(tensor)?,
        )));
    }

    #[cfg(feature = "cuda")]
    if let Some(reverse_state_rev) = try_cast_primitive::<B, CubeTensor<CudaRuntime>>(prim_reverse)
    {
        let query = try_cast_primitive::<B, _>(prim_query)?;
        let params = try_cast_primitive::<B, _>(prim_params)?;
        let tensor =
            recurrent_attention_grad_value_runtime::<CudaRuntime>(reverse_state_rev, query, params);
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(tensor)?,
        )));
    }

    None
}

fn try_direct_recurrent_grad_decay<B: BackendTrait>(
    forward_state: BurnTensor<B, 5>,
    reverse_state_rev: BurnTensor<B, 5>,
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    decay: BurnTensor<B, 1>,
    params: BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 3>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_forward = forward_state.into_primitive().tensor();
    let prim_reverse = reverse_state_rev.into_primitive().tensor();
    let prim_query = query.into_primitive().tensor();
    let prim_value = value.into_primitive().tensor();
    let prim_decay = decay.into_primitive().tensor();
    let prim_params = params.into_primitive().tensor();
    if let Some(forward_state) =
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(prim_forward.clone())
    {
        let reverse_state_rev = try_cast_primitive::<B, _>(prim_reverse.clone())?;
        let query = try_cast_primitive::<B, _>(prim_query.clone())?;
        let value = try_cast_primitive::<B, _>(prim_value.clone())?;
        let decay = try_cast_primitive::<B, _>(prim_decay.clone())?;
        let params = try_cast_primitive::<B, _>(prim_params.clone())?;
        let tensor = recurrent_attention_grad_decay_runtime::<WgpuRuntime>(
            forward_state,
            reverse_state_rev,
            query,
            value,
            decay,
            params,
        );
        return Some(BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(tensor)?,
        )));
    }

    #[cfg(feature = "cuda")]
    if let Some(forward_state) = try_cast_primitive::<B, CubeTensor<CudaRuntime>>(prim_forward) {
        let reverse_state_rev = try_cast_primitive::<B, _>(prim_reverse)?;
        let query = try_cast_primitive::<B, _>(prim_query)?;
        let value = try_cast_primitive::<B, _>(prim_value)?;
        let decay = try_cast_primitive::<B, _>(prim_decay)?;
        let params = try_cast_primitive::<B, _>(prim_params)?;
        let tensor = recurrent_attention_grad_decay_runtime::<CudaRuntime>(
            forward_state,
            reverse_state_rev,
            query,
            value,
            decay,
            params,
        );
        return Some(BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(tensor)?,
        )));
    }

    None
}

fn recurrent_attention_autodiff_custom_wgpu<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let query_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(query.clone().into_primitive().tensor())?;
    let value_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(value.clone().into_primitive().tensor())?;
    let rho_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(rho.clone().into_primitive().tensor())?;
    let decay_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(decay.clone().into_primitive().tensor())?;
    let meta_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(meta.clone().into_primitive().tensor())?;

    let query_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(query_ad.clone());
    let value_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(value_ad.clone());
    let rho_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(rho_ad.clone());
    let decay_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(decay_ad.clone());
    let meta_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(meta_ad);

    let captured = try_direct_path_runtime_with_state_history::<WgpuCubeBackend, WgpuRuntime>(
        &BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            query_inner.clone(),
        )),
        &BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            value_inner.clone(),
        )),
        &BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            rho_inner.clone(),
        )),
        &BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            decay_inner.clone(),
        )),
        &BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            meta_inner.clone(),
        )),
    )?;

    let context_inner = captured.context.into_primitive().tensor();
    let rho_inner_out = captured.rho.into_primitive().tensor();
    let state_history_inner = captured.state_history.into_primitive().tensor();

    let context_ad = match FusedRecurrentAttentionBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            query_ad.node.clone(),
            value_ad.node.clone(),
            rho_ad.node.clone(),
            decay_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            RecurrentAttentionBackwardState {
                query: query_inner,
                value: value_inner,
                rho: rho_inner,
                decay: decay_inner,
                state_history: state_history_inner,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };

    Some(RecurrentAttentionOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(context_ad)?)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
            <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(rho_inner_out),
        )?)),
    })
}

#[cfg(feature = "cuda")]
fn recurrent_attention_autodiff_custom_cuda<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let query_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(query.clone().into_primitive().tensor())?;
    let value_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(value.clone().into_primitive().tensor())?;
    let rho_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(rho.clone().into_primitive().tensor())?;
    let decay_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(decay.clone().into_primitive().tensor())?;
    let meta_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(meta.clone().into_primitive().tensor())?;

    let query_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(query_ad.clone());
    let value_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(value_ad.clone());
    let rho_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(rho_ad.clone());
    let decay_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(decay_ad.clone());
    let meta_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(meta_ad);

    let captured = try_direct_path_runtime_with_state_history::<CudaCubeBackend, CudaRuntime>(
        &BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            query_inner.clone(),
        )),
        &BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            value_inner.clone(),
        )),
        &BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            rho_inner.clone(),
        )),
        &BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            decay_inner.clone(),
        )),
        &BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            meta_inner.clone(),
        )),
    )?;

    let context_inner = captured.context.into_primitive().tensor();
    let rho_inner_out = captured.rho.into_primitive().tensor();
    let state_history_inner = captured.state_history.into_primitive().tensor();

    let context_ad = match FusedRecurrentAttentionBackward::<CudaCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            query_ad.node.clone(),
            value_ad.node.clone(),
            rho_ad.node.clone(),
            decay_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            RecurrentAttentionBackwardState {
                query: query_inner,
                value: value_inner,
                rho: rho_inner,
                decay: decay_inner,
                state_history: state_history_inner,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };

    Some(RecurrentAttentionOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(context_ad)?)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
            <CudaCubeAutodiffBackend as AutodiffBackend>::from_inner(rho_inner_out),
        )?)),
    })
}

pub(super) fn recurrent_attention_autodiff_custom<B: BackendTrait, R: CubeRuntime + 'static>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        return recurrent_attention_autodiff_custom_wgpu(query, value, rho, decay, meta);
    }
    #[cfg(feature = "cuda")]
    if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
        return recurrent_attention_autodiff_custom_cuda(query, value, rho, decay, meta);
    }
    None
}
