#![allow(
    clippy::let_unit_value,
    clippy::too_many_arguments,
    dead_code,
    unused_variables
)]

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{DType, Shape, Tensor as BurnTensor, TensorData};
use burn_cubecl::CubeRuntime;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::cubecl::{self};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::{empty_device, zeros_client};
use burn_cubecl::tensor::CubeTensor;

const GDN2_PARAM_LEN: usize = 7;
const GDN2_FORWARD_WORKGROUP_X: u32 = 32;
const GDN2_BACKWARD_WORKGROUP_X: u32 = 32;
const GDN2_ZERO_WORKGROUP_X: u32 = 256;
const GDN2_INVERSE_REPLAY_INTERVAL: usize = 8;

#[derive(Debug)]
pub(crate) struct Gdn2ForwardRuntimeOutput<R: CubeRuntime> {
    pub(crate) context: CubeTensor<R>,
    pub(crate) final_state: CubeTensor<R>,
    pub(crate) boundary_states: CubeTensor<R>,
}

#[derive(Debug)]
pub(crate) struct Gdn2BackwardRuntimeOutput<R: CubeRuntime> {
    pub(crate) grad_query: CubeTensor<R>,
    pub(crate) grad_key: CubeTensor<R>,
    pub(crate) grad_value: CubeTensor<R>,
    pub(crate) grad_erase: CubeTensor<R>,
    pub(crate) grad_write: CubeTensor<R>,
    pub(crate) grad_log_decay: CubeTensor<R>,
    pub(crate) grad_initial_state: CubeTensor<R>,
}

fn gdn2_params<B: BackendTrait>(
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    dense: usize,
    chunk_size: usize,
    num_chunks: usize,
    device: &B::Device,
) -> BurnTensor<B, 1> {
    BurnTensor::<B, 1>::from_data(
        TensorData::new(
            vec![
                batch as f32,
                heads as f32,
                time as f32,
                latent as f32,
                dense as f32,
                chunk_size.max(1) as f32,
                num_chunks as f32,
            ],
            [GDN2_PARAM_LEN],
        ),
        device,
    )
}

#[cube(launch)]
fn gdn2_prepare_wy_factors_kernel(
    key: &Tensor<f32>,
    erase: &Tensor<f32>,
    log_decay: &Tensor<f32>,
    cumulative_decay: &mut Tensor<f32>,
    wy_lower: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] max_latent: usize,
    #[comptime] max_chunk_size: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let chunk_size = (u32::cast_from(params[5]) as usize).max(1usize);
    let num_chunks = u32::cast_from(params[6]) as usize;

    let chunk = CUBE_POS_X as usize;
    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1usize);
    let h = z % heads.max(1usize);
    if b >= batch
        || h >= heads
        || chunk >= num_chunks
        || latent > max_latent
        || chunk_size > max_chunk_size
        || UNIT_POS_X != 0
    {
        terminate!();
    }

    let zero = f32::cast_from(0u32);
    let chunk_latent_len = max_chunk_size * max_latent;
    let mut cumulative_local = SharedMemory::<f32>::new_aligned(chunk_latent_len, 1usize);
    let chunk_start = chunk * chunk_size;
    let mut chunk_end = chunk_start + chunk_size;
    if chunk_end > time {
        chunk_end = time;
    }
    let chunk_len = chunk_end - chunk_start;

    let mut local = 0usize;
    while local < chunk_size {
        let mut l = 0usize;
        while l < latent {
            let cum_index = b * cumulative_decay.stride(0)
                + h * cumulative_decay.stride(1)
                + chunk * cumulative_decay.stride(2)
                + local * cumulative_decay.stride(3)
                + l * cumulative_decay.stride(4);
            cumulative_decay[cum_index] = zero;
            l += 1usize;
        }

        let mut j = 0usize;
        while j < chunk_size {
            let matrix_index = b * wy_lower.stride(0)
                + h * wy_lower.stride(1)
                + chunk * wy_lower.stride(2)
                + local * wy_lower.stride(3)
                + j * wy_lower.stride(4);
            wy_lower[matrix_index] = zero;
            j += 1usize;
        }
        local += 1usize;
    }

    local = 0usize;
    while local < chunk_len {
        let t = chunk_start + local;
        let mut l = 0usize;
        while l < latent {
            let decay_index = b * log_decay.stride(0)
                + h * log_decay.stride(1)
                + t * log_decay.stride(2)
                + l * log_decay.stride(3);
            let cum_index = b * cumulative_decay.stride(0)
                + h * cumulative_decay.stride(1)
                + chunk * cumulative_decay.stride(2)
                + local * cumulative_decay.stride(3)
                + l * cumulative_decay.stride(4);
            let decay = f32::exp(log_decay[decay_index]);
            let cumulative = if local == 0usize {
                decay
            } else {
                cumulative_local[(local - 1usize) * max_latent + l] * decay
            };
            cumulative_local[local * max_latent + l] = cumulative;
            cumulative_decay[cum_index] = cumulative;
            l += 1usize;
        }
        local += 1usize;
    }

    local = 0usize;
    while local < chunk_len {
        let t = chunk_start + local;
        let mut previous = 0usize;
        while previous < local {
            let t_previous = chunk_start + previous;
            let mut lower = zero;
            let mut l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                let previous_key_index = b * key.stride(0)
                    + h * key.stride(1)
                    + t_previous * key.stride(2)
                    + l * key.stride(3);
                let decay_c = cumulative_local[local * max_latent + l];
                let previous_decay_c = cumulative_local[previous * max_latent + l];
                let m_basis = erase[erase_index] * key[key_index] * decay_c;
                let w_basis = key[previous_key_index] / previous_decay_c;
                lower += m_basis * w_basis;
                l += 1usize;
            }
            let lower_index = b * wy_lower.stride(0)
                + h * wy_lower.stride(1)
                + chunk * wy_lower.stride(2)
                + local * wy_lower.stride(3)
                + previous * wy_lower.stride(4);
            wy_lower[lower_index] = lower;
            previous += 1usize;
        }

        local += 1usize;
    }
}

#[cube(launch)]
fn gdn2_backward_chunk_wy_factored_kernel(
    query: &Tensor<f32>,
    key: &Tensor<f32>,
    value: &Tensor<f32>,
    erase: &Tensor<f32>,
    write: &Tensor<f32>,
    log_decay: &Tensor<f32>,
    boundary_states: &Tensor<f32>,
    grad_output: &Tensor<f32>,
    wy_lower: &Tensor<f32>,
    grad_query: &mut Tensor<Atomic<f32>>,
    grad_key: &mut Tensor<Atomic<f32>>,
    grad_value: &mut Tensor<f32>,
    grad_erase: &mut Tensor<Atomic<f32>>,
    grad_write: &mut Tensor<f32>,
    grad_cumulative_decay: &mut Tensor<Atomic<f32>>,
    grad_lower: &mut Tensor<Atomic<f32>>,
    grad_initial_state: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] max_latent: usize,
    #[comptime] max_chunk_size: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let dense = u32::cast_from(params[4]) as usize;
    let chunk_size = (u32::cast_from(params[5]) as usize).max(1usize);
    let num_chunks = u32::cast_from(params[6]) as usize;

    let z_pos = CUBE_POS_Z as usize;
    let b = z_pos / heads.max(1usize);
    let h = z_pos % heads.max(1usize);
    let d = CUBE_POS_X as usize;
    if b >= batch
        || h >= heads
        || d >= dense
        || latent > max_latent
        || chunk_size > max_chunk_size
        || UNIT_POS_X != 0
    {
        terminate!();
    }

    let zero = f32::cast_from(0u32);
    let one = f32::cast_from(1u32);
    let scale = f32::sqrt(f32::cast_from(latent as u32));
    let inv_scale = one / scale;

    let chunk_latent_len = max_chunk_size * max_latent;
    let mut cumulative_decay_local = SharedMemory::<f32>::new_aligned(chunk_latent_len, 1usize);
    let mut rhs = SharedMemory::<f32>::new_aligned(max_chunk_size, 1usize);
    let mut solved_update = SharedMemory::<f32>::new_aligned(max_chunk_size, 1usize);
    let mut grad_solved_update = SharedMemory::<f32>::new_aligned(max_chunk_size, 1usize);
    let mut grad_cum_local = SharedMemory::<f32>::new_aligned(chunk_latent_len, 1usize);
    let mut transformed_state_local = SharedMemory::<f32>::new_aligned(chunk_latent_len, 1usize);
    let mut carry_boundary = SharedMemory::<f32>::new_aligned(max_latent, 1usize);

    let mut l = 0usize;
    while l < latent {
        carry_boundary[l] = zero;
        l += 1usize;
    }

    let mut chunk_rev = 0usize;
    while chunk_rev < num_chunks {
        let chunk = num_chunks - 1usize - chunk_rev;
        let chunk_start = chunk * chunk_size;
        let mut chunk_end = chunk_start + chunk_size;
        if chunk_end > time {
            chunk_end = time;
        }
        let chunk_len = chunk_end - chunk_start;

        let mut local = 0usize;
        while local < chunk_len {
            let t = chunk_start + local;
            rhs[local] = zero;
            solved_update[local] = zero;
            grad_solved_update[local] = zero;
            l = 0usize;
            while l < latent {
                let decay_index = b * log_decay.stride(0)
                    + h * log_decay.stride(1)
                    + t * log_decay.stride(2)
                    + l * log_decay.stride(3);
                let decay = f32::exp(log_decay[decay_index]);
                cumulative_decay_local[local * max_latent + l] = if local == 0usize {
                    decay
                } else {
                    cumulative_decay_local[(local - 1usize) * max_latent + l] * decay
                };
                grad_cum_local[local * max_latent + l] = zero;
                l += 1usize;
            }
            local += 1usize;
        }

        local = 0usize;
        while local < chunk_len {
            let t = chunk_start + local;
            let value_index = b * value.stride(0)
                + h * value.stride(1)
                + t * value.stride(2)
                + d * value.stride(3);
            let write_index = b * write.stride(0)
                + h * write.stride(1)
                + t * write.stride(2)
                + d * write.stride(3);
            let mut rhs_value = write[write_index] * value[value_index];

            l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                let boundary_index = b * boundary_states.stride(0)
                    + h * boundary_states.stride(1)
                    + chunk * boundary_states.stride(2)
                    + l * boundary_states.stride(3)
                    + d * boundary_states.stride(4);
                let decay_c = cumulative_decay_local[local * max_latent + l];
                let m_basis = erase[erase_index] * key[key_index] * decay_c;
                rhs_value -= m_basis * boundary_states[boundary_index];
                l += 1usize;
            }
            rhs[local] = rhs_value;

            let mut previous = 0usize;
            let mut solved_value = rhs[local];
            while previous < local {
                let lower_index = b * wy_lower.stride(0)
                    + h * wy_lower.stride(1)
                    + chunk * wy_lower.stride(2)
                    + local * wy_lower.stride(3)
                    + previous * wy_lower.stride(4);
                solved_value -= wy_lower[lower_index] * solved_update[previous];
                previous += 1usize;
            }
            solved_update[local] = solved_value;
            local += 1usize;
        }

        l = 0usize;
        while l < latent {
            let boundary_index = b * boundary_states.stride(0)
                + h * boundary_states.stride(1)
                + chunk * boundary_states.stride(2)
                + l * boundary_states.stride(3)
                + d * boundary_states.stride(4);
            let mut transformed_state = boundary_states[boundary_index];
            local = 0usize;
            while local < chunk_len {
                let t_local = chunk_start + local;
                let key_index = b * key.stride(0)
                    + h * key.stride(1)
                    + t_local * key.stride(2)
                    + l * key.stride(3);
                let decay_c = cumulative_decay_local[local * max_latent + l];
                transformed_state += key[key_index] / decay_c * solved_update[local];
                transformed_state_local[local * max_latent + l] = transformed_state;
                local += 1usize;
            }
            l += 1usize;
        }

        if chunk_len > 0usize {
            let last = chunk_len - 1usize;
            l = 0usize;
            while l < latent {
                let transformed_state = transformed_state_local[last * max_latent + l];
                grad_cum_local[last * max_latent + l] += carry_boundary[l] * transformed_state;
                carry_boundary[l] =
                    carry_boundary[l] * cumulative_decay_local[last * max_latent + l];
                l += 1usize;
            }
        }

        let mut local_rev = 0usize;
        while local_rev < chunk_len {
            let local_index = chunk_len - 1usize - local_rev;
            let t = chunk_start + local_index;
            let grad_index = b * grad_output.stride(0)
                + h * grad_output.stride(1)
                + t * grad_output.stride(2)
                + d * grad_output.stride(3);
            let grad_o_scaled = grad_output[grad_index] * inv_scale;
            let z_value = solved_update[local_index];

            l = 0usize;
            while l < latent {
                let transformed_state = transformed_state_local[local_index * max_latent + l];
                let decay_c = cumulative_decay_local[local_index * max_latent + l];
                let state_value = decay_c * transformed_state;
                let query_index = b * query.stride(0)
                    + h * query.stride(1)
                    + t * query.stride(2)
                    + l * query.stride(3);
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let grad_state = query[query_index] * grad_o_scaled;

                let grad_query_index = b * grad_query.stride(0)
                    + h * grad_query.stride(1)
                    + t * grad_query.stride(2)
                    + l * grad_query.stride(3);
                grad_query[grad_query_index].fetch_add(state_value * grad_o_scaled);

                grad_cum_local[local_index * max_latent + l] += grad_state * transformed_state;
                carry_boundary[l] += grad_state * decay_c;

                let grad_w_basis = carry_boundary[l] * z_value;
                let grad_key_index = b * grad_key.stride(0)
                    + h * grad_key.stride(1)
                    + t * grad_key.stride(2)
                    + l * grad_key.stride(3);
                grad_key[grad_key_index].fetch_add(grad_w_basis / decay_c);
                grad_cum_local[local_index * max_latent + l] -=
                    grad_w_basis * key[key_index] / (decay_c * decay_c);
                grad_solved_update[local_index] += carry_boundary[l] * key[key_index] / decay_c;
                l += 1usize;
            }

            local_rev += 1usize;
        }

        local_rev = 0usize;
        while local_rev < chunk_len {
            let local_index = chunk_len - 1usize - local_rev;
            let t = chunk_start + local_index;
            let grad_rhs = grad_solved_update[local_index];

            let mut previous = 0usize;
            while previous < local_index {
                let lower_index = b * wy_lower.stride(0)
                    + h * wy_lower.stride(1)
                    + chunk * wy_lower.stride(2)
                    + local_index * wy_lower.stride(3)
                    + previous * wy_lower.stride(4);
                let lower = wy_lower[lower_index];
                let grad_lower_value = -grad_rhs * solved_update[previous];
                let grad_lower_index = b * grad_lower.stride(0)
                    + h * grad_lower.stride(1)
                    + chunk * grad_lower.stride(2)
                    + local_index * grad_lower.stride(3)
                    + previous * grad_lower.stride(4);
                grad_lower[grad_lower_index].fetch_add(grad_lower_value);
                grad_solved_update[previous] -= grad_rhs * lower;
                previous += 1usize;
            }

            let value_index = b * value.stride(0)
                + h * value.stride(1)
                + t * value.stride(2)
                + d * value.stride(3);
            let write_index = b * write.stride(0)
                + h * write.stride(1)
                + t * write.stride(2)
                + d * write.stride(3);
            let grad_write_index = b * grad_write.stride(0)
                + h * grad_write.stride(1)
                + t * grad_write.stride(2)
                + d * grad_write.stride(3);
            let grad_value_index = b * grad_value.stride(0)
                + h * grad_value.stride(1)
                + t * grad_value.stride(2)
                + d * grad_value.stride(3);
            grad_write[grad_write_index] = grad_rhs * value[value_index];
            grad_value[grad_value_index] = grad_rhs * write[write_index];

            l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                let boundary_index = b * boundary_states.stride(0)
                    + h * boundary_states.stride(1)
                    + chunk * boundary_states.stride(2)
                    + l * boundary_states.stride(3)
                    + d * boundary_states.stride(4);
                let decay_c = cumulative_decay_local[local_index * max_latent + l];
                let m_basis = erase[erase_index] * key[key_index] * decay_c;
                let grad_m_basis = -grad_rhs * boundary_states[boundary_index];
                let grad_key_index = b * grad_key.stride(0)
                    + h * grad_key.stride(1)
                    + t * grad_key.stride(2)
                    + l * grad_key.stride(3);
                let grad_erase_index = b * grad_erase.stride(0)
                    + h * grad_erase.stride(1)
                    + t * grad_erase.stride(2)
                    + l * grad_erase.stride(3);
                grad_key[grad_key_index].fetch_add(grad_m_basis * erase[erase_index] * decay_c);
                grad_erase[grad_erase_index].fetch_add(grad_m_basis * key[key_index] * decay_c);
                grad_cum_local[local_index * max_latent + l] +=
                    grad_m_basis * erase[erase_index] * key[key_index];
                carry_boundary[l] -= grad_rhs * m_basis;
                l += 1usize;
            }

            local_rev += 1usize;
        }

        local = 0usize;
        while local < chunk_len {
            l = 0usize;
            while l < latent {
                let grad_cum_index = b * grad_cumulative_decay.stride(0)
                    + h * grad_cumulative_decay.stride(1)
                    + chunk * grad_cumulative_decay.stride(2)
                    + local * grad_cumulative_decay.stride(3)
                    + l * grad_cumulative_decay.stride(4);
                grad_cumulative_decay[grad_cum_index]
                    .fetch_add(grad_cum_local[local * max_latent + l]);
                l += 1usize;
            }
            local += 1usize;
        }

        chunk_rev += 1usize;
    }

    l = 0usize;
    while l < latent {
        let initial_index = b * grad_initial_state.stride(0)
            + h * grad_initial_state.stride(1)
            + l * grad_initial_state.stride(2)
            + d * grad_initial_state.stride(3);
        grad_initial_state[initial_index] = carry_boundary[l];
        l += 1usize;
    }
}

#[cube(launch)]
fn gdn2_project_wy_lower_grad_kernel(
    key: &Tensor<f32>,
    erase: &Tensor<f32>,
    cumulative_decay: &Tensor<f32>,
    grad_lower: &Tensor<f32>,
    grad_key: &mut Tensor<Atomic<f32>>,
    grad_erase: &mut Tensor<Atomic<f32>>,
    grad_cumulative_decay: &mut Tensor<Atomic<f32>>,
    params: &Tensor<f32>,
    #[comptime] max_latent: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let chunk_size = (u32::cast_from(params[5]) as usize).max(1usize);
    let num_chunks = u32::cast_from(params[6]) as usize;

    let chunk = CUBE_POS_X as usize;
    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1usize);
    let h = z % heads.max(1usize);
    let lane = UNIT_POS_X as usize;
    let lane_count = CUBE_DIM_X as usize;
    if b >= batch || h >= heads || chunk >= num_chunks || latent > max_latent {
        terminate!();
    }

    let chunk_start = chunk * chunk_size;
    let mut chunk_end = chunk_start + chunk_size;
    if chunk_end > time {
        chunk_end = time;
    }
    let chunk_len = chunk_end - chunk_start;

    let mut local = 0usize;
    while local < chunk_len {
        let t = chunk_start + local;
        let mut previous = 0usize;
        while previous < local {
            let t_previous = chunk_start + previous;
            let grad_lower_index = b * grad_lower.stride(0)
                + h * grad_lower.stride(1)
                + chunk * grad_lower.stride(2)
                + local * grad_lower.stride(3)
                + previous * grad_lower.stride(4);
            let grad_lower_value = grad_lower[grad_lower_index];
            let mut l = lane;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                let previous_key_index = b * key.stride(0)
                    + h * key.stride(1)
                    + t_previous * key.stride(2)
                    + l * key.stride(3);
                let cum_index = b * cumulative_decay.stride(0)
                    + h * cumulative_decay.stride(1)
                    + chunk * cumulative_decay.stride(2)
                    + local * cumulative_decay.stride(3)
                    + l * cumulative_decay.stride(4);
                let previous_cum_index = b * cumulative_decay.stride(0)
                    + h * cumulative_decay.stride(1)
                    + chunk * cumulative_decay.stride(2)
                    + previous * cumulative_decay.stride(3)
                    + l * cumulative_decay.stride(4);
                let decay_c = cumulative_decay[cum_index];
                let previous_decay_c = cumulative_decay[previous_cum_index];
                let m_basis = erase[erase_index] * key[key_index] * decay_c;
                let w_basis = key[previous_key_index] / previous_decay_c;

                let grad_m_basis = grad_lower_value * w_basis;
                let grad_key_index = b * grad_key.stride(0)
                    + h * grad_key.stride(1)
                    + t * grad_key.stride(2)
                    + l * grad_key.stride(3);
                let grad_erase_index = b * grad_erase.stride(0)
                    + h * grad_erase.stride(1)
                    + t * grad_erase.stride(2)
                    + l * grad_erase.stride(3);
                let grad_cum_index = b * grad_cumulative_decay.stride(0)
                    + h * grad_cumulative_decay.stride(1)
                    + chunk * grad_cumulative_decay.stride(2)
                    + local * grad_cumulative_decay.stride(3)
                    + l * grad_cumulative_decay.stride(4);
                grad_key[grad_key_index].fetch_add(grad_m_basis * erase[erase_index] * decay_c);
                grad_erase[grad_erase_index].fetch_add(grad_m_basis * key[key_index] * decay_c);
                grad_cumulative_decay[grad_cum_index]
                    .fetch_add(grad_m_basis * erase[erase_index] * key[key_index]);

                let grad_w_basis = grad_lower_value * m_basis;
                let previous_grad_key_index = b * grad_key.stride(0)
                    + h * grad_key.stride(1)
                    + t_previous * grad_key.stride(2)
                    + l * grad_key.stride(3);
                let previous_grad_cum_index = b * grad_cumulative_decay.stride(0)
                    + h * grad_cumulative_decay.stride(1)
                    + chunk * grad_cumulative_decay.stride(2)
                    + previous * grad_cumulative_decay.stride(3)
                    + l * grad_cumulative_decay.stride(4);
                grad_key[previous_grad_key_index].fetch_add(grad_w_basis / previous_decay_c);
                grad_cumulative_decay[previous_grad_cum_index].fetch_add(
                    -grad_w_basis * key[previous_key_index] / (previous_decay_c * previous_decay_c),
                );
                l += lane_count;
            }
            previous += 1usize;
        }
        local += 1usize;
    }
}

#[cube(launch)]
fn gdn2_accumulate_wy_log_decay_grad_kernel(
    cumulative_decay: &Tensor<f32>,
    grad_cumulative_decay: &Tensor<f32>,
    grad_log_decay: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] max_latent: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let chunk_size = (u32::cast_from(params[5]) as usize).max(1usize);
    let num_chunks = u32::cast_from(params[6]) as usize;

    let chunk = CUBE_POS_X as usize;
    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1usize);
    let h = z % heads.max(1usize);
    let lane = UNIT_POS_X as usize;
    let lane_count = CUBE_DIM_X as usize;
    if b >= batch || h >= heads || chunk >= num_chunks || latent > max_latent {
        terminate!();
    }

    let zero = f32::cast_from(0u32);
    let chunk_start = chunk * chunk_size;
    let mut chunk_end = chunk_start + chunk_size;
    if chunk_end > time {
        chunk_end = time;
    }
    let chunk_len = chunk_end - chunk_start;

    let mut l = lane;
    while l < latent {
        let mut suffix = zero;
        let mut local_rev = 0usize;
        while local_rev < chunk_len {
            let local_index = chunk_len - 1usize - local_rev;
            let t = chunk_start + local_index;
            let cum_index = b * cumulative_decay.stride(0)
                + h * cumulative_decay.stride(1)
                + chunk * cumulative_decay.stride(2)
                + local_index * cumulative_decay.stride(3)
                + l * cumulative_decay.stride(4);
            suffix += grad_cumulative_decay[cum_index] * cumulative_decay[cum_index];
            let grad_log_decay_index = b * grad_log_decay.stride(0)
                + h * grad_log_decay.stride(1)
                + t * grad_log_decay.stride(2)
                + l * grad_log_decay.stride(3);
            grad_log_decay[grad_log_decay_index] = suffix;
            local_rev += 1usize;
        }
        l += lane_count;
    }
}

#[cube(launch)]
fn gdn2_forward_kernel(
    query: &Tensor<f32>,
    key: &Tensor<f32>,
    value: &Tensor<f32>,
    erase: &Tensor<f32>,
    write: &Tensor<f32>,
    log_decay: &Tensor<f32>,
    initial_state: &Tensor<f32>,
    context: &mut Tensor<f32>,
    final_state: &mut Tensor<f32>,
    boundary_states: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] max_latent: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let dense = u32::cast_from(params[4]) as usize;
    let chunk_size = (u32::cast_from(params[5]) as usize).max(1usize);
    let num_chunks = u32::cast_from(params[6]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1usize);
    let h = z % heads.max(1usize);
    let d = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let lane = UNIT_POS_X as usize;
    if b >= batch || h >= heads || d >= dense || latent > max_latent {
        terminate!();
    }

    let zero = f32::cast_from(0u32);
    let scale = f32::sqrt(f32::cast_from(latent as u32));
    let inv_scale = f32::cast_from(1u32) / scale;
    let mut state =
        SharedMemory::<f32>::new_aligned(GDN2_FORWARD_WORKGROUP_X as usize * max_latent, 1usize);
    let lane_base = lane * max_latent;

    let mut l = 0usize;
    while l < latent {
        let init_index = b * initial_state.stride(0)
            + h * initial_state.stride(1)
            + l * initial_state.stride(2)
            + d * initial_state.stride(3);
        state[lane_base + l] = initial_state[init_index];
        l += 1usize;
    }

    let mut t = 0usize;
    while t < time {
        if t % chunk_size == 0usize {
            let chunk = t / chunk_size;
            if chunk < num_chunks {
                l = 0usize;
                while l < latent {
                    let boundary_index = b * boundary_states.stride(0)
                        + h * boundary_states.stride(1)
                        + chunk * boundary_states.stride(2)
                        + l * boundary_states.stride(3)
                        + d * boundary_states.stride(4);
                    boundary_states[boundary_index] = state[lane_base + l];
                    l += 1usize;
                }
            }
        }

        let mut erased_value = zero;
        l = 0usize;
        while l < latent {
            let decay_index = b * log_decay.stride(0)
                + h * log_decay.stride(1)
                + t * log_decay.stride(2)
                + l * log_decay.stride(3);
            let key_index =
                b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
            let erase_index = b * erase.stride(0)
                + h * erase.stride(1)
                + t * erase.stride(2)
                + l * erase.stride(3);
            let decayed = state[lane_base + l] * f32::exp(log_decay[decay_index]);
            state[lane_base + l] = decayed;
            erased_value += decayed * erase[erase_index] * key[key_index];
            l += 1usize;
        }

        let value_index =
            b * value.stride(0) + h * value.stride(1) + t * value.stride(2) + d * value.stride(3);
        let write_index =
            b * write.stride(0) + h * write.stride(1) + t * write.stride(2) + d * write.stride(3);
        let update = write[write_index] * value[value_index] - erased_value;

        let mut out_acc = zero;
        l = 0usize;
        while l < latent {
            let key_index =
                b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
            let query_index = b * query.stride(0)
                + h * query.stride(1)
                + t * query.stride(2)
                + l * query.stride(3);
            let next = state[lane_base + l] + key[key_index] * update;
            state[lane_base + l] = next;
            out_acc += query[query_index] * next;
            l += 1usize;
        }

        let context_index = b * context.stride(0)
            + h * context.stride(1)
            + t * context.stride(2)
            + d * context.stride(3);
        context[context_index] = out_acc * inv_scale;

        t += 1usize;
    }

    l = 0usize;
    while l < latent {
        let out_index = b * final_state.stride(0)
            + h * final_state.stride(1)
            + l * final_state.stride(2)
            + d * final_state.stride(3);
        final_state[out_index] = state[lane_base + l];
        let boundary_index = b * boundary_states.stride(0)
            + h * boundary_states.stride(1)
            + num_chunks * boundary_states.stride(2)
            + l * boundary_states.stride(3)
            + d * boundary_states.stride(4);
        boundary_states[boundary_index] = state[lane_base + l];
        l += 1usize;
    }
}

#[cube(launch)]
fn gdn2_zero_backward_kernel(
    grad_query: &mut Tensor<f32>,
    grad_key: &mut Tensor<f32>,
    grad_value: &mut Tensor<f32>,
    grad_erase: &mut Tensor<f32>,
    grad_write: &mut Tensor<f32>,
    grad_log_decay: &mut Tensor<f32>,
    grad_initial_state: &mut Tensor<f32>,
) {
    let index = ABSOLUTE_POS as usize;
    let zero = f32::cast_from(0u32);
    if index < grad_query.len() {
        grad_query[index] = zero;
    }
    if index < grad_key.len() {
        grad_key[index] = zero;
    }
    if index < grad_value.len() {
        grad_value[index] = zero;
    }
    if index < grad_erase.len() {
        grad_erase[index] = zero;
    }
    if index < grad_write.len() {
        grad_write[index] = zero;
    }
    if index < grad_log_decay.len() {
        grad_log_decay[index] = zero;
    }
    if index < grad_initial_state.len() {
        grad_initial_state[index] = zero;
    }
}

#[cube(launch)]
fn gdn2_backward_dense_block_parts_kernel(
    query: &Tensor<f32>,
    key: &Tensor<f32>,
    value: &Tensor<f32>,
    erase: &Tensor<f32>,
    write: &Tensor<f32>,
    log_decay: &Tensor<f32>,
    initial_state: &Tensor<f32>,
    boundary_states: &Tensor<f32>,
    grad_output: &Tensor<f32>,
    grad_query_block_parts: &mut Tensor<f32>,
    grad_key_block_parts: &mut Tensor<f32>,
    grad_value: &mut Tensor<f32>,
    grad_erase_block_parts: &mut Tensor<f32>,
    grad_write: &mut Tensor<f32>,
    grad_log_decay_block_parts: &mut Tensor<f32>,
    grad_initial_state: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] max_latent: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let dense = u32::cast_from(params[4]) as usize;
    let chunk_size = (u32::cast_from(params[5]) as usize).max(1usize);
    let num_chunks = u32::cast_from(params[6]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1usize);
    let h = z % heads.max(1usize);
    let block = CUBE_POS_X as usize;
    let block_d_start = (CUBE_POS_X * CUBE_DIM_X) as usize;
    let d = block_d_start + UNIT_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    if b >= batch || h >= heads || latent > max_latent {
        terminate!();
    }
    let active = d < dense;

    let zero = f32::cast_from(0u32);
    let scale = f32::sqrt(f32::cast_from(latent as u32));
    let inv_scale = f32::cast_from(1u32) / scale;
    let lane_base = lane * max_latent;
    let shared_len = GDN2_BACKWARD_WORKGROUP_X as usize * max_latent;
    let mut carry = SharedMemory::<f32>::new_aligned(shared_len, 1usize);
    let mut state = SharedMemory::<f32>::new_aligned(shared_len, 1usize);
    let mut prev_state = SharedMemory::<f32>::new_aligned(shared_len, 1usize);
    let mut decayed_state = SharedMemory::<f32>::new_aligned(shared_len, 1usize);
    let mut reduction = SharedMemory::<f32>::new_aligned(shared_len, 1usize);

    let mut l = 0usize;
    while l < latent {
        carry[lane_base + l] = zero;
        l += 1usize;
    }

    let mut chunk_rev = 0usize;
    while chunk_rev < num_chunks {
        let chunk = num_chunks - 1usize - chunk_rev;
        let chunk_start = chunk * chunk_size;
        let mut chunk_end = chunk_start + chunk_size;
        if chunk_end > time {
            chunk_end = time;
        }

        l = 0usize;
        while l < latent {
            if active {
                let boundary_index = b * boundary_states.stride(0)
                    + h * boundary_states.stride(1)
                    + (chunk + 1usize) * boundary_states.stride(2)
                    + l * boundary_states.stride(3)
                    + d * boundary_states.stride(4);
                state[lane_base + l] = boundary_states[boundary_index];
            } else {
                state[lane_base + l] = zero;
            }
            l += 1usize;
        }

        let mut t_rev = 0usize;
        while t_rev < chunk_end - chunk_start {
            let t = chunk_end - 1usize - t_rev;

            let value_value = if active {
                let value_index = b * value.stride(0)
                    + h * value.stride(1)
                    + t * value.stride(2)
                    + d * value.stride(3);
                value[value_index]
            } else {
                zero
            };
            let write_value = if active {
                let write_index = b * write.stride(0)
                    + h * write.stride(1)
                    + t * write.stride(2)
                    + d * write.stride(3);
                write[write_index]
            } else {
                zero
            };
            let grad_o = if active {
                let grad_index = b * grad_output.stride(0)
                    + h * grad_output.stride(1)
                    + t * grad_output.stride(2)
                    + d * grad_output.stride(3);
                grad_output[grad_index]
            } else {
                zero
            };
            let write_value_times_value = write_value * value_value;

            let mut denom = f32::cast_from(1u32);
            let mut erased_numer = zero;
            let mut min_decay = f32::cast_from(1u32);
            l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                let decay_index = b * log_decay.stride(0)
                    + h * log_decay.stride(1)
                    + t * log_decay.stride(2)
                    + l * log_decay.stride(3);
                let decay = f32::exp(log_decay[decay_index]);
                let key_value = key[key_index];
                let erased_key = erase[erase_index] * key_value;
                let next_minus_write = state[lane_base + l] - key_value * write_value_times_value;
                prev_state[lane_base + l] = next_minus_write;
                denom -= erased_key * key_value;
                erased_numer += erased_key * next_minus_write;
                if decay < min_decay {
                    min_decay = decay;
                }
                l += 1usize;
            }
            let denom_abs = if denom < zero { -denom } else { denom };
            let denom_epsilon = f32::cast_from(1u32) / f32::cast_from(10000u32);
            let mut erased_value = zero;
            if denom_abs < denom_epsilon
                || min_decay < denom_epsilon
                || t_rev % GDN2_INVERSE_REPLAY_INTERVAL == 0usize
            {
                l = 0usize;
                while l < latent {
                    if active {
                        let boundary_index = b * boundary_states.stride(0)
                            + h * boundary_states.stride(1)
                            + chunk * boundary_states.stride(2)
                            + l * boundary_states.stride(3)
                            + d * boundary_states.stride(4);
                        state[lane_base + l] = boundary_states[boundary_index];
                    } else {
                        state[lane_base + l] = zero;
                    }
                    l += 1usize;
                }

                let mut u = chunk_start;
                let replay_end = t + 1usize;
                while u < replay_end {
                    let mut erased_value_u = zero;
                    l = 0usize;
                    while l < latent {
                        let decay_index = b * log_decay.stride(0)
                            + h * log_decay.stride(1)
                            + u * log_decay.stride(2)
                            + l * log_decay.stride(3);
                        let key_index = b * key.stride(0)
                            + h * key.stride(1)
                            + u * key.stride(2)
                            + l * key.stride(3);
                        let erase_index = b * erase.stride(0)
                            + h * erase.stride(1)
                            + u * erase.stride(2)
                            + l * erase.stride(3);
                        if u == t {
                            prev_state[lane_base + l] = state[lane_base + l];
                        }
                        let decayed = state[lane_base + l] * f32::exp(log_decay[decay_index]);
                        if u == t {
                            decayed_state[lane_base + l] = decayed;
                        }
                        state[lane_base + l] = decayed;
                        erased_value_u += decayed * erase[erase_index] * key[key_index];
                        l += 1usize;
                    }

                    let update_u = if active {
                        let value_index = b * value.stride(0)
                            + h * value.stride(1)
                            + u * value.stride(2)
                            + d * value.stride(3);
                        let write_index = b * write.stride(0)
                            + h * write.stride(1)
                            + u * write.stride(2)
                            + d * write.stride(3);
                        write[write_index] * value[value_index] - erased_value_u
                    } else {
                        zero
                    };
                    l = 0usize;
                    while l < latent {
                        let key_index = b * key.stride(0)
                            + h * key.stride(1)
                            + u * key.stride(2)
                            + l * key.stride(3);
                        state[lane_base + l] = state[lane_base + l] + key[key_index] * update_u;
                        l += 1usize;
                    }
                    if u == t {
                        erased_value = erased_value_u;
                    }
                    u += 1usize;
                }
            } else {
                erased_value = erased_numer / denom;
                l = 0usize;
                while l < latent {
                    let key_index = b * key.stride(0)
                        + h * key.stride(1)
                        + t * key.stride(2)
                        + l * key.stride(3);
                    let decay_index = b * log_decay.stride(0)
                        + h * log_decay.stride(1)
                        + t * log_decay.stride(2)
                        + l * log_decay.stride(3);
                    let decay = f32::exp(log_decay[decay_index]);
                    let decayed = prev_state[lane_base + l] + key[key_index] * erased_value;
                    decayed_state[lane_base + l] = decayed;
                    prev_state[lane_base + l] = decayed / decay;
                    l += 1usize;
                }
            }
            let update = write_value_times_value - erased_value;

            let mut grad_update = zero;
            l = 0usize;
            while l < latent {
                let query_index = b * query.stride(0)
                    + h * query.stride(1)
                    + t * query.stride(2)
                    + l * query.stride(3);
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let grad_next = carry[lane_base + l] + query[query_index] * grad_o * inv_scale;
                carry[lane_base + l] = grad_next;
                grad_update += grad_next * key[key_index];
                reduction[lane_base + l] = state[lane_base + l] * grad_o * inv_scale;
                l += 1usize;
            }
            sync_cube();
            if lane == 0usize {
                l = 0usize;
                while l < latent {
                    let mut sum = zero;
                    let mut compensation = zero;
                    let mut reduce_lane = 0usize;
                    while reduce_lane < GDN2_BACKWARD_WORKGROUP_X as usize {
                        if block_d_start + reduce_lane < dense {
                            let y = reduction[reduce_lane * max_latent + l] - compensation;
                            let next = sum + y;
                            compensation = (next - sum) - y;
                            sum = next;
                        }
                        reduce_lane += 1usize;
                    }
                    let part_index = b * grad_query_block_parts.stride(0)
                        + h * grad_query_block_parts.stride(1)
                        + block * grad_query_block_parts.stride(2)
                        + t * grad_query_block_parts.stride(3)
                        + l * grad_query_block_parts.stride(4);
                    grad_query_block_parts[part_index] = sum;
                    l += 1usize;
                }
            }
            sync_cube();

            if active {
                let grad_write_index = b * grad_write.stride(0)
                    + h * grad_write.stride(1)
                    + t * grad_write.stride(2)
                    + d * grad_write.stride(3);
                let grad_value_index = b * grad_value.stride(0)
                    + h * grad_value.stride(1)
                    + t * grad_value.stride(2)
                    + d * grad_value.stride(3);
                grad_write[grad_write_index] = grad_update * value_value;
                grad_value[grad_value_index] = grad_update * write_value;
            }

            let grad_erased_value = -grad_update;
            l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                let grad_next = carry[lane_base + l];
                let grad_erased_key = decayed_state[lane_base + l] * grad_erased_value;
                let grad_k = grad_next * update + grad_erased_key * erase[erase_index];
                reduction[lane_base + l] = grad_k;
                l += 1usize;
            }
            sync_cube();
            if lane == 0usize {
                l = 0usize;
                while l < latent {
                    let mut sum = zero;
                    let mut compensation = zero;
                    let mut reduce_lane = 0usize;
                    while reduce_lane < GDN2_BACKWARD_WORKGROUP_X as usize {
                        if block_d_start + reduce_lane < dense {
                            let y = reduction[reduce_lane * max_latent + l] - compensation;
                            let next = sum + y;
                            compensation = (next - sum) - y;
                            sum = next;
                        }
                        reduce_lane += 1usize;
                    }
                    let part_index = b * grad_key_block_parts.stride(0)
                        + h * grad_key_block_parts.stride(1)
                        + block * grad_key_block_parts.stride(2)
                        + t * grad_key_block_parts.stride(3)
                        + l * grad_key_block_parts.stride(4);
                    grad_key_block_parts[part_index] = sum;
                    l += 1usize;
                }
            }
            sync_cube();

            l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let grad_erased_key = decayed_state[lane_base + l] * grad_erased_value;
                reduction[lane_base + l] = grad_erased_key * key[key_index];
                l += 1usize;
            }
            sync_cube();
            if lane == 0usize {
                l = 0usize;
                while l < latent {
                    let mut sum = zero;
                    let mut compensation = zero;
                    let mut reduce_lane = 0usize;
                    while reduce_lane < GDN2_BACKWARD_WORKGROUP_X as usize {
                        if block_d_start + reduce_lane < dense {
                            let y = reduction[reduce_lane * max_latent + l] - compensation;
                            let next = sum + y;
                            compensation = (next - sum) - y;
                            sum = next;
                        }
                        reduce_lane += 1usize;
                    }
                    let part_index = b * grad_erase_block_parts.stride(0)
                        + h * grad_erase_block_parts.stride(1)
                        + block * grad_erase_block_parts.stride(2)
                        + t * grad_erase_block_parts.stride(3)
                        + l * grad_erase_block_parts.stride(4);
                    grad_erase_block_parts[part_index] = sum;
                    l += 1usize;
                }
            }
            sync_cube();

            l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                let decay_index = b * log_decay.stride(0)
                    + h * log_decay.stride(1)
                    + t * log_decay.stride(2)
                    + l * log_decay.stride(3);
                let decay = f32::exp(log_decay[decay_index]);
                let erased_key = erase[erase_index] * key[key_index];
                let grad_next = carry[lane_base + l];
                let grad_decayed = grad_next + erased_key * grad_erased_value;
                reduction[lane_base + l] = grad_decayed * decayed_state[lane_base + l];
                carry[lane_base + l] = grad_decayed * decay;
                l += 1usize;
            }
            sync_cube();
            if lane == 0usize {
                l = 0usize;
                while l < latent {
                    let mut sum = zero;
                    let mut compensation = zero;
                    let mut reduce_lane = 0usize;
                    while reduce_lane < GDN2_BACKWARD_WORKGROUP_X as usize {
                        if block_d_start + reduce_lane < dense {
                            let y = reduction[reduce_lane * max_latent + l] - compensation;
                            let next = sum + y;
                            compensation = (next - sum) - y;
                            sum = next;
                        }
                        reduce_lane += 1usize;
                    }
                    let part_index = b * grad_log_decay_block_parts.stride(0)
                        + h * grad_log_decay_block_parts.stride(1)
                        + block * grad_log_decay_block_parts.stride(2)
                        + t * grad_log_decay_block_parts.stride(3)
                        + l * grad_log_decay_block_parts.stride(4);
                    grad_log_decay_block_parts[part_index] = sum;
                    l += 1usize;
                }
            }
            sync_cube();

            l = 0usize;
            while l < latent {
                state[lane_base + l] = prev_state[lane_base + l];
                l += 1usize;
            }

            t_rev += 1usize;
        }
        chunk_rev += 1usize;
    }

    l = 0usize;
    while l < latent {
        if active {
            let initial_index = b * grad_initial_state.stride(0)
                + h * grad_initial_state.stride(1)
                + l * grad_initial_state.stride(2)
                + d * grad_initial_state.stride(3);
            grad_initial_state[initial_index] = carry[lane_base + l];
        }
        l += 1usize;
    }
}

#[cube(launch)]
fn gdn2_backward_column_replay_kernel(
    query: &Tensor<f32>,
    key: &Tensor<f32>,
    value: &Tensor<f32>,
    erase: &Tensor<f32>,
    write: &Tensor<f32>,
    log_decay: &Tensor<f32>,
    initial_state: &Tensor<f32>,
    boundary_states: &Tensor<f32>,
    grad_output: &Tensor<f32>,
    grad_query: &mut Tensor<Atomic<f32>>,
    grad_key: &mut Tensor<Atomic<f32>>,
    grad_value: &mut Tensor<f32>,
    grad_erase: &mut Tensor<Atomic<f32>>,
    grad_write: &mut Tensor<f32>,
    grad_log_decay: &mut Tensor<Atomic<f32>>,
    grad_initial_state: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] max_latent: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let dense = u32::cast_from(params[4]) as usize;
    let chunk_size = (u32::cast_from(params[5]) as usize).max(1usize);
    let num_chunks = u32::cast_from(params[6]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1usize);
    let h = z % heads.max(1usize);
    let d = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || d >= dense || latent > max_latent {
        terminate!();
    }

    let zero = f32::cast_from(0u32);
    let scale = f32::sqrt(f32::cast_from(latent as u32));
    let inv_scale = f32::cast_from(1u32) / scale;
    let mut carry = SharedMemory::<f32>::new_aligned(max_latent, 1usize);
    let mut state = SharedMemory::<f32>::new_aligned(max_latent, 1usize);
    let mut prev_state = SharedMemory::<f32>::new_aligned(max_latent, 1usize);
    let mut decayed_state = SharedMemory::<f32>::new_aligned(max_latent, 1usize);

    let mut l = 0usize;
    while l < latent {
        carry[l] = zero;
        l += 1usize;
    }

    let mut chunk_rev = 0usize;
    while chunk_rev < num_chunks {
        let chunk = num_chunks - 1usize - chunk_rev;
        let chunk_start = chunk * chunk_size;
        let mut chunk_end = chunk_start + chunk_size;
        if chunk_end > time {
            chunk_end = time;
        }

        let mut t_rev = 0usize;
        while t_rev < chunk_end - chunk_start {
            let t = chunk_end - 1usize - t_rev;

            l = 0usize;
            while l < latent {
                let boundary_index = b * boundary_states.stride(0)
                    + h * boundary_states.stride(1)
                    + chunk * boundary_states.stride(2)
                    + l * boundary_states.stride(3)
                    + d * boundary_states.stride(4);
                state[l] = boundary_states[boundary_index];
                l += 1usize;
            }

            let mut u = chunk_start;
            let replay_end = t + 1usize;
            while u < replay_end {
                let mut erased_value_u = zero;
                l = 0usize;
                while l < latent {
                    let decay_index = b * log_decay.stride(0)
                        + h * log_decay.stride(1)
                        + u * log_decay.stride(2)
                        + l * log_decay.stride(3);
                    let key_index = b * key.stride(0)
                        + h * key.stride(1)
                        + u * key.stride(2)
                        + l * key.stride(3);
                    let erase_index = b * erase.stride(0)
                        + h * erase.stride(1)
                        + u * erase.stride(2)
                        + l * erase.stride(3);
                    if u == t {
                        prev_state[l] = state[l];
                    }
                    let decayed = state[l] * f32::exp(log_decay[decay_index]);
                    if u == t {
                        decayed_state[l] = decayed;
                    }
                    state[l] = decayed;
                    erased_value_u += decayed * erase[erase_index] * key[key_index];
                    l += 1usize;
                }

                let value_index = b * value.stride(0)
                    + h * value.stride(1)
                    + u * value.stride(2)
                    + d * value.stride(3);
                let write_index = b * write.stride(0)
                    + h * write.stride(1)
                    + u * write.stride(2)
                    + d * write.stride(3);
                let update_u = write[write_index] * value[value_index] - erased_value_u;
                l = 0usize;
                while l < latent {
                    let key_index = b * key.stride(0)
                        + h * key.stride(1)
                        + u * key.stride(2)
                        + l * key.stride(3);
                    state[l] = state[l] + key[key_index] * update_u;
                    l += 1usize;
                }
                u += 1usize;
            }

            let value_index = b * value.stride(0)
                + h * value.stride(1)
                + t * value.stride(2)
                + d * value.stride(3);
            let write_index = b * write.stride(0)
                + h * write.stride(1)
                + t * write.stride(2)
                + d * write.stride(3);
            let grad_index = b * grad_output.stride(0)
                + h * grad_output.stride(1)
                + t * grad_output.stride(2)
                + d * grad_output.stride(3);
            let grad_o = grad_output[grad_index];

            let mut erased_value = zero;
            l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                erased_value += decayed_state[l] * erase[erase_index] * key[key_index];
                l += 1usize;
            }
            let update = write[write_index] * value[value_index] - erased_value;

            let mut grad_update = zero;
            l = 0usize;
            while l < latent {
                let query_index = b * query.stride(0)
                    + h * query.stride(1)
                    + t * query.stride(2)
                    + l * query.stride(3);
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let grad_next = carry[l] + query[query_index] * grad_o * inv_scale;
                carry[l] = grad_next;
                grad_update += grad_next * key[key_index];

                let grad_q_index = b * grad_query.stride(0)
                    + h * grad_query.stride(1)
                    + t * grad_query.stride(2)
                    + l * grad_query.stride(3);
                grad_query[grad_q_index].fetch_add(state[l] * grad_o * inv_scale);
                l += 1usize;
            }

            let grad_write_index = b * grad_write.stride(0)
                + h * grad_write.stride(1)
                + t * grad_write.stride(2)
                + d * grad_write.stride(3);
            let grad_value_index = b * grad_value.stride(0)
                + h * grad_value.stride(1)
                + t * grad_value.stride(2)
                + d * grad_value.stride(3);
            grad_write[grad_write_index] = grad_update * value[value_index];
            grad_value[grad_value_index] = grad_update * write[write_index];

            let grad_erased_value = -grad_update;
            l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                let decay_index = b * log_decay.stride(0)
                    + h * log_decay.stride(1)
                    + t * log_decay.stride(2)
                    + l * log_decay.stride(3);
                let decay = f32::exp(log_decay[decay_index]);
                let erased_key = erase[erase_index] * key[key_index];
                let grad_next = carry[l];
                let grad_decayed = grad_next + erased_key * grad_erased_value;
                let grad_erased_key = decayed_state[l] * grad_erased_value;
                let grad_k = grad_next * update + grad_erased_key * erase[erase_index];
                let grad_erase_value = grad_erased_key * key[key_index];
                let grad_log_decay_value = grad_decayed * prev_state[l] * decay;

                let grad_k_index = b * grad_key.stride(0)
                    + h * grad_key.stride(1)
                    + t * grad_key.stride(2)
                    + l * grad_key.stride(3);
                let grad_erase_index = b * grad_erase.stride(0)
                    + h * grad_erase.stride(1)
                    + t * grad_erase.stride(2)
                    + l * grad_erase.stride(3);
                let grad_log_decay_index = b * grad_log_decay.stride(0)
                    + h * grad_log_decay.stride(1)
                    + t * grad_log_decay.stride(2)
                    + l * grad_log_decay.stride(3);
                grad_key[grad_k_index].fetch_add(grad_k);
                grad_erase[grad_erase_index].fetch_add(grad_erase_value);
                grad_log_decay[grad_log_decay_index].fetch_add(grad_log_decay_value);
                carry[l] = grad_decayed * decay;
                l += 1usize;
            }

            t_rev += 1usize;
        }
        chunk_rev += 1usize;
    }

    l = 0usize;
    while l < latent {
        let initial_index = b * grad_initial_state.stride(0)
            + h * grad_initial_state.stride(1)
            + l * grad_initial_state.stride(2)
            + d * grad_initial_state.stride(3);
        grad_initial_state[initial_index] = carry[l];
        l += 1usize;
    }
}

#[cube(launch)]
fn gdn2_backward_column_parts_kernel(
    query: &Tensor<f32>,
    key: &Tensor<f32>,
    value: &Tensor<f32>,
    erase: &Tensor<f32>,
    write: &Tensor<f32>,
    log_decay: &Tensor<f32>,
    initial_state: &Tensor<f32>,
    boundary_states: &Tensor<f32>,
    grad_output: &Tensor<f32>,
    grad_query_parts: &mut Tensor<f32>,
    grad_key_parts: &mut Tensor<f32>,
    grad_erase_parts: &mut Tensor<f32>,
    grad_log_decay_parts: &mut Tensor<f32>,
    grad_value: &mut Tensor<f32>,
    grad_write: &mut Tensor<f32>,
    grad_initial_state: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] max_latent: usize,
    #[comptime] max_chunk_size: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let dense = u32::cast_from(params[4]) as usize;
    let chunk_size = (u32::cast_from(params[5]) as usize).max(1usize);
    let num_chunks = u32::cast_from(params[6]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1usize);
    let h = z % heads.max(1usize);
    let d = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || d >= dense || latent > max_latent {
        terminate!();
    }

    let zero = f32::cast_from(0u32);
    let scale = f32::sqrt(f32::cast_from(latent as u32));
    let inv_scale = f32::cast_from(1u32) / scale;
    let mut carry = SharedMemory::<f32>::new_aligned(max_latent, 1usize);
    let mut state = SharedMemory::<f32>::new_aligned(max_latent, 1usize);
    let mut prev_states = SharedMemory::<f32>::new_aligned(max_chunk_size * max_latent, 1usize);

    let mut l = 0usize;
    while l < latent {
        carry[l] = zero;
        l += 1usize;
    }

    let mut chunk_rev = 0usize;
    while chunk_rev < num_chunks {
        let chunk = num_chunks - 1usize - chunk_rev;
        let chunk_start = chunk * chunk_size;
        let mut chunk_end = chunk_start + chunk_size;
        if chunk_end > time {
            chunk_end = time;
        }
        let chunk_len = chunk_end - chunk_start;
        if chunk_len > max_chunk_size {
            terminate!();
        }

        l = 0usize;
        while l < latent {
            let boundary_index = b * boundary_states.stride(0)
                + h * boundary_states.stride(1)
                + chunk * boundary_states.stride(2)
                + l * boundary_states.stride(3)
                + d * boundary_states.stride(4);
            state[l] = boundary_states[boundary_index];
            l += 1usize;
        }

        let mut local = 0usize;
        while local < chunk_len {
            let t = chunk_start + local;
            let state_base = local * max_latent;
            let mut erased_value = zero;
            l = 0usize;
            while l < latent {
                let decay_index = b * log_decay.stride(0)
                    + h * log_decay.stride(1)
                    + t * log_decay.stride(2)
                    + l * log_decay.stride(3);
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                prev_states[state_base + l] = state[l];
                let decayed = state[l] * f32::exp(log_decay[decay_index]);
                state[l] = decayed;
                erased_value += decayed * erase[erase_index] * key[key_index];
                l += 1usize;
            }

            let value_index = b * value.stride(0)
                + h * value.stride(1)
                + t * value.stride(2)
                + d * value.stride(3);
            let write_index = b * write.stride(0)
                + h * write.stride(1)
                + t * write.stride(2)
                + d * write.stride(3);
            let update = write[write_index] * value[value_index] - erased_value;
            l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                state[l] = state[l] + key[key_index] * update;
                l += 1usize;
            }
            local += 1usize;
        }

        let mut t_rev = 0usize;
        while t_rev < chunk_len {
            let local_rev = chunk_len - 1usize - t_rev;
            let t = chunk_start + local_rev;
            let state_base = local_rev * max_latent;

            let value_index = b * value.stride(0)
                + h * value.stride(1)
                + t * value.stride(2)
                + d * value.stride(3);
            let write_index = b * write.stride(0)
                + h * write.stride(1)
                + t * write.stride(2)
                + d * write.stride(3);
            let grad_index = b * grad_output.stride(0)
                + h * grad_output.stride(1)
                + t * grad_output.stride(2)
                + d * grad_output.stride(3);
            let grad_o = grad_output[grad_index];

            let mut erased_value = zero;
            l = 0usize;
            while l < latent {
                let decay_index = b * log_decay.stride(0)
                    + h * log_decay.stride(1)
                    + t * log_decay.stride(2)
                    + l * log_decay.stride(3);
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                let decayed = prev_states[state_base + l] * f32::exp(log_decay[decay_index]);
                erased_value += decayed * erase[erase_index] * key[key_index];
                l += 1usize;
            }
            let update = write[write_index] * value[value_index] - erased_value;

            let mut grad_update = zero;
            l = 0usize;
            while l < latent {
                let query_index = b * query.stride(0)
                    + h * query.stride(1)
                    + t * query.stride(2)
                    + l * query.stride(3);
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let decay_index = b * log_decay.stride(0)
                    + h * log_decay.stride(1)
                    + t * log_decay.stride(2)
                    + l * log_decay.stride(3);
                let decayed = prev_states[state_base + l] * f32::exp(log_decay[decay_index]);
                let next_state = decayed + key[key_index] * update;
                let grad_next = carry[l] + query[query_index] * grad_o * inv_scale;
                carry[l] = grad_next;
                grad_update += grad_next * key[key_index];

                let part_index = b * grad_query_parts.stride(0)
                    + h * grad_query_parts.stride(1)
                    + t * grad_query_parts.stride(2)
                    + l * grad_query_parts.stride(3)
                    + d * grad_query_parts.stride(4);
                grad_query_parts[part_index] = next_state * grad_o * inv_scale;
                l += 1usize;
            }

            let grad_write_index = b * grad_write.stride(0)
                + h * grad_write.stride(1)
                + t * grad_write.stride(2)
                + d * grad_write.stride(3);
            let grad_value_index = b * grad_value.stride(0)
                + h * grad_value.stride(1)
                + t * grad_value.stride(2)
                + d * grad_value.stride(3);
            grad_write[grad_write_index] = grad_update * value[value_index];
            grad_value[grad_value_index] = grad_update * write[write_index];

            let grad_erased_value = -grad_update;
            l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                let decay_index = b * log_decay.stride(0)
                    + h * log_decay.stride(1)
                    + t * log_decay.stride(2)
                    + l * log_decay.stride(3);
                let decay = f32::exp(log_decay[decay_index]);
                let erased_key = erase[erase_index] * key[key_index];
                let grad_next = carry[l];
                let grad_decayed = grad_next + erased_key * grad_erased_value;
                let decayed = prev_states[state_base + l] * decay;
                let grad_erased_key = decayed * grad_erased_value;

                let part_index = b * grad_key_parts.stride(0)
                    + h * grad_key_parts.stride(1)
                    + t * grad_key_parts.stride(2)
                    + l * grad_key_parts.stride(3)
                    + d * grad_key_parts.stride(4);
                grad_key_parts[part_index] =
                    grad_next * update + grad_erased_key * erase[erase_index];
                grad_erase_parts[part_index] = grad_erased_key * key[key_index];
                grad_log_decay_parts[part_index] = grad_decayed * decayed;

                carry[l] = grad_decayed * decay;
                l += 1usize;
            }

            t_rev += 1usize;
        }
        chunk_rev += 1usize;
    }

    l = 0usize;
    while l < latent {
        let initial_index = b * grad_initial_state.stride(0)
            + h * grad_initial_state.stride(1)
            + l * grad_initial_state.stride(2)
            + d * grad_initial_state.stride(3);
        grad_initial_state[initial_index] = carry[l];
        l += 1usize;
    }
}

#[cube(launch)]
fn gdn2_backward_chunk_wy_kernel(
    query: &Tensor<f32>,
    key: &Tensor<f32>,
    value: &Tensor<f32>,
    erase: &Tensor<f32>,
    write: &Tensor<f32>,
    log_decay: &Tensor<f32>,
    boundary_states: &Tensor<f32>,
    grad_output: &Tensor<f32>,
    grad_query: &mut Tensor<Atomic<f32>>,
    grad_key: &mut Tensor<Atomic<f32>>,
    grad_value: &mut Tensor<f32>,
    grad_erase: &mut Tensor<Atomic<f32>>,
    grad_write: &mut Tensor<f32>,
    grad_log_decay: &mut Tensor<Atomic<f32>>,
    grad_initial_state: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] max_latent: usize,
    #[comptime] max_chunk_size: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let dense = u32::cast_from(params[4]) as usize;
    let chunk_size = (u32::cast_from(params[5]) as usize).max(1usize);
    let num_chunks = u32::cast_from(params[6]) as usize;

    let z_pos = CUBE_POS_Z as usize;
    let b = z_pos / heads.max(1usize);
    let h = z_pos % heads.max(1usize);
    let d = CUBE_POS_X as usize;
    if b >= batch || h >= heads || d >= dense || latent > max_latent || UNIT_POS_X != 0 {
        terminate!();
    }

    let zero = f32::cast_from(0u32);
    let one = f32::cast_from(1u32);
    let scale = f32::sqrt(f32::cast_from(latent as u32));
    let inv_scale = one / scale;

    let chunk_latent_len = max_chunk_size * max_latent;
    let mut cumulative_decay = SharedMemory::<f32>::new_aligned(chunk_latent_len, 1usize);
    let mut grad_cumulative_decay = SharedMemory::<f32>::new_aligned(chunk_latent_len, 1usize);
    let mut solved_update = SharedMemory::<f32>::new_aligned(max_chunk_size, 1usize);
    let mut grad_solved_update = SharedMemory::<f32>::new_aligned(max_chunk_size, 1usize);
    let mut carry_boundary = SharedMemory::<f32>::new_aligned(max_latent, 1usize);

    let mut l = 0usize;
    while l < latent {
        carry_boundary[l] = zero;
        l += 1usize;
    }

    let mut chunk_rev = 0usize;
    while chunk_rev < num_chunks {
        let chunk = num_chunks - 1usize - chunk_rev;
        let chunk_start = chunk * chunk_size;
        let mut chunk_end = chunk_start + chunk_size;
        if chunk_end > time {
            chunk_end = time;
        }
        let chunk_len = chunk_end - chunk_start;
        if chunk_len > max_chunk_size {
            terminate!();
        }

        let mut local = 0usize;
        while local < chunk_len {
            let t = chunk_start + local;
            l = 0usize;
            while l < latent {
                let decay_index = b * log_decay.stride(0)
                    + h * log_decay.stride(1)
                    + t * log_decay.stride(2)
                    + l * log_decay.stride(3);
                let decay = f32::exp(log_decay[decay_index]);
                let basis_index = local * max_latent + l;
                cumulative_decay[basis_index] = if local == 0usize {
                    decay
                } else {
                    cumulative_decay[(local - 1usize) * max_latent + l] * decay
                };
                grad_cumulative_decay[basis_index] = zero;
                l += 1usize;
            }
            solved_update[local] = zero;
            grad_solved_update[local] = zero;
            local += 1usize;
        }

        local = 0usize;
        while local < chunk_len {
            let t = chunk_start + local;
            let value_index = b * value.stride(0)
                + h * value.stride(1)
                + t * value.stride(2)
                + d * value.stride(3);
            let write_index = b * write.stride(0)
                + h * write.stride(1)
                + t * write.stride(2)
                + d * write.stride(3);
            let mut rhs = write[write_index] * value[value_index];

            l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                let boundary_index = b * boundary_states.stride(0)
                    + h * boundary_states.stride(1)
                    + chunk * boundary_states.stride(2)
                    + l * boundary_states.stride(3)
                    + d * boundary_states.stride(4);
                let decay_c = cumulative_decay[local * max_latent + l];
                let m_basis = erase[erase_index] * key[key_index] * decay_c;
                rhs -= m_basis * boundary_states[boundary_index];
                l += 1usize;
            }

            let mut previous = 0usize;
            while previous < local {
                let t_previous = chunk_start + previous;
                let mut lower = zero;
                l = 0usize;
                while l < latent {
                    let key_index = b * key.stride(0)
                        + h * key.stride(1)
                        + t * key.stride(2)
                        + l * key.stride(3);
                    let erase_index = b * erase.stride(0)
                        + h * erase.stride(1)
                        + t * erase.stride(2)
                        + l * erase.stride(3);
                    let previous_key_index = b * key.stride(0)
                        + h * key.stride(1)
                        + t_previous * key.stride(2)
                        + l * key.stride(3);
                    let m_basis = erase[erase_index]
                        * key[key_index]
                        * cumulative_decay[local * max_latent + l];
                    let w_basis =
                        key[previous_key_index] / cumulative_decay[previous * max_latent + l];
                    lower += m_basis * w_basis;
                    l += 1usize;
                }
                rhs -= lower * solved_update[previous];
                previous += 1usize;
            }
            solved_update[local] = rhs;
            local += 1usize;
        }

        if chunk_len > 0usize {
            let last = chunk_len - 1usize;
            l = 0usize;
            while l < latent {
                let boundary_index = b * boundary_states.stride(0)
                    + h * boundary_states.stride(1)
                    + chunk * boundary_states.stride(2)
                    + l * boundary_states.stride(3)
                    + d * boundary_states.stride(4);
                let mut transformed_state = boundary_states[boundary_index];
                let mut j = 0usize;
                while j < chunk_len {
                    let t_j = chunk_start + j;
                    let key_index = b * key.stride(0)
                        + h * key.stride(1)
                        + t_j * key.stride(2)
                        + l * key.stride(3);
                    transformed_state +=
                        key[key_index] / cumulative_decay[j * max_latent + l] * solved_update[j];
                    j += 1usize;
                }
                grad_cumulative_decay[last * max_latent + l] +=
                    carry_boundary[l] * transformed_state;
                carry_boundary[l] = carry_boundary[l] * cumulative_decay[last * max_latent + l];
                l += 1usize;
            }
        }

        let mut local_rev = 0usize;
        while local_rev < chunk_len {
            let local_index = chunk_len - 1usize - local_rev;
            let t = chunk_start + local_index;
            let grad_index = b * grad_output.stride(0)
                + h * grad_output.stride(1)
                + t * grad_output.stride(2)
                + d * grad_output.stride(3);
            let grad_o_scaled = grad_output[grad_index] * inv_scale;
            let z_value = solved_update[local_index];

            l = 0usize;
            while l < latent {
                let boundary_index = b * boundary_states.stride(0)
                    + h * boundary_states.stride(1)
                    + chunk * boundary_states.stride(2)
                    + l * boundary_states.stride(3)
                    + d * boundary_states.stride(4);
                let mut transformed_state = boundary_states[boundary_index];
                let mut j = 0usize;
                while j <= local_index {
                    let t_j = chunk_start + j;
                    let key_j_index = b * key.stride(0)
                        + h * key.stride(1)
                        + t_j * key.stride(2)
                        + l * key.stride(3);
                    transformed_state +=
                        key[key_j_index] / cumulative_decay[j * max_latent + l] * solved_update[j];
                    j += 1usize;
                }
                let decay_c = cumulative_decay[local_index * max_latent + l];
                let state_value = decay_c * transformed_state;
                let query_index = b * query.stride(0)
                    + h * query.stride(1)
                    + t * query.stride(2)
                    + l * query.stride(3);
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let grad_state = query[query_index] * grad_o_scaled;

                let grad_query_index = b * grad_query.stride(0)
                    + h * grad_query.stride(1)
                    + t * grad_query.stride(2)
                    + l * grad_query.stride(3);
                grad_query[grad_query_index].fetch_add(state_value * grad_o_scaled);

                grad_cumulative_decay[local_index * max_latent + l] +=
                    grad_state * transformed_state;
                carry_boundary[l] += grad_state * decay_c;

                let grad_w_basis = carry_boundary[l] * z_value;
                let grad_key_index = b * grad_key.stride(0)
                    + h * grad_key.stride(1)
                    + t * grad_key.stride(2)
                    + l * grad_key.stride(3);
                grad_key[grad_key_index].fetch_add(grad_w_basis / decay_c);
                grad_cumulative_decay[local_index * max_latent + l] -=
                    grad_w_basis * key[key_index] / (decay_c * decay_c);
                grad_solved_update[local_index] += carry_boundary[l] * key[key_index] / decay_c;
                l += 1usize;
            }

            local_rev += 1usize;
        }

        local_rev = 0usize;
        while local_rev < chunk_len {
            let local_index = chunk_len - 1usize - local_rev;
            let t = chunk_start + local_index;
            let grad_rhs = grad_solved_update[local_index];

            let mut previous = 0usize;
            while previous < local_index {
                let t_previous = chunk_start + previous;
                let mut lower = zero;
                l = 0usize;
                while l < latent {
                    let key_index = b * key.stride(0)
                        + h * key.stride(1)
                        + t * key.stride(2)
                        + l * key.stride(3);
                    let erase_index = b * erase.stride(0)
                        + h * erase.stride(1)
                        + t * erase.stride(2)
                        + l * erase.stride(3);
                    let previous_key_index = b * key.stride(0)
                        + h * key.stride(1)
                        + t_previous * key.stride(2)
                        + l * key.stride(3);
                    let m_basis = erase[erase_index]
                        * key[key_index]
                        * cumulative_decay[local_index * max_latent + l];
                    let w_basis =
                        key[previous_key_index] / cumulative_decay[previous * max_latent + l];
                    lower += m_basis * w_basis;
                    l += 1usize;
                }

                let grad_lower = -grad_rhs * solved_update[previous];
                grad_solved_update[previous] -= grad_rhs * lower;

                l = 0usize;
                while l < latent {
                    let key_index = b * key.stride(0)
                        + h * key.stride(1)
                        + t * key.stride(2)
                        + l * key.stride(3);
                    let erase_index = b * erase.stride(0)
                        + h * erase.stride(1)
                        + t * erase.stride(2)
                        + l * erase.stride(3);
                    let previous_key_index = b * key.stride(0)
                        + h * key.stride(1)
                        + t_previous * key.stride(2)
                        + l * key.stride(3);
                    let decay_c = cumulative_decay[local_index * max_latent + l];
                    let previous_decay_c = cumulative_decay[previous * max_latent + l];
                    let m_basis = erase[erase_index] * key[key_index] * decay_c;
                    let w_basis = key[previous_key_index] / previous_decay_c;

                    let grad_m_basis = grad_lower * w_basis;
                    let grad_key_index = b * grad_key.stride(0)
                        + h * grad_key.stride(1)
                        + t * grad_key.stride(2)
                        + l * grad_key.stride(3);
                    let grad_erase_index = b * grad_erase.stride(0)
                        + h * grad_erase.stride(1)
                        + t * grad_erase.stride(2)
                        + l * grad_erase.stride(3);
                    grad_key[grad_key_index].fetch_add(grad_m_basis * erase[erase_index] * decay_c);
                    grad_erase[grad_erase_index].fetch_add(grad_m_basis * key[key_index] * decay_c);
                    grad_cumulative_decay[local_index * max_latent + l] +=
                        grad_m_basis * erase[erase_index] * key[key_index];

                    let grad_w_basis = grad_lower * m_basis;
                    let previous_grad_key_index = b * grad_key.stride(0)
                        + h * grad_key.stride(1)
                        + t_previous * grad_key.stride(2)
                        + l * grad_key.stride(3);
                    grad_key[previous_grad_key_index].fetch_add(grad_w_basis / previous_decay_c);
                    grad_cumulative_decay[previous * max_latent + l] -= grad_w_basis
                        * key[previous_key_index]
                        / (previous_decay_c * previous_decay_c);
                    l += 1usize;
                }
                previous += 1usize;
            }

            let value_index = b * value.stride(0)
                + h * value.stride(1)
                + t * value.stride(2)
                + d * value.stride(3);
            let write_index = b * write.stride(0)
                + h * write.stride(1)
                + t * write.stride(2)
                + d * write.stride(3);
            let grad_write_index = b * grad_write.stride(0)
                + h * grad_write.stride(1)
                + t * grad_write.stride(2)
                + d * grad_write.stride(3);
            let grad_value_index = b * grad_value.stride(0)
                + h * grad_value.stride(1)
                + t * grad_value.stride(2)
                + d * grad_value.stride(3);
            grad_write[grad_write_index] = grad_rhs * value[value_index];
            grad_value[grad_value_index] = grad_rhs * write[write_index];

            l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                let boundary_index = b * boundary_states.stride(0)
                    + h * boundary_states.stride(1)
                    + chunk * boundary_states.stride(2)
                    + l * boundary_states.stride(3)
                    + d * boundary_states.stride(4);
                let decay_c = cumulative_decay[local_index * max_latent + l];
                let m_basis = erase[erase_index] * key[key_index] * decay_c;
                let grad_m_basis = -grad_rhs * boundary_states[boundary_index];
                let grad_key_index = b * grad_key.stride(0)
                    + h * grad_key.stride(1)
                    + t * grad_key.stride(2)
                    + l * grad_key.stride(3);
                let grad_erase_index = b * grad_erase.stride(0)
                    + h * grad_erase.stride(1)
                    + t * grad_erase.stride(2)
                    + l * grad_erase.stride(3);
                grad_key[grad_key_index].fetch_add(grad_m_basis * erase[erase_index] * decay_c);
                grad_erase[grad_erase_index].fetch_add(grad_m_basis * key[key_index] * decay_c);
                grad_cumulative_decay[local_index * max_latent + l] +=
                    grad_m_basis * erase[erase_index] * key[key_index];
                carry_boundary[l] -= grad_rhs * m_basis;
                l += 1usize;
            }

            local_rev += 1usize;
        }

        l = 0usize;
        while l < latent {
            let mut suffix = zero;
            local_rev = 0usize;
            while local_rev < chunk_len {
                let local_index = chunk_len - 1usize - local_rev;
                let t = chunk_start + local_index;
                suffix += grad_cumulative_decay[local_index * max_latent + l]
                    * cumulative_decay[local_index * max_latent + l];
                let grad_log_decay_index = b * grad_log_decay.stride(0)
                    + h * grad_log_decay.stride(1)
                    + t * grad_log_decay.stride(2)
                    + l * grad_log_decay.stride(3);
                grad_log_decay[grad_log_decay_index].fetch_add(suffix);
                local_rev += 1usize;
            }
            l += 1usize;
        }

        chunk_rev += 1usize;
    }

    l = 0usize;
    while l < latent {
        let initial_index = b * grad_initial_state.stride(0)
            + h * grad_initial_state.stride(1)
            + l * grad_initial_state.stride(2)
            + d * grad_initial_state.stride(3);
        grad_initial_state[initial_index] = carry_boundary[l];
        l += 1usize;
    }
}

#[cube(launch)]
fn gdn2_reduce_dense_block_parts_kernel(
    grad_query_block_parts: &Tensor<f32>,
    grad_key_block_parts: &Tensor<f32>,
    grad_erase_block_parts: &Tensor<f32>,
    grad_log_decay_block_parts: &Tensor<f32>,
    grad_query: &mut Tensor<f32>,
    grad_key: &mut Tensor<f32>,
    grad_erase: &mut Tensor<f32>,
    grad_log_decay: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let dense = u32::cast_from(params[4]) as usize;
    let dense_blocks = dense.div_ceil(GDN2_BACKWARD_WORKGROUP_X as usize);
    let index = ABSOLUTE_POS as usize;
    let total = batch * heads * time * latent;
    if index >= total {
        terminate!();
    }

    let l = index % latent;
    let t = (index / latent) % time;
    let h = (index / (latent * time)) % heads;
    let b = index / (latent * time * heads);
    let zero = f32::cast_from(0u32);
    let mut grad_q = zero;
    let mut grad_k = zero;
    let mut grad_e = zero;
    let mut grad_ld = zero;
    let mut grad_q_comp = zero;
    let mut grad_k_comp = zero;
    let mut grad_e_comp = zero;
    let mut grad_ld_comp = zero;

    let mut block = 0usize;
    while block < dense_blocks {
        let part_index = b * grad_query_block_parts.stride(0)
            + h * grad_query_block_parts.stride(1)
            + block * grad_query_block_parts.stride(2)
            + t * grad_query_block_parts.stride(3)
            + l * grad_query_block_parts.stride(4);
        let q_y = grad_query_block_parts[part_index] - grad_q_comp;
        let q_next = grad_q + q_y;
        grad_q_comp = (q_next - grad_q) - q_y;
        grad_q = q_next;
        let k_y = grad_key_block_parts[part_index] - grad_k_comp;
        let k_next = grad_k + k_y;
        grad_k_comp = (k_next - grad_k) - k_y;
        grad_k = k_next;
        let e_y = grad_erase_block_parts[part_index] - grad_e_comp;
        let e_next = grad_e + e_y;
        grad_e_comp = (e_next - grad_e) - e_y;
        grad_e = e_next;
        let ld_y = grad_log_decay_block_parts[part_index] - grad_ld_comp;
        let ld_next = grad_ld + ld_y;
        grad_ld_comp = (ld_next - grad_ld) - ld_y;
        grad_ld = ld_next;
        block += 1usize;
    }

    let grad_index = b * grad_query.stride(0)
        + h * grad_query.stride(1)
        + t * grad_query.stride(2)
        + l * grad_query.stride(3);
    grad_query[grad_index] = grad_q;
    grad_key[grad_index] = grad_k;
    grad_erase[grad_index] = grad_e;
    grad_log_decay[grad_index] = grad_ld;
}

#[cube(launch)]
fn gdn2_reduce_dense_parts_kernel(
    grad_query_parts: &Tensor<f32>,
    grad_key_parts: &Tensor<f32>,
    grad_erase_parts: &Tensor<f32>,
    grad_log_decay_parts: &Tensor<f32>,
    grad_query: &mut Tensor<f32>,
    grad_key: &mut Tensor<f32>,
    grad_erase: &mut Tensor<f32>,
    grad_log_decay: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let dense = u32::cast_from(params[4]) as usize;
    let index = ABSOLUTE_POS as usize;
    let total = batch * heads * time * latent;
    if index >= total {
        terminate!();
    }

    let l = index % latent;
    let t = (index / latent) % time;
    let h = (index / (latent * time)) % heads;
    let b = index / (latent * time * heads);
    let zero = f32::cast_from(0u32);
    let mut grad_q = zero;
    let mut grad_k = zero;
    let mut grad_e = zero;
    let mut grad_ld = zero;

    let mut d = 0usize;
    while d < dense {
        let part_index = b * grad_query_parts.stride(0)
            + h * grad_query_parts.stride(1)
            + t * grad_query_parts.stride(2)
            + l * grad_query_parts.stride(3)
            + d * grad_query_parts.stride(4);
        grad_q += grad_query_parts[part_index];
        grad_k += grad_key_parts[part_index];
        grad_e += grad_erase_parts[part_index];
        grad_ld += grad_log_decay_parts[part_index];
        d += 1usize;
    }

    let grad_index = b * grad_query.stride(0)
        + h * grad_query.stride(1)
        + t * grad_query.stride(2)
        + l * grad_query.stride(3);
    grad_query[grad_index] = grad_q;
    grad_key[grad_index] = grad_k;
    grad_erase[grad_index] = grad_e;
    grad_log_decay[grad_index] = grad_ld;
}

#[cube(launch)]
fn gdn2_backward_serial_v2_kernel(
    query: &Tensor<f32>,
    key: &Tensor<f32>,
    value: &Tensor<f32>,
    erase: &Tensor<f32>,
    write: &Tensor<f32>,
    log_decay: &Tensor<f32>,
    initial_state: &Tensor<f32>,
    boundary_states: &Tensor<f32>,
    grad_output: &Tensor<f32>,
    scratch_state: &mut Tensor<f32>,
    scratch_prev: &mut Tensor<f32>,
    scratch_decayed: &mut Tensor<f32>,
    grad_query: &mut Tensor<f32>,
    grad_key: &mut Tensor<f32>,
    grad_value: &mut Tensor<f32>,
    grad_erase: &mut Tensor<f32>,
    grad_write: &mut Tensor<f32>,
    grad_log_decay: &mut Tensor<f32>,
    grad_initial_state: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let dense = u32::cast_from(params[4]) as usize;
    let chunk_size = (u32::cast_from(params[5]) as usize).max(1usize);

    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1usize);
    let h = z % heads.max(1usize);
    if b >= batch || h >= heads || UNIT_POS_X != 0 {
        terminate!();
    }

    let zero = f32::cast_from(0u32);
    let scale = f32::sqrt(f32::cast_from(latent as u32));
    let inv_scale = f32::cast_from(1u32) / scale;

    let mut t_clear = 0usize;
    while t_clear < time {
        let mut l_clear = 0usize;
        while l_clear < latent {
            let latent_index = b * grad_query.stride(0)
                + h * grad_query.stride(1)
                + t_clear * grad_query.stride(2)
                + l_clear * grad_query.stride(3);
            grad_query[latent_index] = zero;
            grad_key[latent_index] = zero;
            grad_erase[latent_index] = zero;
            grad_log_decay[latent_index] = zero;
            l_clear += 1usize;
        }
        let mut d_clear = 0usize;
        while d_clear < dense {
            let dense_index = b * grad_value.stride(0)
                + h * grad_value.stride(1)
                + t_clear * grad_value.stride(2)
                + d_clear * grad_value.stride(3);
            grad_value[dense_index] = zero;
            grad_write[dense_index] = zero;
            d_clear += 1usize;
        }
        t_clear += 1usize;
    }

    let mut l_clear = 0usize;
    while l_clear < latent {
        let mut d_clear = 0usize;
        while d_clear < dense {
            let carry_index = b * grad_initial_state.stride(0)
                + h * grad_initial_state.stride(1)
                + l_clear * grad_initial_state.stride(2)
                + d_clear * grad_initial_state.stride(3);
            grad_initial_state[carry_index] = zero;
            d_clear += 1usize;
        }
        l_clear += 1usize;
    }

    let mut t_rev = 0usize;
    while t_rev < time {
        let t = time - 1usize - t_rev;
        let chunk = t / chunk_size;
        let chunk_start = chunk * chunk_size;

        let mut l = 0usize;
        while l < latent {
            let mut d = 0usize;
            while d < dense {
                let boundary_index = b * boundary_states.stride(0)
                    + h * boundary_states.stride(1)
                    + chunk * boundary_states.stride(2)
                    + l * boundary_states.stride(3)
                    + d * boundary_states.stride(4);
                let scratch_index = b * scratch_state.stride(0)
                    + h * scratch_state.stride(1)
                    + l * scratch_state.stride(2)
                    + d * scratch_state.stride(3);
                scratch_state[scratch_index] = boundary_states[boundary_index];
                d += 1usize;
            }
            l += 1usize;
        }

        let mut u = chunk_start;
        let replay_end = t + 1usize;
        while u < replay_end {
            let mut d = 0usize;
            while d < dense {
                let mut erased_value = zero;
                l = 0usize;
                while l < latent {
                    let scratch_index = b * scratch_state.stride(0)
                        + h * scratch_state.stride(1)
                        + l * scratch_state.stride(2)
                        + d * scratch_state.stride(3);
                    let prev = scratch_state[scratch_index];
                    let decay_index = b * log_decay.stride(0)
                        + h * log_decay.stride(1)
                        + u * log_decay.stride(2)
                        + l * log_decay.stride(3);
                    let decay = f32::exp(log_decay[decay_index]);
                    let decayed = prev * decay;
                    if u == t {
                        let prev_index = b * scratch_prev.stride(0)
                            + h * scratch_prev.stride(1)
                            + l * scratch_prev.stride(2)
                            + d * scratch_prev.stride(3);
                        let decayed_index = b * scratch_decayed.stride(0)
                            + h * scratch_decayed.stride(1)
                            + l * scratch_decayed.stride(2)
                            + d * scratch_decayed.stride(3);
                        scratch_prev[prev_index] = prev;
                        scratch_decayed[decayed_index] = decayed;
                    }
                    scratch_state[scratch_index] = decayed;
                    let key_index = b * key.stride(0)
                        + h * key.stride(1)
                        + u * key.stride(2)
                        + l * key.stride(3);
                    let erase_index = b * erase.stride(0)
                        + h * erase.stride(1)
                        + u * erase.stride(2)
                        + l * erase.stride(3);
                    erased_value += decayed * erase[erase_index] * key[key_index];
                    l += 1usize;
                }

                let value_index = b * value.stride(0)
                    + h * value.stride(1)
                    + u * value.stride(2)
                    + d * value.stride(3);
                let write_index = b * write.stride(0)
                    + h * write.stride(1)
                    + u * write.stride(2)
                    + d * write.stride(3);
                let update = write[write_index] * value[value_index] - erased_value;
                l = 0usize;
                while l < latent {
                    let scratch_index = b * scratch_state.stride(0)
                        + h * scratch_state.stride(1)
                        + l * scratch_state.stride(2)
                        + d * scratch_state.stride(3);
                    let key_index = b * key.stride(0)
                        + h * key.stride(1)
                        + u * key.stride(2)
                        + l * key.stride(3);
                    scratch_state[scratch_index] =
                        scratch_state[scratch_index] + key[key_index] * update;
                    l += 1usize;
                }
                d += 1usize;
            }
            u += 1usize;
        }

        let mut d = 0usize;
        while d < dense {
            let grad_index = b * grad_output.stride(0)
                + h * grad_output.stride(1)
                + t * grad_output.stride(2)
                + d * grad_output.stride(3);
            let grad_o = grad_output[grad_index];
            let value_index = b * value.stride(0)
                + h * value.stride(1)
                + t * value.stride(2)
                + d * value.stride(3);
            let write_index = b * write.stride(0)
                + h * write.stride(1)
                + t * write.stride(2)
                + d * write.stride(3);

            let mut erased_value = zero;
            l = 0usize;
            while l < latent {
                let decayed_index = b * scratch_decayed.stride(0)
                    + h * scratch_decayed.stride(1)
                    + l * scratch_decayed.stride(2)
                    + d * scratch_decayed.stride(3);
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                erased_value +=
                    scratch_decayed[decayed_index] * erase[erase_index] * key[key_index];
                l += 1usize;
            }
            let update = write[write_index] * value[value_index] - erased_value;

            let mut grad_update = zero;
            l = 0usize;
            while l < latent {
                let scratch_index = b * scratch_state.stride(0)
                    + h * scratch_state.stride(1)
                    + l * scratch_state.stride(2)
                    + d * scratch_state.stride(3);
                let query_index = b * query.stride(0)
                    + h * query.stride(1)
                    + t * query.stride(2)
                    + l * query.stride(3);
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let carry_index = b * grad_initial_state.stride(0)
                    + h * grad_initial_state.stride(1)
                    + l * grad_initial_state.stride(2)
                    + d * grad_initial_state.stride(3);
                let next_state = scratch_state[scratch_index];
                let grad_next =
                    grad_initial_state[carry_index] + query[query_index] * grad_o * inv_scale;
                scratch_state[scratch_index] = grad_next;
                grad_update += grad_next * key[key_index];
                let grad_q_index = b * grad_query.stride(0)
                    + h * grad_query.stride(1)
                    + t * grad_query.stride(2)
                    + l * grad_query.stride(3);
                grad_query[grad_q_index] =
                    grad_query[grad_q_index] + next_state * grad_o * inv_scale;
                l += 1usize;
            }

            let grad_write_index = b * grad_write.stride(0)
                + h * grad_write.stride(1)
                + t * grad_write.stride(2)
                + d * grad_write.stride(3);
            let grad_value_index = b * grad_value.stride(0)
                + h * grad_value.stride(1)
                + t * grad_value.stride(2)
                + d * grad_value.stride(3);
            grad_write[grad_write_index] = grad_update * value[value_index];
            grad_value[grad_value_index] = grad_update * write[write_index];

            let grad_erased_value = -grad_update;
            l = 0usize;
            while l < latent {
                let key_index =
                    b * key.stride(0) + h * key.stride(1) + t * key.stride(2) + l * key.stride(3);
                let erase_index = b * erase.stride(0)
                    + h * erase.stride(1)
                    + t * erase.stride(2)
                    + l * erase.stride(3);
                let decay_index = b * log_decay.stride(0)
                    + h * log_decay.stride(1)
                    + t * log_decay.stride(2)
                    + l * log_decay.stride(3);
                let prev_index = b * scratch_prev.stride(0)
                    + h * scratch_prev.stride(1)
                    + l * scratch_prev.stride(2)
                    + d * scratch_prev.stride(3);
                let decayed_index = b * scratch_decayed.stride(0)
                    + h * scratch_decayed.stride(1)
                    + l * scratch_decayed.stride(2)
                    + d * scratch_decayed.stride(3);
                let scratch_index = b * scratch_state.stride(0)
                    + h * scratch_state.stride(1)
                    + l * scratch_state.stride(2)
                    + d * scratch_state.stride(3);
                let carry_index = b * grad_initial_state.stride(0)
                    + h * grad_initial_state.stride(1)
                    + l * grad_initial_state.stride(2)
                    + d * grad_initial_state.stride(3);
                let decay = f32::exp(log_decay[decay_index]);
                let erased_key = erase[erase_index] * key[key_index];
                let grad_next = scratch_state[scratch_index];
                let grad_decayed = grad_next + erased_key * grad_erased_value;
                let grad_erased_key = scratch_decayed[decayed_index] * grad_erased_value;
                let grad_k_value = grad_next * update + grad_erased_key * erase[erase_index];
                let grad_erase_value = grad_erased_key * key[key_index];
                let grad_log_decay_value = grad_decayed * scratch_prev[prev_index] * decay;

                let grad_k_index = b * grad_key.stride(0)
                    + h * grad_key.stride(1)
                    + t * grad_key.stride(2)
                    + l * grad_key.stride(3);
                let grad_erase_index = b * grad_erase.stride(0)
                    + h * grad_erase.stride(1)
                    + t * grad_erase.stride(2)
                    + l * grad_erase.stride(3);
                let grad_log_decay_index = b * grad_log_decay.stride(0)
                    + h * grad_log_decay.stride(1)
                    + t * grad_log_decay.stride(2)
                    + l * grad_log_decay.stride(3);
                grad_key[grad_k_index] = grad_key[grad_k_index] + grad_k_value;
                grad_erase[grad_erase_index] = grad_erase[grad_erase_index] + grad_erase_value;
                grad_log_decay[grad_log_decay_index] =
                    grad_log_decay[grad_log_decay_index] + grad_log_decay_value;
                grad_initial_state[carry_index] = grad_decayed * decay;
                l += 1usize;
            }
            d += 1usize;
        }
        t_rev += 1usize;
    }
}

pub(crate) fn gdn2_forward_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    erase: CubeTensor<R>,
    write: CubeTensor<R>,
    log_decay: CubeTensor<R>,
    initial_state: CubeTensor<R>,
    params: CubeTensor<R>,
    _num_chunks: usize,
) -> Gdn2ForwardRuntimeOutput<R> {
    let query = into_contiguous(query);
    let key = into_contiguous(key);
    let value = into_contiguous(value);
    let erase = into_contiguous(erase);
    let write = into_contiguous(write);
    let log_decay = into_contiguous(log_decay);
    let initial_state = into_contiguous(initial_state);
    let params = into_contiguous(params);

    let [batch, heads, time, latent] = query.meta.shape.dims::<4>();
    let dense = value.meta.shape.dims::<4>()[3];
    let client = query.client.clone();
    let device = query.device.clone();
    let context = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, dense]),
    );
    let final_state = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, latent, dense]),
    );

    let boundary_states = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([
            batch,
            heads,
            _num_chunks.max(1usize) + 1usize,
            latent,
            dense,
        ]),
    );
    let cube_dim = CubeDim::new_1d(GDN2_FORWARD_WORKGROUP_X);
    let count_x = dense.div_ceil(GDN2_FORWARD_WORKGROUP_X as usize) as u32;
    let count_z = (batch * heads) as u32;
    let _ = gdn2_forward_kernel::launch::<R>(
        &client,
        CubeCount::Static(count_x, 1, count_z),
        cube_dim,
        query.into_tensor_arg(),
        key.into_tensor_arg(),
        value.into_tensor_arg(),
        erase.into_tensor_arg(),
        write.into_tensor_arg(),
        log_decay.into_tensor_arg(),
        initial_state.into_tensor_arg(),
        context.clone().into_tensor_arg(),
        final_state.clone().into_tensor_arg(),
        boundary_states.clone().into_tensor_arg(),
        params.into_tensor_arg(),
        latent,
    );

    Gdn2ForwardRuntimeOutput {
        context,
        final_state,
        boundary_states,
    }
}

pub(crate) fn gdn2_forward_params<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    chunk_size: usize,
) -> BurnTensor<B, 1> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let dense = value.shape().dims::<4>()[3];
    let num_chunks = time.div_ceil(chunk_size.max(1));
    gdn2_params::<B>(
        batch,
        heads,
        time,
        latent,
        dense,
        chunk_size,
        num_chunks,
        &query.device(),
    )
}

pub(crate) fn gdn2_wy_factors_runtime<R: CubeRuntime>(
    key: CubeTensor<R>,
    erase: CubeTensor<R>,
    log_decay: CubeTensor<R>,
    params: CubeTensor<R>,
    chunk_size_hint: usize,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let key = into_contiguous(key);
    let erase = into_contiguous(erase);
    let log_decay = into_contiguous(log_decay);
    let params = into_contiguous(params);

    let [batch, heads, time, latent] = key.meta.shape.dims::<4>();
    let runtime_chunk_size = chunk_size_hint.max(1usize);
    let runtime_num_chunks = time.div_ceil(runtime_chunk_size);
    let client = key.client.clone();
    let device = key.device.clone();
    let factor_latent_shape =
        Shape::new([batch, heads, runtime_num_chunks, runtime_chunk_size, latent]);
    let factor_matrix_shape = Shape::new([
        batch,
        heads,
        runtime_num_chunks,
        runtime_chunk_size,
        runtime_chunk_size,
    ]);
    let cumulative_decay =
        empty_device::<R, f32>(client.clone(), device.clone(), factor_latent_shape);
    let wy_lower = empty_device::<R, f32>(client.clone(), device, factor_matrix_shape);

    let factor_count = CubeCount::Static(runtime_num_chunks as u32, 1, (batch * heads) as u32);
    let _ = gdn2_prepare_wy_factors_kernel::launch::<R>(
        &client,
        factor_count,
        CubeDim::new_1d(1),
        key.into_tensor_arg(),
        erase.into_tensor_arg(),
        log_decay.into_tensor_arg(),
        cumulative_decay.clone().into_tensor_arg(),
        wy_lower.clone().into_tensor_arg(),
        params.into_tensor_arg(),
        latent,
        runtime_chunk_size,
    );

    (cumulative_decay, wy_lower)
}

fn gdn2_serial_backward_enabled() -> bool {
    matches!(
        std::env::var("BURN_DRAGON_GDN2_CUDA_SERIAL_BACKWARD")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON")
    )
}

fn gdn2_column_parts_backward_enabled() -> bool {
    matches!(
        std::env::var("BURN_DRAGON_GDN2_CUDA_COLUMN_PARTS_BACKWARD")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON")
    )
}

fn gdn2_block_parts_backward_enabled() -> bool {
    matches!(
        std::env::var("BURN_DRAGON_GDN2_CUDA_BLOCK_PARTS_BACKWARD")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON")
    )
}

pub(crate) fn gdn2_backward_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    erase: CubeTensor<R>,
    write: CubeTensor<R>,
    log_decay: CubeTensor<R>,
    initial_state: CubeTensor<R>,
    boundary_states: CubeTensor<R>,
    grad_output: CubeTensor<R>,
    params: CubeTensor<R>,
    chunk_size_hint: usize,
) -> Gdn2BackwardRuntimeOutput<R> {
    let query = into_contiguous(query);
    let key = into_contiguous(key);
    let value = into_contiguous(value);
    let erase = into_contiguous(erase);
    let write = into_contiguous(write);
    let log_decay = into_contiguous(log_decay);
    let initial_state = into_contiguous(initial_state);
    let boundary_states = into_contiguous(boundary_states);
    let grad_output = into_contiguous(grad_output);
    let params = into_contiguous(params);

    let [batch, heads, time, latent] = query.meta.shape.dims::<4>();
    let dense = value.meta.shape.dims::<4>()[3];
    let client = query.client.clone();
    let device = query.device.clone();
    let grad_query = zeros_client::<R>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, latent]),
        DType::F32,
    );
    let grad_key = zeros_client::<R>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, latent]),
        DType::F32,
    );
    let grad_value = zeros_client::<R>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, dense]),
        DType::F32,
    );
    let grad_erase = zeros_client::<R>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, latent]),
        DType::F32,
    );
    let grad_write = zeros_client::<R>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, dense]),
        DType::F32,
    );
    let grad_log_decay = zeros_client::<R>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, latent]),
        DType::F32,
    );
    let grad_initial_state = zeros_client::<R>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, latent, dense]),
        DType::F32,
    );

    let count_z = (batch * heads) as u32;
    if gdn2_serial_backward_enabled() {
        let scratch_state = empty_device::<R, f32>(
            client.clone(),
            device.clone(),
            Shape::new([batch, heads, latent, dense]),
        );
        let scratch_prev = empty_device::<R, f32>(
            client.clone(),
            device.clone(),
            Shape::new([batch, heads, latent, dense]),
        );
        let scratch_decayed = empty_device::<R, f32>(
            client.clone(),
            device,
            Shape::new([batch, heads, latent, dense]),
        );
        let _ = gdn2_backward_serial_v2_kernel::launch::<R>(
            &client,
            CubeCount::Static(1, 1, count_z),
            CubeDim::new_1d(1),
            query.into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            erase.into_tensor_arg(),
            write.into_tensor_arg(),
            log_decay.clone().into_tensor_arg(),
            initial_state.into_tensor_arg(),
            boundary_states.into_tensor_arg(),
            grad_output.into_tensor_arg(),
            scratch_state.into_tensor_arg(),
            scratch_prev.into_tensor_arg(),
            scratch_decayed.into_tensor_arg(),
            grad_query.clone().into_tensor_arg(),
            grad_key.clone().into_tensor_arg(),
            grad_value.clone().into_tensor_arg(),
            grad_erase.clone().into_tensor_arg(),
            grad_write.clone().into_tensor_arg(),
            grad_log_decay.clone().into_tensor_arg(),
            grad_initial_state.clone().into_tensor_arg(),
            params.into_tensor_arg(),
        );
    } else if gdn2_column_parts_backward_enabled() {
        let part_shape = Shape::new([batch, heads, time, latent, dense]);
        let grad_query_parts =
            empty_device::<R, f32>(client.clone(), device.clone(), part_shape.clone());
        let grad_key_parts =
            empty_device::<R, f32>(client.clone(), device.clone(), part_shape.clone());
        let grad_erase_parts =
            empty_device::<R, f32>(client.clone(), device.clone(), part_shape.clone());
        let grad_log_decay_parts =
            empty_device::<R, f32>(client.clone(), device.clone(), part_shape);

        let _ = gdn2_backward_column_parts_kernel::launch::<R>(
            &client,
            CubeCount::Static(dense as u32, 1, count_z),
            CubeDim::new_1d(1),
            query.into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            erase.into_tensor_arg(),
            write.into_tensor_arg(),
            log_decay.clone().into_tensor_arg(),
            initial_state.into_tensor_arg(),
            boundary_states.into_tensor_arg(),
            grad_output.into_tensor_arg(),
            grad_query_parts.clone().into_tensor_arg(),
            grad_key_parts.clone().into_tensor_arg(),
            grad_erase_parts.clone().into_tensor_arg(),
            grad_log_decay_parts.clone().into_tensor_arg(),
            grad_value.clone().into_tensor_arg(),
            grad_write.clone().into_tensor_arg(),
            grad_initial_state.clone().into_tensor_arg(),
            params.clone().into_tensor_arg(),
            latent,
            chunk_size_hint.max(1usize).min(time.max(1usize)),
        );

        let reduce_total = (batch * heads * time * latent) as u32;
        let reduce_count = reduce_total.div_ceil(GDN2_ZERO_WORKGROUP_X);
        let _ = gdn2_reduce_dense_parts_kernel::launch::<R>(
            &client,
            CubeCount::Static(reduce_count, 1, 1),
            CubeDim::new_1d(GDN2_ZERO_WORKGROUP_X),
            grad_query_parts.into_tensor_arg(),
            grad_key_parts.into_tensor_arg(),
            grad_erase_parts.into_tensor_arg(),
            grad_log_decay_parts.into_tensor_arg(),
            grad_query.clone().into_tensor_arg(),
            grad_key.clone().into_tensor_arg(),
            grad_erase.clone().into_tensor_arg(),
            grad_log_decay.clone().into_tensor_arg(),
            params.into_tensor_arg(),
        );
    } else if gdn2_block_parts_backward_enabled() {
        let dense_blocks = dense.div_ceil(GDN2_BACKWARD_WORKGROUP_X as usize);
        let part_shape = Shape::new([batch, heads, dense_blocks, time, latent]);
        let grad_query_block_parts =
            empty_device::<R, f32>(client.clone(), device.clone(), part_shape.clone());
        let grad_key_block_parts =
            empty_device::<R, f32>(client.clone(), device.clone(), part_shape.clone());
        let grad_erase_block_parts =
            empty_device::<R, f32>(client.clone(), device.clone(), part_shape.clone());
        let grad_log_decay_block_parts =
            empty_device::<R, f32>(client.clone(), device.clone(), part_shape);

        let _ = gdn2_backward_dense_block_parts_kernel::launch::<R>(
            &client,
            CubeCount::Static(dense_blocks as u32, 1, count_z),
            CubeDim::new_1d(GDN2_BACKWARD_WORKGROUP_X),
            query.into_tensor_arg(),
            key.into_tensor_arg(),
            value.into_tensor_arg(),
            erase.into_tensor_arg(),
            write.into_tensor_arg(),
            log_decay.into_tensor_arg(),
            initial_state.into_tensor_arg(),
            boundary_states.into_tensor_arg(),
            grad_output.into_tensor_arg(),
            grad_query_block_parts.clone().into_tensor_arg(),
            grad_key_block_parts.clone().into_tensor_arg(),
            grad_value.clone().into_tensor_arg(),
            grad_erase_block_parts.clone().into_tensor_arg(),
            grad_write.clone().into_tensor_arg(),
            grad_log_decay_block_parts.clone().into_tensor_arg(),
            grad_initial_state.clone().into_tensor_arg(),
            params.clone().into_tensor_arg(),
            latent,
        );

        let reduce_total = (batch * heads * time * latent) as u32;
        let reduce_count = reduce_total.div_ceil(GDN2_ZERO_WORKGROUP_X);
        let _ = gdn2_reduce_dense_block_parts_kernel::launch::<R>(
            &client,
            CubeCount::Static(reduce_count, 1, 1),
            CubeDim::new_1d(GDN2_ZERO_WORKGROUP_X),
            grad_query_block_parts.into_tensor_arg(),
            grad_key_block_parts.into_tensor_arg(),
            grad_erase_block_parts.into_tensor_arg(),
            grad_log_decay_block_parts.into_tensor_arg(),
            grad_query.clone().into_tensor_arg(),
            grad_key.clone().into_tensor_arg(),
            grad_erase.clone().into_tensor_arg(),
            grad_log_decay.clone().into_tensor_arg(),
            params.into_tensor_arg(),
        );
    } else {
        let runtime_chunk_size = chunk_size_hint.max(1usize);
        let runtime_num_chunks = time.div_ceil(runtime_chunk_size);
        let factor_latent_shape =
            Shape::new([batch, heads, runtime_num_chunks, runtime_chunk_size, latent]);
        let grad_cumulative_decay = zeros_client::<R>(
            client.clone(),
            device.clone(),
            factor_latent_shape,
            DType::F32,
        );
        let factor_matrix_shape = Shape::new([
            batch,
            heads,
            runtime_num_chunks,
            runtime_chunk_size,
            runtime_chunk_size,
        ]);
        let grad_lower = zeros_client::<R>(
            client.clone(),
            device.clone(),
            factor_matrix_shape,
            DType::F32,
        );

        let (cumulative_decay, wy_lower) = gdn2_wy_factors_runtime::<R>(
            key.clone(),
            erase.clone(),
            log_decay.clone(),
            params.clone(),
            runtime_chunk_size,
        );
        let factor_count = CubeCount::Static(runtime_num_chunks as u32, 1, count_z);

        let _ = gdn2_backward_chunk_wy_factored_kernel::launch::<R>(
            &client,
            CubeCount::Static(dense as u32, 1, count_z),
            CubeDim::new_1d(1),
            query.into_tensor_arg(),
            key.clone().into_tensor_arg(),
            value.into_tensor_arg(),
            erase.clone().into_tensor_arg(),
            write.into_tensor_arg(),
            log_decay.clone().into_tensor_arg(),
            boundary_states.into_tensor_arg(),
            grad_output.into_tensor_arg(),
            wy_lower.clone().into_tensor_arg(),
            grad_query.clone().into_tensor_arg(),
            grad_key.clone().into_tensor_arg(),
            grad_value.clone().into_tensor_arg(),
            grad_erase.clone().into_tensor_arg(),
            grad_write.clone().into_tensor_arg(),
            grad_cumulative_decay.clone().into_tensor_arg(),
            grad_lower.clone().into_tensor_arg(),
            grad_initial_state.clone().into_tensor_arg(),
            params.clone().into_tensor_arg(),
            latent,
            runtime_chunk_size,
        );

        let _ = gdn2_project_wy_lower_grad_kernel::launch::<R>(
            &client,
            factor_count.clone(),
            CubeDim::new_1d(GDN2_BACKWARD_WORKGROUP_X),
            key.into_tensor_arg(),
            erase.into_tensor_arg(),
            cumulative_decay.clone().into_tensor_arg(),
            grad_lower.into_tensor_arg(),
            grad_key.clone().into_tensor_arg(),
            grad_erase.clone().into_tensor_arg(),
            grad_cumulative_decay.clone().into_tensor_arg(),
            params.clone().into_tensor_arg(),
            latent,
        );

        let _ = gdn2_accumulate_wy_log_decay_grad_kernel::launch::<R>(
            &client,
            factor_count,
            CubeDim::new_1d(GDN2_BACKWARD_WORKGROUP_X),
            cumulative_decay.into_tensor_arg(),
            grad_cumulative_decay.into_tensor_arg(),
            grad_log_decay.clone().into_tensor_arg(),
            params.into_tensor_arg(),
            latent,
        );
    }

    Gdn2BackwardRuntimeOutput {
        grad_query,
        grad_key,
        grad_value,
        grad_erase,
        grad_write,
        grad_log_decay,
        grad_initial_state,
    }
}

pub(crate) fn try_cast_primitive<B: BackendTrait, T: 'static>(
    value: B::FloatTensorPrimitive,
) -> Option<T>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn std::any::Any> = Box::new(value);
    boxed.downcast::<T>().ok().map(|boxed| *boxed)
}

pub(crate) fn try_cast_backend<B: BackendTrait, T: 'static>(
    value: T,
) -> Option<B::FloatTensorPrimitive>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn std::any::Any> = Box::new(value);
    boxed
        .downcast::<B::FloatTensorPrimitive>()
        .ok()
        .map(|boxed| *boxed)
}

#[cfg(feature = "cuda")]
pub(crate) fn try_gdn2_forward_runtime_cuda<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    key: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    erase: BurnTensor<B, 4>,
    write: BurnTensor<B, 4>,
    log_decay: BurnTensor<B, 4>,
    initial_state: BurnTensor<B, 4>,
    chunk_size: usize,
) -> Option<Gdn2ForwardRuntimeOutput<CudaRuntime>>
where
    B::FloatTensorPrimitive: 'static,
{
    let params = gdn2_forward_params(&query, &value, chunk_size);
    let time = query.shape().dims::<4>()[2];
    let num_chunks = time.div_ceil(chunk_size.max(1));
    let query_raw = query.into_primitive().tensor();
    let key_raw = key.into_primitive().tensor();
    let value_raw = value.into_primitive().tensor();
    let erase_raw = erase.into_primitive().tensor();
    let write_raw = write.into_primitive().tensor();
    let log_decay_raw = log_decay.into_primitive().tensor();
    let initial_state_raw = initial_state.into_primitive().tensor();
    let params_raw = params.into_primitive().tensor();

    Some(gdn2_forward_runtime::<CudaRuntime>(
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(query_raw)?,
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(key_raw)?,
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(value_raw)?,
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(erase_raw)?,
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(write_raw)?,
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(log_decay_raw)?,
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(initial_state_raw)?,
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(params_raw)?,
        num_chunks,
    ))
}
