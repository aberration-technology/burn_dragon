pub use crate::dense_causal_attention::{
    CompiledDenseCausalAttentionPlan, supports_dense_causal_attention_backend,
    try_fused_dense_causal_attention_wgpu, try_fused_dense_causal_attention_wgpu_with_plan,
};
pub use crate::recurrent::{
    CompiledRecurrentAttentionPlan, RecurrentAttentionOutput, RecurrentProfileSnapshot,
    recurrent_profile_reset, recurrent_profile_snapshot,
    supports_backend as supports_recurrent_backend, try_fused_recurrent_attention_wgpu,
    try_fused_recurrent_attention_wgpu_with_plan,
};
