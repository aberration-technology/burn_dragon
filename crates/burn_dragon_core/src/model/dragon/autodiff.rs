//! Autodiff-specific detached auxiliary forwarding.

use super::*;

impl<B: AutodiffBackend> DragonModel<B> {
    /// Run a read-only auxiliary backbone pass without building an autodiff graph or sampling
    /// training-time dropout.
    ///
    /// `Tensor::inner` and `Tensor::from_inner` preserve the backend allocation, so CUDA callers do
    /// not cross the host boundary. The returned tensor is a detached feature view suitable for a
    /// task head whose parameters remain on the autodiff model.
    pub fn forward_hidden_deterministic_auxiliary(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let hidden = self.valid().forward_hidden(tokens.inner());
        Tensor::from_inner(hidden)
    }
}
