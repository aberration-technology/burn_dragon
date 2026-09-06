//! Tokenizer-derived eligibility, independent of answer-token supervision.

use burn::tensor::{Int, Tensor, backend::Backend};

#[derive(Debug, Clone, Default)]
pub(crate) struct NextLatentTokenLayout {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
}

impl NextLatentTokenLayout {
    pub(crate) fn from_tokenizer(tokenizer: &dyn crate::tokenizer::Tokenizer) -> Self {
        Self {
            bos: tokenizer.bos_id(),
            eos: tokenizer.eos_id(),
            pad: tokenizer.pad_id(),
        }
    }

    pub(crate) fn source_mask<B: Backend>(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
        let mut mask = Tensor::ones(tokens.shape(), &tokens.device());
        for id in [self.eos, self.pad].into_iter().flatten() {
            mask = mask * tokens.clone().not_equal_elem(id as i64).int();
        }
        mask
    }

    pub(crate) fn destination_mask<B: Backend>(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2, Int> {
        let mut mask = self.source_mask(tokens.clone());
        if let Some(bos) = self.bos {
            mask = mask * tokens.not_equal_elem(bos as i64).int();
        }
        mask
    }
}
