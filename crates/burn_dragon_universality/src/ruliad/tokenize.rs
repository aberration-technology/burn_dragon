use anyhow::{Result, anyhow};

use crate::manifest::{TokenizerFamily, UniversalityTokenizerManifest};
use crate::ruliad::config::RuliadTokenizationConfig;

#[derive(Debug, Clone)]
pub struct RuliadByteTokenizer {
    vocab_size: usize,
    eos_id: Option<u32>,
}

impl RuliadByteTokenizer {
    pub fn from_config(config: &RuliadTokenizationConfig) -> Result<Self> {
        match config {
            RuliadTokenizationConfig::Gpt2ByteCompatible { vocab_size, eos_id } => {
                if *vocab_size < 257 {
                    return Err(anyhow!("ruliad byte tokenizer vocab_size must be >= 257"));
                }
                Ok(Self {
                    vocab_size: *vocab_size,
                    eos_id: *eos_id,
                })
            }
        }
    }

    pub fn manifest(&self) -> UniversalityTokenizerManifest {
        UniversalityTokenizerManifest {
            family: TokenizerFamily::Gpt2ByteCompatible,
            vocab_size: self.vocab_size,
            bos_id: None,
            eos_id: self.eos_id,
            frame_special_tokens: false,
            pad_id: None,
            unk_id: None,
            tokenizer_id: "ruliad-byte-v1".to_string(),
        }
    }

    pub fn encode_document(&self, text: &str, document_tokens: usize) -> Vec<u32> {
        let eos_tokens = usize::from(self.eos_id.is_some());
        let payload_len = document_tokens.saturating_sub(eos_tokens);
        let mut tokens = Vec::with_capacity(document_tokens);
        tokens.extend(text.bytes().take(payload_len).map(u32::from));
        const FILLER: &[u8] = b"\n.\n";
        let mut filler_index = 0usize;
        while tokens.len() < payload_len {
            tokens.push(u32::from(FILLER[filler_index % FILLER.len()]));
            filler_index += 1;
        }
        if let Some(eos_id) = self.eos_id
            && tokens.len() < document_tokens
        {
            tokens.push(eos_id);
        }
        tokens.truncate(document_tokens);
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_tokenizer_emits_exact_document_length() {
        let tokenizer =
            RuliadByteTokenizer::from_config(&RuliadTokenizationConfig::default()).expect("tok");
        let tokens = tokenizer.encode_document("abc", 8);
        assert_eq!(tokens.len(), 8);
        assert_eq!(tokens[0], u32::from(b'a'));
        assert_eq!(tokens[7], 50_256);
    }
}
