pub mod pretokenized;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use pretokenized::PretokenizedTokenizer;
use serde::{Deserialize, Serialize};

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
    fn decode_with_options(&self, ids: &[u32], _stop_at_eos: bool) -> String {
        self.decode(ids)
    }
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn bos_id(&self) -> Option<u32>;
    fn eos_id(&self) -> Option<u32>;
    fn pad_id(&self) -> Option<u32>;
    fn unk_id(&self) -> Option<u32>;
    fn as_any(&self) -> &dyn std::any::Any;
}

pub type SharedTokenizer = Arc<dyn Tokenizer>;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TokenizerConfig {
    #[serde(default)]
    pub vocab_path: Option<PathBuf>,
    #[serde(flatten)]
    pub kind: TokenizerKind,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_path: None,
            kind: TokenizerKind::Pretokenized(PretokenizedTokenizerConfig::default()),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TokenizerKind {
    Pretokenized(PretokenizedTokenizerConfig),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct PretokenizedTokenizerConfig {
    pub vocab_size: usize,
    #[serde(default)]
    pub bos_id: Option<u32>,
    #[serde(default)]
    pub eos_id: Option<u32>,
    #[serde(default)]
    pub pad_id: Option<u32>,
    #[serde(default)]
    pub unk_id: Option<u32>,
}

impl Default for PretokenizedTokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50_257,
            bos_id: None,
            eos_id: Some(50_256),
            pad_id: None,
            unk_id: None,
        }
    }
}

impl TokenizerConfig {
    pub fn storage_path(&self, _cache_dir: &Path) -> Option<PathBuf> {
        None
    }

    pub fn load(&self, _path: &Path) -> Result<SharedTokenizer> {
        let TokenizerKind::Pretokenized(config) = &self.kind;
        Ok(Arc::new(PretokenizedTokenizer::new(
            config.vocab_size,
            config.bos_id,
            config.eos_id,
            config.pad_id,
            config.unk_id,
        )) as SharedTokenizer)
    }

    pub fn fit<'a, I>(&self, _texts: I) -> Result<SharedTokenizer>
    where
        I: Iterator<Item = &'a str>,
    {
        self.load(Path::new(""))
    }

    pub fn save(&self, _tokenizer: &dyn Tokenizer, _path: &Path) -> Result<()> {
        Ok(())
    }

    pub fn requires_strict_coverage(&self) -> bool {
        false
    }

    pub fn validate_corpus(&self, _tokenizer: &dyn Tokenizer, _text: &str) -> Result<()> {
        Ok(())
    }

    pub fn kind_name(&self) -> &'static str {
        "pretokenized"
    }
}
