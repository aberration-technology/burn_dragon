use anyhow::{Result, anyhow};

use crate::manifest::{TokenizerFamily, UniversalityTokenizerManifest};
use crate::ruliad::config::RuliadTokenizationConfig;

const RAW_BYTE_TOKEN_COUNT: usize = 256;
const SYMBOLIC_STRUCTURAL_TOKEN_COUNT: usize = 10;
const STRUCTURED_CLASS_TOKEN_START: usize = RAW_BYTE_TOKEN_COUNT + SYMBOLIC_STRUCTURAL_TOKEN_COUNT;
const STRUCTURED_CLASS_TOKEN_COUNT: usize = 5;
const STRUCTURED_RESERVED_TOKEN_COUNT: usize =
    STRUCTURED_CLASS_TOKEN_START + STRUCTURED_CLASS_TOKEN_COUNT;
const STRUCTURED_MAX_LEXEME_BYTES: usize = 32;

#[derive(Debug, Clone)]
pub struct RuliadByteTokenizer {
    mode: RuliadTokenizerMode,
}

#[derive(Debug, Clone)]
enum RuliadTokenizerMode {
    ByteCompatible {
        vocab_size: usize,
        eos_id: Option<u32>,
    },
    Symbolic {
        vocab_size: usize,
        eos_id: Option<u32>,
    },
    StructuredSymbolic {
        vocab_size: usize,
        eos_id: Option<u32>,
    },
}

impl RuliadByteTokenizer {
    pub fn from_config(config: &RuliadTokenizationConfig) -> Result<Self> {
        match config {
            RuliadTokenizationConfig::Gpt2ByteCompatible { vocab_size, eos_id } => {
                if *vocab_size < 257 {
                    return Err(anyhow!("ruliad byte tokenizer vocab_size must be >= 257"));
                }
                Ok(Self {
                    mode: RuliadTokenizerMode::ByteCompatible {
                        vocab_size: *vocab_size,
                        eos_id: *eos_id,
                    },
                })
            }
            RuliadTokenizationConfig::Symbolic { vocab_size, eos_id } => {
                if *vocab_size < 512 {
                    return Err(anyhow!(
                        "ruliad symbolic tokenizer vocab_size must be >= 512"
                    ));
                }
                Ok(Self {
                    mode: RuliadTokenizerMode::Symbolic {
                        vocab_size: *vocab_size,
                        eos_id: *eos_id,
                    },
                })
            }
            RuliadTokenizationConfig::StructuredSymbolic { vocab_size, eos_id } => {
                if *vocab_size < STRUCTURED_RESERVED_TOKEN_COUNT + 1 {
                    return Err(anyhow!(
                        "ruliad structured symbolic tokenizer vocab_size must be >= 272"
                    ));
                }
                if matches!(eos_id, Some(id) if *id < STRUCTURED_RESERVED_TOKEN_COUNT as u32) {
                    return Err(anyhow!(
                        "ruliad structured symbolic eos_id must not collide with byte, structural, or class tokens"
                    ));
                }
                Ok(Self {
                    mode: RuliadTokenizerMode::StructuredSymbolic {
                        vocab_size: *vocab_size,
                        eos_id: *eos_id,
                    },
                })
            }
        }
    }

    pub fn manifest(&self) -> UniversalityTokenizerManifest {
        match self.mode {
            RuliadTokenizerMode::ByteCompatible { vocab_size, eos_id } => {
                UniversalityTokenizerManifest {
                    family: TokenizerFamily::Gpt2ByteCompatible,
                    vocab_size,
                    bos_id: None,
                    eos_id,
                    frame_special_tokens: false,
                    pad_id: None,
                    unk_id: None,
                    tokenizer_id: "ruliad-byte-v1".to_string(),
                }
            }
            RuliadTokenizerMode::Symbolic { vocab_size, eos_id } => UniversalityTokenizerManifest {
                family: TokenizerFamily::RuliadSymbolic,
                vocab_size,
                bos_id: None,
                eos_id,
                frame_special_tokens: false,
                pad_id: None,
                unk_id: None,
                tokenizer_id: format!("ruliad-symbolic-v6:{vocab_size}"),
            },
            RuliadTokenizerMode::StructuredSymbolic { vocab_size, eos_id } => {
                UniversalityTokenizerManifest {
                    family: TokenizerFamily::RuliadSymbolic,
                    vocab_size,
                    bos_id: None,
                    eos_id,
                    frame_special_tokens: false,
                    pad_id: None,
                    unk_id: None,
                    tokenizer_id: format!("ruliad-structured-symbolic-v3:{vocab_size}"),
                }
            }
        }
    }

    pub fn payload_token_capacity(&self, document_tokens: usize) -> usize {
        document_tokens.saturating_sub(usize::from(self.eos_id().is_some()))
    }

    pub fn payload_token_count(&self, text: &str) -> usize {
        match self.mode {
            RuliadTokenizerMode::ByteCompatible { .. } => text.len(),
            RuliadTokenizerMode::Symbolic { .. } => symbolic_payload_tokens(text, &self.mode).len(),
            RuliadTokenizerMode::StructuredSymbolic { .. } => {
                structured_symbolic_payload_tokens(text, &self.mode).len()
            }
        }
    }

    pub fn encode_payload(&self, text: &str) -> Vec<u32> {
        match self.mode {
            RuliadTokenizerMode::ByteCompatible { .. } => text.bytes().map(u32::from).collect(),
            RuliadTokenizerMode::Symbolic { .. } => symbolic_payload_tokens(text, &self.mode),
            RuliadTokenizerMode::StructuredSymbolic { .. } => {
                structured_symbolic_payload_tokens(text, &self.mode)
            }
        }
    }

    pub fn decode_payload(&self, tokens: &[u32], stop_at_eos: bool) -> String {
        let mut text = String::new();
        for token in tokens {
            if Some(*token) == self.eos_id() {
                if stop_at_eos {
                    break;
                }
                continue;
            }
            match self.mode {
                RuliadTokenizerMode::ByteCompatible { .. } => {
                    if let Some(ch) = char::from_u32(*token) {
                        text.push(ch);
                    }
                }
                RuliadTokenizerMode::Symbolic { .. } => {
                    decode_symbolic_token(&mut text, *token, false);
                }
                RuliadTokenizerMode::StructuredSymbolic { .. } => {
                    decode_symbolic_token(&mut text, *token, true);
                }
            }
        }
        text
    }

    pub fn eos_id(&self) -> Option<u32> {
        match self.mode {
            RuliadTokenizerMode::ByteCompatible { eos_id, .. }
            | RuliadTokenizerMode::Symbolic { eos_id, .. }
            | RuliadTokenizerMode::StructuredSymbolic { eos_id, .. } => eos_id,
        }
    }

    pub fn encode_document(&self, text: &str, document_tokens: usize) -> Vec<u32> {
        let payload_len = self.payload_token_capacity(document_tokens);
        let mut tokens = Vec::with_capacity(document_tokens);
        match self.mode {
            RuliadTokenizerMode::ByteCompatible { .. } => {
                tokens.extend(text.bytes().take(payload_len).map(u32::from));
            }
            RuliadTokenizerMode::Symbolic { .. } => {
                let mut payload = symbolic_payload_tokens(text, &self.mode);
                payload.truncate(payload_len);
                tokens.extend(payload);
            }
            RuliadTokenizerMode::StructuredSymbolic { .. } => {
                let mut payload = structured_symbolic_payload_tokens(text, &self.mode);
                payload.truncate(payload_len);
                tokens.extend(payload);
            }
        }
        if let Some(eos_id) = self.eos_id()
            && tokens.len() < document_tokens
        {
            tokens.push(eos_id);
        }
        while tokens.len() < document_tokens {
            tokens.push(self.fill_token());
        }
        tokens.truncate(document_tokens);
        tokens
    }

    fn fill_token(&self) -> u32 {
        match self.mode {
            RuliadTokenizerMode::ByteCompatible { eos_id, .. } => {
                eos_id.unwrap_or(u32::from(b'\n'))
            }
            RuliadTokenizerMode::Symbolic { eos_id, .. }
            | RuliadTokenizerMode::StructuredSymbolic { eos_id, .. } => {
                eos_id.unwrap_or_else(|| {
                    symbolic_structural_token(SymbolicStructuralToken::DocumentEnd, &self.mode)
                })
            }
        }
    }
}

fn decode_symbolic_token(text: &mut String, token: u32, structured: bool) {
    match token {
        0..=255 if structured => {
            if let Some(ch) = char::from_u32(token) {
                text.push(ch);
            }
        }
        0..=255 => {}
        token
            if token
                == RAW_BYTE_TOKEN_COUNT as u32 + SymbolicStructuralToken::TraceStart.index() =>
        {
            push_marker(text, "[T");
        }
        token
            if token
                == RAW_BYTE_TOKEN_COUNT as u32 + SymbolicStructuralToken::TraceNode.index() =>
        {
            push_marker(text, "N<");
        }
        token
            if token == RAW_BYTE_TOKEN_COUNT as u32 + SymbolicStructuralToken::TraceEnd.index() =>
        {
            push_marker(text, "[/T]");
        }
        token
            if token
                == RAW_BYTE_TOKEN_COUNT as u32 + SymbolicStructuralToken::DocumentStart.index() =>
        {
            push_marker(text, "[R2");
        }
        token
            if token == RAW_BYTE_TOKEN_COUNT as u32 + SymbolicStructuralToken::Metadata.index() =>
        {
            push_marker(text, "S:");
        }
        token if token == RAW_BYTE_TOKEN_COUNT as u32 + SymbolicStructuralToken::Data.index() => {
            push_marker(text, "G:");
        }
        token if token == RAW_BYTE_TOKEN_COUNT as u32 + SymbolicStructuralToken::Query.index() => {
            push_marker(text, "?:");
        }
        token
            if token
                == RAW_BYTE_TOKEN_COUNT as u32 + SymbolicStructuralToken::ProofStep.index() =>
        {
            push_marker(text, ">");
        }
        token if token == RAW_BYTE_TOKEN_COUNT as u32 + SymbolicStructuralToken::Answer.index() => {
            push_marker(text, "!:");
        }
        token
            if token
                == RAW_BYTE_TOKEN_COUNT as u32 + SymbolicStructuralToken::DocumentEnd.index() =>
        {
            push_marker(text, "[/R2]");
        }
        _ => {}
    }
}

fn push_marker(text: &mut String, marker: &str) {
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(marker);
}

#[derive(Debug, Clone, Copy)]
enum SymbolicClass {
    Alpha,
    Number,
    Hex,
    Mixed,
    #[allow(dead_code)]
    Filler,
}

#[derive(Debug, Clone, Copy)]
enum SymbolicStructuralToken {
    TraceStart,
    TraceNode,
    TraceEnd,
    DocumentStart,
    Metadata,
    Data,
    Query,
    ProofStep,
    Answer,
    DocumentEnd,
}

impl SymbolicStructuralToken {
    const fn index(self) -> u32 {
        match self {
            Self::TraceStart => 0,
            Self::TraceNode => 1,
            Self::TraceEnd => 2,
            Self::DocumentStart => 3,
            Self::Metadata => 4,
            Self::Data => 5,
            Self::Query => 6,
            Self::ProofStep => 7,
            Self::Answer => 8,
            Self::DocumentEnd => 9,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::TraceStart => "trace_start",
            Self::TraceNode => "trace_node",
            Self::TraceEnd => "trace_end",
            Self::DocumentStart => "doc_start",
            Self::Metadata => "metadata",
            Self::Data => "data",
            Self::Query => "query",
            Self::ProofStep => "proof_step",
            Self::Answer => "answer",
            Self::DocumentEnd => "doc_end",
        }
    }
}

fn symbolic_payload_tokens(text: &str, mode: &RuliadTokenizerMode) -> Vec<u32> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::with_capacity(text.len().min(1024));
    let mut index = 0usize;
    while index < bytes.len() {
        if let Some((kind, consumed)) = symbolic_structural_marker(&bytes[index..]) {
            tokens.push(symbolic_structural_token(kind, mode));
            index += consumed;
            continue;
        }
        let byte = bytes[index];
        if byte.is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if is_symbolic_separator_byte(byte) {
            index += 1;
            continue;
        }
        if is_symbolic_lexeme_byte(byte) {
            let start = index;
            index += 1;
            while index < bytes.len() && is_symbolic_lexeme_byte(bytes[index]) {
                index += 1;
            }
            let lexeme = &text[start..index];
            if is_symbolic_boilerplate_lexeme(lexeme) {
                continue;
            }
            tokens.push(symbolic_lexeme_token(lexeme, mode));
            continue;
        }
        push_byte_token(&mut tokens, byte);
        index += 1;
    }
    tokens
}

fn structured_symbolic_payload_tokens(text: &str, mode: &RuliadTokenizerMode) -> Vec<u32> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::with_capacity(text.len().min(2048));
    let mut index = 0usize;
    while index < bytes.len() {
        if let Some((kind, consumed)) = structured_symbolic_structural_marker(bytes, index) {
            tokens.push(symbolic_structural_token(kind, mode));
            index += consumed;
            continue;
        }
        let byte = bytes[index];
        if byte.is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if is_symbolic_separator_byte(byte) {
            push_structured_separator_byte(&mut tokens, byte);
            index += 1;
            continue;
        }
        if is_symbolic_lexeme_byte(byte) {
            let start = index;
            index += 1;
            while index < bytes.len() && is_symbolic_lexeme_byte(bytes[index]) {
                index += 1;
            }
            let lexeme = &text[start..index];
            if !is_symbolic_boilerplate_lexeme(lexeme) {
                push_structured_lexeme_tokens(&mut tokens, lexeme);
            }
            continue;
        }
        push_structured_payload_byte(&mut tokens, byte);
        index += 1;
    }
    tokens
}

fn structured_symbolic_structural_marker(
    bytes: &[u8],
    index: usize,
) -> Option<(SymbolicStructuralToken, usize)> {
    let at_line_start = index == 0 || bytes.get(index.saturating_sub(1)) == Some(&b'\n');
    if !at_line_start {
        return match bytes.get(index) {
            Some(b'[') => {
                if bytes[index..].starts_with(b"[/R2]") {
                    Some((SymbolicStructuralToken::DocumentEnd, 5))
                } else if bytes[index..].starts_with(b"[/T]") {
                    Some((SymbolicStructuralToken::TraceEnd, 4))
                } else {
                    None
                }
            }
            _ => None,
        };
    }
    symbolic_structural_marker(&bytes[index..])
}

fn symbolic_structural_marker(bytes: &[u8]) -> Option<(SymbolicStructuralToken, usize)> {
    if bytes.starts_with(b"[T") {
        Some((SymbolicStructuralToken::TraceStart, 2))
    } else if bytes.starts_with(b"[/T]") {
        Some((SymbolicStructuralToken::TraceEnd, 4))
    } else if let Some(consumed) = trace_node_marker_len(bytes) {
        Some((SymbolicStructuralToken::TraceNode, consumed))
    } else if bytes.starts_with(b"[R2") {
        Some((SymbolicStructuralToken::DocumentStart, 3))
    } else if bytes.starts_with(b"[/R2]") {
        Some((SymbolicStructuralToken::DocumentEnd, 5))
    } else if bytes.starts_with(b"S:") {
        Some((SymbolicStructuralToken::Metadata, 2))
    } else if bytes.starts_with(b"G:") {
        Some((SymbolicStructuralToken::Data, 2))
    } else if bytes.starts_with(b"?:") {
        Some((SymbolicStructuralToken::Query, 2))
    } else if bytes.starts_with(b"!:") {
        Some((SymbolicStructuralToken::Answer, 2))
    } else if bytes.first() == Some(&b'>') {
        Some((SymbolicStructuralToken::ProofStep, 1))
    } else {
        None
    }
}

fn trace_node_marker_len(bytes: &[u8]) -> Option<usize> {
    if bytes.first() != Some(&b'N') {
        return None;
    }
    let mut index = 1usize;
    while bytes.get(index).is_some_and(|byte| byte.is_ascii_digit()) {
        index = index.saturating_add(1);
    }
    if index > 1 && bytes.get(index) == Some(&b'<') {
        Some(1)
    } else {
        None
    }
}

fn symbolic_structural_token(kind: SymbolicStructuralToken, mode: &RuliadTokenizerMode) -> u32 {
    let eos_id = match *mode {
        RuliadTokenizerMode::Symbolic { eos_id, .. }
        | RuliadTokenizerMode::StructuredSymbolic { eos_id, .. } => eos_id,
        RuliadTokenizerMode::ByteCompatible { .. } => unreachable!("structural token in byte mode"),
    };
    let token = RAW_BYTE_TOKEN_COUNT as u32 + kind.index();
    if Some(token) == eos_id {
        symbolic_bucket_token(SymbolicClass::Mixed, stable_text_hash(kind.label()), mode)
    } else {
        token
    }
}

fn push_structured_lexeme_tokens(tokens: &mut Vec<u32>, lexeme: &str) {
    for byte in lexeme
        .bytes()
        .filter_map(normalized_structured_lexeme_byte)
        .take(STRUCTURED_MAX_LEXEME_BYTES)
    {
        push_structured_payload_byte(tokens, byte);
    }
}

fn push_structured_payload_byte(tokens: &mut Vec<u32>, byte: u8) {
    tokens.push(u32::from(byte));
}

fn push_structured_separator_byte(tokens: &mut Vec<u32>, byte: u8) {
    if matches!(byte, b'[' | b']') {
        return;
    }
    tokens.push(u32::from(byte));
}

fn push_byte_token(tokens: &mut Vec<u32>, byte: u8) {
    tokens.push(u32::from(byte));
}

fn is_symbolic_lexeme_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')
}

fn is_symbolic_separator_byte(byte: u8) -> bool {
    matches!(
        byte,
        b',' | b';'
            | b':'
            | b'='
            | b'|'
            | b'('
            | b')'
            | b'.'
            | b'*'
            | b'+'
            | b'@'
            | b'/'
            | b'<'
            | b'>'
            | b'['
            | b']'
            | b'?'
            | b'!'
    )
}

fn is_symbolic_boilerplate_lexeme(lexeme: &str) -> bool {
    if lexeme.is_empty() || lexeme.bytes().all(|byte| byte == b'-') {
        return true;
    }
    if lexeme.len() == 1 && lexeme.bytes().all(|byte| byte.is_ascii_uppercase()) {
        return true;
    }
    if lexeme == "R2" {
        return true;
    }
    if is_hash_boilerplate_lexeme(lexeme) {
        return true;
    }
    if let Some(rest) = lexeme.strip_prefix('N') {
        return !rest.is_empty() && rest.bytes().all(|byte| byte.is_ascii_digit());
    }
    if let Some(rest) = lexeme.strip_prefix('n') {
        return !rest.is_empty() && rest.bytes().all(|byte| byte.is_ascii_digit());
    }
    if let Some(rest) = lexeme.strip_prefix('v') {
        return !rest.is_empty() && rest.bytes().all(|byte| byte.is_ascii_digit());
    }
    false
}

fn is_hash_boilerplate_lexeme(lexeme: &str) -> bool {
    lexeme.len() >= 8 && lexeme.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn symbolic_lexeme_token(lexeme: &str, mode: &RuliadTokenizerMode) -> u32 {
    symbolic_bucket_token(
        symbolic_lexeme_class(lexeme),
        stable_text_hash(lexeme),
        mode,
    )
}

fn symbolic_lexeme_class(lexeme: &str) -> SymbolicClass {
    if lexeme.bytes().all(|byte| byte.is_ascii_digit()) {
        SymbolicClass::Number
    } else if lexeme.len() >= 6 && lexeme.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        SymbolicClass::Hex
    } else if lexeme
        .bytes()
        .all(|byte| byte.is_ascii_alphabetic() || byte == b'_')
    {
        SymbolicClass::Alpha
    } else {
        SymbolicClass::Mixed
    }
}

fn normalized_structured_lexeme_byte(byte: u8) -> Option<u8> {
    if byte.is_ascii_alphanumeric() {
        Some(byte.to_ascii_lowercase())
    } else if matches!(byte, b'_' | b'-') {
        Some(byte)
    } else {
        None
    }
}

fn symbolic_bucket_token(class: SymbolicClass, hash: u64, mode: &RuliadTokenizerMode) -> u32 {
    let (vocab_size, eos_id) = match *mode {
        RuliadTokenizerMode::Symbolic { vocab_size, eos_id } => (vocab_size, eos_id),
        RuliadTokenizerMode::ByteCompatible { .. } => unreachable!("symbolic bucket in byte mode"),
        RuliadTokenizerMode::StructuredSymbolic { .. } => {
            unreachable!("hash bucket in structured symbolic mode")
        }
    };
    let bucket_start = RAW_BYTE_TOKEN_COUNT + SYMBOLIC_STRUCTURAL_TOKEN_COUNT;
    let bucket_count = vocab_size.saturating_sub(bucket_start).max(1);
    let class_index = match class {
        SymbolicClass::Alpha => 0u64,
        SymbolicClass::Number => 1,
        SymbolicClass::Hex => 2,
        SymbolicClass::Mixed => 3,
        SymbolicClass::Filler => 4,
    };
    let mixed = hash ^ class_index.wrapping_mul(0xD1B5_4A32_D192_ED03);
    let mut token = bucket_start as u32 + (mixed % bucket_count as u64) as u32;
    if Some(token) == eos_id {
        token = bucket_start as u32 + ((u64::from(token) + 1) % bucket_count as u64) as u32;
        if Some(token) == eos_id {
            token = bucket_start as u32;
        }
    }
    token
}

fn stable_text_hash(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
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
        assert_eq!(tokens[3], 50_256);
        assert_eq!(tokens[7], 50_256);
        assert!(
            !tokens
                .windows(3)
                .any(|window| window == [u32::from(b'\n'), u32::from(b'.'), u32::from(b'\n')]),
            "ruliad padding must not expose the period-3 pattern observed in collapsed rollouts"
        );
    }

    #[test]
    fn symbolic_tokenizer_groups_lexemes_and_uses_deterministic_eos_padding() {
        let tokenizer = RuliadByteTokenizer::from_config(&RuliadTokenizationConfig::Symbolic {
            vocab_size: 1025,
            eos_id: Some(1024),
        })
        .expect("tok");
        let tokens = tokenizer.encode_document("G:O=48;I=u48:habc123:z9\n?:cp\n", 24);
        assert_eq!(tokens.len(), 24);
        assert!(
            tokens
                .iter()
                .any(|token| *token >= RAW_BYTE_TOKEN_COUNT as u32)
        );
        assert_eq!(
            tokens.iter().position(|token| *token == 1024).expect("eos"),
            tokenizer.payload_token_count("G:O=48;I=u48:habc123:z9\n?:cp\n")
        );
        let eos_pos = tokens.iter().position(|token| *token == 1024).expect("eos");
        assert!(
            tokens[eos_pos..].iter().all(|token| *token == 1024),
            "symbolic padding should be deterministic eos tokens after payload end: {tokens:?}"
        );
    }

    #[test]
    fn symbolic_tokenizer_suppresses_low_information_syntax() {
        let tokenizer = RuliadByteTokenizer::from_config(&RuliadTokenizationConfig::Symbolic {
            vocab_size: 1025,
            eos_id: Some(1024),
        })
        .expect("tok");
        let tokens = tokenizer.encode_document(
            "[T] G:A=ABC|R=AA>A,AB>C\n?:nf(x0)<=96\n>0:a,b;c=d\n!:ok=1\n",
            64,
        );
        for separator in [
            b',', b';', b':', b'=', b'|', b'(', b')', b'.', b'*', b'+', b'@', b'/', b'<', b'>',
            b'[', b']', b'?', b'!', b'\n',
        ] {
            assert!(
                !tokens.contains(&u32::from(separator)),
                "symbolic token stream should not expose separator byte {}",
                separator
            );
        }
        assert!(
            tokens
                .iter()
                .take_while(|token| **token != 1024)
                .all(|token| *token >= RAW_BYTE_TOKEN_COUNT as u32),
            "symbolic payload should be lexeme buckets, not raw syntax bytes"
        );
    }

    #[test]
    fn symbolic_tokenizer_preserves_abstract_phase_markers() {
        let tokenizer = RuliadByteTokenizer::from_config(&RuliadTokenizationConfig::Symbolic {
            vocab_size: 1025,
            eos_id: Some(1024),
        })
        .expect("tok");
        let tokens =
            tokenizer.encode_document("[R2 h v1 a/b/c]\nS:x\nG:y\n?:q\n>p\n!:a\n[/R2]\n", 64);
        for expected in 259..=265 {
            assert!(
                tokens.contains(&expected),
                "missing symbolic structural token {expected}: {tokens:?}"
            );
        }
    }

    #[test]
    fn symbolic_tokenizer_preserves_trace_markers_without_hash_ids() {
        let tokenizer = RuliadByteTokenizer::from_config(&RuliadTokenizationConfig::Symbolic {
            vocab_size: 65_536,
            eos_id: Some(65_535),
        })
        .expect("tok");
        let tokens = tokenizer.encode_document(
            "[T h=0123456789abcdef n=2]\nN0<- p/t @abcdef012345\n[R2 abcdef0123456789 v7 p/t/c]\nS:x\nG:y\n?:q\n>p\n!:a\n[/R2]\n[/T]\n",
            96,
        );
        for expected in 256..=265 {
            assert!(
                tokens.contains(&expected),
                "missing symbolic trace/record token {expected}: {tokens:?}"
            );
        }
        let hash_token = symbolic_lexeme_token("0123456789abcdef", &tokenizer.mode);
        assert!(
            !tokens.contains(&hash_token),
            "hash-like verifier identifiers should be suppressed from symbolic training tokens"
        );
    }

    #[test]
    fn symbolic_tokenizer_preserves_trace_node_indices() {
        let tokenizer = RuliadByteTokenizer::from_config(&RuliadTokenizationConfig::Symbolic {
            vocab_size: 65_536,
            eos_id: Some(65_535),
        })
        .expect("tok");
        let tokens = tokenizer.encode_document("[T n=2]\nN0<-\nN1<n0\n[/T]\n", 32);
        let zero = symbolic_lexeme_token("0", &tokenizer.mode);
        let one = symbolic_lexeme_token("1", &tokenizer.mode);
        assert_eq!(
            tokens.iter().filter(|token| **token == 257).count(),
            2,
            "expected one trace-node marker per N-indexed node: {tokens:?}"
        );
        assert!(tokens.contains(&zero), "missing N0 index token: {tokens:?}");
        assert!(tokens.contains(&one), "missing N1 index token: {tokens:?}");
    }

    #[test]
    fn structured_symbolic_tokenizer_preserves_compositional_lexeme_bytes() {
        let tokenizer =
            RuliadByteTokenizer::from_config(&RuliadTokenizationConfig::StructuredSymbolic {
                vocab_size: 272,
                eos_id: Some(271),
            })
            .expect("tok");
        let tokens = tokenizer.encode_document("G:Compose=F12\n?:Goal_7\n>Step-A\n!:OK\n", 96);
        let eos = tokens.iter().position(|token| *token == 271).expect("eos");
        let payload = &tokens[..eos];
        assert!(payload.contains(&261), "missing data marker: {payload:?}");
        assert!(payload.contains(&262), "missing query marker: {payload:?}");
        assert!(
            payload.contains(&263),
            "missing proof-step marker: {payload:?}"
        );
        assert!(payload.contains(&264), "missing answer marker: {payload:?}");
        for expected in b"composef12goal_7step-aok" {
            assert!(
                payload.contains(&u32::from(*expected)),
                "missing normalized lexeme byte `{}` in {payload:?}",
                char::from(*expected)
            );
        }
    }

    #[test]
    fn structured_symbolic_tokenizer_does_not_emit_reserved_class_or_hash_buckets_before_eos() {
        let tokenizer =
            RuliadByteTokenizer::from_config(&RuliadTokenizationConfig::StructuredSymbolic {
                vocab_size: 272,
                eos_id: Some(271),
            })
            .expect("tok");
        let tokens = tokenizer.encode_document(
            "[T h=0123456789abcdef n=2]\nN0<- p/t @abcdef012345\n[R2 abcdef0123456789 v7 p/t/c]\nS:x\nG:y\n?:q\n>p\n!:a\n[/R2]\n[/T]\n",
            160,
        );
        let eos = tokens.iter().position(|token| *token == 271).expect("eos");
        let payload = &tokens[..eos];
        assert!(
            payload.iter().all(|token| {
                let token = *token as usize;
                token < STRUCTURED_CLASS_TOKEN_START
                    || token >= STRUCTURED_CLASS_TOKEN_START + STRUCTURED_CLASS_TOKEN_COUNT
            }),
            "structured payload should not emit reserved class-marker targets: {payload:?}"
        );
        for expected in 256..=265 {
            assert!(
                payload.contains(&expected),
                "missing structured trace/record token {expected}: {payload:?}"
            );
        }
        let hash_run = b"0123456789abcdef"
            .iter()
            .map(|byte| u32::from(*byte))
            .collect::<Vec<_>>();
        assert!(
            !payload
                .windows(hash_run.len())
                .any(|window| window == hash_run.as_slice()),
            "hash-like verifier identifiers should be suppressed from structured tokens"
        );
    }

    #[test]
    fn structured_symbolic_tokenizer_preserves_formal_separator_bytes() {
        let tokenizer =
            RuliadByteTokenizer::from_config(&RuliadTokenizationConfig::StructuredSymbolic {
                vocab_size: 272,
                eos_id: Some(271),
            })
            .expect("tok");
        let tokens = tokenizer.encode_document(
            "[T] G:A=ABC|R=AA>A,AB>C\n?:nf(x0)<=96\n>0:a,b;c=d\n!:ok=1\n",
            96,
        );
        let eos = tokens.iter().position(|token| *token == 271).expect("eos");
        let payload = &tokens[..eos];
        for separator in [b',', b';', b':', b'=', b'|', b'(', b')', b'<', b'>'] {
            assert!(
                payload.contains(&u32::from(separator)),
                "structured symbolic token stream should preserve formal separator byte {}",
                separator
            );
        }
        assert!(
            !payload.contains(&u32::from(b'\n')),
            "structured symbolic token stream should not expose whitespace"
        );
    }

    #[test]
    fn structured_symbolic_tokenizer_distinguishes_operator_gt_from_proof_step_marker() {
        let tokenizer =
            RuliadByteTokenizer::from_config(&RuliadTokenizationConfig::StructuredSymbolic {
                vocab_size: 272,
                eos_id: Some(271),
            })
            .expect("tok");
        let tokens = tokenizer.encode_document("G:AA>A\n>Step\n", 64);
        let eos = tokens.iter().position(|token| *token == 271).expect("eos");
        let payload = &tokens[..eos];
        assert!(
            payload.contains(&u32::from(b'>')),
            "in-expression > should remain an operator byte: {payload:?}"
        );
        assert_eq!(
            payload.iter().filter(|token| **token == 263).count(),
            1,
            "only the line-start proof step should become a proof-step marker: {payload:?}"
        );
    }

    #[test]
    fn structured_symbolic_decode_recovers_verifier_completion_shape() {
        let tokenizer =
            RuliadByteTokenizer::from_config(&RuliadTokenizationConfig::StructuredSymbolic {
                vocab_size: 272,
                eos_id: Some(271),
            })
            .expect("tok");
        let tokens = vec![
            264,
            u32::from(b'o'),
            u32::from(b'k'),
            u32::from(b'='),
            u32::from(b'1'),
            265,
            271,
            u32::from(b'x'),
        ];
        let decoded = tokenizer.decode_payload(&tokens, true);
        assert!(decoded.contains("!:ok=1"), "{decoded}");
        assert!(decoded.contains("[/R2]"), "{decoded}");
        assert!(!decoded.contains('x'), "{decoded}");
    }
}
