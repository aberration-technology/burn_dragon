use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub fn stable_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec(value).context("serialize stable json")
}

pub fn stable_json_hash<T: Serialize>(value: &T) -> Result<String> {
    Ok(sha256_hex(&stable_json_bytes(value)?))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize)]
    struct Fixture {
        a: u32,
        b: &'static str,
    }

    #[test]
    fn stable_hash_is_repeatable() {
        let fixture = Fixture { a: 4, b: "x" };
        assert_eq!(
            stable_json_hash(&fixture).expect("hash"),
            stable_json_hash(&fixture).expect("hash")
        );
    }
}
