# burn_dragon_tokenizer

Tokenizer primitives for the `burn_dragon` workspace.

This crate currently provides a compact GPT-style byte-pair tokenizer used by
Dragon language experiments. It is a Rust library crate in this workspace; the
optional PyO3 bindings in the code are for local experimentation and are not the
published package contract.

## features

- GPT-4 style regex pre-tokenization by default
- parallel BPE training with `rayon`
- encode/decode helpers over byte-level token ids
- optional `python-bindings` and `extension-module` features for local PyO3
  experiments

## rust use

```rust
use burn_dragon_tokenizer::Tokenizer;

let mut tokenizer = Tokenizer::new();
tokenizer.train_from_texts(["dragon training text"], 4096, None)?;
let ids = tokenizer.encode("dragon training text");
let text = tokenizer.decode_to_string(&ids)?;
```

## local checks

```bash
cargo test -p burn_dragon_tokenizer
cargo test -p burn_dragon_tokenizer --features python-bindings
```

The workspace-level smoke and CI commands remain the source of truth for release
readiness.
