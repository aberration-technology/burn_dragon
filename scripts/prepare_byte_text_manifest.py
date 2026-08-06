#!/usr/bin/env python3
"""Prepare a deterministic byte-tokenized text corpus for local training.

The output follows the universality manifest/chunk format so experiments reuse
the production memory-mapped loader without introducing a text-only data path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path


MANIFEST_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 or arbitrary byte input file")
    parser.add_argument("output", type=Path, help="Output corpus directory")
    parser.add_argument("--name", default="byte-text-fixed-holdout")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--chunk-tokens", type=int, default=262_144)
    parser.add_argument("--document-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def entropy_bits(payload: bytes) -> float:
    if not payload:
        return 0.0
    total = len(payload)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in Counter(payload).values()
    )


def write_chunks(
    chunk_dir: Path,
    payload: bytes,
    split: str,
    token_offset: int,
    chunk_tokens: int,
    document_tokens: int,
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for index, start in enumerate(range(0, len(payload), chunk_tokens)):
        part = payload[start : start + chunk_tokens]
        file_name = f"{split}-chunk-{index:05d}.u32le"
        with (chunk_dir / file_name).open("wb") as handle:
            handle.write(struct.pack(f"<{len(part)}I", *part))
        chunks.append(
            {
                "file_name": file_name,
                "split": split,
                "token_offset": token_offset + start,
                "token_count": len(part),
                "sample_count": len(part) // document_tokens,
            }
        )
    return chunks


def prepare(args: argparse.Namespace) -> Path:
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1)")
    if args.chunk_tokens <= 0:
        raise ValueError("--chunk-tokens must be positive")
    if args.document_tokens <= 1:
        raise ValueError("--document-tokens must be greater than one")
    if args.chunk_tokens % args.document_tokens != 0:
        raise ValueError("--chunk-tokens must be divisible by --document-tokens")

    payload = args.input.read_bytes()
    if len(payload) < 4:
        raise ValueError("input must contain at least four bytes")
    split_at = min(max(2, int(len(payload) * args.train_ratio)), len(payload) - 2)
    source_digest = hashlib.sha256(payload).hexdigest()
    train = payload[:split_at]
    validation = payload[split_at:]
    train = train[: len(train) - (len(train) % args.document_tokens)]
    validation = validation[
        : len(validation) - (len(validation) % args.document_tokens)
    ]
    if not train or not validation:
        raise ValueError("both splits must contain at least one complete document")
    materialized = train + validation

    args.output.mkdir(parents=True, exist_ok=True)
    chunk_dir = args.output / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks = write_chunks(
        chunk_dir,
        train,
        "train",
        0,
        args.chunk_tokens,
        args.document_tokens,
    )
    chunks.extend(
        write_chunks(
            chunk_dir,
            validation,
            "validation",
            len(train),
            args.chunk_tokens,
            args.document_tokens,
        )
    )

    digest = hashlib.sha256(materialized).hexdigest()
    sample_records = args.output / "sample_records.jsonl"
    records = []
    document_entropies = []
    for split, split_offset, split_payload in (
        ("train", 0, train),
        ("validation", len(train), validation),
    ):
        for start in range(0, len(split_payload), args.document_tokens):
            part = split_payload[start : start + args.document_tokens]
            entropy = entropy_bits(part)
            document_entropies.append(entropy)
            records.append(
                {
                    "sample_index": len(records),
                    "split": split,
                    "family": "byte_text",
                    "complexity_band": "natural_text",
                    "complexity_filter_matched": True,
                    "identity_bias": 0.0,
                    "temperature": 0.0,
                    "step_stride": 1,
                    "start_step": 0,
                    "token_offset": split_offset + start,
                    "token_count": len(part),
                    "preview_path": None,
                    "serialized_char_count": len(part),
                    "stats": {
                        "grid_width": 0,
                        "grid_height": 0,
                        "steps": 0,
                        "state_count": 256,
                        "patch_count_per_frame": 0,
                        "patch_token_count": len(part),
                        "mean_entropy_bits": entropy,
                        "mean_transition_rate": 0.0,
                        "active_ratio_mean": 0.0,
                        "unique_frames": 0,
                        "unique_patch_count": len(set(part)),
                        "frame_uniqueness_ratio": 0.0,
                        "patch_uniqueness_ratio": len(set(part)) / len(part),
                        "gzip_complexity_ratio": 0.0,
                        "complexity_score": entropy,
                    },
                    "math_domains": [],
                    "reasoning_modes": [],
                }
            )
    sample_records.write_text("".join(json.dumps(record) + "\n" for record in records))

    combined_entropy = entropy_bits(materialized)
    train_samples = len(train) // args.document_tokens
    validation_samples = len(validation) // args.document_tokens
    stats = {
        "total_samples": train_samples + validation_samples,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "total_token_count": len(materialized),
        "mean_token_count": args.document_tokens,
        "mean_entropy_bits": combined_entropy,
        "mean_transition_rate": 0.0,
        "mean_active_ratio": 0.0,
        "mean_gzip_complexity_ratio": 0.0,
        "min_gzip_complexity_ratio": 0.0,
        "max_gzip_complexity_ratio": 0.0,
        "mean_complexity_score": combined_entropy,
        "min_complexity_score": min(document_entropies),
        "max_complexity_score": max(document_entropies),
        "family_counts": {"byte_text": train_samples + validation_samples},
        "complexity_histogram": [],
    }
    manifest = {
        "version": MANIFEST_VERSION,
        "corpus_kind": "nca",
        "dataset_name": args.name,
        "seed": args.seed,
        "train_token_count": len(train),
        "val_token_count": len(validation),
        "token_count": len(materialized),
        "chunk_token_capacity": args.chunk_tokens,
        "tokenizer": {
            "family": "gpt2_byte_compatible",
            "vocab_size": 256,
            "bos_id": None,
            "eos_id": None,
            "frame_special_tokens": False,
            "pad_id": None,
            "unk_id": None,
            "tokenizer_id": (
                f"raw-byte-v1:source-sha256:{source_digest}:materialized-sha256:{digest}"
            ),
        },
        "chunk_dir": "chunks",
        "preview_dir": "preview",
        "sample_records_path": "sample_records.jsonl",
        "chunks": chunks,
        "stats": stats,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"prepared {manifest_path}: tokens={len(materialized)} train={len(train)} "
        f"validation={len(validation)} documents={len(records)} sha256={digest}"
    )
    return manifest_path


if __name__ == "__main__":
    prepare(parse_args())
