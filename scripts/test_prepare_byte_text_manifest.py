#!/usr/bin/env python3
"""Tests for the deterministic byte-text corpus materializer."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import prepare_byte_text_manifest as corpus_builder


class PrepareByteTextManifestTests(unittest.TestCase):
    def _prepare(self, root: Path, output_name: str) -> tuple[Path, bytes]:
        payload = bytes(range(256)) * 8
        source = root / "source.bin"
        source.write_bytes(payload)
        output = root / output_name
        manifest = corpus_builder.prepare(
            argparse.Namespace(
                input=source,
                output=output,
                name="test-byte-text",
                train_ratio=0.75,
                chunk_tokens=512,
                document_tokens=128,
                seed=17,
            )
        )
        return manifest, payload

    @staticmethod
    def _materialized_tokens(manifest_path: Path) -> bytes:
        manifest = json.loads(manifest_path.read_text())
        tokens: list[int] = []
        for chunk in manifest["chunks"]:
            raw = (manifest_path.parent / manifest["chunk_dir"] / chunk["file_name"]).read_bytes()
            tokens.extend(struct.unpack(f"<{len(raw) // 4}I", raw))
        return bytes(tokens)

    def test_manifest_is_aligned_reconstructable_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first_path, payload = self._prepare(root, "first")
            second_path, _ = self._prepare(root, "second")
            first = json.loads(first_path.read_text())
            second = json.loads(second_path.read_text())

            self.assertEqual(first, second)
            self.assertEqual(first["train_token_count"], 1536)
            self.assertEqual(first["val_token_count"], 512)
            self.assertEqual(first["stats"]["train_samples"], 12)
            self.assertEqual(first["stats"]["validation_samples"], 4)
            self.assertTrue(all(chunk["token_count"] % 128 == 0 for chunk in first["chunks"]))
            self.assertEqual(self._materialized_tokens(first_path), payload)
            expected_digest = hashlib.sha256(payload).hexdigest()
            self.assertIn(expected_digest, first["tokenizer"]["tokenizer_id"])

    def test_rejects_chunk_size_that_splits_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            source.write_bytes(bytes(range(64)) * 8)
            with self.assertRaisesRegex(ValueError, "divisible"):
                corpus_builder.prepare(
                    argparse.Namespace(
                        input=source,
                        output=root / "output",
                        name="test-byte-text",
                        train_ratio=0.75,
                        chunk_tokens=192,
                        document_tokens=128,
                        seed=17,
                    )
                )


if __name__ == "__main__":
    unittest.main()
