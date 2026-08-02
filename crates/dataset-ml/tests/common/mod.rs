//! Shared helpers for the `dataset-ml` integration tests.
//!
//! This file sits under `tests/common/`. Cargo treats it as a plain module,
//! included via `mod common;`, not as its own test binary.

use dataset_core::DatasetError;
use std::path::Path;

/// Verify that a file's SHA256 hash matches an expected value (case-insensitive).
///
/// This is a thin wrapper over `dataset-core`'s
/// [`verify_sha256`](dataset_core::verify_sha256). The loaders run this same check
/// during acquisition. A test that asserts a downloaded or overwritten file is
/// correct then exercises that real code path, not a second implementation of it.
pub fn file_sha256_matches(path: &Path, expected_hex: &str) -> Result<bool, DatasetError> {
    dataset_core::verify_sha256(path, expected_hex)
}
