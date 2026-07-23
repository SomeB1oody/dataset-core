//! Shared helpers for the `dataset-ml` integration tests.
//!
//! Lives under `tests/common/` so Cargo treats it as a plain module included via
//! `mod common;` rather than compiling it as its own test binary.

use dataset_core::DatasetError;
use std::path::Path;

/// Verify that a file's SHA256 hash matches an expected value (case-insensitive).
///
/// A thin wrapper over `dataset-core`'s
/// [`verify_sha256`](dataset_core::verify_sha256) — the very check the loaders run
/// during acquisition — so a test asserting that a downloaded or overwritten file
/// is the expected one exercises that code path instead of a second implementation
/// of it.
pub fn file_sha256_matches(path: &Path, expected_hex: &str) -> Result<bool, DatasetError> {
    dataset_core::verify_sha256(path, expected_hex)
}
