//! Shared helpers for the `dataset-ml` integration tests.
//!
//! Lives under `tests/common/` so Cargo treats it as a plain module included via
//! `mod common;` rather than compiling it as its own test binary.

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Verify that a file's SHA256 hash matches an expected value (case-insensitive).
///
/// Mirrors the verification `dataset-core` performs internally during acquisition;
/// the tests use it to assert that a downloaded/overwritten file is the expected one.
pub fn file_sha256_matches(path: &Path, expected_hex: &str) -> io::Result<bool> {
    let mut file = File::open(path)?;

    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }

    let digest = hasher.finalize();
    let mut actual_hex = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write as _;
        // Writing formatted bytes into a `String` is infallible.
        let _ = write!(actual_hex, "{:02x}", b);
    }
    Ok(actual_hex.eq_ignore_ascii_case(expected_hex))
}
