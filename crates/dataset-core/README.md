[简体中文](https://github.com/SomeB1oody/dataset-core/blob/master/crates/dataset-core/README.zh-CN.md) | English

# dataset-core

A generic, thread-safe dataset container with lazy loading and caching for Rust.

[![Rust Version](https://img.shields.io/badge/Rust-1.88+-brown)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/dataset-core.svg)](https://crates.io/crates/dataset-core)

## Overview

`dataset-core` provides `Dataset<T, E>`, a lightweight wrapper that pairs a storage directory with a lazily-initialized value of any type `T`. The actual loading logic is supplied by the caller through a closure stored at construction time, so `Dataset<T, E>` works with any data source — local files, remote URLs, databases, or in-memory generation. (`E` is the loader's error type, chosen freely by the caller.)

The first call to `load()` executes the closure and caches the result via `OnceLock`; every subsequent call returns a reference to the cached value with zero overhead, even across threads.

On top of this core type, an **optional** feature-gated module is available:

- **`utils`** — helpers for downloading files, extracting archives, verifying SHA-256 hashes, and managing temporary directories.

Looking for ready-to-use loaders for classic ML datasets (Iris, Breast Cancer, Boston/California Housing, Diabetes, Titanic, Palmer Penguins, Wine Recognition, Wine Quality)? They live in the companion crate [`dataset-ml`](https://crates.io/crates/dataset-ml), which depends on `dataset-core` with the `utils` feature enabled.

## Installation

**Core only** (zero dependencies):

```toml
[dependencies]
dataset-core = "0.3"
```

**With utilities**:

```toml
[dependencies]
dataset-core = { version = "0.3", features = ["utils"] }
```

## Feature Flags

| Feature  | What it enables                                              | Extra dependencies               |
|----------|--------------------------------------------------------------|----------------------------------|
| *(none)* | `Dataset<T, E>` only                                         | none                             |
| `utils`  | Download, unzip, gunzip, temp dirs, SHA-256 validation, error types | ureq, zip, flate2, tempfile, sha2, thiserror |

## Core Usage

```rust
use dataset_core::Dataset;

fn my_loader(dir: &str) -> Result<Vec<String>, std::io::Error> {
    // In a real use case you would read/download files from `dir`.
    Ok(vec!["hello".to_string(), "world".to_string()])
}

fn main() {
    // The loader is supplied once, at construction time.
    let ds: Dataset<Vec<String>, std::io::Error> = Dataset::new("./my_data", my_loader);

    // First call runs the loader and caches the result.
    let data = ds.load().unwrap();
    assert_eq!(data.len(), 2);

    // Subsequent calls return the cached reference instantly.
    let data_again = ds.load().unwrap();
    assert!(std::ptr::eq(data, data_again)); // same reference, no reload
}
```

### `Dataset<T, E>` API

| Method               | Returns         | Description                                                         |
|----------------------|-----------------|--------------------------------------------------------------------|
| `new(dir, loader)`   | `Dataset<T, E>` | Create an instance, storing the loader (no I/O)                    |
| `load()`             | `Result<&T, E>` | Run the stored loader on first call, return cached `&T` thereafter |
| `set_loader(loader)` | `()`            | Replace the loader and invalidate the cache (lazy re-parse)        |
| `invalidate()`       | `()`            | Drop the cached value, keep the loader (next `load` re-runs it)    |
| `is_loaded()`        | `bool`          | Whether data has been loaded                                       |
| `storage_dir()`      | `&str`          | The storage directory path                                         |

## Utility Functions (feature `utils`)

| Function              | Purpose                                                                                |
|-----------------------|----------------------------------------------------------------------------------------|
| `download_to`         | Download a remote file into a directory                                                |
| `unzip`               | Extract a ZIP archive                                                                  |
| `gunzip`              | Decompress a gzip (`.gz`) file into a single output file                               |
| `acquire_dataset`     | Cache-aware acquisition: reuse valid local file, prepare in temp dir, hash check, move |

## Building Your Own Dataset

`Dataset<T>` is designed to be wrapped. The companion crate [`dataset-ml`](https://crates.io/crates/dataset-ml) demonstrates the recommended pattern; here is a simplified outline:

```rust,ignore
use dataset_core::Dataset;

pub struct MyDataset {
    inner: Dataset<(Vec<f64>, Vec<String>), MyError>,
}

impl MyDataset {
    pub fn new(storage_dir: &str) -> Self {
        Self {
            inner: Dataset::new(storage_dir, |dir| {
                // Download / read / parse files from `dir` ...
                Ok((vec![1.0, 2.0], vec!["a".into(), "b".into()]))
            }),
        }
    }

    pub fn data(&self) -> Result<&(Vec<f64>, Vec<String>), MyError> {
        self.inner.load()
    }
}
```

See the [`dataset-ml`](https://crates.io/crates/dataset-ml) source for complete, real-world examples including downloading, CSV parsing, SHA-256 validation, and ndarray integration.

## Performance Considerations

- **First access**: runs the loader once (potentially network + parse), caches the result.
- **Subsequent accesses**: return a reference to the cached data — zero allocation, zero I/O.
- **Cross-thread safety**: `Dataset<T, E>` is `Send + Sync` whenever `T` is (the stored loader is always `Send + Sync`); the internal `OnceLock` guarantees the loader runs at most once even under concurrent calls.

## License

This project is licensed under the MIT License — see [LICENSE](../../LICENSE) for details.

## Author

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
