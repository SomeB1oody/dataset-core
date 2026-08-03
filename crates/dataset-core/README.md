[简体中文](https://github.com/SomeB1oody/dataset-core/blob/master/crates/dataset-core/README.zh-CN.md) | English

# dataset-core

A generic, thread-safe dataset container with lazy loading and caching for Rust.

[![rustc](https://img.shields.io/badge/rustc-1.88%2B-brown)](https://www.rust-lang.org/) [![edition](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/) [![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/dataset-core/blob/master/LICENSE) [![crates.io](https://img.shields.io/crates/v/dataset-core.svg)](https://crates.io/crates/dataset-core)

[![CI](https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/ci.yml?branch=master&label=CI)](https://github.com/SomeB1oody/dataset-core/actions/workflows/ci.yml)

## Overview

`dataset-core` provides `Dataset<T, E>`, a lightweight wrapper that pairs a storage directory with a lazily-initialized value of any type `T`. The caller supplies the loading logic through a closure stored at construction time. As a result, `Dataset<T, E>` works with any data source: local files, remote URLs, databases, or in-memory generation. (The caller freely chooses `E`, the loader's error type.)

The first call to `load()` executes the closure and caches the result with `OnceLock`. Every subsequent call returns a reference to the cached value with zero overhead, even across threads.

On top of this core type, the crate provides one **optional**, feature-gated module:

- **`utils`**: helpers for downloading files, extracting archives, verifying SHA-256 hashes, and managing temporary directories.

For ready-to-use loaders for classic ML datasets, see the companion crate [`dataset-ml`](https://crates.io/crates/dataset-ml). It ships 29 loaders, from Iris, Breast Cancer, and California Housing to Covertype, KDD Cup '99, and 20 Newsgroups. It depends on `dataset-core` with the `utils` feature enabled.

## Installation

**Core only** (zero dependencies):

```toml
[dependencies]
dataset-core = "0.5"
```

**With utilities**:

```toml
[dependencies]
dataset-core = { version = "0.5", features = ["utils"] }
```

## Feature Flags

| Feature  | What it enables                                              | Extra dependencies               |
|----------|--------------------------------------------------------------|----------------------------------|
| *(none)* | `Dataset<T, E>` only                                         | none                             |
| `utils`  | Download (with optional retries), unzip, gunzip, untar, untar_gz, temp dirs, SHA-256 hashing & validation, Latin-1 reading, error types | ureq, zip, flate2, tar, tempfile, sha2, thiserror |

## Core Usage

```rust
use dataset_core::Dataset;

fn my_loader(dir: &str) -> Result<Vec<String>, std::io::Error> {
    // A real loader would read or download files from `dir`.
    Ok(vec!["hello".to_string(), "world".to_string()])
}

fn main() {
    // You supply the loader once, at construction time.
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
| `new(dir, loader)`   | `Dataset<T, E>` | Create an instance and store the loader (no I/O)                   |
| `load()`             | `Result<&T, E>` | Run the stored loader on first call, return cached `&T` thereafter |
| `load_mut()`         | `Result<&mut T, E>` | Load if needed, then borrow the cached value mutably for in-place edits |
| `get()` / `get_mut()`| `Option<&T>` / `Option<&mut T>` | Borrow the cached value **without** loading                |
| `take()`             | `Option<T>`     | Move the cached value out and leave the container reusable         |
| `into_inner()`       | `Option<T>`     | Consume the container and return the cached value                  |
| `set_loader(loader)` | `()`            | Replace the loader and invalidate the cache (lazy re-parse)        |
| `invalidate()`       | `()`            | Drop the cached value, keep the loader (next `load` re-runs it)    |
| `is_loaded()`        | `bool`          | Whether the dataset has already loaded its data                    |
| `storage_dir()`      | `&str`          | The storage directory path                                         |

## Utility Functions (feature `utils`)

| Function              | Purpose                                                                                |
|-----------------------|----------------------------------------------------------------------------------------|
| `download_to`         | Download a remote file into a directory                                                |
| `download_to_with_retries` | Like `download_to`, but retries transient failures with exponential backoff    |
| `unzip`               | Extract a ZIP archive                                                                  |
| `gunzip`              | Decompress a gzip (`.gz`) file into a single output file                               |
| `untar`               | Extract a tar (`.tar`) archive into a directory                                        |
| `untar_gz`            | Extract a gzip-compressed tar (`.tar.gz` / `.tgz`) archive into a directory, and stream it so no intermediate file touches disk |
| `sha256_file`         | Compute a file's SHA-256 digest as hex. Use it to pin a new dataset's hash             |
| `verify_sha256`       | Check a file against a hash you already have                                           |
| `read_latin1`         | Read a file as Latin-1 text, losslessly and without failing on non-UTF-8 bytes         |
| `acquire_dataset`     | Cache-aware acquisition: reuse a valid local file, prepare it in a temp dir, check the hash, then move it into place |

## Building Your Own Dataset

You can wrap `Dataset<T, E>` in your own type. The companion crate [`dataset-ml`](https://crates.io/crates/dataset-ml) demonstrates the recommended pattern. Here is a simplified outline:

```rust,ignore
use dataset_core::Dataset;

pub struct MyDataset {
    inner: Dataset<(Vec<f64>, Vec<String>), MyError>,
}

impl MyDataset {
    pub fn new(storage_dir: &str) -> Self {
        Self {
            inner: Dataset::new(storage_dir, |dir| {
                // Download, read, or parse files from `dir` here.
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
- **Subsequent accesses**: return a reference to the cached data, with zero allocation and zero I/O.
- **Cross-thread safety**: `Dataset<T, E>` is `Send + Sync` whenever `T` is (the stored loader is always `Send + Sync`). The loader runs at most once, even under concurrent calls. An internal mutex serializes the first load. A thread that arrives mid-load waits for the result and shares it, instead of starting its own download.

## License

This project uses the MIT License. See [LICENSE](../../LICENSE) for details.

## Author

**SomeB1oody**: [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
