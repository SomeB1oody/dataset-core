# dataset-core

A generic, thread-safe dataset container with lazy loading and caching for Rust.

[![Rust Version](https://img.shields.io/badge/Rust-1.88+-brown)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/dataset-core.svg)](https://crates.io/crates/dataset-core)

## Overview

`dataset-core` provides `Dataset<T>`, a lightweight wrapper that pairs a storage directory with a lazily-initialized value of any type `T`. The actual loading logic is supplied by the caller through a closure, so `Dataset<T>` works with any data source — local files, remote URLs, databases, or in-memory generation.

The first call to `load()` executes the closure and caches the result via `OnceLock`; every subsequent call returns a reference to the cached value with zero overhead, even across threads.

On top of this core type, two **optional** feature-gated modules are available:

- **`utils`** — helpers for downloading files, extracting archives, verifying SHA-256 hashes, and managing temporary directories.
- **`datasets`** — ready-to-use loaders for classic ML datasets that also serve as reference implementations showing how to wrap `Dataset<T>`.

## Installation

**Core only** (zero dependencies):

```toml
[dependencies]
dataset-core = "*"
```

**With utilities**:

```toml
[dependencies]
dataset-core = { version = "*", features = ["utils"] }
```

**With built-in datasets** (implies `utils`):

```toml
[dependencies]
dataset-core = { version = "*", features = ["datasets"] }
```

## Feature Flags

| Feature    | What it enables                                                  | Extra dependencies                        |
|------------|------------------------------------------------------------------|-------------------------------------------|
| *(none)*   | `Dataset<T>` only                                                | none                                      |
| `utils`    | Download, unzip, temp dirs, SHA-256 validation, error types      | downloader, zip, tempfile, sha2           |
| `datasets` | All built-in dataset loaders (implies `utils`)                   | ndarray, csv (+ everything in `utils`)    |

## Core Usage

```rust
use dataset_core::Dataset;

fn my_loader(dir: &str) -> Result<Vec<String>, std::io::Error> {
    // In a real use case you would read/download files from `dir`.
    Ok(vec!["hello".to_string(), "world".to_string()])
}

fn main() {
    let ds: Dataset<Vec<String>> = Dataset::new("./my_data");

    // First call runs the loader and caches the result.
    let data = ds.load(my_loader).unwrap();
    assert_eq!(data.len(), 2);

    // Subsequent calls return the cached reference instantly.
    let data_again = ds.load(my_loader).unwrap();
    assert!(std::ptr::eq(data, data_again)); // same reference, no reload    
}
```

### `Dataset<T>` API

| Method          | Returns       | Description                                              |
|-----------------|---------------|----------------------------------------------------------|
| `new(dir)`      | `Dataset<T>`  | Create an instance (no I/O)                              |
| `load(loader)`  | `Result<&T, E>` | Run `loader` on first call, return cached `&T` thereafter |
| `is_loaded()`   | `bool`        | Whether data has been loaded                             |
| `storage_dir()` | `&str`        | The storage directory path                               |

## Built-in Datasets (feature `datasets`)

| Dataset              | Samples | Features | Task Type      | Source              |
|----------------------|---------|----------|----------------|---------------------|
| Iris                 | 150     | 4        | Classification | UCI ML Repository   |
| Boston Housing       | 506     | 13       | Regression     | UCI ML Repository   |
| Diabetes             | 768     | 8        | Classification | Kaggle              |
| Titanic              | 891     | 11       | Classification | Kaggle              |
| Wine Quality (Red)   | 1,599   | 11       | Regression     | UCI ML Repository   |
| Wine Quality (White) | 4,898   | 11       | Regression     | UCI ML Repository   |

```rust
use dataset_core::datasets::iris::Iris;

fn main() {
    let iris = Iris::new("./data");

    // Lazy: downloads and parses on first access, then cached.
    let features = iris.features().unwrap();  // &Array2<f64>
    let labels   = iris.labels().unwrap();    // &Array1<String>

    // Or get both at once:
    let (features, labels) = iris.data().unwrap();

    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);

    // Call .to_owned() when you need a mutable copy.
    let mut owned = features.to_owned();
    owned[[0, 0]] = 5.5;
}
```

Each built-in dataset struct follows the same pattern:

- `new(storage_dir)` — create instance (no I/O)
- `features()` — reference to feature matrix
- `labels()` / `targets()` — reference to label/target vector
- `data()` — all references at once

> **Note**: Titanic's `features()` returns `(&Array2<String>, &Array2<f64>)` (string + numeric features), and `data()` returns a triple.

## Utility Functions (feature `utils`)

| Function               | Purpose                                                       |
|------------------------|---------------------------------------------------------------|
| `download_to`          | Download a remote file into a directory                       |
| `unzip`                | Extract a ZIP archive                                         |
| `create_temp_dir`      | Create a self-cleaning temporary directory                    |
| `file_sha256_matches`  | Verify a file's SHA-256 hash                                  |
| `download_dataset_with`| End-to-end acquisition: temp dir, download, hash check, move  |

## Building Your Own Dataset

The built-in datasets in the `datasets` module demonstrate the recommended pattern for wrapping `Dataset<T>`. Here is a simplified outline:

```rust,ignore
use dataset_core::Dataset;

pub struct MyDataset {
    inner: Dataset<(Vec<f64>, Vec<String>)>,
}

impl MyDataset {
    pub fn new(storage_dir: &str) -> Self {
        Self { inner: Dataset::new(storage_dir) }
    }

    pub fn data(&self) -> Result<&(Vec<f64>, Vec<String>), MyError> {
        self.inner.load(|dir| {
            // Download / read / parse files from `dir` ...
            Ok((vec![1.0, 2.0], vec!["a".into(), "b".into()]))
        })
    }
}
```

See `src/datasets/iris.rs` and others for complete, real-world examples including downloading, CSV parsing, SHA-256 validation, and ndarray integration.

## Performance Considerations

- **First access**: downloads the file (if not on disk), validates SHA-256, parses, and caches in memory.
- **Subsequent accesses**: return a reference to the cached data — zero allocation, zero I/O.
- **`.to_owned()`**: clones cached data into a new owned value — use only when mutation is needed.
- **Offline**: once downloaded, datasets are stored on disk; no network required on subsequent runs.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Datasets Attribution

The built-in datasets are classic machine learning datasets widely used for educational and research purposes:

- **Iris**: Fisher's Iris dataset (1936)
- **Boston Housing**: Harrison & Rubinfeld (1978)
- **Diabetes**: Pima Indians Diabetes Database
- **Titanic**: Kaggle Titanic dataset
- **Wine Quality**: UCI Machine Learning Repository

## Author

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
