[简体中文](https://github.com/SomeB1oody/dataset-core/blob/master/crates/dataset-ml/README.zh-CN.md) | English

# dataset-ml

Ready-to-use loaders for classic machine learning datasets, built on [`dataset-core`](https://crates.io/crates/dataset-core).

[![Rust Version](https://img.shields.io/badge/Rust-1.88+-brown)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/dataset-ml.svg)](https://crates.io/crates/dataset-ml)

## Overview

`dataset-ml` ships with loaders for six classic ML datasets. Each loader:

- Downloads the source file on first access (with `ureq`).
- Verifies a pinned SHA-256 hash to detect corruption or upstream changes.
- Parses the CSV into [`ndarray`](https://crates.io/crates/ndarray) `Array1` / `Array2`.
- Caches the parsed result in memory via `dataset_core::Dataset<T>` — subsequent accesses return a `&` reference with zero I/O.

Each module is also a complete reference implementation of the pattern for wrapping `Dataset<T>` for a concrete data source.

## Installation

```toml
[dependencies]
dataset-ml = "0.1"
```

## Datasets

| Struct                                     | Module path                                        | Samples | Features | Task Type      | Source            |
|--------------------------------------------|----------------------------------------------------|---------|----------|----------------|-------------------|
| `Iris`                                     | `dataset_ml::iris`                                 | 150     | 4        | Classification | UCI ML Repository |
| `BostonHousing`                            | `dataset_ml::boston_housing`                       | 506     | 13       | Regression     | UCI ML Repository |
| `Diabetes`                                 | `dataset_ml::diabetes`                             | 768     | 8        | Classification | Kaggle            |
| `Titanic`                                  | `dataset_ml::titanic`                              | 891     | 11       | Classification | Kaggle            |
| `RedWineQuality`                           | `dataset_ml::wine_quality::red_wine_quality`       | 1,599   | 11       | Regression     | UCI ML Repository |
| `WhiteWineQuality`                         | `dataset_ml::wine_quality::white_wine_quality`     | 4,898   | 11       | Regression     | UCI ML Repository |

All structs are also re-exported at the crate root, so `dataset_ml::Iris`, `dataset_ml::RedWineQuality`, etc. work too.

## Usage

```rust
use dataset_ml::iris::Iris;

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

Each dataset struct follows the same pattern:

- `new(storage_dir)` — create instance (no I/O)
- `features()` — reference to feature matrix
- `labels()` / `targets()` — reference to label/target vector
- `data()` — all references at once

> **Note**: Titanic's `features()` returns `(&Array2<String>, &Array2<f64>)` (string + numeric features), and `data()` returns a triple.

## Migration from `dataset-core` 0.1.x

If you used the `datasets` feature of `dataset-core` 0.1.x, switch to this crate:

```diff
- dataset-core = { version = "0.1", features = ["datasets"] }
+ dataset-ml = "0.1"
```

| Old path                                                                       | New path                                                       |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|
| `dataset_core::datasets::iris::Iris`                                           | `dataset_ml::iris::Iris`                                       |
| `dataset_core::datasets::boston_housing::BostonHousing`                        | `dataset_ml::boston_housing::BostonHousing`                    |
| `dataset_core::datasets::diabetes::Diabetes`                                   | `dataset_ml::diabetes::Diabetes`                               |
| `dataset_core::datasets::titanic::Titanic`                                     | `dataset_ml::titanic::Titanic`                                 |
| `dataset_core::datasets::wine_quality::red_wine_quality::RedWineQuality`       | `dataset_ml::wine_quality::red_wine_quality::RedWineQuality`   |
| `dataset_core::datasets::wine_quality::white_wine_quality::WhiteWineQuality`   | `dataset_ml::wine_quality::white_wine_quality::WhiteWineQuality` |

`dataset_core::utils::*` and `dataset_core::DatasetError` are unchanged — they remain in [`dataset-core`](https://crates.io/crates/dataset-core) under the `utils` feature.

## Performance Considerations

- **First access**: downloads the file (if not on disk), validates SHA-256, parses, caches in memory.
- **Subsequent accesses**: return a reference to the cached data — zero allocation, zero I/O.
- **`.to_owned()`**: clones cached data into a new owned value — use only when mutation is needed.
- **Offline**: once downloaded, datasets stay on disk; no network needed on subsequent runs.

## License

This project is licensed under the MIT License — see [LICENSE](../../LICENSE) for details.

## Datasets Attribution

The bundled datasets are classic machine learning datasets widely used for educational and research purposes:

- **Iris**: Fisher's Iris dataset (1936)
- **Boston Housing**: Harrison & Rubinfeld (1978)
- **Diabetes**: Pima Indians Diabetes Database
- **Titanic**: Kaggle Titanic dataset
- **Wine Quality**: UCI Machine Learning Repository

## Author

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
