[简体中文](https://github.com/SomeB1oody/dataset-core/blob/master/crates/dataset-ml/README.zh-CN.md) | English

# dataset-ml

Ready-to-use loaders for classic machine learning datasets, built on [`dataset-core`](https://crates.io/crates/dataset-core).

[![rustc](https://img.shields.io/badge/rustc-1.88%2B-brown)](https://www.rust-lang.org/) [![edition](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/) [![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/dataset-core/blob/master/LICENSE) [![crates.io](https://img.shields.io/crates/v/dataset-ml.svg)](https://crates.io/crates/dataset-ml)

[![CI](https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/ci.yml?branch=master&label=CI)](https://github.com/SomeB1oody/dataset-core/actions/workflows/ci.yml)

## Overview

`dataset-ml` includes loaders for 29 classic ML datasets. Each loader:

- Downloads the source file on first access with `ureq`, and retries transient network failures.
- Verifies a pinned SHA-256 hash to detect corruption or upstream changes.
- Parses the source (CSV, or raw documents extracted from an archive for the text corpora) into [`ndarray`](https://crates.io/crates/ndarray) `Array1` / `Array2`.
- Caches the parsed result in memory using `dataset_core::Dataset<T, E>`. Later accesses return a `&` reference with zero I/O.

Each module is also a complete reference implementation of the pattern for wrapping `Dataset<T, E>` for a concrete data source.

Two modules apply to every dataset rather than to one of them:

- [`preprocessing`](#preprocessing): seeded train/test and k-fold splits (plain or class-stratified), feature scaling, one-hot encoding, and label encoding.
- [`traits`](#the-mldataset-trait): the `MlDataset` trait every loader implements, for code written generically over "some dataset".

## Installation

```toml
[dependencies]
dataset-ml = "0.4"
```

## Feature flags

| Feature         | Default | What it enables                                                                                                                                                            |
|-----------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dataset`       | yes     | The `dataset` module and its 29 loaders, the crate-root re-export of every loader struct, and `DOWNLOAD_RETRIES`. It adds the `csv`, `serde`, and `tempfile` dependencies. |
| `preprocessing` | yes     | The `preprocessing` module: seeded splits, feature scaling, one-hot encoding, and label encoding. It adds no dependencies.                                                 |

The `traits` module is always available, whichever features you pick. It holds `MlDataset` and `NumSamples`, so you can write a loader of your own against the same interface with both features off.

To take only what you need, turn the default off:

```toml
[dependencies]
dataset-ml = { version = "0.4", default-features = false, features = ["dataset"] }
```

With `dataset` off, the only direct dependencies left are `dataset-core` and `ndarray`.

## Datasets

For the full list, with the sample count, feature count, and task type of every dataset, see the [dataset overview on docs.rs](https://docs.rs/dataset-ml/latest/dataset_ml/dataset/index.html#datasets).

## Usage

```rust
use dataset_ml::Iris;

fn main() {
    let iris = Iris::new("./data");

    // Lazy: downloads and parses on first access, then cached.
    let features = iris.features().unwrap();  // &Array2<f64>
    let labels   = iris.labels().unwrap();    // &Array1<&'static str>

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

- `new(storage_dir)`: create instance (no I/O)
- `features()`: reference to feature matrix
- `labels()` / `targets()`: reference to label/target vector
- `data()`: all references at once

> The text loaders **SmsSpam**, **YoutubeSpam**, **SentimentSentences**, **Newsgroups20**, and **MovieReviewPolarity** are the exception. A text corpus has no fixed feature matrix, so instead of `features()` they expose `texts()` (an `Array1<String>` of raw documents). **SentimentSentences** also exposes `sources()` (the review site each sentence came from). **Newsgroups20** is the only **multi-class** text loader (20 classes) and offers `new`/`new_test`/`new_all` subset constructors.

## The `MlDataset` trait

Every loader implements `dataset_ml::traits::MlDataset`, which covers the container operations that are the same whatever the loader parses into. This lets you write a function over "some dataset" instead of one concrete struct:

```rust
use dataset_ml::traits::MlDataset;
use dataset_ml::{Iris, SmsSpam};

fn describe<D: MlDataset>(dataset: &D) -> String {
    format!("{} ({} samples)", D::NAME, dataset.n_samples().unwrap())
}

fn main() {
    println!("{}", describe(&Iris::new("./data")));     // iris (150 samples)
    println!("{}", describe(&SmsSpam::new("./data")));  // sms_spam (5574 samples)
}
```

| Method                          | Description                                                                     |
|---------------------------------|---------------------------------------------------------------------------------|
| `load()` / `load_mut()`         | Load if needed, then borrow the parsed data (`load_mut` for in-place edits)     |
| `peek()`                        | Borrow the parsed data **without** triggering a load                            |
| `unload()`                      | Move the parsed data out, leaving the loader reusable                           |
| `n_samples()`                   | Sample count, uniform across pair- and triple-shaped datasets                   |
| `is_loaded()` / `storage_dir()` | Inspect the loader without touching the data                                    |
| `invalidate()`                  | Drop the in-memory cache to free the memory a large dataset holds               |

The trait's names deliberately differ from the inherent `data()` / `get_data()` / `take_data()`, so neither set ever shadows the other. Both are always available and always agree.

## Preprocessing

`dataset_ml::preprocessing` turns what the loaders return into what a model consumes. Everything is deterministic given a seed and needs no extra crates.

```rust
use dataset_ml::preprocessing::{stratified_split, standardize, label_encode};
use dataset_ml::Iris;
use ndarray::Axis;

fn main() {
    let iris = Iris::new("./data");
    let (features, labels) = iris.data().unwrap();

    // Split with each species proportionally represented on both sides.
    let (train, test) = stratified_split(labels.as_slice().unwrap(), 0.2, 42).unwrap();

    // Fit the scaler on the training rows only, then replay it on the test rows.
    let (train_x, scaler) = standardize(&features.select(Axis(0), &train)).unwrap();
    let (codes, classes) = label_encode(&labels.select(Axis(0), &train)).unwrap();

    assert_eq!(train_x.nrows(), 120);
    assert_eq!(classes.len(), 3);
}
```

| Function                                    | Purpose                                                                        |
|---------------------------------------------|--------------------------------------------------------------------------------|
| `train_test_split(n, ratio, seed)`          | Shuffled train/test row indices                                                |
| `stratified_split(labels, ratio, seed)`     | The same, but each class keeps its proportion. Use it for imbalanced datasets  |
| `k_fold_indices(n, k, seed)`                | `k` `(train, validation)` index pairs. Each sample appears in validation once  |
| `shuffled_indices(n, seed)`                 | A deterministic permutation of `0..n`                                          |
| `standardize` / `min_max_scale`             | Per-column scaling, returning the fitted `Scaler`                              |
| `apply_scaler(features, &scaler)`           | Replay a fitted scaler on new data, without refitting                          |
| `one_hot_encode(categorical, names)`        | Expand the categorical `Array2<String>` into indicator columns                 |
| `label_encode(labels)` / `class_counts`     | Map labels to `0..n_classes` codes and count samples per class                 |

The splitting functions return **row indices**, not arrays, because a sample spans two or three parallel arrays. One index list keeps every sample aligned across them. To get arrays, use ndarray's `select(Axis(0), &indices)`. The scalers compute their statistics over the **finite** values of each column. So the `NaN` that marks a missing value in `Titanic`, `PalmerPenguins`, and `HeartDisease` stays missing, instead of skewing the column's statistics.

## Performance Considerations

- **First access**: downloads the file (if not on disk), validates SHA-256, parses, caches in memory.
- **Later accesses**: return a reference to the cached data, with zero allocation and zero I/O.
- **`.to_owned()`**: clones cached data into a new owned value. Use it only when you need to mutate the data.
- **Offline**: after the initial download, datasets stay on disk. Later runs need no network access.

## License

This project is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.

## Author

**SomeB1oody**: [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
