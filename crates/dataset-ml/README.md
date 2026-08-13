[简体中文](https://github.com/SomeB1oody/dataset-core/blob/master/crates/dataset-ml/README.zh-CN.md) | English

# dataset-ml

`dataset-ml` provides ready-to-use loaders for classic machine learning datasets, built on [`dataset-core`](https://crates.io/crates/dataset-core).

[![rustc](https://img.shields.io/badge/rustc-1.88%2B-brown)](https://www.rust-lang.org/) [![edition](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/) [![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/dataset-core/blob/master/LICENSE) [![crates.io](https://img.shields.io/crates/v/dataset-ml.svg)](https://crates.io/crates/dataset-ml)

[![CI](https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/ci.yml?branch=master&label=CI)](https://github.com/SomeB1oody/dataset-core/actions/workflows/ci.yml)

## Overview

`dataset-ml` includes loaders for classic ML datasets. Each loader:

- Downloads the source file on first access with `ureq`, and retries transient network failures.
- Verifies a pinned SHA-256 hash to detect corruption or upstream changes.
- Parses the source into a `Table`: one named, typed column per source column.
- Caches the parsed `Table` in memory using `dataset_core::Dataset<T, E>`. Later accesses return a `&` reference with zero I/O.

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

| Feature         | Default | What it enables                                                                                  |
|-----------------|---------|--------------------------------------------------------------------------------------------------|
| `dataset`       | yes     | The `dataset` module and its loaders, the crate-root re-export of every loader struct            |
| `preprocessing` | yes     | The `preprocessing` module: seeded splits, feature scaling, one-hot encoding, and label encoding |

The `table` and `traits` modules are always available, whichever features you pick. They hold `Table` and `MlDataset`. You can write a loader of your own against the same interface with both features off.

To take only what you need, turn the default off:

```toml
[dependencies]
dataset-ml = { version = "0.4", default-features = false, features = ["dataset"] }
```

With `dataset` off, the only direct dependencies left are `dataset-core` and `ndarray`.

## Datasets

See the [dataset overview on docs.rs](https://docs.rs/dataset-ml/latest/dataset_ml/dataset/index.html#datasets) for the full list. It shows the sample count, feature count, and task type of every dataset.

## Usage

```rust
use dataset_ml::Iris;

fn main() {
    let iris = Iris::new("./data");

    // Lazy: downloads and parses on first access, then cached.
    let table = iris.data().unwrap();

    assert_eq!(table.n_samples(), 150);
    assert_eq!(table.n_columns(), 5);

    // Name the columns you want in the matrix, when you want it.
    let features = table.numeric_matrix(&Iris::FEATURE_NAMES).unwrap();
    assert_eq!(features.shape(), &[150, 4]);

    // Or reach one column by name, whatever its position.
    let species = table.column(Iris::TARGET).unwrap().as_string().unwrap();
    assert_eq!(species[0], "setosa");
}
```

Every dataset struct exposes the same six methods, whatever it holds:

- `new(storage_dir)`: create instance (no I/O). Some datasets add `new_test` / `new_all` / `new_full` for their subsets
- `data()`: reference to the parsed `Table`
- `get_data()` / `get_data_mut()`: borrow the cached `Table` **without** loading
- `into_data()` / `take_data()`: move the owned `Table` out, with no clone

## The `Table`

Every loader returns a `Table`: one `Column` per source column, each with its own name and its values
in the type the source uses.

`Table::new` checks its columns, so a loader cannot hand you misaligned data:

- the table holds at least one column
- every column holds the same number of samples
- no two columns share a name

| `ColumnData` | What one column holds                                                  |
|--------------|------------------------------------------------------------------------|
| `Numeric`    | one `f64` per sample. A missing value is `NaN`                         |
| `Integer`    | one `i64` per sample                                                   |
| `String`     | one `String` per sample, spelled as the source spells it               |
| `Bytes`      | one fixed-width row of `u8` per sample, such as the pixels of an image |

Each loader names its columns in associated constants. `FEATURE_NAMES` lists the columns the source designates as the model inputs, and `TARGET` names the label column. A source that designates more than one label column uses `TARGET_NAMES` in place of `TARGET`. A dataset without a label has neither constant. You reach every other column by its name.

```rust
use dataset_ml::Iris;

fn main() {
    let iris = Iris::new("./data");
    let table = iris.data().unwrap();

    // Every column, with its name and its type.
    for column in table.columns() {
        println!("{} {}", column.name(), column.data().kind());
    }

    // Materialize a matrix only when you need one.
    let features = table.numeric_matrix(&Iris::FEATURE_NAMES).unwrap();

    // The matrix follows the order you name, which is not always the source order.
    let petals = table.numeric_matrix(&["petal_width", "petal_length"]).unwrap();

    // Strings stay as the source spells them. Encoding is your decision.
    let species = table.column(Iris::TARGET).unwrap().as_string().unwrap();
}
```

A `String` column has no numeric reading, so `numeric_matrix` returns an error if you name one. `numeric_matrix` allocates on every call, so call it once and keep the result.

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
| `n_samples()`                   | Sample count, uniform whatever shape a loader parses into                       |
| `is_loaded()` / `storage_dir()` | Inspect the loader without touching the data                                    |
| `invalidate()`                  | Drop the in-memory cache to free the memory a large dataset holds               |

The trait's names deliberately differ from the inherent `data()` / `get_data()` / `take_data()`. This way, neither set ever shadows the other. Both are always available and always agree.

## Preprocessing

`dataset_ml::preprocessing` turns what the loaders return into what a model consumes. Everything is deterministic given a seed and needs no extra crates.

```rust
use dataset_ml::preprocessing::{label_encode, standardize, stratified_split};
use dataset_ml::Iris;
use ndarray::Axis;

fn main() {
    let iris = Iris::new("./data");
    let table = iris.data().unwrap();

    let features = table.numeric_matrix(&Iris::FEATURE_NAMES).unwrap();
    let species = table.column(Iris::TARGET).unwrap().as_string().unwrap();

    // Split with each species proportionally represented on both sides.
    let (train, test) = stratified_split(species.as_slice().unwrap(), 0.2, 42).unwrap();

    // Fit the scaler on the training rows only, then replay it on the test rows.
    let (train_x, scaler) = standardize(&features.select(Axis(0), &train)).unwrap();
    let (codes, classes) = label_encode(&species.select(Axis(0), &train)).unwrap();

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

The splitting functions return **row indices**, not arrays, because a sample spans every column of the table. One index list keeps every sample aligned across them. To get arrays, use ndarray's `select(Axis(0), &indices)`. The scalers compute their statistics over the **finite** values of each column. As a result, the `NaN` that marks a missing value in `Titanic`, `PalmerPenguins`, and `HeartDisease` stays missing. It does not skew the column's statistics.

## Performance Considerations

- **First access**: downloads the file (if not on disk), validates SHA-256, parses, caches in memory.
- **Later accesses**: return a reference to the cached data, with zero allocation and zero I/O.
- **`numeric_matrix()`**: allocates a new matrix out of the columns you name. Call it once and keep the result.
- **`take_data()` / `into_data()`**: move the owned `Table` out with no clone. `get_data_mut()` edits it in place.
- **Offline**: after the initial download, datasets stay on disk. Later runs need no network access.

## License

This project uses the MIT License. See [LICENSE](../../LICENSE) for details.

## Author

**SomeB1oody**: [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
