[简体中文](https://github.com/SomeB1oody/dataset-core/blob/master/crates/dataset-ml/README.zh-CN.md) | English

# dataset-ml

Ready-to-use loaders for classic machine learning datasets, built on [`dataset-core`](https://crates.io/crates/dataset-core).

<p align="center">
  <a href="https://www.rust-lang.org/"><img alt="rustc" src="https://img.shields.io/badge/rustc-1.88%2B-brown"></a>
  <a href="https://doc.rust-lang.org/edition-guide/"><img alt="edition" src="https://img.shields.io/badge/edition-2024-orange"></a>
  <a href="https://github.com/SomeB1oody/dataset-core/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a>
  <a href="https://crates.io/crates/dataset-ml"><img alt="crates.io" src="https://img.shields.io/crates/v/dataset-ml.svg"></a>
  <br>
  <a href="https://github.com/SomeB1oody/dataset-core/actions/workflows/fmt.yml"><img alt="fmt" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/fmt.yml?branch=master&label=fmt"></a>
  <a href="https://github.com/SomeB1oody/dataset-core/actions/workflows/clippy.yml"><img alt="clippy" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/clippy.yml?branch=master&label=clippy"></a>
  <a href="https://github.com/SomeB1oody/dataset-core/actions/workflows/test.yml"><img alt="test" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/test.yml?branch=master&label=test"></a>
  <a href="https://github.com/SomeB1oody/dataset-core/actions/workflows/doc.yml"><img alt="doc" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/doc.yml?branch=master&label=doc"></a>
</p>

## Overview

`dataset-ml` ships with loaders for 26 classic ML datasets. Each loader:

- Downloads the source file on first access (with `ureq`).
- Verifies a pinned SHA-256 hash to detect corruption or upstream changes.
- Parses the source (CSV, or raw documents extracted from an archive for the text corpora) into [`ndarray`](https://crates.io/crates/ndarray) `Array1` / `Array2`.
- Caches the parsed result in memory via `dataset_core::Dataset<T, E>` — subsequent accesses return a `&` reference with zero I/O.

Each module is also a complete reference implementation of the pattern for wrapping `Dataset<T, E>` for a concrete data source.

## Installation

```toml
[dependencies]
dataset-ml = "0.3"
```

## Datasets

| Struct                                     | Module path                                        | Samples | Features | Task Type      | Source            |
|--------------------------------------------|----------------------------------------------------|---------|----------|----------------|-------------------|
| `Abalone`                                  | `dataset_ml::abalone`                              | 4,177   | 8        | Regression     | UCI ML Repository |
| `Adult`                                    | `dataset_ml::adult`                                | 32,561  | 14       | Classification | UCI ML Repository |
| `BankMarketing`                            | `dataset_ml::bank_marketing`                       | 45,211  | 16       | Classification | UCI ML Repository |
| `Iris`                                     | `dataset_ml::iris`                                 | 150     | 4        | Classification | UCI ML Repository |
| `BreastCancer`                             | `dataset_ml::breast_cancer`                        | 569     | 30       | Classification | UCI ML Repository |
| `BostonHousing`                            | `dataset_ml::boston_housing`                       | 506     | 13       | Regression     | UCI ML Repository |
| `CaliforniaHousing`                        | `dataset_ml::california_housing`                   | 20,640  | 8        | Regression     | StatLib (1990 census) |
| `CarEvaluation`                            | `dataset_ml::car_evaluation`                       | 1,728   | 6        | Classification | UCI ML Repository |
| `Covtype`                                  | `dataset_ml::covtype`                              | 581,012 | 54       | Classification | UCI ML Repository |
| `Diabetes`                                 | `dataset_ml::diabetes`                             | 442     | 10       | Regression     | Efron et al. (2004) |
| `Digits`                                   | `dataset_ml::digits`                               | 1,797   | 64       | Classification | UCI ML Repository |
| `HeartDisease`                             | `dataset_ml::heart_disease`                        | 303     | 13       | Classification | UCI ML Repository |
| `Ionosphere`                               | `dataset_ml::ionosphere`                           | 351     | 34       | Classification | UCI ML Repository |
| `Kddcup99`                                 | `dataset_ml::kddcup99`                             | 494,021 / 4,898,431 | 41 | Classification | UCI KDD Archive   |
| `Linnerud`                                 | `dataset_ml::linnerud`                             | 20      | 3        | Regression (multi-output) | scikit-learn |
| `Mushroom`                                 | `dataset_ml::mushroom`                             | 8,124   | 22       | Classification | UCI ML Repository |
| `Titanic`                                  | `dataset_ml::titanic`                              | 891     | 11       | Classification | Kaggle            |
| `PalmerPenguins`                           | `dataset_ml::palmer_penguins`                      | 344     | 7        | Classification | palmerpenguins    |
| `SmsSpam`                                  | `dataset_ml::sms_spam`                             | 5,574   | text     | Classification | UCI ML Repository |
| `WineRecognition`                          | `dataset_ml::wine_recognition`                     | 178     | 13       | Classification | UCI ML Repository |
| `RedWineQuality`                           | `dataset_ml::wine_quality::red_wine_quality`       | 1,599   | 11       | Regression     | UCI ML Repository |
| `WhiteWineQuality`                         | `dataset_ml::wine_quality::white_wine_quality`     | 4,898   | 11       | Regression     | UCI ML Repository |
| `YoutubeSpam`                              | `dataset_ml::youtube_spam`                         | 1,956   | text     | Classification | UCI ML Repository |
| `SentimentSentences`                       | `dataset_ml::sentiment_sentences`                  | 3,000   | text     | Classification | UCI ML Repository |
| `Newsgroups20`                             | `dataset_ml::newsgroups20`                         | 11,314 / 18,846 | text | Classification | Jason Rennie / 20 Newsgroups |
| `MovieReviewPolarity`                      | `dataset_ml::movie_review_polarity`                | 2,000   | text     | Classification | Cornell (Pang & Lee) |

All structs are also re-exported at the crate root, so `dataset_ml::Iris`, `dataset_ml::RedWineQuality`, etc. work too.

## Usage

```rust
use dataset_ml::iris::Iris;

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

- `new(storage_dir)` — create instance (no I/O)
- `features()` — reference to feature matrix
- `labels()` / `targets()` — reference to label/target vector
- `data()` — all references at once

> The text loaders **SmsSpam**, **YoutubeSpam**, **SentimentSentences**, **Newsgroups20**, and **MovieReviewPolarity** are the exception: instead of `features()` they expose `texts()` (an `Array1<String>` of raw documents), since a text corpus has no fixed feature matrix. **SentimentSentences** additionally exposes `sources()` (the review site each sentence came from); **Newsgroups20** is the only **multi-class** text loader (20 classes) and offers `new`/`new_test`/`new_all` subset constructors.

## Migration from `dataset-core` 0.1.x

If you used the `datasets` feature of `dataset-core` 0.1.x, switch to this crate:

```diff
- dataset-core = { version = "0.1", features = ["datasets"] }
+ dataset-ml = "0.3"
```

| Old path                                                                     | New path                                                         |
|------------------------------------------------------------------------------|------------------------------------------------------------------|
| `dataset_core::datasets::iris::Iris`                                         | `dataset_ml::iris::Iris`                                         |
| `dataset_core::datasets::boston_housing::BostonHousing`                      | `dataset_ml::boston_housing::BostonHousing`                      |
| `dataset_core::datasets::diabetes::Diabetes`                                 | `dataset_ml::diabetes::Diabetes`                                 |
| `dataset_core::datasets::titanic::Titanic`                                   | `dataset_ml::titanic::Titanic`                                   |
| `dataset_core::datasets::wine_quality::red_wine_quality::RedWineQuality`     | `dataset_ml::wine_quality::red_wine_quality::RedWineQuality`     |
| `dataset_core::datasets::wine_quality::white_wine_quality::WhiteWineQuality` | `dataset_ml::wine_quality::white_wine_quality::WhiteWineQuality` |

`dataset_core::utils::*` and `dataset_core::DatasetError` are unchanged — they remain in [`dataset-core`](https://crates.io/crates/dataset-core) under the `utils` feature.

## Performance Considerations

- **First access**: downloads the file (if not on disk), validates SHA-256, parses, caches in memory.
- **Subsequent accesses**: return a reference to the cached data — zero allocation, zero I/O.
- **`.to_owned()`**: clones cached data into a new owned value — use only when mutation is needed.
- **Offline**: once downloaded, datasets stay on disk; no network needed on subsequent runs.

## License

This project is licensed under the MIT License — see [LICENSE](../../LICENSE) for details.

## Author

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
