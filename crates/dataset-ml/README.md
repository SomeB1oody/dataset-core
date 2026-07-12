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

`dataset-ml` ships with loaders for 25 classic ML datasets. Each loader:

- Downloads the source file on first access (with `ureq`).
- Verifies a pinned SHA-256 hash to detect corruption or upstream changes.
- Parses the CSV into [`ndarray`](https://crates.io/crates/ndarray) `Array1` / `Array2`.
- Caches the parsed result in memory via `dataset_core::Dataset<T, E>` — subsequent accesses return a `&` reference with zero I/O.

Each module is also a complete reference implementation of the pattern for wrapping `Dataset<T, E>` for a concrete data source.

## Installation

```toml
[dependencies]
dataset-ml = "0.2"
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

> The text loaders **SmsSpam**, **YoutubeSpam**, **SentimentSentences**, and **Newsgroups20** are the exception: instead of `features()` they expose `texts()` (an `Array1<String>` of raw documents), since a text corpus has no fixed feature matrix. **SentimentSentences** additionally exposes `sources()` (the review site each sentence came from); **Newsgroups20** is the only **multi-class** text loader (20 classes) and offers `new`/`new_test`/`new_all` subset constructors.

> **Note**: Titanic, Palmer Penguins, Adult, BankMarketing, Kddcup99, and Abalone are mixed-type: `features()` returns `(&Array2<String>, &Array2<f64>)` (string + numeric features), and `data()` returns a triple. All are classification (a `labels()` accessor) **except Abalone**, which is regression — its third element is an `Array1<f64>` target exposed via `targets()`. Palmer Penguins also represents missing values as `NaN` (numeric) or `""` (string).
>
> **Note**: Abalone is the first **mixed-type regression** loader: a single categorical `sex` feature (`M`/`F`/`I`, a `(4177, 1)` `&Array2<String>`) plus 7 numeric measurements (`length`, `diameter`, `height`, `whole_weight`, `shucked_weight`, `viscera_weight`, `shell_weight`, a `(4177, 7)` `&Array2<f64>`), predicting `rings` — the `Array1<f64>` regression target (the age in years is `rings + 1.5`). It has no missing values.
>
> **Note**: Adult (Census Income) reproduces the classic UCI dataset for predicting whether income exceeds $50K/year: 8 categorical features (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`), 6 numeric features (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`), and an `Array1<String>` income label kept verbatim (`<=50K` or `>50K`). It loads the canonical `adult.data` training partition (32,561 records); the source's `?` missing categorical token is mapped to empty strings `""`.
>
> **Note**: BankMarketing reproduces the classic UCI Bank Marketing dataset (a Portuguese bank's phone campaigns) for predicting term-deposit subscription: 9 categorical features (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`), 7 numeric features (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`), and an `Array1<String>` label kept verbatim (`yes` or `no`). It loads the full `bank-full.csv` partition (45,211 records) from a ZIP archive; the categorical `unknown` label is kept verbatim (it is a documented level, e.g. `poutcome = unknown` means no previous contact), rather than mapped to an empty string.
>
> **Note**: Mushroom is an **all-categorical** dataset (like Car Evaluation): all 22 features are single-letter string codes, so `features()` returns a single `&Array2<String>` (no numeric matrix) and `data()` returns a `(features, labels)` pair. The `Array1<String>` label is kept verbatim (`e` = edible, `p` = poisonous). The source's `?` missing token (only in `stalk-root`) is mapped to empty strings `""`.
>
> **Note**: Car Evaluation is also **all-categorical** (like Mushroom): all 6 features are string codes (`buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`), so `features()` returns a single `&Array2<String>` and `data()` returns a `(features, labels)` pair. The `Array1<String>` label is kept verbatim, one of the four acceptability classes `unacc` / `acc` / `good` / `vgood`. The 1,728 records enumerate the full cartesian product of the six attributes, so there are no missing values.
>
> **Note**: California Housing reproduces scikit-learn's `fetch_california_housing` features by deriving them from the raw census columns (e.g. `AveRooms = total_rooms / households`) and scaling the target by `1/100000`. Its 207 missing `total_bedrooms` values surface as `NaN` in `AveBedrms`.
>
> **Note**: Diabetes reproduces scikit-learn's `load_diabetes` (default output): the 10 feature columns are standardized (mean-centered, divided by their L2 norm, so each column's sum of squares is 1) and the regression target is left unscaled.
>
> **Note**: Covtype reproduces scikit-learn's `fetch_covtype`: 581,012 samples with 54 features (10 quantitative variables, 4 one-hot `Wilderness_Area` columns, 40 one-hot `Soil_Type` columns) and an `Array1<u8>` cover-type label (`1`–`7`). It is the first loader sourced from a gzip-compressed file: the loader downloads `covtype.data.gz` and decompresses it with `dataset-core`'s `gunzip` helper.
>
> **Note**: Digits reproduces scikit-learn's `load_digits`: each of the 64 features is an integer pixel intensity in `0..=16` (an 8×8 image flattened row-major), and `labels()` returns an `Array1<u8>` of the digit classes (`0`–`9`). It is sourced from the UCI static ZIP package, using the `optdigits.tes` test partition (the same 1,797 samples scikit-learn uses).
>
> **Note**: Linnerud reproduces scikit-learn's `load_linnerud` (multi-output regression): `features()` returns the three exercise variables (`Chins`, `Situps`, `Jumps`) and `targets()` returns the three physiological variables (`Weight`, `Waist`, `Pulse`), so both are `Array2<f64>` with shape `(20, 3)`. It is sourced from two whitespace-separated files distributed with scikit-learn.
>
> **Note**: Kddcup99 reproduces scikit-learn's `fetch_kddcup99`. Like scikit-learn, `Kddcup99::new` loads the **default 10% subset** (494,021 connections, `percent10=True`) and `Kddcup99::new_full` loads the **full set** (4,898,431 connections, `percent10=False`); both share the same 41-feature schema and 23 classes. It is mixed-type like Titanic: `features()` returns `(&Array2<String>, &Array2<f64>)` — 3 categorical features (`protocol_type`, `service`, `flag`) and 38 numeric features — and `labels()` returns an `Array1<String>` of the connection class kept verbatim including the trailing period (e.g. `"normal."`, `"smurf."`). Like Covtype it is sourced from a gzip-compressed file decompressed with `gunzip`. **Heads-up:** the full set's decompressed source is ~743 MB and the parsed in-memory arrays are several GB, so `new_full` takes noticeable time and memory; the default subset is ~10× smaller.
>
> **Note**: Heart Disease (Cleveland) is all-numeric with missing values: `features()` returns a single `&Array2<f64>` of shape `(303, 13)` (several columns are integer-coded categoricals kept as `f64`), and the `?` tokens in `ca` (4) and `thal` (2) become `NaN` (like Titanic/Palmer Penguins). `labels()` returns an `Array1<u8>` diagnosis `num` in `0..=4` (`0` = absence, `1`–`4` = increasing presence), commonly binarized to `0` vs `> 0`. It loads the canonical `processed.cleveland.data` partition (the 14-column subset used by virtually all published experiments).
>
> **Note**: SMS Spam is the first **text** dataset. It has no feature matrix: `texts()` returns an `Array1<String>` of the 5,574 raw SMS message bodies (vectorize them yourself — bag-of-words, TF-IDF, embeddings, …), and `labels()` returns an `Array1<&'static str>` of `"ham"` / `"spam"`. `data()` returns the `(texts, labels)` pair. It is sourced from a **ZIP archive** (like Digits/BankMarketing): the loader downloads `smsspamcollection.zip`, extracts the tab-separated `SMSSpamCollection` file (cached as `sms_spam.csv`), and parses it with quote handling disabled (the messages are free text that can contain `"` and `,`).
>
> **Note**: YouTube Spam is a second **text** dataset and a sibling of SMS Spam (same authors). Like SMS Spam it has no feature matrix: `texts()` returns an `Array1<String>` of the 1,956 raw YouTube comment bodies (the source `CONTENT` column) and `labels()` returns an `Array1<&'static str>` of `"ham"` / `"spam"` (mapped from the source `CLASS` codes `0` / `1`); `data()` returns the `(texts, labels)` pair. It is sourced from a **ZIP archive** of **five** per-video CSVs (comments on music videos by Psy, Katy Perry, LMFAO, Eminem, and Shakira); the loader concatenates them in a fixed order into a single `youtube_spam.csv` covered by one pinned SHA-256, then parses standard comma-separated CSV with quote handling **enabled** (unlike SMS Spam — the comments are properly quoted, one even spanning a newline) and skips each file's repeated header row. The per-comment metadata columns (`COMMENT_ID`, `AUTHOR`, `DATE`) are not exposed.
>
> **Note**: Sentiment Labelled Sentences is a third **text** dataset and the first to carry per-sample **metadata**. It has 3,000 review sentences (1,000 each from Amazon, IMDb, and Yelp; 500 positive + 500 negative per site, so it is perfectly balanced). `texts()` returns an `Array1<String>` of the sentences and `labels()` returns an `Array1<&'static str>` of `"positive"` / `"negative"` (mapped from the source labels `1` / `0`); in addition, `sources()` returns an `Array1<&'static str>` of `"amazon"` / `"imdb"` / `"yelp"` so you can slice the corpus by domain (or set up cross-domain transfer experiments). Because of that extra column, `SentimentSentencesData` is a **triple** `(texts, sources, labels)` and `data()` returns all three. It is sourced from a **ZIP archive** of three per-site `sentence<TAB>label` files; since those files carry no source column, the loader tags each line with its site and combines them into a single `source<TAB>sentence<TAB>label` corpus (`sentiment_sentences.csv`, covered by one pinned SHA-256), parsed tab-separated with quote handling disabled (like SMS Spam).
>
> **Note**: 20 Newsgroups is the first **multi-class** text dataset (20 classes) and the framework-agnostic analogue of scikit-learn's `fetch_20newsgroups`. `texts()` returns an `Array1<String>` of the full raw Usenet posts — **including** the email-style headers, nothing stripped, matching scikit-learn's default — and `labels()` returns an `Array1<&'static str>` of the 20 newsgroup names (e.g. `"sci.space"`). Mirroring scikit-learn's `subset` argument there are three constructors, all sharing one cached archive: `new` (train, 11,314 posts — the default), `new_test` (test, 7,532), and `new_all` (18,846). It uses the canonical **"bydate"** `.tar.gz`, decompressed with `dataset-core`'s `untar_gz`; unlike the other text loaders (which cache a combined CSV), the posts are multi-line raw documents, so the loader caches the tarball as-is (its SHA-256 is the integrity check) and re-extracts it in memory on load, decoding each file as Latin-1 (like scikit-learn) so non-UTF-8 bytes are preserved. Samples are walked in a deterministic (category-then-file lexicographic) order.

## Migration from `dataset-core` 0.1.x

If you used the `datasets` feature of `dataset-core` 0.1.x, switch to this crate:

```diff
- dataset-core = { version = "0.1", features = ["datasets"] }
+ dataset-ml = "0.2"
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

## Datasets Attribution

The bundled datasets are classic machine learning datasets widely used for educational and research purposes:

- **Abalone**: Nash, Sellers, Talbot, Cawthorn & Ford (1994), UCI Machine Learning Repository, from a study of blacklip abalone in Tasmania
- **Adult / Census Income**: Becker & Kohavi (1996), UCI Machine Learning Repository, extracted from the 1994 US Census
- **Bank Marketing**: Moro, Rita & Cortez (2012), UCI Machine Learning Repository, from a Portuguese bank's direct marketing campaigns
- **Iris**: Fisher's Iris dataset (1936)
- **Breast Cancer Wisconsin (Diagnostic)**: Wolberg, Mangasarian, Street & Street (1995)
- **Boston Housing**: Harrison & Rubinfeld (1978)
- **California Housing**: Pace & Barry (1997), from the 1990 U.S. census
- **Forest Cover Type**: Blackard & Dean (1999), UCI Machine Learning Repository, via scikit-learn's `fetch_covtype`
- **Heart Disease**: Janosi, Steinbrunn, Pfisterer & Detrano (1988), UCI Machine Learning Repository, Cleveland Clinic Foundation partition
- **Ionosphere**: Sigillito, Wing, Hutton & Baker (1989), UCI Machine Learning Repository, radar returns collected in Goose Bay, Labrador
- **Car Evaluation**: Bohanec (1988), UCI Machine Learning Repository, derived from a DEX hierarchical decision model
- **KDD Cup 1999**: Stolfo, Fan, Lee, Prodromidis & Chan (1999/2000), UCI KDD Archive, via scikit-learn's `fetch_kddcup99`
- **Diabetes**: Efron, Hastie, Johnstone & Tibshirani (2004), via scikit-learn's `load_diabetes`
- **Linnerud**: A. C. Linnerud (NCSU), via scikit-learn's `load_linnerud`
- **Mushroom**: UCI Machine Learning Repository (1987), from *The Audubon Society Field Guide to North American Mushrooms* (1981)
- **Titanic**: Kaggle Titanic dataset
- **Palmer Penguins**: Horst, Hill & Gorman (2020); data by Gorman, Williams & Fraser (2014)
- **SMS Spam Collection**: Almeida & Hidalgo (2011), UCI Machine Learning Repository, from Grumbletext, the NUS SMS Corpus, and a PhD thesis collection
- **YouTube Spam Collection**: Alberto, Lochter & Almeida (2017), UCI Machine Learning Repository, comments collected from five popular music videos
- **Sentiment Labelled Sentences**: Kotzias, Denil, de Freitas & Smyth (2015), UCI Machine Learning Repository, sentences from Amazon, IMDb, and Yelp reviews
- **20 Newsgroups**: Lang (1995), the `bydate` version curated by Jason Rennie (<http://qwone.com/~jason/20Newsgroups/>), the same tarball scikit-learn's `fetch_20newsgroups` uses
- **Wine Recognition**: Aeberhard & Forina (1991), UCI Machine Learning Repository
- **Wine Quality**: UCI Machine Learning Repository

## Author

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
