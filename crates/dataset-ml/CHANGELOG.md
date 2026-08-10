# Changelog for `dataset-ml`

This file documents all notable changes to the `dataset-ml` crate.

This crate provides loaders for 31 classic machine learning datasets, built on [`dataset-core`](https://crates.io/crates/dataset-core). These include tabular benchmarks (Iris, Breast Cancer, California Housing, Diabetes, Adult, Covtype, …) and text corpora (SMS Spam, 20 Newsgroups, Movie Review Polarity, …). The crate also includes the `preprocessing` and `traits` modules, which apply to every loader.

See [SomeB1oody/dataset-core](https://github.com/SomeB1oody/dataset-core) for more information.

This changelog groups entries by release and lists only each version's notable changes. It omits routine dependency bumps, doc-only tweaks, and minor internal refactors. It summarizes new loaders to their essentials. The crate also re-exports every loader struct at the crate root (for example, `dataset_ml::Iris`).

## [Unreleased]
### Added
- **Bike Sharing** (UCI, Fanaee-T 2013): rental counts of the Capital Bikeshare system in Washington, D.C., over 2011 and 2012, with the calendar attributes and the weather of each period. This is the crate's first dataset with a **time axis**. A new `dates()` accessor returns the calendar date of each record as `YYYY-MM-DD`, and the rows stay in chronological order, so a split by time is possible. One ZIP source holds two aggregations of the same rental log, and each one has its own loader and its own cached file:
  - `BikeSharingHourly`: 17,379 records, 12 numeric features
  - `BikeSharingDaily`: 731 records, 11 numeric features (no `hr` column)

  Both use the multi-output regression target `(casual, registered, cnt)` as an `Array2<f64>` with 3 columns, where `cnt` is the sum of the other two. For the usual single-target task, take `targets.column(2)`.
- Two feature flags, both on by default, so a user can compile only the half they need:
  - `dataset`: the `dataset` module and its 31 loaders, the crate-root re-export of every loader struct, and `DOWNLOAD_RETRIES`. It gates the `csv`, `serde`, and `tempfile` dependencies, which are now optional.
  - `preprocessing`: the `preprocessing` module. It adds no dependencies.

  With `dataset` off, the only direct dependencies left are `dataset-core` and `ndarray`. The `traits` module stays available under every feature combination, so a downstream loader can implement `MlDataset` with both features off.

### Changed
- Dataset loaders moved from the crate root into a new `dataset` module. `dataset_ml::iris::Iris` is now `dataset_ml::dataset::iris::Iris`, and the same one-level shift applies to all 28 loader modules. The crate root now holds three modules: `dataset`, `preprocessing`, and `traits`.
- The crate root still re-exports every loader struct, so `dataset_ml::Iris` and the other struct names are unchanged. Code that imports a struct from the crate root needs no edit.
- The dataset overview table moved from the crate root docs to the `dataset` module docs, which is where the modules it lists now live.

### Testing
- Each integration test file starts with the feature gate it needs: `#![cfg(feature = "dataset")]` for the loader tests and `traits_test.rs`, `#![cfg(feature = "preprocessing")]` for `preprocessing_test.rs`. A test binary compiles to zero tests when its feature is off.

## [0.4.0] - 2026-08-02
### Added
- `traits::MlDataset` (re-exported as `dataset_ml::MlDataset`): every loader now implements this trait, the first uniform surface over "some dataset". It adds container operations (`invalidate`, `is_loaded`, `storage_dir`, `n_samples`) and the data accessors `load`, `load_mut`, `peek`, and `unload`. These accessor names deliberately avoid shadowing a loader's inherent `data` / `get_data` / … methods. A companion `NumSamples` trait (blanket-implemented for the loaders' array pairs and triples) backs `n_samples`.
- `preprocessing` module: helpers that turn loader output into model input, with **no new dependencies**. The shuffle uses a built-in seeded SplitMix64 generator, so splits reproduce across platforms. The module adds `train_test_split`, `stratified_split`, `k_fold_indices`, `shuffled_indices`, `standardize` / `min_max_scale` / `apply_scaler` (plus the `Scaler` they fit), `one_hot_encode`, `label_encode`, and `class_counts`. Splits return row indices. This keeps parallel arrays aligned. Scalers compute statistics over finite values only, so `NaN` missing-value markers stay untouched.
- **Spambase** (UCI, 1999): 4,601 emails (2,788 ham / 1,813 spam), 57 numeric features, `"ham"`/`"spam"` label. It is the feature-engineered counterpart to the raw-text spam corpora. ZIP source.
- **Letter Recognition** (UCI, 1991): 20,000 samples, 16 integer features, the 26 capital letters as an `Array1<char>` label (the crate's widest classification by class count). ZIP source.
- **Banknote Authentication** (UCI, 2012): 1,372 samples, 4 continuous features, raw `0`/`1` target. It is the most compact pure-numeric benchmark. ZIP source.
- `DOWNLOAD_RETRIES`: the crate's shared download-retry policy (2 extra tries), exposed as a public constant.

### Changed
- Every loader now downloads through `dataset-core`'s `download_to_with_retries` with `DOWNLOAD_RETRIES` extra tries, so a transient timeout on a university archive no longer fails a run.
- `wine_quality::WineData` is now `pub` (was `pub(crate)`), so the wine loaders can name it as their `MlDataset::Data`. The type itself is unchanged.

### Testing
- Added `tests/preprocessing_test.rs` (27 tests, network-free) and `tests/traits_test.rs` (6 tests). `tests/common` now uses `dataset-core`'s `verify_sha256` instead of reimplementing SHA-256. This drops the `sha2` dev-dependency.

## [0.3.0] - 2026-07-12
### Added
**Text corpora** (the crate's first text loaders: `texts()` in place of `features()`):
- **SMS Spam** (UCI, 2011): 5,574 messages, `"ham"`/`"spam"`. The first text loader.
- **YouTube Spam** (UCI, 2017): 1,956 comments, `"ham"`/`"spam"`. Combined from five per-video CSVs.
- **Sentiment Labelled Sentences** (UCI, 2015): 3,000 sentences from Amazon/IMDb/Yelp, `"positive"`/`"negative"`. Adds a `sources()` accessor (data is a triple).
- **20 Newsgroups** (bydate): ~18,846 posts across 20 groups, multi-class. First `.tar.gz` source (decoded as Latin-1), with `new` / `new_test` / `new_all` subsets.
- **Movie Review Polarity** (Pang & Lee, 2004): 2,000 full reviews, `"positive"`/`"negative"`.

**Tabular loaders:**
- **Digits** (scikit-learn `load_digits`): 1,797 samples, 64 pixel features, digit `0`–`9`. First ZIP source and first `u8` label.
- **Linnerud** (scikit-learn): 20 samples, multi-output regression (3 features → 3 targets). First `Array2<f64>` target.
- **Covtype** (scikit-learn `fetch_covtype`): 581,012 samples, 54 features, 7 cover types. First gzip source.
- **KDD Cup 1999**: 41 mixed features, 23 intrusion classes. `new` (10% subset, 494,021) and `new_full` (4,898,431).
- **Adult / Census Income** (UCI, 1996): 32,561 samples, 14 mixed features, `<=50K`/`>50K`.
- **Bank Marketing** (UCI, 2012): 45,211 samples, 16 mixed features, `yes`/`no`. ZIP source.
- **Mushroom** (UCI, 1987): 8,124 samples, 22 categorical features, edible/poisonous. First all-categorical loader.
- **Ionosphere** (UCI, 1989): 351 samples, 34 continuous features, `"good"`/`"bad"`.
- **Car Evaluation** (UCI, 1988): 1,728 samples, 6 categorical features, 4-class.
- **Heart Disease (Cleveland)** (UCI, 1988): 303 samples, 13 numeric features (missing `?` → `NaN`), diagnosis `0`–`4`.
- **Abalone** (UCI, 1994): 4,177 samples, 8 mixed features, regression target `rings`. First mixed-type regression loader.

### Changed
- **Breaking:** `diabetes::Diabetes` now loads scikit-learn's `load_diabetes` (442 samples × 10 standardized features, **regression**) instead of the Pima Indians Diabetes dataset (768 × 8, classification). This change renames its label accessor from `labels()` to `targets()`.

## [0.2.0] - 2026-06-05
### Added
- **Breast Cancer Wisconsin (Diagnostic)** (UCI): 569 samples, 30 numeric features, `"malignant"`/`"benign"`.
- **Wine recognition** (scikit-learn `load_wine`): 178 samples, 13 numeric features, 3 cultivars (distinct from the Wine Quality regression datasets).
- **Palmer Penguins**: 344 samples, mixed features (2 categorical + 5 numeric), 3 species. `NA` → `NaN`/`""`.
- **California Housing**: 20,640 samples, 8 numeric features, regression target `MedHouseVal`. Reproduces scikit-learn's engineered features and replaces Boston Housing.
- `into_data` / `take_data` (owned arrays, no clone) and `get_data` / `get_data_mut` (borrow the cache without loading) on every loader.

### Changed
- Adapted every loader to `dataset-core`'s loader-on-construction API (fields become `Dataset<XData, DatasetError>`). Each loader's public API is unchanged.
- Refactored CSV parsing in every loader to Serde. Positional `#[derive(Deserialize)]` records replace the manual per-field parsing and column-count checks. Adds `serde` as a direct dependency.
- `data()` now returns a reference to the cached tuple (for example, `&IrisData`) instead of a tuple of references. Each content type gained a named alias (`IrisData`, `TitanicData`, …).

## [0.1.0] - 2026-05-27
### Added
- Initial release as a standalone crate, separated from `dataset-core` 0.1.x. That crate's `datasets` feature had previously gated the loaders.
- Loaders: `Iris`, `BostonHousing`, `Diabetes`, `Titanic`, `RedWineQuality`, `WhiteWineQuality`.

### Migration from `dataset-core` 0.1.x with the `datasets` feature

| Old path (`dataset-core` 0.1.x) | New path (`dataset-ml` 0.1.0) |
|---|---|
| `dataset_core::datasets::iris::Iris` | `dataset_ml::iris::Iris` |
| `dataset_core::datasets::boston_housing::BostonHousing` | `dataset_ml::boston_housing::BostonHousing` |
| `dataset_core::datasets::diabetes::Diabetes` | `dataset_ml::diabetes::Diabetes` |
| `dataset_core::datasets::titanic::Titanic` | `dataset_ml::titanic::Titanic` |
| `dataset_core::datasets::wine_quality::red_wine_quality::RedWineQuality` | `dataset_ml::wine_quality::red_wine_quality::RedWineQuality` |
| `dataset_core::datasets::wine_quality::white_wine_quality::WhiteWineQuality` | `dataset_ml::wine_quality::white_wine_quality::WhiteWineQuality` |

## History before the split
The project developed these loaders inside `dataset-core` 0.1.x before this crate existed. This section summarizes the milestones instead of listing them day-by-day:
- Scaffolded the starter datasets: Iris, Diabetes, Boston Housing, Titanic, and Wine Quality.
- Replaced the hardcoded/bundled data with dynamic download and SHA-256 validation.
- Refactored every loader to lazy loading and caching on the `Dataset` container.
- Switched manual CSV parsing to the `csv` crate.
- Split Wine Quality into separate red and white datasets and added semantic/consistency tests.
