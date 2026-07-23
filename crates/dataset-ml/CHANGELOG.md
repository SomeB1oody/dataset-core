# Changelog — `dataset-ml`

All notable changes to the `dataset-ml` crate will be documented in this file.

This crate provides ready-to-use loaders for 29 classic machine learning datasets — tabular benchmarks (Iris, Breast Cancer, California Housing, Diabetes, Adult, Covtype, …) and text corpora (SMS Spam, 20 Newsgroups, Movie Review Polarity, …) — built on top of [`dataset-core`](https://crates.io/crates/dataset-core), plus the `preprocessing` and `traits` modules that apply to all of them.

Please view [SomeB1oody/dataset-core](https://github.com/SomeB1oody/dataset-core) for more info.

Entries are grouped by release and list only each version's notable changes; routine dependency bumps, doc-only tweaks, and minor internal refactors are omitted. New loaders are summarized to their essentials — every loader struct is also re-exported at the crate root (e.g. `dataset_ml::Iris`).

## [Unreleased]
### Added
- `traits::MlDataset` (re-exported as `dataset_ml::MlDataset`) — a trait implemented by **every** loader, the first uniform surface over "some dataset": container operations (`invalidate`, `is_loaded`, `storage_dir`, `n_samples`) plus `load` / `load_mut` / `peek` / `unload` data accessors (deliberately named so they never shadow a loader's inherent `data` / `get_data` / … methods). A companion `NumSamples` trait (blanket-implemented for the loaders' array pairs/triples) backs `n_samples`.
- `preprocessing` module — helpers that turn loader output into model input, with **no new dependencies** (shuffling uses a built-in seeded SplitMix64 for platform-reproducible splits): `train_test_split`, `stratified_split`, `k_fold_indices`, `shuffled_indices`, `standardize` / `min_max_scale` / `apply_scaler` (plus the `Scaler` they fit), `one_hot_encode`, `label_encode`, and `class_counts`. Splits return row indices (keeping parallel arrays aligned); scalers compute statistics over finite values only, leaving `NaN` missing-value markers untouched.
- **Spambase** (UCI, 1999) — 4,601 emails (2,788 ham / 1,813 spam), 57 numeric features, `"ham"`/`"spam"` label; the feature-engineered counterpart to the raw-text spam corpora. ZIP source.
- **Letter Recognition** (UCI, 1991) — 20,000 samples, 16 integer features, the 26 capital letters as an `Array1<char>` label (the crate's widest classification by class count). ZIP source.
- **Banknote Authentication** (UCI, 2012) — 1,372 samples, 4 continuous features, raw `0`/`1` target; the most compact pure-numeric benchmark. ZIP source.
- `DOWNLOAD_RETRIES` — the crate's shared download-retry policy (2 extra attempts), exposed as a public constant.

### Changed
- Every loader now downloads through `dataset-core`'s `download_to_with_retries` with `DOWNLOAD_RETRIES` extra attempts, so a transient timeout on a university archive no longer fails a run.
- `wine_quality::WineData` is now `pub` (was `pub(crate)`) so the wine loaders can name it as their `MlDataset::Data`; the type itself is unchanged.

### Testing
- Added `tests/preprocessing_test.rs` (27 tests, network-free) and `tests/traits_test.rs` (6 tests). `tests/common` now uses `dataset-core`'s `verify_sha256` instead of reimplementing SHA-256, dropping the `sha2` dev-dependency.

## [0.3.0] - 2026-07-12
### Added
**Text corpora** (the crate's first text loaders — `texts()` in place of `features()`):
- **SMS Spam** (UCI, 2011) — 5,574 messages, `"ham"`/`"spam"`; the first text loader.
- **YouTube Spam** (UCI, 2017) — 1,956 comments, `"ham"`/`"spam"`; combined from five per-video CSVs.
- **Sentiment Labelled Sentences** (UCI, 2015) — 3,000 sentences from Amazon/IMDb/Yelp, `"positive"`/`"negative"`; adds a `sources()` accessor (data is a triple).
- **20 Newsgroups** (bydate) — ~18,846 posts across 20 groups, multi-class; first `.tar.gz` source (decoded as Latin-1), with `new` / `new_test` / `new_all` subsets.
- **Movie Review Polarity** (Pang & Lee, 2004) — 2,000 full reviews, `"positive"`/`"negative"`.

**Tabular loaders:**
- **Digits** (scikit-learn `load_digits`) — 1,797 samples, 64 pixel features, digit `0`–`9`; first ZIP source and first `u8` label.
- **Linnerud** (scikit-learn) — 20 samples, multi-output regression (3 features → 3 targets); first `Array2<f64>` target.
- **Covtype** (scikit-learn `fetch_covtype`) — 581,012 samples, 54 features, 7 cover types; first gzip source.
- **KDD Cup 1999** — 41 mixed features, 23 intrusion classes; `new` (10% subset, 494,021) and `new_full` (4,898,431).
- **Adult / Census Income** (UCI, 1996) — 32,561 samples, 14 mixed features, `<=50K`/`>50K`.
- **Bank Marketing** (UCI, 2012) — 45,211 samples, 16 mixed features, `yes`/`no`; ZIP source.
- **Mushroom** (UCI, 1987) — 8,124 samples, 22 categorical features, edible/poisonous; first all-categorical loader.
- **Ionosphere** (UCI, 1989) — 351 samples, 34 continuous features, `"good"`/`"bad"`.
- **Car Evaluation** (UCI, 1988) — 1,728 samples, 6 categorical features, 4-class.
- **Heart Disease (Cleveland)** (UCI, 1988) — 303 samples, 13 numeric features (missing `?` → `NaN`), diagnosis `0`–`4`.
- **Abalone** (UCI, 1994) — 4,177 samples, 8 mixed features, regression target `rings`; first mixed-type regression loader.

### Changed
- **Breaking:** `diabetes::Diabetes` now loads scikit-learn's `load_diabetes` (442 samples × 10 standardized features, **regression**) instead of the Pima Indians Diabetes dataset (768 × 8, classification); its label accessor is renamed `labels()` → `targets()`.

## [0.2.0] - 2026-06-05
### Added
- **Breast Cancer Wisconsin (Diagnostic)** (UCI) — 569 samples, 30 numeric features, `"malignant"`/`"benign"`.
- **Wine recognition** (scikit-learn `load_wine`) — 178 samples, 13 numeric features, 3 cultivars (distinct from the Wine Quality regression datasets).
- **Palmer Penguins** — 344 samples, mixed features (2 categorical + 5 numeric), 3 species; `NA` → `NaN`/`""`.
- **California Housing** — 20,640 samples, 8 numeric features, regression target `MedHouseVal`; reproduces scikit-learn's engineered features and replaces Boston Housing.
- `into_data` / `take_data` (owned arrays, no clone) and `get_data` / `get_data_mut` (borrow the cache without loading) on every loader.

### Changed
- Adapted every loader to `dataset-core`'s loader-on-construction API (fields become `Dataset<XData, DatasetError>`); each loader's public API is unchanged.
- Refactored CSV parsing in every loader to Serde — positional `#[derive(Deserialize)]` records replacing the manual per-field parsing and column-count checks. Adds `serde` as a direct dependency.
- `data()` now returns a reference to the cached tuple (e.g. `&IrisData`) instead of a tuple of references; each content type gained a named alias (`IrisData`, `TitanicData`, …).

## [0.1.0] - 2026-05-27
### Added
- Initial release as a standalone crate, split out from `dataset-core` 0.1.x (the loaders previously lived behind that crate's `datasets` feature).
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

## History prior to the split
These loaders were developed inside `dataset-core` 0.1.x before this crate existed; the milestones, summarized rather than listed day-by-day:
- Scaffolded the starter datasets: Iris, Diabetes, Boston Housing, Titanic, and Wine Quality.
- Replaced the hardcoded/bundled data with dynamic download and SHA-256 validation.
- Refactored every loader to lazy loading and caching on the `Dataset` container.
- Switched manual CSV parsing to the `csv` crate.
- Split Wine Quality into separate red and white datasets and added semantic/consistency tests.
