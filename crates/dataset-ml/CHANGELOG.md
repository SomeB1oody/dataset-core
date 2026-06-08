# Changelog — `dataset-ml`

All notable changes to the `dataset-ml` crate will be documented in this file.

This crate provides ready-to-use loaders for classic machine learning datasets (Iris, Breast Cancer, Boston/California Housing, Diabetes, Titanic, Palmer Penguins, Wine Recognition, Wine Quality), built on top of [`dataset-core`](https://crates.io/crates/dataset-core).

Please view [SomeB1oody/dataset-core](https://github.com/SomeB1oody/dataset-core) for more info.

## [Unreleased] - 2026-06-07
### Added
- `digits::Digits` loader for the Optical Recognition of Handwritten Digits dataset (scikit-learn's `load_digits`): 1,797 samples, 64 numeric pixel features (an 8×8 grayscale image flattened in row-major order, each intensity an integer in `0..=16`), and a `u8` digit label (`0`–`9`) for multi-class classification. This is the first loader with an `Array1<u8>` target and the first to extract its source from a **ZIP archive**: it downloads the UCI static package, unzips it, and uses the `optdigits.tes` test partition (the same partition scikit-learn uses, hence the 1,797 sample count), with SHA-256 verification. The struct is also re-exported at the crate root as `dataset_ml::Digits`.

## [Unreleased] - 2026-06-06
### Changed
- **BREAKING**: `diabetes::Diabetes` now loads the **scikit-learn `load_diabetes`** dataset (Efron, Hastie, Johnstone & Tibshirani, 2004) instead of the Pima Indians Diabetes dataset — changing it from **768 samples × 8 features, binary classification** to **442 samples × 10 features, regression**. The ten features are standardized to reproduce scikit-learn's default `load_diabetes()` output (each column mean-centered and divided by its L2 norm, so its sum of squares is 1); the target is the unscaled measure of disease progression one year after baseline (integer-valued, range 25–346). The label accessor is renamed `labels()` → `targets()` to match the other regression loaders (`CaliforniaHousing`, `BostonHousing`); `DiabetesData` remains `(Array2<f64>, Array1<f64>)` but its `.1` is now the regression target. The source URL and pinned SHA-256 now point at the original tab-separated `diabetes.tab` file, with SHA-256 verification.

## [0.2.0] - 2026-06-05
### Added
- `california_housing::CaliforniaHousing` loader for the California Housing dataset: 20,640 samples, 8 numeric features, and an `f64` regression target (`MedHouseVal`, median house value in units of $100,000). Unlike the other loaders, it does **feature engineering** rather than exposing raw columns — it reproduces scikit-learn's `fetch_california_housing` features (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`) by deriving the per-household ratios from Géron's `housing.csv` and scaling the target by `1/100000`. The source's 207 missing `total_bedrooms` values surface as `NaN` in `AveBedrms` (sklearn's complete upstream has none). A modern replacement for Boston Housing. Sourced with SHA-256 verification; the struct is also re-exported at the crate root as `dataset_ml::CaliforniaHousing`.

## [0.2.0] - 2026-06-04
### Added
- `palmer_penguins::PalmerPenguins` loader for the Palmer Penguins dataset: 344 samples with mixed features — 2 categorical string features (`island`, `sex`) and 5 numeric features (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `year`) — and a `&'static str` species label (`"Adelie"` / `"Chinstrap"` / `"Gentoo"`) for multi-class classification. Like `Titanic`, `features()` returns `(&Array2<String>, &Array2<f64>)` and `data()` is a triple. The source encodes missing values as the literal token `NA`, which becomes `NaN` (numeric) or `""` (string). Sourced from the palmerpenguins R package (`penguins.csv`) with SHA-256 verification. The struct is also re-exported at the crate root as `dataset_ml::PalmerPenguins`.

## [0.2.0] - 2026-06-03
### Added
- `wine_recognition::WineRecognition` loader for the Wine recognition dataset (scikit-learn's `load_wine`): 178 samples, 13 numeric features (the chemical constituents), and a `&'static str` cultivar label (`"class_1"` / `"class_2"` / `"class_3"`) for multi-class classification. Sourced from the UCI Machine Learning Repository (`wine.data`) with SHA-256 verification. This is **distinct** from the `wine_quality` datasets, which are a regression task on quality scores. The struct is also re-exported at the crate root as `dataset_ml::WineRecognition`.

## [0.2.0] - 2026-06-02
### Added
- `breast_cancer::BreastCancer` loader for the Breast Cancer Wisconsin (Diagnostic) dataset: 569 samples, 30 numeric features (the `mean`, `se`, and `worst` statistics for 10 cell-nucleus measurements), and a `&'static str` diagnosis label (`"malignant"` / `"benign"`) for binary classification. Sourced from the UCI Machine Learning Repository (`wdbc.data`) with SHA-256 verification. The struct is also re-exported at the crate root as `dataset_ml::BreastCancer`.

## [0.2.0] - 2026-05-29
### Changed
- Adapted to `dataset-core`'s loader-on-construction API: the loader is now stored on the `Dataset` and supplied at construction. Every loader's `Dataset<XData>` field becomes `Dataset<XData, DatasetError>`, `new` passes `Self::load_data` to `Dataset::new`, and the accessor methods call `self.dataset.load()` (no argument). The public API of each loader (`Iris::new(dir)`, `features()`, `labels()`, `data()`, etc.) is unchanged.
- Refactored CSV parsing in every dataset loader to use Serde. Each loader now defines a `#[derive(Deserialize)]` record struct and parses with `csv::Reader::deserialize()`, replacing the manual per-field `record[i].parse()` loops, the `num_features` inference, and the explicit column-count checks (the `csv` reader now enforces a consistent column count). Records are deserialized **positionally**, so parsing is independent of the exact CSV header spelling and of any byte-order mark on the header row. Missing Titanic numeric fields deserialize to `None` and are mapped to `NaN` exactly as before.
- The error type and the download → SHA-256 verify → cache → reuse workflow are unchanged; the Serde change is an internal parsing refactor.
- Each loader's content type now has a named alias (`IrisData`, `BostonHousingData`, `DiabetesData`, `TitanicData`, and the shared `WineData` for both wine subsets), used for the `Dataset<…>` field and the owned/borrowed accessor return types.
- `data()` on every loader now returns a reference to the cached data tuple (`&IrisData`, `&TitanicData`, …) instead of a tuple of references (`(&Array2, &Array1)`). Call-site destructuring (`let (features, labels) = ds.data()?`) is unchanged thanks to match ergonomics.

### Added
- `into_data(self)` and `take_data(&mut self)` on every dataset loader (`Iris`, `BostonHousing`, `Diabetes`, `Titanic`, `RedWineQuality`, `WhiteWineQuality`), returning **owned** arrays instead of borrows — no `to_owned()` clone needed. `into_data` consumes the loader; `take_data` leaves it reusable (a later accessor call reloads). Built on the new `Dataset::into_inner` / `Dataset::take` in `dataset-core`.
- `get_data(&self) -> Option<&XData>` and `get_data_mut(&mut self) -> Option<&mut XData>` on every dataset loader, returning a reference to the cached data tuple **without** triggering loading (they return `None` if the data has not been loaded yet). `get_data` borrows it; `get_data_mut` allows editing it in place — no `to_owned()` clone, no reload, and the change persists in the cache. Built on the new `Dataset::get` / `Dataset::get_mut` in `dataset-core`.
- `serde` (with the `derive` feature) as a direct dependency, used for record deserialization.

## [0.1.0] - 2026-05-27
### Added
- Initial release as a standalone crate.
- Split out from `dataset-core` v0.1.x as part of the workspace reorganization. The dataset loaders previously lived behind the `datasets` feature in `dataset-core`; they are now their own crate that depends on `dataset-core` with the `utils` feature.
- Public surface: `iris::Iris`, `boston_housing::BostonHousing`, `diabetes::Diabetes`, `titanic::Titanic`, `wine_quality::red_wine_quality::RedWineQuality`, `wine_quality::white_wine_quality::WhiteWineQuality`. The structs themselves are also re-exported at the crate root.

### Migration from `dataset-core` 0.1.x with `datasets` feature

| Old path (`dataset-core` 0.1.x)                                              | New path (`dataset-ml` 0.1.0)                                    |
|------------------------------------------------------------------------------|------------------------------------------------------------------|
| `dataset_core::datasets::iris::Iris`                                         | `dataset_ml::iris::Iris`                                         |
| `dataset_core::datasets::boston_housing::BostonHousing`                      | `dataset_ml::boston_housing::BostonHousing`                      |
| `dataset_core::datasets::diabetes::Diabetes`                                 | `dataset_ml::diabetes::Diabetes`                                 |
| `dataset_core::datasets::titanic::Titanic`                                   | `dataset_ml::titanic::Titanic`                                   |
| `dataset_core::datasets::wine_quality::red_wine_quality::RedWineQuality`     | `dataset_ml::wine_quality::red_wine_quality::RedWineQuality`     |
| `dataset_core::datasets::wine_quality::white_wine_quality::WhiteWineQuality` | `dataset_ml::wine_quality::white_wine_quality::WhiteWineQuality` |

## History prior to the split

The following entries trace the development of the dataset loaders while they were part of `dataset-core` 0.1.x. They are reproduced here for continuity.

### 2026-04-14
- Remove formatting of record data in dataset validations.

### 2026-04-13
- Add comprehensive semantic tests for datasets, including value constraints, consistency checks.

### 2026-04-12
- Add detailed module-level dataset descriptions for all dataset submodules.

### 2026-04-11
- Split Wine Quality dataset into red and white datasets.

### 2026-04-07
- Add feature gating for `datasets`, update examples and tests.

### 2026-04-06
- Rename all dataset test files to `dataset_*_test.rs`.

### 2026-04-05
- Move dataset implementations to the `datasets` module.

### 2026-04-02
- Streamline dataset loader methods so that downloading and parsing happen in one method.

### 2026-04-01
- Refactor dataset modules to use `Dataset` for unified lazy loading and caching logic.

### 2026-03-29
- Refactor dataset modules to return `PathBuf` from `download_dataset` and accept it in `parse_dataset` for improved API clarity.

### 2026-03-28
- Separate downloading and parsing logic for improved clarity and maintainability.

### 2026-03-27
- Remove hardcoded feature counts from dataset modules and infer dynamically.

### 2026-03-26
- Refactor dataset error handling to use `empty_dataset` helper for improved clarity.

### 2026-03-25
- Remove hardcoded dataset sample sizes for dynamic determination.

### 2026-03-18
- Implement `Clone` and `Debug` traits for dataset structs.

### 2026-03-17
- Use type alias for Titanic dataset.

### 2026-03-16
- Replace manual CSV parsing with the `csv` crate for Titanic and Wine Quality datasets.

### 2026-03-14
- Replace manual CSV parsing with the `csv` crate for Iris.

### 2026-03-13
- Replace manual CSV parsing with the `csv` crate for Diabetes.

### 2026-03-12
- Replace manual CSV parsing with the `csv` crate for Boston Housing.

### 2026-03-11
- Improve error handling and code readability across dataset modules.

### 2026-03-09
- Refactor Wine Quality dataset handling to enable lazy loading and improve modularity.

### 2026-03-08
- Refactor Titanic dataset handling to enable lazy loading and improve modularity.

### 2026-03-07
- Refactor Iris dataset handling to enable lazy loading and improve modularity.

### 2026-03-06
- Refactor Diabetes dataset handling to enable lazy loading and improve modularity.

### 2026-03-05
- Refactor Boston Housing dataset handling to enable lazy loading and improve modularity.

### 2026-03-04
- Update dataset documentation for clarity and consistency.

### 2026-03-02
- Add SHA-256 validation for Wine Quality dataset download.

### 2026-03-01
- Add SHA-256 validation for Titanic dataset download.

### 2026-02-28
- Add SHA-256 validation for Iris dataset download.

### 2026-02-27
- Add SHA-256 validation for Diabetes dataset download.

### 2026-02-26
- Add SHA-256 validation for Boston Housing dataset download.

### 2026-02-24
- Replace hardcoded dataset configurations with reusable constants.

### 2026-02-23
- Replace hardcoded Wine Quality dataset with dynamic download and processing.

### 2026-02-22
- Replace hardcoded Titanic dataset with dynamic download and processing.

### 2026-02-20
- Replace hardcoded Boston Housing dataset with dynamic download and processing.

### 2026-02-19
- Replace hardcoded Diabetes dataset with dynamic download and processing.

### 2026-02-18
- Replace hardcoded Iris dataset with dynamic download and processing.

### 2026-02-14
- Initial project scaffolding with starter datasets and documentation.
