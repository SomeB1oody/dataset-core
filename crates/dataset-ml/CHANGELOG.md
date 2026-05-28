# Changelog — `dataset-ml`

All notable changes to the `dataset-ml` crate will be documented in this file.

This crate provides ready-to-use loaders for classic machine learning datasets (Iris, Boston Housing, Diabetes, Titanic, Wine Quality), built on top of [`dataset-core`](https://crates.io/crates/dataset-core).

Please view [SomeB1oody/dataset-core](https://github.com/SomeB1oody/dataset-core) for more info.

## [0.1.0] - 2026-5-27
### Added
- Initial release as a standalone crate.
- Split out from `dataset-core` v0.1.x as part of the workspace reorganization. The dataset loaders previously lived behind the `datasets` feature in `dataset-core`; they are now their own crate that depends on `dataset-core` with the `utils` feature.
- Public surface: `iris::Iris`, `boston_housing::BostonHousing`, `diabetes::Diabetes`, `titanic::Titanic`, `wine_quality::red_wine_quality::RedWineQuality`, `wine_quality::white_wine_quality::WhiteWineQuality`. The structs themselves are also re-exported at the crate root.

### Migration from `dataset-core` 0.1.x with `datasets` feature

| Old path (`dataset-core` 0.1.x)                                    | New path (`dataset-ml` 0.1.0)                              |
|--------------------------------------------------------------------|-------------------------------------------------------------|
| `dataset_core::datasets::iris::Iris`                               | `dataset_ml::iris::Iris`                                    |
| `dataset_core::datasets::boston_housing::BostonHousing`            | `dataset_ml::boston_housing::BostonHousing`                 |
| `dataset_core::datasets::diabetes::Diabetes`                       | `dataset_ml::diabetes::Diabetes`                            |
| `dataset_core::datasets::titanic::Titanic`                         | `dataset_ml::titanic::Titanic`                              |
| `dataset_core::datasets::wine_quality::red_wine_quality::RedWineQuality` | `dataset_ml::wine_quality::red_wine_quality::RedWineQuality` |
| `dataset_core::datasets::wine_quality::white_wine_quality::WhiteWineQuality` | `dataset_ml::wine_quality::white_wine_quality::WhiteWineQuality` |

## History prior to the split

The following entries trace the development of the dataset loaders while they were part of `dataset-core` 0.1.x. They are reproduced here for continuity.

### 2026-4-14
- Remove formatting of record data in dataset validations.

### 2026-4-13
- Add comprehensive semantic tests for datasets, including value constraints, consistency checks.

### 2026-4-12
- Add detailed module-level dataset descriptions for all dataset submodules.

### 2026-4-11
- Split Wine Quality dataset into red and white datasets.

### 2026-4-7
- Add feature gating for `datasets`, update examples and tests.

### 2026-4-6
- Rename all dataset test files to `dataset_*_test.rs`.

### 2026-4-5
- Move dataset implementations to the `datasets` module.

### 2026-4-2
- Streamline dataset loader methods so that downloading and parsing happen in one method.

### 2026-4-1
- Refactor dataset modules to use `Dataset` for unified lazy loading and caching logic.

### 2026-3-29
- Refactor dataset modules to return `PathBuf` from `download_dataset` and accept it in `parse_dataset` for improved API clarity.

### 2026-3-28
- Separate downloading and parsing logic for improved clarity and maintainability.

### 2026-3-27
- Remove hardcoded feature counts from dataset modules and infer dynamically.

### 2026-3-26
- Refactor dataset error handling to use `empty_dataset` helper for improved clarity.

### 2026-3-25
- Remove hardcoded dataset sample sizes for dynamic determination.

### 2026-3-18
- Implement `Clone` and `Debug` traits for dataset structs.

### 2026-3-17
- Use type alias for Titanic dataset.

### 2026-3-16
- Replace manual CSV parsing with the `csv` crate for Titanic and Wine Quality datasets.

### 2026-3-14
- Replace manual CSV parsing with the `csv` crate for Iris.

### 2026-3-13
- Replace manual CSV parsing with the `csv` crate for Diabetes.

### 2026-3-12
- Replace manual CSV parsing with the `csv` crate for Boston Housing.

### 2026-3-11
- Improve error handling and code readability across dataset modules.

### 2026-3-9
- Refactor Wine Quality dataset handling to enable lazy loading and improve modularity.

### 2026-3-8
- Refactor Titanic dataset handling to enable lazy loading and improve modularity.

### 2026-3-7
- Refactor Iris dataset handling to enable lazy loading and improve modularity.

### 2026-3-6
- Refactor Diabetes dataset handling to enable lazy loading and improve modularity.

### 2026-3-5
- Refactor Boston Housing dataset handling to enable lazy loading and improve modularity.

### 2026-3-4
- Update dataset documentation for clarity and consistency.

### 2026-3-2
- Add SHA-256 validation for Wine Quality dataset download.

### 2026-3-1
- Add SHA-256 validation for Titanic dataset download.

### 2026-2-28
- Add SHA-256 validation for Iris dataset download.

### 2026-2-27
- Add SHA-256 validation for Diabetes dataset download.

### 2026-2-26
- Add SHA-256 validation for Boston Housing dataset download.

### 2026-2-24
- Replace hardcoded dataset configurations with reusable constants.

### 2026-2-23
- Replace hardcoded Wine Quality dataset with dynamic download and processing.

### 2026-2-22
- Replace hardcoded Titanic dataset with dynamic download and processing.

### 2026-2-20
- Replace hardcoded Boston Housing dataset with dynamic download and processing.

### 2026-2-19
- Replace hardcoded Diabetes dataset with dynamic download and processing.

### 2026-2-18
- Replace hardcoded Iris dataset with dynamic download and processing.

### 2026-2-14
- Initial project scaffolding with starter datasets and documentation.
