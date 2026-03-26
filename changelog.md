# Changelog

All notable changes to this project will be documented in this file.
This change log records updates after 2026-2-13.

Please view [SomeB1oody/RustyML-dataset](https://github.com/SomeB1oody/RustyML-dataset) for more info.

## [0.1.0] - 2026-3-26
### Changed
- Refactor dataset error handling to use `empty_dataset` helper for improved clarity

## [0.1.0] - 2026-3-25
### Changed
- Remove hardcoded dataset sample sizes for dynamic determination

## [0.1.0] - 2026-3-24
### Changed
- Update zip crate version to 8.3.1
- Rename `storage_path` to `storage_dir` across all dataset modules for improved clarity

## [0.1.0] - 2026-3-22
### Changed
- Update zip crate version to 8.3.0

## [0.1.0] - 2026-3-21
### Changed
- Refactor dataset modules to streamline error handling

## [0.1.0] - 2026-3-20
### Added
- Add `Thread Safety` documentation to dataset modules

## [0.1.0] - 2026-3-19
### Changed
- Ensure temporary directory cleanup on SHA256 hash validation failure for all datasets

## [0.1.0] - 2026-3-18
### Changed
- Implement `Clone` and `Debug` traits for dataset structs to improve usability and debugging

## [0.1.0] - 2026-3-17
### Changed
- Update rust version to 1.88.0
- Use type alias for Titanic dataset

## [0.1.0] - 2026-3-16
### Changed
- Refactor Titanic dataset handling to replace manual CSV parsing with `csv` crate for improved reliability and readability
- Refactor Wine Quality dataset handling to replace manual CSV parsing with `csv` crate for improved reliability and readability

## [0.1.0] - 2026-3-14
### Changed
- Refactor Iris dataset handling to replace manual CSV parsing with `csv` crate for improved reliability and readability

## [0.1.0] - 2026-3-13
### Changed
- Refactor Diabetes dataset handling to replace manual CSV parsing with `csv` crate for improved reliability and readability

## [0.1.0] - 2026-3-12
### Changed
- Update tempfile version to 3.27.0
- Refactor Boston Housing dataset handling to replace manual CSV parsing with `csv` crate for improved reliability and readability

## [0.1.0] - 2026-3-11
### Changed
- Refactor dataset modules to improve error handling and enhance code readability

## [0.1.0] - 2026-3-10
### Changed
- Update crate documentation to reflect the struct-based API

## [0.1.0] - 2026-3-9
### Changed
- Refactor Wine Quality dataset handling to enable lazy loading and improve modularity

## [0.1.0] - 2026-3-8
### Changed
- Refactor Titanic dataset handling to enable lazy loading and improve modularity

## [0.1.0] - 2026-3-7
### Changed
- Refactor Iris dataset handling to enable lazy loading and improve modularity

## [0.1.0] - 2026-3-6
### Changed
- Refactor Diabetes dataset handling to enable lazy loading and improve modularity

## [0.1.0] - 2026-3-5
### Changed
- Refactor Boston Housing dataset handling to enable lazy loading and improve modularity

## [0.1.0] - 2026-3-4
### Changed
- Update dataset documentation for clarity and consistency

## [0.1.0] - 2026-3-3
### Changed
- Refactor dataset modules to use `prepare_download_dir` for unified download and validation logic

## [0.1.0] - 2026-3-2
### Changed
- Add SHA256 validation for Wine Quality dataset download

## [0.1.0] - 2026-3-1
### Changed
- Add SHA256 validation for Titanic dataset download

## [0.1.0] - 2026-2-28
### Changed
- Add SHA256 validation for Iris dataset download

## [0.1.0] - 2026-2-27
### Changed
- Add SHA256 validation for Diabetes dataset download

## [0.1.0] - 2026-2-26
### Changed
- Add SHA256 validation for Boston Housing dataset download

## [0.1.0] - 2026-2-25
### Changed
- Update documentation to reflect automatic downloading, caching, and expanded feature set
- Update `tempfile` version to 3.26.0

## [0.1.0] - 2026-2-24
### Changed
- Replace hardcoded dataset configurations with reusable constants across datasets

## [0.1.0] - 2026-2-23
### Changed
- Replace hardcoded Wine Quality dataset with dynamic download and processing

## [0.1.0] - 2026-2-22
### Changed
- Replace hardcoded Titanic dataset with dynamic download and processing

## [0.1.0] - 2026-2-21
### Changed
- Automatically create dataset directories if they do not exist

## [0.1.0] - 2026-2-20
### Changed
- Replace hardcoded Boston Housing dataset with dynamic download and processing

## [0.1.0] - 2026-2-19
### Changed
- Replace hardcoded Diabetes dataset with dynamic download and processing

## [0.1.0] - 2026-2-18
### Changed
- Replace hardcoded Iris dataset with dynamic download and processing

## [0.1.0] - 2026-2-17
### Added
- Add dependencies and functions for file downloading and zip extraction

## [0.1.0] - 2026-2-14
### Added
- Add project scaffolding with initial datasets and documentation

## [0.1.0] - 2026-2-13
### Added
- Initial commit