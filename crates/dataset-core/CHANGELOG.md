# Changelog — `dataset-core`

All notable changes to the `dataset-core` crate will be documented in this file.

This crate provides `Dataset<T>` plus the optional `utils` feature (download / unzip / temp dir / SHA-256 / `acquire_dataset`) and the `error` module.

Please view [SomeB1oody/dataset-core](https://github.com/SomeB1oody/dataset-core) for more info.

## [0.2.0] - 2026-5-27
### Changed
- Split the project into a Cargo workspace. `dataset-core` now contains only the architecture layer (`Dataset<T>`, `utils`, `error`). The built-in dataset loaders have moved to the new companion crate [`dataset-ml`](https://crates.io/crates/dataset-ml).
- Removed the `datasets` feature flag (the loaders that depended on it are in `dataset-ml`).

## [0.2.0] - 2026-4-14
### Changed
- Remove formatting of record data in error messages, simplify `DataFormatError` structures.

## [0.1.0] - 2026-4-11
### Changed
- Replace `downloader` with `ureq` for file downloads.
- Refactor `download_to` to provide optional file naming.
- Refactor `utils` module: improve formatting, rename functions, update docs and examples for clarity.

## [0.1.0] - 2026-4-10
### Added
- Integrate `thiserror` for structured error handling, and refactor error implementations accordingly.

## [0.1.0] - 2026-4-8
### Changed
- Migrate from `rustyml-dataset` to `dataset-core`, update import paths, refactor README and documentation.
- Update `zip` to v8.5.1.

## [0.1.0] - 2026-4-7
### Changed
- Add feature gating for `utils`; modularize optional dependencies.

## [0.1.0] - 2026-4-6
### Added
- Add tests for utility functions in the `utils` module.

## [0.1.0] - 2026-4-5
### Changed
- Separate utility functions into the `utils` module.

## [0.1.0] - 2026-4-4
### Changed
- Add usage examples to dataset utility functions.

## [0.1.0] - 2026-4-3
### Changed
- Update `zip` to v8.5.0.
- Put `String` initialization of `field` in error handling.

## [0.1.0] - 2026-4-2
### Changed
- Streamline `Dataset` loader methods and simplify the `Dataset` trait bound; combine downloading and parsing in a single method.

## [0.1.0] - 2026-4-1
### Added
- Introduce the `Dataset` struct for unified lazy loading and caching logic.

## [0.1.0] - 2026-3-31
### Changed
- Update `zip` to v8.4.0 and `sha2` to v0.11.0.

## [0.1.0] - 2026-3-30
### Changed
- Introduce `download_dataset_with` for streamlined, reusable download workflows.
- Make `expected_sha256` optional for flexibility in validation logic.
- Remove the specified prefix for temporary directory creation.

## [0.1.0] - 2026-3-24
### Changed
- Update `zip` to v8.3.1.
- Rename `storage_path` to `storage_dir` for improved clarity.

## [0.1.0] - 2026-3-22
### Changed
- Update `zip` to v8.3.0.

## [0.1.0] - 2026-3-21
### Changed
- Streamline error handling.

## [0.1.0] - 2026-3-20
### Added
- Add `Thread Safety` documentation.

## [0.1.0] - 2026-3-19
### Changed
- Ensure temporary directory cleanup on SHA-256 hash validation failure.

## [0.1.0] - 2026-3-17
### Changed
- Update Rust version to 1.88.0.

## [0.1.0] - 2026-3-12
### Changed
- Update `tempfile` to v3.27.0.

## [0.1.0] - 2026-3-10
### Changed
- Update crate documentation to reflect the struct-based API.

## [0.1.0] - 2026-3-3
### Added
- Add `prepare_download_dir` for unified download and validation logic.

## [0.1.0] - 2026-2-25
### Changed
- Update `tempfile` to v3.26.0.
- Update documentation to reflect automatic downloading, caching, and expanded feature set.

## [0.1.0] - 2026-2-21
### Changed
- Automatically create dataset directories if they do not exist.

## [0.1.0] - 2026-2-17
### Added
- Add dependencies and functions for file downloading and zip extraction.

## [0.1.0] - 2026-2-13
### Added
- Initial commit.
