# Changelog — `dataset-core`

All notable changes to the `dataset-core` crate will be documented in this file.

This crate provides `Dataset<T, E>` plus the optional `utils` feature (download / unzip / gunzip / untar / untar_gz / temp dir / SHA-256 / `acquire_dataset`) and the `error` module.

Please view [SomeB1oody/dataset-core](https://github.com/SomeB1oody/dataset-core) for more info.

## [Unreleased]
### Added
- `untar(file_path, extract_dir)` and `untar_gz(file_path, extract_dir)` in the `utils` module (feature `utils`): extract a tar archive, and a gzip-compressed tar (`.tar.gz` / `.tgz`) archive, into a directory. `untar` is the tar counterpart to `unzip`; `untar_gz` composes the gzip and tar layers in one streaming pass (bytes flow through `flate2::read::GzDecoder` straight into `tar::Archive`), so the intermediate uncompressed `.tar` is never written to disk — suitable for large `.tar.gz` datasets (e.g. 20 Newsgroups). Both are re-exported at the crate root (`dataset_core::untar`, `dataset_core::untar_gz`). This adds `tar` (pure-Rust) as an optional dependency enabled by the `utils` feature. Extraction failures surface as `DatasetError::IoError`, so no new error variant was introduced.
- `gunzip(file_path, output_path)` in the `utils` module (feature `utils`): decompresses a gzip (`.gz`) file into a single output file, streaming through `flate2::read::GzDecoder` so the whole file is never held in memory at once. It is the gzip counterpart to `unzip` and is re-exported at the crate root (`dataset_core::gunzip`). This adds `flate2` (pure-Rust `miniz_oxide` backend) as an optional dependency enabled by the `utils` feature. Decompression failures surface as `DatasetError::IoError`, so no new error variant was introduced.

## [0.3.0] - 2026-06-01
### Removed
- **Breaking:** `create_temp_dir` and `file_sha256_matches` are no longer part of the public API. In 0.2.x they were re-exported at the crate root (`dataset_core::create_temp_dir`, `dataset_core::file_sha256_matches`) and reachable through `dataset_core::utils::`; they are now private implementation details. The internal `evaluate_storage` helper was likewise folded away. Call `acquire_dataset` — which performs temp-dir creation, SHA-256 verification, and the atomic rename for you — instead of composing these helpers by hand.

### Changed
- `download_to` now validates the URL and strips any query string and fragment before deriving the output filename from the URL. An explicit `filename` argument is still used verbatim, and the public signature is unchanged.
- Raised the minimum `ureq` to 3.3.0, `thiserror` to 2.0.18, and `zip` to 8.6.0 (all within their existing major versions; `utils` feature only).

## [0.3.0] - 2026-05-29
### Changed
- **Breaking:** the loader is now stored on the container and supplied once at construction. `Dataset<T>` becomes `Dataset<T, E>` (the loader's error type `E` is now a type parameter), `new` takes the loader (`new(dir, loader)`), and `load()` no longer takes a loader argument. The stored loader is `Box<dyn Fn(&str) -> Result<T, E> + Send + Sync>`, so it must be `Send + Sync + 'static` (capture by value/clone, not by borrow). `Dataset<T, E>` remains `Send + Sync` whenever `T` is.

### Added
- `Dataset::set_loader(&mut self, loader)` to replace the stored loader and invalidate the cache, so the next `load` lazily re-parses with the new loader (no immediate I/O).
- `Dataset::invalidate(&mut self)` to drop the cached value while keeping the current loader, so the next `load` re-runs it (e.g. after the underlying files change on disk).
- `Dataset::into_inner(self) -> Option<T>` and `Dataset::take(&mut self) -> Option<T>` for moving the cached value out of a `Dataset` without cloning. `into_inner` consumes the container; `take` leaves it reusable, resetting it to the unloaded state so a later `load` re-runs the loader. Both return `None` if the dataset was never loaded, and neither triggers loading.
- `Dataset::get(&self) -> Option<&T>` and `Dataset::get_mut(&mut self) -> Option<&mut T>` for accessing the cached value without triggering loading. `get` borrows it (the reference-returning companion of `is_loaded`); `get_mut` allows editing it in place — no clone, no reload, and the change persists in the cache. Both return `None` if the dataset was never loaded.

## [0.2.0] - 2026-05-27
### Changed
- Split the project into a Cargo workspace. `dataset-core` now contains only the architecture layer (`Dataset<T>`, `utils`, `error`). The built-in dataset loaders have moved to the new companion crate [`dataset-ml`](https://crates.io/crates/dataset-ml).
- Removed the `datasets` feature flag (the loaders that depended on it are in `dataset-ml`).

## [0.2.0] - 2026-04-14
### Changed
- Remove formatting of record data in error messages, simplify `DataFormatError` structures.

## [0.1.0] - 2026-04-11
### Changed
- Replace `downloader` with `ureq` for file downloads.
- Refactor `download_to` to provide optional file naming.
- Refactor `utils` module: improve formatting, rename functions, update docs and examples for clarity.

## [0.1.0] - 2026-04-10
### Added
- Integrate `thiserror` for structured error handling, and refactor error implementations accordingly.

## [0.1.0] - 2026-04-08
### Changed
- Migrate from `rustyml-dataset` to `dataset-core`, update import paths, refactor README and documentation.
- Update `zip` to v8.5.1.

## [0.1.0] - 2026-04-07
### Changed
- Add feature gating for `utils`; modularize optional dependencies.

## [0.1.0] - 2026-04-06
### Added
- Add tests for utility functions in the `utils` module.

## [0.1.0] - 2026-04-05
### Changed
- Separate utility functions into the `utils` module.

## [0.1.0] - 2026-04-04
### Changed
- Add usage examples to dataset utility functions.

## [0.1.0] - 2026-04-03
### Changed
- Update `zip` to v8.5.0.
- Put `String` initialization of `field` in error handling.

## [0.1.0] - 2026-04-02
### Changed
- Streamline `Dataset` loader methods and simplify the `Dataset` trait bound; combine downloading and parsing in a single method.

## [0.1.0] - 2026-04-01
### Added
- Introduce the `Dataset` struct for unified lazy loading and caching logic.

## [0.1.0] - 2026-03-31
### Changed
- Update `zip` to v8.4.0 and `sha2` to v0.11.0.

## [0.1.0] - 2026-03-30
### Changed
- Introduce `download_dataset_with` for streamlined, reusable download workflows.
- Make `expected_sha256` optional for flexibility in validation logic.
- Remove the specified prefix for temporary directory creation.

## [0.1.0] - 2026-03-24
### Changed
- Update `zip` to v8.3.1.
- Rename `storage_path` to `storage_dir` for improved clarity.

## [0.1.0] - 2026-03-22
### Changed
- Update `zip` to v8.3.0.

## [0.1.0] - 2026-03-21
### Changed
- Streamline error handling.

## [0.1.0] - 2026-03-20
### Added
- Add `Thread Safety` documentation.

## [0.1.0] - 2026-03-19
### Changed
- Ensure temporary directory cleanup on SHA-256 hash validation failure.

## [0.1.0] - 2026-03-17
### Changed
- Update Rust version to 1.88.0.

## [0.1.0] - 2026-03-12
### Changed
- Update `tempfile` to v3.27.0.

## [0.1.0] - 2026-03-10
### Changed
- Update crate documentation to reflect the struct-based API.

## [0.1.0] - 2026-03-03
### Added
- Add `prepare_download_dir` for unified download and validation logic.

## [0.1.0] - 2026-02-25
### Changed
- Update `tempfile` to v3.26.0.
- Update documentation to reflect automatic downloading, caching, and expanded feature set.

## [0.1.0] - 2026-02-21
### Changed
- Automatically create dataset directories if they do not exist.

## [0.1.0] - 2026-02-17
### Added
- Add dependencies and functions for file downloading and zip extraction.

## [0.1.0] - 2026-02-13
### Added
- Initial commit.
