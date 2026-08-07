# Changelog: `dataset-core`

This file documents all notable changes to the `dataset-core` crate.

This crate provides `Dataset<T, E>` and the `error` module. It also provides an optional `utils` feature with download, unzip, gunzip, untar, untar_gz, temp dir, SHA-256, Latin-1, and `acquire_dataset`.

See [SomeB1oody/dataset-core](https://github.com/SomeB1oody/dataset-core) for more information.

This changelog groups entries by release and lists only each version's notable changes. It omits routine dependency bumps, doc-only tweaks, and minor internal refactors.

## [0.5.0] - 2026-08-02
### Added
- `Dataset::load_mut(&mut self) -> Result<&mut T, E>`: the loading counterpart of `get_mut`. If needed, it loads the dataset. It then returns a mutable reference to the cached value.
- `sha256_file` and `verify_sha256` (feature `utils`, re-exported at the crate root): `sha256_file` computes a file's SHA-256 digest as 64 lowercase hex characters. This is useful for pinning a new dataset's hash. `verify_sha256` checks a file against an expected digest, and the check ignores case.
- `read_latin1` (feature `utils`, re-exported at the crate root): it reads a file as Latin-1 text. It never loses data and never substitutes `U+FFFD`. Use it for the raw-document corpora (20 Newsgroups, Movie Review Polarity).
- `download_to_with_retries` (feature `utils`, re-exported at the crate root): this works like `download_to`, but adds bounded retries with exponential backoff for flaky archive hosts. If you set `retries = 0`, it behaves the same as `download_to`.

### Fixed
- `Dataset::load` now runs the loader **at most once** under concurrent access. Previously, every thread that found the cache empty ran the loader. For a typical loader that downloads into `storage_dir`, this meant N concurrent downloads of the same file. The system discarded N-1 of the results. An internal mutex now serializes the first load, and later arrivals block and share its result. The already-loaded fast path still takes no lock, and the system does not cache a failing loader, so a later `load` retries.

### Testing
- Added `tests/dataset_test.rs` (12 tests). The suite includes a 16-thread regression test that confirms the loader runs exactly once, and a test that confirms a failed load stays uncached.

## [0.4.0] - 2026-07-17
### Added
- `untar` / `untar_gz` (feature `utils`): these extract a tar archive and a gzip-compressed tar (`.tar.gz` / `.tgz`) into a directory. `untar_gz` streams through gzip straight into tar, so no intermediate `.tar` file hits disk. This adds the pure-Rust `tar` dependency.
- `gunzip` (feature `utils`): it decompresses a `.gz` file into a single output. It streams the data, so the whole file never sits in memory at once. This adds the pure-Rust `flate2` dependency.

## [0.3.0] - 2026-06-01
### Changed
- **Breaking:** the container now stores the loader and takes it once, at construction. `Dataset<T>` becomes `Dataset<T, E>`, `new` takes the loader (`new(dir, loader)`), and `load()` no longer takes a loader argument.
- **Breaking:** `create_temp_dir` and `file_sha256_matches` are no longer part of the public API. Call `acquire_dataset` instead (it does temp-dir creation, SHA-256 verification, and the atomic rename).
- `download_to` now validates the URL and strips any query string and fragment before deriving the output filename.

### Added
- `set_loader` replaces the stored loader. `invalidate` drops the cached value. `into_inner` and `take` move the cached value out. `get` and `get_mut` borrow or edit the value in place, and neither call triggers a load.

## [0.2.0] - 2026-05-27
### Changed
- Split the project into a Cargo workspace. `dataset-core` now contains only the architecture layer (`Dataset`, `utils`, `error`). The built-in dataset loaders moved to the new companion crate [`dataset-ml`](https://crates.io/crates/dataset-ml). This release removed the `datasets` feature flag.
- Simplified the `DataFormatError` structures and removed record-data formatting from error messages.

## [0.1.0] - 2026-04-11
### Added
- Initial release: the `Dataset` struct provides unified lazy loading and caching. This struct came from the earlier `rustyml-dataset` project.
- `utils` module (feature `utils`): file download (via `ureq`), ZIP extraction, SHA-256 verification, and temporary-directory helpers, with a reusable download-and-validate workflow.
- Structured error handling via `thiserror` (the `error` module).
