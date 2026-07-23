# Changelog — `dataset-core`

All notable changes to the `dataset-core` crate will be documented in this file.

This crate provides `Dataset<T, E>` plus the optional `utils` feature (download / unzip / gunzip / untar / untar_gz / temp dir / SHA-256 / Latin-1 / `acquire_dataset`) and the `error` module.

Please view [SomeB1oody/dataset-core](https://github.com/SomeB1oody/dataset-core) for more info.

Entries are grouped by release and list only each version's notable changes; routine dependency bumps, doc-only tweaks, and minor internal refactors are omitted.

## [Unreleased]
### Added
- `Dataset::load_mut(&mut self) -> Result<&mut T, E>` — the loading counterpart of `get_mut`: loads the dataset if needed, then returns a mutable reference to the cached value.
- `sha256_file` and `verify_sha256` (feature `utils`, re-exported at the crate root) — compute a file's SHA-256 as 64 lowercase hex chars (for pinning a new dataset's hash) and check a file against an expected digest (case-insensitive).
- `read_latin1` (feature `utils`, re-exported at the crate root) — read a file as Latin-1, lossless and never substituting `U+FFFD`, for the raw-document corpora (20 Newsgroups, Movie Review Polarity).
- `download_to_with_retries` (feature `utils`, re-exported at the crate root) — `download_to` with bounded retries and exponential backoff for flaky archive hosts; `retries = 0` is equivalent to `download_to`.

### Fixed
- `Dataset::load` now genuinely runs the loader **at most once** under concurrent access. Previously every thread that found the cache empty ran the loader — for the typical download-into-`storage_dir` loader, that meant N concurrent downloads of the same file with N-1 results thrown away. An internal mutex now serializes the first load; later arrivals block and share its result. The already-loaded fast path still takes no lock, and a failing loader is still not cached (a later `load` retries).

### Testing
- Added `tests/dataset_test.rs` (12 tests), including a 16-thread regression test asserting the loader runs exactly once and a test that a failed load is not cached.

## [0.4.0] - 2026-07-17
### Added
- `untar` / `untar_gz` (feature `utils`) — extract a tar archive and a gzip-compressed tar (`.tar.gz` / `.tgz`) into a directory; `untar_gz` streams through gzip straight into tar, so no intermediate `.tar` hits disk. Adds the pure-Rust `tar` dependency.
- `gunzip` (feature `utils`) — decompress a `.gz` file into a single output, streaming so the whole file is never held in memory. Adds the pure-Rust `flate2` dependency.

## [0.3.0] - 2026-06-01
### Changed
- **Breaking:** the loader is now stored on the container and supplied once at construction — `Dataset<T>` becomes `Dataset<T, E>`, `new` takes the loader (`new(dir, loader)`), and `load()` no longer takes a loader argument.
- **Breaking:** `create_temp_dir` and `file_sha256_matches` are no longer part of the public API; call `acquire_dataset` (which does temp-dir creation, SHA-256 verification, and the atomic rename) instead.
- `download_to` now validates the URL and strips any query string and fragment before deriving the output filename.

### Added
- `set_loader`, `invalidate`, `into_inner` / `take`, and `get` / `get_mut` — replace the stored loader, drop the cached value, move the cached value out, or borrow/edit it in place without triggering a load.

## [0.2.0] - 2026-05-27
### Changed
- Split the project into a Cargo workspace: `dataset-core` now contains only the architecture layer (`Dataset`, `utils`, `error`); the built-in dataset loaders moved to the new companion crate [`dataset-ml`](https://crates.io/crates/dataset-ml), and the `datasets` feature flag was removed.
- Simplified the `DataFormatError` structures and removed record-data formatting from error messages.

## [0.1.0] - 2026-04-11
### Added
- Initial release: the `Dataset` struct for unified lazy loading and caching, migrated from the earlier `rustyml-dataset` project.
- `utils` module (feature `utils`): file download (via `ureq`), ZIP extraction, SHA-256 verification, and temporary-directory helpers, with a reusable download-and-validate workflow.
- Structured error handling via `thiserror` (the `error` module).
