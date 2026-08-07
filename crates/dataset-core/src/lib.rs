//! A generic, thread-safe dataset container with lazy loading and caching.
//!
//! `dataset-core` provides [`Dataset<T, E>`], a lightweight wrapper that pairs a storage
//! directory with a lazily-initialized value of any type `T`. The caller supplies the
//! download and parse logic through a loader closure stored at construction time. This
//! makes `Dataset<T, E>` suitable for any data source: local files, remote URLs,
//! databases, or in-memory generation.
//!
//! On top of this core type, the crate offers an **optional** feature-gated module:
//!
//! - **`utils`**: helper functions to download files, extract archives, verify
//!   SHA-256 hashes, and manage temporary directories.
//!
//! Ready-to-use loaders for 26 classic ML datasets live in the companion crate
//! [`dataset-ml`](https://crates.io/crates/dataset-ml). Examples include Iris, Breast
//! Cancer, Titanic, Forest CoverType, KDD Cup '99, and 20 Newsgroups. The crate depends
//! on `dataset-core` with the `utils` feature enabled and serves as the reference
//! implementation that wraps `Dataset<T, E>`.
//!
//! # Feature Flags
//!
//! | Feature | What it enables                                                                            |
//! |---------|--------------------------------------------------------------------------------------------|
//! | `utils` | `download_to`, `download_to_with_retries`, `unzip`, `gunzip`, `untar`, `untar_gz`, `sha256_file`, `verify_sha256`, `read_latin1`, `acquire_dataset`, and the `error` module |
//!
//! With no features enabled, only `Dataset<T, E>` is available. It depends only on
//! `std::sync::OnceLock`.
//!
//! # Quick Start: `Dataset<T, E>`
//!
//! ```rust
//! use dataset_core::Dataset;
//!
//! fn my_loader(dir: &str) -> Result<Vec<String>, std::io::Error> {
//!     // A real use case downloads or reads files from `dir`.
//!     Ok(vec!["hello".to_string(), "world".to_string()])
//! }
//!
//! // The caller supplies the loader once, at construction time.
//! let mut ds: Dataset<Vec<String>, std::io::Error> = Dataset::new("./my_data", my_loader);
//!
//! // The first call runs the loader. Later calls return the cached reference.
//! let data = ds.load().unwrap();
//! assert_eq!(data.len(), 2);
//!
//! let data_again = ds.load().unwrap();
//! assert!(std::ptr::eq(data, data_again)); // same reference, no reload
//!
//! // `get` borrows the cached value. It does not run the loader.
//! // `get_mut` edits the value in place, with no clone or reload. The change stays cached.
//! assert!(ds.get().is_some());
//! if let Some(v) = ds.get_mut() {
//!     v[0] = "HELLO".to_string();
//! }
//! assert_eq!(ds.get().unwrap()[0], "HELLO");
//!
//! // Move the cached value out without cloning. `take` leaves `ds` reusable.
//! // A later `load` call re-runs the loader. `into_inner` consumes `ds`.
//! let owned = ds.take().unwrap();
//! assert_eq!(owned.len(), 2);
//! assert!(!ds.is_loaded());
//!
//! ds.load().unwrap(); // `take` reset the cache, so this reloads
//! let owned = ds.into_inner().unwrap();
//! assert_eq!(owned.len(), 2);
//! ```
//!
//! # Swapping the loader
//!
//! Because the loader lives inside the `Dataset`, [`Dataset::set_loader`] lets you
//! change *how* the loader parses the data. It also invalidates the cache, so the
//! next access re-parses with the new loader. If the file on disk changes, use
//! [`Dataset::invalidate`] to re-run the **same** loader.
//!
//! ```rust
//! use dataset_core::Dataset;
//!
//! let mut ds: Dataset<i32, std::convert::Infallible> = Dataset::new("./data", |_| Ok(1));
//! assert_eq!(*ds.load().unwrap(), 1);
//!
//! ds.set_loader(|_| Ok(2)); // swap the loader and drop the old cache
//! assert!(!ds.is_loaded());
//! assert_eq!(*ds.load().unwrap(), 2); // next load uses the new loader
//! ```
//!
//! # Utility Functions (feature `utils`)
//!
//! - `download_to` - download a remote file into a directory
//! - `download_to_with_retries` - same as `download_to`, but retries transient failures
//!   with backoff
//! - `unzip` - extract a ZIP archive
//! - `gunzip` - decompress a gzip (`.gz`) file into a single output file
//! - `untar` - extract a tar (`.tar`) archive into a directory
//! - `untar_gz` - extract a gzip-compressed tar (`.tar.gz` / `.tgz`) archive as a stream
//! - `sha256_file` - compute a file's SHA-256 digest, to pin as an expected hash
//! - `verify_sha256` - check a file against a hash you already have
//! - `read_latin1` - read a file as Latin-1 text, with no data loss and no failure on
//!   non-UTF-8 bytes
//! - `acquire_dataset` - cache-aware dataset acquisition workflow
//!   (temp dir → prepare → optional hash check → move to final location)
//!
//! `acquire_dataset` is the single entry point for caching a dataset file. It creates
//! a temporary directory and verifies the SHA-256 hash internally. Use `sha256_file`
//! and `verify_sha256` only outside that workflow. Use them to pin a new dataset's
//! hash, or to check which file is on disk after a test runs.

#[cfg(feature = "utils")]
pub use error::{DataFormatErrorKind, DatasetError};
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "utils")]
pub use utils::{
    acquire_dataset, download_to, download_to_with_retries, gunzip, read_latin1, sha256_file,
    untar, untar_gz, unzip, verify_sha256,
};

/// The boxed loader stored inside a [`Dataset`].
///
/// A loader takes the storage directory path and returns the parsed dataset, or an
/// error. `Dataset` stores it behind a `Box<dyn Fn ...>`, so the concrete closure
/// type does not leak into `Dataset`'s type parameters. The `Send + Sync` bound keeps
/// `Dataset<T, E>` shareable across threads. The implied `'static` bound means the
/// loader must not borrow from its environment: it must capture by value or clone.
type Loader<T, E> = Box<dyn Fn(&str) -> Result<T, E> + Send + Sync>;

/// A generic, thread-safe dataset container with lazy loading and in-memory caching.
///
/// `Dataset<T, E>` is a thin caching wrapper. It holds a `storage_dir`, where the
/// loader stores dataset files, a loader closure, and a lazily-initialized value of
/// type `T`. The caller supplies the download and parse logic through the loader
/// passed to [`Dataset::new`]. [`Dataset::load`] runs this loader on first access.
///
/// This struct serves as the building block for the loaders in the companion crate
/// [`dataset-ml`](https://crates.io/crates/dataset-ml) and for custom datasets that
/// external users define.
///
/// # Type Parameters
///
/// - `T` - The type of the parsed dataset. It can be any type, such as
///   `(Array2<f64>, Array1<f64>)`, a custom struct, or another data shape.
///   `T` must implement `Send + Sync` so that threads can share `Dataset<T, E>`.
/// - `E` - The error type that the loader returns. Callers choose it freely, for
///   example `std::io::Error`, a crate-specific `DatasetError`, or
///   `std::convert::Infallible` for loaders that cannot fail.
///
/// # Thread Safety
///
/// `Dataset<T, E>` is `Send + Sync` when `T` is `Send + Sync` (the stored loader is
/// always `Send + Sync`). The loader runs at most once even when multiple threads
/// call [`Dataset::load`] concurrently. An internal mutex serializes the first load.
/// Late arrivals wait for it, then share its result. Each thread does not start its
/// own download.
///
/// # Example
///
/// ```rust
/// use dataset_core::Dataset;
///
/// // Define a simple loader that reads a value from the storage directory path.
/// // The loader can return any error type you choose.
/// fn my_loader(dir: &str) -> Result<Vec<String>, std::io::Error> {
///     // A real use case downloads or reads files from `dir`.
///     // This shows the caching behavior.
///     Ok(vec!["hello".to_string(), "world".to_string()])
/// }
///
/// // The caller supplies the loader once, at construction time.
/// let mut dataset: Dataset<Vec<String>, std::io::Error> = Dataset::new("./my_data", my_loader);
///
/// // The first call to `load` triggers the loader.
/// let data = dataset.load().unwrap();
/// assert_eq!(data.len(), 2);
///
/// // Later calls return the cached reference.
/// let data_again = dataset.load().unwrap();
/// assert!(std::ptr::eq(data, data_again)); // same reference, no reload
///
/// // Check whether the data is loaded.
/// assert!(dataset.is_loaded());
///
/// // `get_mut` edits the cached value in place, with no reload.
/// if let Some(v) = dataset.get_mut() {
///     v[0] = "HELLO".to_string();
/// }
/// assert_eq!(dataset.get().unwrap()[0], "HELLO");
///
/// // Move the cached value out without cloning.
/// // `take` leaves `dataset` reusable. `into_inner` consumes it.
/// let owned = dataset.take().unwrap();
/// assert_eq!(owned.len(), 2);
/// assert!(!dataset.is_loaded()); // `take` resets it to unloaded
///
/// dataset.load().unwrap(); // this reloads, because `take` cleared the cache
/// let owned = dataset.into_inner().unwrap();
/// assert_eq!(owned.len(), 2);
/// ```
pub struct Dataset<T, E> {
    storage_dir: String,
    loader: Loader<T, E>,
    data: OnceLock<T>,
    /// Serializes the loader so that concurrent [`Dataset::load`] calls run it
    /// **once** rather than racing to produce a value only one of them keeps.
    ///
    /// The mutex guards no data of its own. It exists only to make the check-run-store
    /// sequence in `load` atomic. So `load` recovers from a poisoned lock (caused by a
    /// loader that panicked on another thread) rather than propagating it.
    init_lock: Mutex<()>,
}

impl<T, E> Dataset<T, E> {
    /// Create a new `Dataset` instance without loading any data.
    ///
    /// This is a lightweight operation that only stores the storage directory path
    /// and the loader. It performs no I/O or network requests until [`Dataset::load`]
    /// runs.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - The directory where the loader stores dataset files. If the
    ///   directory does not yet exist, the loader creates it automatically when it
    ///   runs.
    /// - `loader` - A closure or function that takes the storage directory path (`&str`)
    ///   and returns `Result<T, E>`. This is where you download data, handle file I/O,
    ///   and parse it. It runs at most once (see [`Dataset::load`]). `Dataset::new`
    ///   stores it behind `Box<dyn Fn ...>`, so it must be `Send + Sync + 'static`.
    ///   Capture owned values or clones instead of borrowing from the environment.
    ///
    /// # Returns
    ///
    /// A new `Dataset<T, E>` instance ready for lazy loading.
    pub fn new(
        storage_dir: &str,
        loader: impl Fn(&str) -> Result<T, E> + Send + Sync + 'static,
    ) -> Self {
        Dataset {
            storage_dir: storage_dir.to_string(),
            loader: Box::new(loader),
            data: OnceLock::new(),
            init_lock: Mutex::new(()),
        }
    }

    /// Load the dataset. The first call runs the stored loader and caches the result.
    ///
    /// On the first call, `load` runs the loader supplied to [`Dataset::new`] (or last
    /// set via [`Dataset::set_loader`]) with the storage directory path. It caches the
    /// returned value. Later calls, from any thread, return a reference to the cached
    /// value without running the loader again.
    ///
    /// # Concurrency
    ///
    /// The loader runs **at most once**, even when several threads call `load`
    /// simultaneously. Threads that arrive while a load is in flight block until it
    /// finishes, then share its result. This matters for the typical loader, which
    /// downloads into `storage_dir`: concurrent callers would otherwise each start
    /// their own download of the same file.
    ///
    /// A loader that returns `Err` leaves the `Dataset` unloaded, so a later `load`
    /// retries it. `load` does not cache the error.
    ///
    /// # Returns
    ///
    /// - `Ok(&T)` - A reference to the cached dataset.
    ///
    /// # Errors
    ///
    /// Returns any error the loader produces on the first call. After the first
    /// successful load, this method never returns an error.
    pub fn load(&self) -> Result<&T, E> {
        // Fast path: already loaded, no locking needed.
        if let Some(data) = self.data.get() {
            return Ok(data);
        }

        // A poisoned lock only means that some other thread's loader panicked. The
        // guard protects no invariant of its own, so recover from the poison and
        // continue.
        let _guard = self.init_lock.lock().unwrap_or_else(|e| e.into_inner());

        // Check again: another thread may have loaded while this one waited for the lock.
        if let Some(data) = self.data.get() {
            return Ok(data);
        }

        let value = (self.loader)(&self.storage_dir)?;
        let _ = self.data.set(value);

        Ok(self
            .data
            .get()
            .expect("data should be set after successful load"))
    }

    /// Load the dataset if needed, then return a **mutable** reference to it.
    ///
    /// This is the loading counterpart of [`Dataset::get_mut`]. Unlike `get_mut`,
    /// which returns `None` when nothing is cached yet, `load_mut` runs the loader
    /// first. It always returns a mutable reference on success. Use it to load the
    /// data and adjust it in one step. For example, normalize features right after
    /// parsing, rather than calling [`Dataset::load`] and [`Dataset::get_mut`]
    /// separately.
    ///
    /// As with `get_mut`, you edit the value in place, and the change persists in
    /// the cache.
    ///
    /// # Returns
    ///
    /// - `Ok(&mut T)` - A mutable reference to the cached dataset.
    ///
    /// # Errors
    ///
    /// Returns any error the loader produces on the first call.
    ///
    /// # Example
    ///
    /// ```rust
    /// use dataset_core::Dataset;
    ///
    /// let mut ds: Dataset<Vec<i32>, std::convert::Infallible> =
    ///     Dataset::new("./data", |_| Ok(vec![1, 2, 3]));
    ///
    /// // Loads on first call, then returns a mutable reference.
    /// ds.load_mut().unwrap().push(4);
    /// assert_eq!(ds.get(), Some(&vec![1, 2, 3, 4])); // the change persisted
    /// ```
    pub fn load_mut(&mut self) -> Result<&mut T, E> {
        // Make sure the value is present. The shared borrow ends with this statement.
        self.load()?;

        Ok(self
            .data
            .get_mut()
            .expect("data should be set after successful load"))
    }

    /// Replace the loader and invalidate any cached data.
    ///
    /// Use this when the parsing logic itself needs to change. `set_loader` does not
    /// run the new loader right away. It only swaps the loader and drops the cached
    /// value, which resets the `Dataset` to its unloaded state. The next
    /// [`Dataset::load`] call then re-parses the data with the new loader. This keeps
    /// the "no I/O until access" contract intact.
    ///
    /// To re-run the *same* loader instead, use [`Dataset::invalidate`].
    ///
    /// # Parameters
    ///
    /// - `loader` - The replacement loader. Like the one given to [`Dataset::new`],
    ///   it must be `Send + Sync + 'static`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use dataset_core::Dataset;
    ///
    /// let mut ds: Dataset<i32, std::convert::Infallible> = Dataset::new("./data", |_| Ok(1));
    /// assert_eq!(*ds.load().unwrap(), 1);
    ///
    /// ds.set_loader(|_| Ok(2)); // swaps the loader and drops the old cache
    /// assert!(!ds.is_loaded());
    /// assert_eq!(*ds.load().unwrap(), 2); // next load uses the new loader
    /// ```
    pub fn set_loader(&mut self, loader: impl Fn(&str) -> Result<T, E> + Send + Sync + 'static) {
        self.loader = Box::new(loader);
        self.invalidate();
    }

    /// Drop the cached value. This keeps the current loader.
    ///
    /// This resets the `Dataset` to its unloaded state, so the next [`Dataset::load`]
    /// call re-runs the **current** loader from scratch. If the underlying files
    /// change on disk and you want to re-parse them, call this method. To swap in a
    /// *different* loader, use [`Dataset::set_loader`].
    ///
    /// Unlike [`Dataset::take`], this does not return the cached value. It simply
    /// discards it.
    ///
    /// # Example
    ///
    /// ```rust
    /// use dataset_core::Dataset;
    ///
    /// let mut ds: Dataset<i32, std::convert::Infallible> = Dataset::new("./data", |_| Ok(1));
    /// ds.load().unwrap();
    /// assert!(ds.is_loaded());
    ///
    /// ds.invalidate(); // drop the cache, keep the loader
    /// assert!(!ds.is_loaded());
    /// assert_eq!(*ds.load().unwrap(), 1); // reloads with the same loader
    /// ```
    pub fn invalidate(&mut self) {
        let _ = self.data.take();
    }

    /// Check whether the dataset is loaded into memory.
    ///
    /// # Returns
    ///
    /// `true` after a successful call to [`Dataset::load`], `false` otherwise.
    pub fn is_loaded(&self) -> bool {
        self.data.get().is_some()
    }

    /// Get the storage directory path.
    ///
    /// # Returns
    ///
    /// The storage directory path as a string slice.
    pub fn storage_dir(&self) -> &str {
        &self.storage_dir
    }

    /// Get a reference to the cached value **without** triggering loading.
    ///
    /// Unlike [`Dataset::load`], this never runs the loader. If the dataset is not
    /// loaded yet, it returns `None` instead of downloading or parsing anything. When
    /// you want data only if it is already in memory, use `get`. This avoids the
    /// loader's I/O cost when the data is not cached. For example, a fast path can
    /// fall back to other work when the dataset is not yet cached.
    ///
    /// This is the reference-returning companion of [`Dataset::is_loaded`]:
    /// `is_loaded()` answers *whether* the value is cached, and `get()` returns the
    /// cached reference when it is.
    ///
    /// # Returns
    ///
    /// - `Some(&T)` - a reference to the cached value, if the dataset is loaded.
    /// - `None` - if the dataset is not loaded.
    ///
    /// # Example
    ///
    /// ```rust
    /// use dataset_core::Dataset;
    ///
    /// let ds: Dataset<Vec<i32>, std::convert::Infallible> =
    ///     Dataset::new("./data", |_| Ok(vec![1, 2, 3]));
    /// assert!(ds.get().is_none()); // not loaded yet, no loader runs
    ///
    /// ds.load().unwrap();
    /// assert_eq!(ds.get(), Some(&vec![1, 2, 3]));
    /// ```
    pub fn get(&self) -> Option<&T> {
        self.data.get()
    }

    /// Get a mutable reference to the cached value for **in-place** editing.
    ///
    /// This is the only way to mutate the cached value without moving it out. You can
    /// tweak the loaded data, for example to normalize features, add missing values,
    /// or augment samples. The changes persist in the cache, so later [`Dataset::load`]
    /// and [`Dataset::get`] calls observe them.
    ///
    /// Because it needs unique access (`&mut self`), there is no risk of aliasing or a
    /// race. Unlike both [`take`](Dataset::take) and [`into_inner`](Dataset::into_inner),
    /// it neither clones nor removes the value. The `Dataset` stays loaded.
    ///
    /// Like [`Dataset::get`], this does **not** trigger loading. It returns `None` if
    /// the dataset is not loaded. If you need the value to be present, call
    /// [`Dataset::load`] first.
    ///
    /// # Returns
    ///
    /// - `Some(&mut T)` - a mutable reference to the cached value, if the dataset is
    ///   loaded.
    /// - `None` - if the dataset is not loaded.
    ///
    /// # Example
    ///
    /// ```rust
    /// use dataset_core::Dataset;
    ///
    /// let mut ds: Dataset<Vec<i32>, std::convert::Infallible> =
    ///     Dataset::new("./data", |_| Ok(vec![1, 2, 3]));
    /// assert!(ds.get_mut().is_none()); // not loaded yet, no loader runs
    ///
    /// ds.load().unwrap();
    /// if let Some(data) = ds.get_mut() {
    ///     data.push(4); // edit the cached value in place, no clone, no reload
    /// }
    /// assert_eq!(ds.get(), Some(&vec![1, 2, 3, 4])); // the change persisted
    /// ```
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.data.get_mut()
    }

    /// Consume the `Dataset` and return the cached value, if any.
    ///
    /// This **moves** the cached `T` out of the container. There is no clone.
    /// Because it takes `self` by value, this consumes the `Dataset`. You cannot use
    /// it afterward.
    ///
    /// This method does **not** trigger loading. It returns `None` if the dataset was
    /// never loaded. If you need the value to be present, call [`Dataset::load`]
    /// first.
    ///
    /// # `into_inner` vs [`take`](Dataset::take)
    ///
    /// Both move the cached value out without cloning. The difference is what
    /// happens to the container:
    ///
    /// - [`into_inner`](Dataset::into_inner) takes `self` and **consumes** the
    ///   `Dataset`. Use it when you are done with the container.
    /// - [`take`](Dataset::take) takes `&mut self`, leaving the `Dataset`
    ///   **reusable** in its unloaded state (a later [`load`](Dataset::load)
    ///   re-runs the loader).
    ///
    /// # Returns
    ///
    /// - `Some(T)` - the cached value, if the dataset is loaded.
    /// - `None` - if the dataset was never loaded.
    ///
    /// # Example
    ///
    /// ```rust
    /// use dataset_core::Dataset;
    ///
    /// let ds: Dataset<Vec<i32>, std::convert::Infallible> =
    ///     Dataset::new("./data", |_| Ok(vec![1, 2, 3]));
    /// ds.load().unwrap();
    ///
    /// let owned: Vec<i32> = ds.into_inner().unwrap();
    /// assert_eq!(owned, vec![1, 2, 3]);
    /// // `into_inner` consumed `ds`. You can no longer use it.
    ///
    /// // A dataset that was never loaded yields `None`.
    /// let empty: Dataset<Vec<i32>, std::convert::Infallible> =
    ///     Dataset::new("./data", |_| Ok(vec![1, 2, 3]));
    /// assert!(empty.into_inner().is_none());
    /// ```
    #[must_use = "this consumes the Dataset; discarding the returned value drops the loaded data"]
    pub fn into_inner(self) -> Option<T> {
        self.data.into_inner()
    }

    /// Take the cached value out of the `Dataset`, leaving it reusable.
    ///
    /// This **moves** the cached `T` out. There is no clone. It also resets the
    /// `Dataset` to its unloaded state. Unlike [`into_inner`](Dataset::into_inner),
    /// `take` leaves the container intact. You can use it again, and a later
    /// [`Dataset::load`] call runs the loader from scratch.
    ///
    /// This method does **not** trigger loading. It returns `None` if the dataset is
    /// not loaded.
    ///
    /// # `take` vs [`into_inner`](Dataset::into_inner)
    ///
    /// Both move the cached value out without cloning. The difference is what
    /// happens to the container:
    ///
    /// - [`take`](Dataset::take) takes `&mut self` and keeps the `Dataset`
    ///   **reusable** (reset to unloaded) after extracting the value.
    /// - [`into_inner`](Dataset::into_inner) takes `self` and **consumes** the
    ///   container entirely.
    ///
    /// # Returns
    ///
    /// - `Some(T)` - the cached value, if the dataset is loaded.
    /// - `None` - if the dataset is not loaded.
    ///
    /// # Example
    ///
    /// ```rust
    /// use dataset_core::Dataset;
    ///
    /// let mut ds: Dataset<i32, std::convert::Infallible> = Dataset::new("./data", |_| Ok(1));
    /// ds.load().unwrap();
    /// assert!(ds.is_loaded());
    ///
    /// let taken = ds.take().unwrap();
    /// assert_eq!(taken, 1);
    /// assert!(!ds.is_loaded()); // reset to unloaded, but `ds` is still usable
    ///
    /// // Because it was reset, `load` runs the loader again:
    /// let reloaded = ds.load().unwrap();
    /// assert_eq!(*reloaded, 1);
    /// ```
    #[must_use = "discarding the returned value drops the data taken out of the Dataset"]
    pub fn take(&mut self) -> Option<T> {
        self.data.take()
    }
}

impl<T, E> std::fmt::Debug for Dataset<T, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dataset")
            .field("storage_dir", &self.storage_dir)
            .field("data_loaded", &self.is_loaded())
            .finish()
    }
}

/// Error handling module.
///
/// This module provides structured error types for dataset loading operations, such
/// as download failures, validation errors, and I/O errors. It also provides detailed
/// data format errors with line numbers and context for debugging.
#[cfg(feature = "utils")]
pub mod error;

/// Utility functions for dataset authors.
///
/// Provides helpers to download files, extract archives, verify SHA-256 hashes, and
/// manage the dataset acquisition workflow.
#[cfg(feature = "utils")]
pub mod utils;
