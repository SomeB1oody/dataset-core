//! A generic, thread-safe dataset container with lazy loading and caching.
//!
//! `dataset-core` provides [`Dataset<T, E>`], a lightweight wrapper that pairs a storage
//! directory with a lazily-initialized value of any type `T`. The actual downloading
//! and parsing logic is supplied by the caller through a loader closure stored at
//! construction time, making `Dataset<T, E>` suitable for any data source — local
//! files, remote URLs, databases, or in-memory generation.
//!
//! On top of this core type, the crate offers an **optional** feature-gated module:
//!
//! - **`utils`** — helper functions for downloading files, extracting archives,
//!   verifying SHA-256 hashes, and managing temporary directories.
//!
//! Ready-to-use loaders for classic ML datasets (Iris, Boston Housing, Diabetes,
//! Titanic, Wine Quality) live in the companion crate
//! [`dataset-ml`](https://crates.io/crates/dataset-ml), which depends on
//! `dataset-core` with the `utils` feature enabled and serves as the reference
//! implementation for wrapping `Dataset<T, E>`.
//!
//! # Feature Flags
//!
//! | Feature | What it enables                                                            |
//! |---------|----------------------------------------------------------------------------|
//! | `utils` | `download_to`, `unzip`, `gunzip`, `acquire_dataset`, and the `error` module |
//!
//! With no features enabled, only `Dataset<T, E>` is available — depending only on
//! `std::sync::OnceLock`.
//!
//! # Quick Start — `Dataset<T, E>`
//!
//! ```rust
//! use dataset_core::Dataset;
//!
//! fn my_loader(dir: &str) -> Result<Vec<String>, std::io::Error> {
//!     // In a real use case you would read/download files from `dir`.
//!     Ok(vec!["hello".to_string(), "world".to_string()])
//! }
//!
//! // The loader is supplied once, at construction time.
//! let mut ds: Dataset<Vec<String>, std::io::Error> = Dataset::new("./my_data", my_loader);
//!
//! // First call runs the loader; subsequent calls return the cached reference.
//! let data = ds.load().unwrap();
//! assert_eq!(data.len(), 2);
//!
//! let data_again = ds.load().unwrap();
//! assert!(std::ptr::eq(data, data_again)); // same reference, no reload
//!
//! // `get` borrows the cached value without ever running the loader;
//! // `get_mut` edits it in place (no clone, no reload — the change stays cached).
//! assert!(ds.get().is_some());
//! if let Some(v) = ds.get_mut() {
//!     v[0] = "HELLO".to_string();
//! }
//! assert_eq!(ds.get().unwrap()[0], "HELLO");
//!
//! // Move the cached value out without cloning. `take` leaves `ds` reusable
//! // (a later `load` re-runs the loader); `into_inner` consumes `ds`.
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
//! Because the loader lives inside the `Dataset`, you change *how* the data is
//! parsed with [`Dataset::set_loader`], which also invalidates the cache so the
//! next access re-parses with the new loader. To re-run the **same** loader
//! (e.g. the file on disk changed), use [`Dataset::invalidate`].
//!
//! ```rust
//! use dataset_core::Dataset;
//!
//! let mut ds: Dataset<i32, std::convert::Infallible> = Dataset::new("./data", |_| Ok(1));
//! assert_eq!(*ds.load().unwrap(), 1);
//!
//! ds.set_loader(|_| Ok(2)); // swap the loader; old cache is dropped
//! assert!(!ds.is_loaded());
//! assert_eq!(*ds.load().unwrap(), 2); // next load uses the new loader
//! ```
//!
//! # Utility Functions (feature `utils`)
//!
//! - `download_to` — download a remote file into a directory
//! - `unzip` — extract a ZIP archive
//! - `gunzip` — decompress a gzip (`.gz`) file into a single output file
//! - `acquire_dataset` — cache-aware dataset acquisition workflow
//!   (temp dir → prepare → optional hash check → move to final location)
//!
//! `acquire_dataset` is the single entry point for caching a dataset file; temp-dir
//! creation and SHA-256 verification are internal steps it performs for you.

#[cfg(feature = "utils")]
pub use error::{DataFormatErrorKind, DatasetError};
use std::sync::OnceLock;
#[cfg(feature = "utils")]
pub use utils::{acquire_dataset, download_to, gunzip, unzip};

/// The boxed loader stored inside a [`Dataset`].
///
/// A loader takes the storage directory path and returns the parsed dataset (or
/// an error). It is stored behind a `Box<dyn Fn ...>` so the concrete closure
/// type does not leak into `Dataset`'s type parameters. The `Send + Sync` bound
/// keeps `Dataset<T, E>` shareable across threads (matching the guarantee given
/// by the internal `OnceLock`); the implied `'static` bound means the loader may
/// not borrow from its environment — capture by value or clone instead.
type Loader<T, E> = Box<dyn Fn(&str) -> Result<T, E> + Send + Sync>;

/// A generic, thread-safe dataset container with lazy loading and in-memory caching.
///
/// `Dataset<T, E>` is a thin caching wrapper that holds a `storage_dir` (the directory
/// where dataset files are stored on disk), a loader closure, and a lazily-initialized
/// value of type `T`. The downloading and parsing logic is provided by the caller
/// through the loader passed to [`Dataset::new`] and run on first access by
/// [`Dataset::load`].
///
/// This struct is designed to be the building block for both the built-in datasets
/// shipped with this crate and any custom datasets defined by external users.
///
/// # Type Parameters
///
/// - `T` - The type of the parsed dataset. Can be any type, such as
///   `(Array2<f64>, Array1<f64>)`, a custom struct, or any other data representation.
///   `T` must implement `Send + Sync` for `Dataset<T, E>` to be shared across threads.
/// - `E` - The error type returned by the loader. Callers choose it freely (e.g.
///   `std::io::Error`, a crate-specific `DatasetError`, or `std::convert::Infallible`
///   for loaders that cannot fail).
///
/// # Thread Safety
///
/// `Dataset<T, E>` is `Send + Sync` when `T` is `Send + Sync` (the stored loader is
/// always `Send + Sync`). The internal `OnceLock` ensures that the loader runs at
/// most once, even when multiple threads call [`Dataset::load`] concurrently.
///
/// # Example
///
/// ```rust
/// use dataset_core::Dataset;
///
/// // Define a simple loader that reads a value from the storage directory path.
/// // The loader can return any error type you choose.
/// fn my_loader(dir: &str) -> Result<Vec<String>, std::io::Error> {
///     // In a real use case, you would download/read files from `dir`.
///     // Here we just demonstrate the caching behavior.
///     Ok(vec!["hello".to_string(), "world".to_string()])
/// }
///
/// // The loader is bound to the dataset at construction time.
/// let mut dataset: Dataset<Vec<String>, std::io::Error> = Dataset::new("./my_data", my_loader);
///
/// // The first call to `load` triggers the loader
/// let data = dataset.load().unwrap();
/// assert_eq!(data.len(), 2);
///
/// // Subsequent calls return the cached reference instantly
/// let data_again = dataset.load().unwrap();
/// assert!(std::ptr::eq(data, data_again)); // same reference, no re-load
///
/// // Check whether data has been loaded
/// assert!(dataset.is_loaded());
///
/// // Borrow the cached value without reloading, or edit it in place via `get_mut`.
/// if let Some(v) = dataset.get_mut() {
///     v[0] = "HELLO".to_string();
/// }
/// assert_eq!(dataset.get().unwrap()[0], "HELLO");
///
/// // Move the cached value out without cloning.
/// // `take` leaves `dataset` reusable; `into_inner` consumes it.
/// let owned = dataset.take().unwrap();
/// assert_eq!(owned.len(), 2);
/// assert!(!dataset.is_loaded()); // `take` reset it to unloaded
///
/// dataset.load().unwrap(); // reloads, since `take` cleared the cache
/// let owned = dataset.into_inner().unwrap();
/// assert_eq!(owned.len(), 2);
/// ```
pub struct Dataset<T, E> {
    storage_dir: String,
    loader: Loader<T, E>,
    data: OnceLock<T>,
}

impl<T, E> Dataset<T, E> {
    /// Create a new `Dataset` instance without loading any data.
    ///
    /// This is a lightweight operation that only stores the storage directory path
    /// and the loader. No I/O or network requests are performed until
    /// [`Dataset::load`] is called.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory where dataset files will be stored. The directory
    ///   will be created automatically when the loader runs if it does not exist.
    /// - `loader` - A closure or function that takes the storage directory path (`&str`)
    ///   and returns `Result<T, E>`. This is where you perform downloading, file I/O,
    ///   and parsing. It runs at most once (see [`Dataset::load`]). Because it is
    ///   stored behind `Box<dyn Fn ...>`, it must be `Send + Sync + 'static` —
    ///   capture owned values or clones rather than borrowing from the environment.
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
        }
    }

    /// Load the dataset, executing the stored loader on first call and caching the result.
    ///
    /// On the first call, the loader supplied to [`Dataset::new`] (or last set via
    /// [`Dataset::set_loader`]) is invoked with the storage directory path. The
    /// returned value is cached internally. All subsequent calls — from any thread —
    /// return a reference to the cached value without running the loader again.
    ///
    /// # Returns
    ///
    /// - `Ok(&T)` - A reference to the cached dataset.
    ///
    /// # Errors
    ///
    /// Returns any error produced by the loader on first invocation. Once data is
    /// successfully loaded and cached, this method never returns an error.
    pub fn load(&self) -> Result<&T, E> {
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

    /// Replace the loader and invalidate any cached data.
    ///
    /// Use this when the parsing logic itself needs to change. The new loader is
    /// **not** run immediately: this method only swaps the loader and drops the
    /// cached value (resetting the `Dataset` to its unloaded state), so the next
    /// [`Dataset::load`] lazily re-parses with the new loader. This keeps the
    /// "no I/O until access" contract intact.
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
    /// ds.set_loader(|_| Ok(2)); // swap the loader; the old cache is dropped
    /// assert!(!ds.is_loaded());
    /// assert_eq!(*ds.load().unwrap(), 2); // next load uses the new loader
    /// ```
    pub fn set_loader(&mut self, loader: impl Fn(&str) -> Result<T, E> + Send + Sync + 'static) {
        self.loader = Box::new(loader);
        self.invalidate();
    }

    /// Drop the cached value, keeping the current loader.
    ///
    /// Resets the `Dataset` to its unloaded state so the next [`Dataset::load`]
    /// re-runs the **current** loader from scratch — useful when the underlying
    /// files have changed on disk and you want to re-parse them. To swap in a
    /// *different* loader, use [`Dataset::set_loader`].
    ///
    /// Unlike [`Dataset::take`], this does not hand the cached value back; it simply
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

    /// Check whether the dataset has been loaded into memory.
    ///
    /// # Returns
    ///
    /// `true` if [`Dataset::load`] has been called successfully at least once,
    /// `false` otherwise.
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
    /// Unlike [`Dataset::load`], this never runs the loader: if the dataset has
    /// not been loaded yet, it returns `None` rather than downloading/parsing.
    /// Use it when you only want the data if it is already in memory and want to
    /// avoid paying the loader's I/O cost otherwise — for example a fast path
    /// that falls back to other work when the dataset is not yet cached.
    ///
    /// This is the reference-returning companion of [`Dataset::is_loaded`]:
    /// `is_loaded()` answers *whether* the value is cached, `get()` hands you the
    /// cached reference when it is.
    ///
    /// # Returns
    ///
    /// - `Some(&T)` - a reference to the cached value, if the dataset had been loaded.
    /// - `None` - if the dataset has not been loaded.
    ///
    /// # Example
    ///
    /// ```rust
    /// use dataset_core::Dataset;
    ///
    /// let ds: Dataset<Vec<i32>, std::convert::Infallible> =
    ///     Dataset::new("./data", |_| Ok(vec![1, 2, 3]));
    /// assert!(ds.get().is_none()); // not loaded yet — no loader is run
    ///
    /// ds.load().unwrap();
    /// assert_eq!(ds.get(), Some(&vec![1, 2, 3]));
    /// ```
    pub fn get(&self) -> Option<&T> {
        self.data.get()
    }

    /// Get a mutable reference to the cached value for **in-place** editing.
    ///
    /// This is the only way to mutate the cached value without moving it out:
    /// you can tweak the loaded data (e.g. normalize features, fill in missing
    /// entries, augment samples) and the changes persist in the cache, so later
    /// [`Dataset::load`] / [`Dataset::get`] calls observe them.
    ///
    /// Because it requires unique access (`&mut self`), there is no aliasing or
    /// race concern. And unlike [`take`](Dataset::take) /
    /// [`into_inner`](Dataset::into_inner), it neither clones nor removes the
    /// value — the `Dataset` stays loaded.
    ///
    /// Like [`Dataset::get`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call [`Dataset::load`] first if
    /// you need to ensure the value is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut T)` - a mutable reference to the cached value, if the dataset
    ///   had been loaded.
    /// - `None` - if the dataset has not been loaded.
    ///
    /// # Example
    ///
    /// ```rust
    /// use dataset_core::Dataset;
    ///
    /// let mut ds: Dataset<Vec<i32>, std::convert::Infallible> =
    ///     Dataset::new("./data", |_| Ok(vec![1, 2, 3]));
    /// assert!(ds.get_mut().is_none()); // not loaded yet — no loader is run
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
    /// This **moves** the cached `T` out of the container — there is no clone.
    /// Because it takes `self` by value, the `Dataset` is consumed and cannot be
    /// used afterwards.
    ///
    /// This method does **not** trigger loading: it returns `None` if the dataset
    /// was never loaded. Call [`Dataset::load`] first if you need to ensure the
    /// value is present.
    ///
    /// # `into_inner` vs [`take`](Dataset::take)
    ///
    /// Both move the cached value out without cloning; the difference is what
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
    /// - `Some(T)` - the cached value, if the dataset had been loaded.
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
    /// // `ds` has been consumed and can no longer be used.
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
    /// This **moves** the cached `T` out — there is no clone — and resets the
    /// `Dataset` to its unloaded state. Unlike [`into_inner`](Dataset::into_inner),
    /// the container is left intact: it can be used again, and a later
    /// [`Dataset::load`] will run the loader from scratch.
    ///
    /// This method does **not** trigger loading: it returns `None` if the dataset
    /// was not loaded.
    ///
    /// # `take` vs [`into_inner`](Dataset::into_inner)
    ///
    /// Both move the cached value out without cloning; the difference is what
    /// happens to the container:
    ///
    /// - [`take`](Dataset::take) takes `&mut self` and keeps the `Dataset`
    ///   **reusable** (reset to unloaded) after extracting the value.
    /// - [`into_inner`](Dataset::into_inner) takes `self` and **consumes** the
    ///   container entirely.
    ///
    /// # Returns
    ///
    /// - `Some(T)` - the cached value, if the dataset had been loaded.
    /// - `None` - if the dataset was not loaded.
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
/// Provides structured error types for dataset loading operations including
/// download failures, validation errors, I/O errors, and detailed data format
/// errors with line numbers and contextual information for debugging.
#[cfg(feature = "utils")]
pub mod error;

/// Utility functions for dataset authors.
///
/// Provides helpers for downloading files, extracting archives, verifying
/// SHA256 hashes, and managing the dataset acquisition workflow.
#[cfg(feature = "utils")]
pub mod utils;
