//! A generic, thread-safe dataset container with lazy loading and caching.
//!
//! `dataset-core` provides [`Dataset<T>`], a lightweight wrapper that pairs a storage
//! directory with a lazily-initialized value of any type `T`. The actual downloading
//! and parsing logic is supplied by the caller through a loader closure, making
//! `Dataset<T>` suitable for any data source — local files, remote URLs, databases,
//! or in-memory generation.
//!
//! On top of this core type, the crate offers **optional** feature-gated modules:
//!
//! - **`utils`** — helper functions for downloading files, extracting archives,
//!   verifying SHA-256 hashes, and managing temporary directories.
//! - **`datasets`** — ready-to-use loaders for classic ML datasets (Iris, Boston
//!   Housing, Diabetes, Titanic, Wine Quality). These also serve as reference
//!   implementations showing how to wrap `Dataset<T>` for a concrete use case.
//!
//! # Feature Flags
//!
//! | Feature    | What it enables                                                  |
//! |------------|------------------------------------------------------------------|
//! | `utils`    | [`download_to`], [`unzip`], [`create_temp_dir`], [`file_sha256_matches`], [`download_dataset_with`], and the [`error`] module |
//! | `datasets` | All built-in dataset loaders (implies `utils`)                   |
//!
//! With no features enabled, only `Dataset<T>` is available — only depend on `std::sync::OnceLock`.
//!
//! # Quick Start — `Dataset<T>`
//!
//! ```rust
//! use dataset_core::Dataset;
//!
//! fn my_loader(dir: &str) -> Result<Vec<String>, std::io::Error> {
//!     // In a real use case you would read/download files from `dir`.
//!     Ok(vec!["hello".to_string(), "world".to_string()])
//! }
//!
//! let ds: Dataset<Vec<String>> = Dataset::new("./my_data");
//!
//! // First call runs the loader; subsequent calls return the cached reference.
//! let data = ds.load(my_loader).unwrap();
//! assert_eq!(data.len(), 2);
//!
//! let data_again = ds.load(my_loader).unwrap();
//! assert!(std::ptr::eq(data, data_again)); // same reference, no reload
//! ```
//!
//! # Built-in Datasets (feature `datasets`)
//!
//! | Dataset              | Samples | Features | Task Type      |
//! |----------------------|---------|----------|----------------|
//! | Iris                 | 150     | 4        | Classification |
//! | Boston Housing       | 506     | 13       | Regression     |
//! | Diabetes             | 768     | 8        | Classification |
//! | Titanic              | 891     | 11       | Classification |
//! | Wine Quality (Red)   | 1,599   | 11       | Regression     |
//! | Wine Quality (White) | 4,898   | 11       | Regression     |
//!
//! ```rust,ignore
//! use dataset_core::datasets::iris::Iris;
//!
//! let iris = Iris::new("./data");
//! let (features, labels) = iris.data().unwrap();
//! assert_eq!(features.shape(), &[150, 4]);
//! ```
//!
//! # Utility Functions (feature `utils`)
//!
//! - [`download_to`] — download a remote file into a directory
//! - [`unzip`] — extract a ZIP archive
//! - [`create_temp_dir`] — create a self-cleaning temporary directory
//! - [`file_sha256_matches`] — verify a file's SHA-256 hash
//! - [`download_dataset_with`] — end-to-end dataset acquisition workflow
//!   (temp dir → download → optional hash check → move to final location)

use std::sync::OnceLock;
#[cfg(feature = "utils")]
pub use error::{DatasetError, DataFormatErrorKind};
#[cfg(feature = "utils")]
pub use utils::{download_to, unzip, create_temp_dir, file_sha256_matches, download_dataset_with};

/// A generic, thread-safe dataset container with lazy loading and in-memory caching.
///
/// `Dataset<T>` is a thin caching wrapper that holds a `storage_dir` (the directory
/// where dataset files are stored on disk) and a lazily-initialized value of type `T`.
/// The actual downloading and parsing logic is provided by the caller through a loader
/// closure passed to [`Dataset::load`].
///
/// This struct is designed to be the building block for both the built-in datasets
/// shipped with this crate and any custom datasets defined by external users.
///
/// # Type Parameter
///
/// - `T` - The type of the parsed dataset. Can be any type, such as
///   `(Array2<f64>, Array1<f64>)`, a custom struct, or any other data representation.
///   `T` must implement `Send + Sync` for `Dataset<T>` to be shared across threads.
///
/// # Thread Safety
///
/// `Dataset<T>` is `Send + Sync` when `T` is `Send + Sync`. The internal `OnceLock`
/// ensures that the loader closure runs at most once, even when multiple threads call
/// [`Dataset::load`] concurrently.
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
/// let dataset: Dataset<Vec<String>> = Dataset::new("./my_data");
///
/// // The first call to `load` triggers the loader
/// let data = dataset.load(my_loader).unwrap();
/// assert_eq!(data.len(), 2);
///
/// // Subsequent calls return the cached reference instantly
/// let data_again = dataset.load(my_loader).unwrap();
/// assert!(std::ptr::eq(data, data_again)); // same reference, no re-load
///
/// // Check whether data has been loaded
/// assert!(dataset.is_loaded());
/// ```
pub struct Dataset<T> {
    storage_dir: String,
    data: OnceLock<T>,
}

impl<T> Dataset<T> {
    /// Create a new `Dataset` instance without loading any data.
    ///
    /// This is a lightweight operation that only stores the storage directory path.
    /// No I/O or network requests are performed until [`Dataset::load`] is called.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory where dataset files will be stored. The directory
    ///   will be created automatically when the loader runs if it does not exist.
    ///
    /// # Returns
    ///
    /// A new `Dataset<T>` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Dataset {
            storage_dir: storage_dir.to_string(),
            data: OnceLock::new(),
        }
    }

    /// Load the dataset, executing the loader on first call and caching the result.
    ///
    /// On the first call, the `loader` closure is invoked with the storage directory
    /// path. The returned value is cached internally. All subsequent calls — from any
    /// thread — return a reference to the cached value without running the loader again.
    ///
    /// # Parameters
    ///
    /// - `loader` - A closure or function that takes the storage directory path (`&str`)
    ///   and returns `Result<T, E>`. This is where you perform downloading,
    ///   file I/O, and parsing. The loader is only called once; if the data is already
    ///   cached, it is ignored.
    ///
    /// # Returns
    ///
    /// - `Ok(&T)` - A reference to the cached dataset.
    ///
    /// # Errors
    ///
    /// Returns any error produced by the `loader` closure on first invocation.
    /// Once data is successfully loaded and cached, this method never returns an error.
    pub fn load<E>(&self, loader: impl FnOnce(&str) -> Result<T, E>) -> Result<&T, E> {
        if let Some(data) = self.data.get() {
            return Ok(data);
        }

        let value = loader(&self.storage_dir)?;
        let _ = self.data.set(value);

        Ok(self.data.get().expect("data should be set after successful load"))
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
}

impl<T> std::fmt::Debug for Dataset<T> {
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

/// Built-in dataset implementations.
///
/// Contains ready-to-use loaders for common machine learning datasets.
/// Each submodule also serves as an example of how to wrap [`Dataset<T>`]
/// to implement a custom dataset loader.
#[cfg(feature = "datasets")]
pub mod datasets;
