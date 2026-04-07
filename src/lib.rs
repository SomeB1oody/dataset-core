//! A collection of classic machine learning datasets with automatic download and caching for Rust.
//!
//! `rustyml-dataset` provides easy access to popular machine learning datasets with built-in
//! support for the `ndarray` crate. Datasets are automatically downloaded from their original
//! sources on first use and cached in memory using thread-safe `OnceLock` for optimal performance.
//!
//! # Features
//!
//! - **Automatic downloading**: Datasets are fetched from original sources on demand
//! - **Thread-safe memoization**: Uses `OnceLock` for lazy initialization and caching
//! - **ndarray integration**: All data returned as `ndarray` types (`Array1`, `Array2`)
//! - **Struct-based API**: Each dataset is a struct with lazy-loading accessor methods
//! - **Local storage**: Downloaded datasets are stored locally for offline access
//!
//! # Available Datasets
//!
//! | Dataset              | Samples | Features | Task Type      |
//! |----------------------|---------|----------|----------------|
//! | Iris                 | 150     | 4        | Classification |
//! | Boston Housing       | 506     | 13       | Regression     |
//! | Diabetes             | 768     | 8        | Classification |
//! | Titanic              | 891     | 11       | Classification |
//! | Wine Quality (Red)   | 1599    | 11       | Regression     |
//! | Wine Quality (White) | 4898    | 11       | Regression     |
//!
//! # Quick Start
//!
//! ```rust, ignore
//! use rustyml_dataset::datasets::iris::Iris;
//!
//! let download_dir = "./data"; // the code will create the directory if it doesn't exist
//!
//! let dataset = Iris::new(download_dir);
//! let features = dataset.features().unwrap();
//! let labels = dataset.labels().unwrap();
//!
//! let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
//! // use `.to_owned()` to get owned copies of the data that can be modified
//! let mut features_owned = features.to_owned();
//! let mut labels_owned = labels.to_owned();
//!
//! // Example: Modify feature values
//! features_owned[[0, 0]] = 5.5;
//! labels_owned[0] = "setosa-modified";
//!
//! assert_eq!(features.shape(), &[150, 4]);
//! assert_eq!(labels.len(), 150);
//!
//! // clean up: remove the downloaded files
//! std::fs::remove_dir_all(download_dir).unwrap();
//! ```
//!
//! # API Patterns
//!
//! Each dataset is represented as a struct with lazy loading. Data is not fetched until
//! you call one of the accessor methods:
//!
//! - **`new(storage_path)`**: Create a dataset instance (lightweight, no I/O)
//! - **`features()`**: Return a reference to the feature matrix
//! - **`labels()` / `targets()`**: Return a reference to the label/target vector
//! - **`data()`**: Return references to both features and labels/targets at once
//!
//! Call `.to_owned()` on any returned reference to get an owned, mutable copy.
//!
//! # Performance Considerations
//!
//! The first call to any accessor method downloads, parses, and caches the dataset.
//! Subsequent calls return references to the cached data with zero overhead. Use the
//! reference accessors when possible for better performance; call `.to_owned()` only
//! when you need to modify the data.

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
/// use rustyml_dataset::Dataset;
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
