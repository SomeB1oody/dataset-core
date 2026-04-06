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
//! ```rust
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
pub use error::{DatasetError, DataFormatErrorKind};
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
/// use rustyml_dataset::{Dataset, DatasetError, download_dataset_with, download_to, unzip};
///
/// // Step 1: Define constants for your dataset.
/// //
/// // These describe where to download the file, its expected filename after
/// // extraction, an identifier for error messages, and an optional SHA256 hash
/// // for integrity validation (optional).
/// const IRIS_DATA_URL: &str = "https://archive.ics.uci.edu/static/public/53/iris.zip";
/// const IRIS_ZIP_FILENAME: &str = "iris.zip";
/// const IRIS_FILENAME: &str = "iris.data";
/// const IRIS_DATASET_NAME: &str = "iris";
/// const IRIS_SHA256: &str = "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0";
///
/// // Step 2: Write a loader function that downloads and parses the dataset.
/// //
/// // The loader receives the storage directory path as `&str`. It should:
/// //   1. Download the file (using `download_dataset_with` and other helpers).
/// //   2. Parse the file into your desired output type `T`.
/// //   3. Return `Ok(T)` on success.
/// //
/// // Here we parse the Iris CSV into a simple `Vec<Vec<String>>` where each
/// // inner Vec holds the fields of one row. You can return any type you want.
/// // In fact, this example only performed simple data processing (using String as the element type, without converting it into a numeric type).
/// // This crate also provides a complete implementation of the Iris dataset, and you can refer to that implementation.
/// fn my_loader(dir: &str) -> Result<Vec<Vec<String>>, DatasetError> {
///     // Use `download_dataset_with` to handle download, SHA256 validation,
///     // and file caching. The closure receives a temporary directory, and you
///     // return the path to the prepared file.
///     let file_path = download_dataset_with(
///         dir,
///         IRIS_FILENAME,
///         IRIS_DATASET_NAME,
///         Some(IRIS_SHA256),
///         |temp_path| {
///             // Download the zip archive into the temporary directory.
///             download_to(IRIS_DATA_URL, temp_path)?;
///             // Extract the archive so we can access the CSV inside.
///             unzip(&temp_path.join(IRIS_ZIP_FILENAME), temp_path)?;
///             // Return the path to the extracted dataset file.
///             Ok(temp_path.join(IRIS_FILENAME))
///         },
///     )?;
///
///     // Parse the CSV into Vec<Vec<String>>.
///     let file = std::fs::File::open(&file_path)?;
///     let mut rdr = csv::ReaderBuilder::new()
///         .has_headers(false)
///         .from_reader(file);
///     let mut rows = Vec::new();
///     for result in rdr.records() {
///         let record = result.map_err(|e| DatasetError::csv_read_error(IRIS_DATASET_NAME, e))?;
///         let row: Vec<String> = record.iter().map(|field| field.to_string()).collect();
///         rows.push(row);
///     }
///     Ok(rows)
/// }
///
/// // Step 3: Create a `Dataset` instance and load the data
/// let dataset: Dataset<Vec<Vec<String>>> = Dataset::new("./my_iris_data");
///
/// // The first call to `load` triggers the download and parse
/// let data = dataset.load(my_loader).unwrap();
/// assert_eq!(data.len(), 150); // 150 samples in the Iris dataset
///
/// // Subsequent calls return the cached reference instantly
/// let data_again = dataset.load(my_loader).unwrap();
/// assert!(std::ptr::eq(data, data_again)); // same reference, no re-download
///
/// // Check whether data has been loaded
/// assert!(dataset.is_loaded());
///
/// // Clean up (Dispensable)
/// std::fs::remove_dir_all("./my_iris_data").unwrap();
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
    ///   and returns `Result<T, DatasetError>`. This is where you perform downloading,
    ///   file I/O, and parsing. The loader is only called once; if the data is already
    ///   cached, it is ignored.
    ///
    /// # Returns
    ///
    /// - `Ok(&T)` - A reference to the cached dataset.
    ///
    /// # Errors
    ///
    /// Returns any `DatasetError` produced by the `loader` closure on first invocation.
    /// Once data is successfully loaded and cached, this method never returns an error.
    pub fn load(&self, loader: impl FnOnce(&str) -> Result<T, DatasetError>) -> Result<&T, DatasetError> {
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
pub mod error;

/// Utility functions for dataset authors.
///
/// Provides helpers for downloading files, extracting archives, verifying
/// SHA256 hashes, and managing the dataset acquisition workflow.
pub mod utils;

/// Built-in dataset implementations.
///
/// Contains ready-to-use loaders for common machine learning datasets.
/// Each submodule also serves as an example of how to wrap [`Dataset<T>`]
/// to implement a custom dataset loader.
pub mod datasets;
