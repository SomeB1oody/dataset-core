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
//! use rustyml_dataset::iris::Iris;
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

use downloader::Download;
use downloader::downloader::Builder;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use zip::ZipArchive;
use zip::result::ZipError;
pub use error::{DatasetError, DataFormatErrorKind};

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

/// Download a remote file into the given directory.
///
/// This is a small wrapper around the [`downloader`] crate used by dataset loaders.
/// It downloads the content at `url` into `storage_path` using the downloader's
/// default file naming behavior.
///
/// # Parameters
///
/// - `url` - The URL to download.
/// - `storage_path` - The directory to store the downloaded file in.
///
/// # Errors
///
/// - `DatasetError` - Returned when the downloader cannot be built or when the download fails.
///
/// # Example
/// ```rust
/// use rustyml_dataset::download_to;
/// use std::path::Path;
///
/// let download_dir = "./download_example";
/// std::fs::create_dir_all(download_dir).unwrap();
///
/// // Download a file from the internet
/// let url = "https://archive.ics.uci.edu/static/public/53/iris.zip";
/// download_to(url, Path::new(download_dir)).unwrap();
///
/// // The file will be saved with the name from the URL (iris.zip)
/// assert!(Path::new(download_dir).join("iris.zip").exists());
///
/// // Clean up (dispensable)
/// std::fs::remove_dir_all(download_dir).unwrap();
/// ```
pub fn download_to(url: &str, storage_path: &Path) -> Result<(), DatasetError> {
    let data = Download::new(url);

    let mut dl = Builder::default()
        .connect_timeout(std::time::Duration::from_secs(10))
        .download_folder(storage_path)
        .build()?;

    let response = dl.download(&[data])?;

    for r in response {
        if let Err(e) = r {
            return Err(e.into());
        }
    }

    Ok(())
}

/// Extract a zip archive into a target directory using [`ZipArchive`] in [`zip`] crate.
///
/// # Parameters
///
/// - `file_path` - Path to the `.zip` file to extract.
/// - `extract_dir` - Directory to extract the archive contents into.
///
/// # Errors
///
/// - `DatasetError` - Returned when opening the zip file fails or when extraction fails.
///
/// # Example
/// ```rust
/// use rustyml_dataset::{download_to, unzip};
/// use std::path::Path;
///
/// let work_dir = "./unzip_example";
/// std::fs::create_dir_all(work_dir).unwrap();
///
/// // First download a zip file
/// let url = "https://archive.ics.uci.edu/static/public/53/iris.zip";
/// download_to(url, Path::new(work_dir)).unwrap();
///
/// // Extract the zip archive
/// let zip_path = Path::new(work_dir).join("iris.zip");
/// unzip(&zip_path, Path::new(work_dir)).unwrap();
///
/// // The extracted files are now in the work directory
/// assert!(Path::new(work_dir).join("iris.data").exists());
///
/// // Clean up (dispensable)
/// std::fs::remove_dir_all(work_dir).unwrap();
/// ```
pub fn unzip(file_path: &Path, extract_dir: &Path) -> Result<(), DatasetError> {
    let file = File::open(file_path).map_err(|e| DatasetError::from(ZipError::Io(e)))?;

    ZipArchive::new(file)?.extract(extract_dir)?;

    Ok(())
}

/// Create a temporary directory under the given parent directory.
///
/// This is a small wrapper around [`tempfile::Builder`] used by dataset loaders to
/// keep intermediate download/extraction artifacts isolated. The created directory
/// is removed automatically when the returned [`tempfile::TempDir`] is dropped.
///
/// # Parameters
///
/// - `tempdir_in` - The parent directory in which the temporary directory will be created.
///
/// # Errors
///
/// - `DatasetError` - Returned if the temporary directory cannot be created.
///
/// # Example
/// ```rust
/// use rustyml_dataset::create_temp_dir;
/// use std::path::Path;
///
/// let parent_dir = "./temp_dir_example";
/// std::fs::create_dir_all(parent_dir).unwrap();
///
/// // Create a temporary directory
/// let temp_dir = create_temp_dir(Path::new(parent_dir)).unwrap();
/// let temp_path = temp_dir.path();
///
/// // Use the temporary directory for intermediate operations
/// let temp_file = temp_path.join("temp_file.txt");
/// std::fs::write(&temp_file, "temporary content").unwrap();
/// assert!(temp_file.exists());
///
/// // The temporary directory is automatically removed when `temp_dir` is dropped
/// drop(temp_dir);
///
/// // Clean up parent directory
/// std::fs::remove_dir_all(parent_dir).unwrap();
/// ```
pub fn create_temp_dir(tempdir_in: &Path) -> Result<tempfile::TempDir, DatasetError> {
    let temp_dir = tempfile::Builder::new()
        .tempdir_in(tempdir_in)?;

    Ok(temp_dir)
}

/// Verify that a file's SHA256 hash matches an expected value.
///
/// This function computes the SHA256 hash of the file at the given path and compares
/// it with the expected hexadecimal hash string (case-insensitive). It is used by
/// dataset loaders to validate downloaded files before parsing.
///
/// # Parameters
///
/// - `path` - Path to the file to verify.
/// - `expected_hex` - Expected SHA256 hash as a hexadecimal string.
///
/// # Returns
///
/// - `bool` - true if the computed hash matches the expected hash, false if the hashes don't match
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when file I/O operations fail (opening file, reading data).
///
/// # Example
/// ```rust
/// use rustyml_dataset::file_sha256_matches;
/// use std::path::Path;
/// use std::io::Write;
///
/// let test_dir = "./sha256_example";
/// std::fs::create_dir_all(test_dir).unwrap();
///
/// // Create a test file with known content
/// let file_path = Path::new(test_dir).join("test.txt");
/// let mut file = std::fs::File::create(&file_path).unwrap();
/// file.write_all(b"hello world").unwrap();
/// drop(file);
///
/// // SHA256 of "hello world" is:
/// // b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
/// let expected_hash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
///
/// // Verify the hash matches
/// assert!(file_sha256_matches(&file_path, expected_hash).unwrap());
///
/// // Case-insensitive comparison also works
/// let upper_hash = "B94D27B9934D3E08A52E52D7DA7DABFAC484EFE37A5380EE9088F7ACE2EFCDE9";
/// assert!(file_sha256_matches(&file_path, upper_hash).unwrap());
///
/// // Wrong hash returns false
/// assert!(!file_sha256_matches(&file_path, "0000000000000000000000000000000000000000000000000000000000000000").unwrap());
///
/// // Clean up (dispensable)
/// std::fs::remove_dir_all(test_dir).unwrap();
/// ```
pub fn file_sha256_matches(path: &Path, expected_hex: &str) -> Result<bool, DatasetError> {
    let mut file = File::open(path)?;

    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];

    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }

    let digest = hasher.finalize();
    let actual_hex = digest.iter().map(|b| format!("{:02x}", b)).collect::<String>();
    Ok(actual_hex.eq_ignore_ascii_case(expected_hex))
}

/// Prepare a dataset download directory and determine if download/overwrite is needed.
///
/// This helper ensures the target directory exists and checks whether the destination
/// file already matches the expected SHA256 hash. If `expected_sha256` is `None`,
/// the file is accepted if it exists without validation.
///
/// # Parameters
///
/// - `path` - Directory path where the dataset will be stored.
/// - `dst` - Destination file path for the dataset.
/// - `expected_sha256` - Optional expected SHA256 hash for the dataset file. If `None`,
///   any existing file at `dst` is accepted without validation.
///
/// # Returns
///
/// - `(need_download, need_overwrite)` - Flags indicating whether to download and
///   whether an existing file should be overwritten.
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when creating the directory fails or when
///   file I/O operations fail during hash verification.
///
/// # Example
/// ```rust
/// use rustyml_dataset::prepare_download_dir;
/// use std::path::Path;
/// use std::io::Write;
///
/// let test_dir = "./prepare_download_example";
/// let dir_path = Path::new(test_dir);
/// let file_path = dir_path.join("data.txt");
///
/// // Case 1: Directory doesn't exist yet
/// let (need_download, need_overwrite) = prepare_download_dir(
///     dir_path,
///     &file_path,
///     None,
/// ).unwrap();
/// assert!(need_download);    // File doesn't exist, need to download
/// assert!(!need_overwrite);  // Nothing to overwrite
///
/// // Case 2: File exists with correct hash
/// let mut file = std::fs::File::create(&file_path).unwrap();
/// file.write_all(b"hello world").unwrap();
/// drop(file);
///
/// let correct_hash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
/// let (need_download, need_overwrite) = prepare_download_dir(
///     dir_path,
///     &file_path,
///     Some(correct_hash),
/// ).unwrap();
/// assert!(!need_download);   // File exists with correct hash
/// assert!(!need_overwrite);
///
/// // Case 3: File exists but hash doesn't match
/// let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
/// let (need_download, need_overwrite) = prepare_download_dir(
///     dir_path,
///     &file_path,
///     Some(wrong_hash),
/// ).unwrap();
/// assert!(need_download);    // Hash mismatch, need to download
/// assert!(need_overwrite);   // Existing file needs to be replaced
///
/// // Clean up (dispensable)
/// std::fs::remove_dir_all(test_dir).unwrap();
/// ```
pub fn prepare_download_dir(
    path: &Path,
    dst: &Path,
    expected_sha256: Option<&str>,
) -> Result<(bool, bool), DatasetError> {
    let mut need_download = true;
    let mut need_overwrite = false;

    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }

    if dst.exists() {
        if let Some(hash) = expected_sha256 {
            // SHA256 validation enabled
            if file_sha256_matches(dst, hash)? {
                need_download = false;
            } else {
                need_overwrite = true;
            }
        } else {
            // No SHA256 validation: accept existing file
            need_download = false;
        }
    }

    Ok((need_download, need_overwrite))
}

/// Generic dataset download framework that handles common download workflow.
///
/// This function manages the complete dataset acquisition workflow: checking if download
/// is needed, creating a temporary directory, delegating file preparation to a user-provided
/// closure, optionally validating the file with SHA256, and moving it to the final destination.
///
/// # Parameters
///
/// - `dir` - Target storage directory path
/// - `filename` - Final dataset filename (will be stored as `dir/filename`)
/// - `dataset_name` - Dataset name for error messages
/// - `expected_sha256` - Optional expected SHA256 hash of the dataset file. If `None`,
///   any existing file at the destination is accepted without validation, and newly
///   prepared files skip SHA256 verification.
/// - `prepare_file` - Closure that prepares the dataset file in the temporary directory
///   - **Input**: `temp_dir: &Path` - Path to the temporary directory
///   - **Output**: `Result<PathBuf, DatasetError>` - Path to the prepared dataset file
///   - **Responsibility**: This closure can perform any operations needed to obtain the
///     dataset file, such as downloading (you can use [`download_to`] provided in this crate), extracting archives
///     (you can use [`unzip`] provided in this crate), or locating files within extracted folders. The returned
///     `PathBuf` must point to the final dataset file ready for validation.
///   - Note: The file you provide will be moved to the final destination (`dir/filename`), not copied.
///
/// # Returns
///
/// - `PathBuf` - Path to the final dataset file (`dir/filename`)
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when directory creation, file operations, or
///   hash verification fails
/// - `DatasetError::Sha256ValidationFailed` - Returned when `expected_sha256` is provided
///   and the prepared file's SHA256 hash does not match it
/// - Any error returned by the `prepare_file` closure
///
/// # Example
/// ```rust
/// // Implement the downloading process for iris dataset
///
/// /// The URL for the Iris dataset.
/// ///
/// /// # Citation
/// ///
/// /// R. A. Fisher. "Iris," UCI Machine Learning Repository, \[Online\].
/// /// Available: <https://doi.org/10.24432/C56C76>
/// const IRIS_DATA_URL: &str = "https://archive.ics.uci.edu/static/public/53/iris.zip";
///
/// /// The name of the zip file downloaded.
/// const IRIS_ZIP_FILENAME: &str = "iris.zip";
///
/// /// The name of the file in the zip after extraction.
/// const IRIS_FILENAME: &str = "iris.data";
///
/// /// The SHA256 hash of the Iris dataset file.
/// const IRIS_SHA256: &str = "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0";
///
/// /// The name of the dataset
/// const IRIS_DATASET_NAME: &str = "iris";
///
/// use rustyml_dataset::download_dataset_with;
/// use rustyml_dataset::download_to;
/// use rustyml_dataset::unzip;
///
/// fn main() {
///     let dir = "./somewhere";
///
///     let file_path = download_dataset_with(
///             // Target storage directory path
///             dir,
///             // Final dataset filename (will be stored as `dir/filename`)
///             IRIS_FILENAME,
///             // Dataset name for error messages
///             IRIS_DATASET_NAME,
///             // Expected SHA256 hash of the dataset file
///             Some(IRIS_SHA256),
///             // Closure that prepares the dataset file in the temporary directory
///             |temp_path| {
///                 // Download and extract the dataset
///                 download_to(IRIS_DATA_URL, temp_path)?;
///                 unzip(&temp_path.join(IRIS_ZIP_FILENAME), temp_path)?;
///                 // Return the path to the extracted dataset file
///                 Ok(temp_path.join(IRIS_FILENAME))
///             },
///         ).unwrap();
///
///     // `file_path` is now the path to the downloaded and extracted Iris dataset file
///     // it can be used to give the path of the dataset or parse data
///
///     // cleanup (dispensable)
///     std::fs::remove_dir_all(dir).unwrap();
/// }
/// ```
pub fn download_dataset_with<F>(
    dir: &str,
    filename: &str,
    dataset_name: &str,
    expected_sha256: Option<&str>,
    prepare_file: F,
) -> Result<PathBuf, DatasetError>
where
    F: FnOnce(&Path) -> Result<PathBuf, DatasetError>,
{
    let dir_path = Path::new(dir);
    let dst = dir_path.join(filename);
    let (need_download, need_overwrite) = prepare_download_dir(dir_path, &dst, expected_sha256)?;

    if need_download {
        let temp_dir = create_temp_dir(dir_path)?;
        let temp_path = temp_dir.path();

        // Call user closure: prepare the dataset file in temporary directory
        let src = prepare_file(temp_path)?;

        // Validate SHA256 hash if provided
        if let Some(hash) = expected_sha256 {
            if !file_sha256_matches(&src, hash)? {
                drop(temp_dir); // Clean up temporary directory
                return Err(DatasetError::sha256_validation_failed(
                    dataset_name,
                    filename,
                ));
            }
        }

        // Move file to final destination
        if need_overwrite {
            std::fs::remove_file(&dst)?;
        }
        std::fs::rename(&src, &dst)?;
    }

    Ok(dst)
}

/// Error handling module.
///
/// Provides structured error types for dataset loading operations including
/// download failures, validation errors, I/O errors, and detailed data format
/// errors with line numbers and contextual information for debugging.
pub mod error;

/// Boston Housing dataset module.
///
/// Contains the Boston Housing dataset for predicting median house values
/// in Boston suburbs based on various features like crime rate, room count,
/// and accessibility to highways.
pub mod boston_housing;

/// Diabetes dataset module.
///
/// Contains the Pima Indians Diabetes dataset for binary classification
/// based on 8 diagnostic measurements.
pub mod diabetes;

/// Iris flower dataset module.
///
/// Contains the classic Iris dataset for classifying iris flowers into
/// three species (setosa, versicolor, virginica) based on sepal and petal
/// measurements.
pub mod iris;

/// Titanic dataset module.
///
/// Contains data about Titanic passengers for predicting survival based
/// on features like passenger class, sex, age, and fare.
pub mod titanic;

/// Wine Quality dataset module.
///
/// Contains wine quality assessment data for predicting quality scores
/// based on physicochemical properties like acidity, sugar content, and
/// alcohol percentage.
pub mod wine_quality;
