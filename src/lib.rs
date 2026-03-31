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
use zip::ZipArchive;
use zip::result::ZipError;
pub use error::{DatasetError, DataFormatErrorKind};

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
    let actual_hex = format!("{:x}", digest);
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
