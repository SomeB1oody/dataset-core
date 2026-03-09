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
//! - **Flexible API**: Both reference-based and owned data access patterns
//! - **Local storage**: Downloaded datasets are stored locally for offline access
//!
//! # Available Datasets
//!
//! | Dataset              | Samples | Features | Task Type      | Source                    |
//! |----------------------|---------|----------|----------------|---------------------------|
//! | Iris                 | 150     | 4        | Classification | UCI ML Repository         |
//! | Boston Housing       | 506     | 13       | Regression     | UCI ML Repository         |
//! | Diabetes             | 768     | 8        | Classification | Kaggle                    |
//! | Titanic              | 891     | 11       | Classification | Kaggle                    |
//! | Wine Quality (Red)   | 1599    | 11       | Regression     | UCI ML Repository         |
//! | Wine Quality (White) | 4898    | 11       | Regression     | UCI ML Repository         |
//!
//! # Quick Start
//!
//! ```rust
//! use rustyml_dataset::iris::Iris;
//!
//! let download_dir = "./iris"; // the code will create the directory if it doesn't exist
//!
//! let dataset = Iris::new(download_dir);
//! let features = dataset.features().unwrap();
//! let labels = dataset.labels().unwrap();
//!
//! let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
//! // you can use `.to_owned()` to get owned copies of the data
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
//! Each dataset module provides two loading functions:
//!
//! - **Reference functions** (`load_*()`): Return static references to cached data (zero allocation)
//! - **Owned functions** (`load_*_owned()`): Return cloned copies suitable for mutation
//!
//! # Performance Considerations
//!
//! The first call to any `load_*` function downloads, parses, and caches the dataset.
//! Subsequent calls return references to the cached data with zero overhead. Use reference
//! functions when possible for better performance; use owned functions when you need to
//! modify the data.

use zip::result::ZipError;
use downloader::downloader::Builder;
use downloader::Download;
use std::path::Path;
use std::io::Read;
use sha2::{Digest, Sha256};
use zip::ZipArchive;
use std::fs::File;

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
        .build()
        .map_err(|e| DatasetError::DownloadError(e))?;

    let response = dl
        .download(&[data])
        .map_err(|e| DatasetError::DownloadError(e))?;

    for r in response {
        if let Err(e) = r {
            return Err(DatasetError::DownloadError(e));
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
    let file = File::open(file_path).map_err(|e| DatasetError::UnzipError(ZipError::Io(e)))?;

    ZipArchive::new(file)
        .map_err(|e| DatasetError::UnzipError(e))?
        .extract(extract_dir)
        .map_err(|e| DatasetError::UnzipError(e))?;

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
/// - `temp_dir_name` - Prefix used for the temporary directory name.
///
/// # Errors
///
/// - `DatasetError` - Returned if the temporary directory cannot be created.
pub fn create_temp_dir(tempdir_in: &Path, temp_dir_name: &str) -> Result<tempfile::TempDir, DatasetError> {
    let temp_dir = tempfile::Builder::new()
        .prefix(temp_dir_name)
        .tempdir_in(tempdir_in)
        .map_err(|e| DatasetError::TempFileError(e))?;

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
/// - `DatasetError::StdIoError` - Returned when file I/O operations fail (opening file, reading data).
pub fn file_sha256_matches(path: &Path, expected_hex: &str) -> Result<bool, DatasetError> {
    let mut file = File::open(path).map_err(|e| DatasetError::StdIoError(e))?;

    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];

    loop {
        let read = file.read(&mut buf).map_err(|e| DatasetError::StdIoError(e))?;
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
/// file already matches the expected SHA256 hash.
///
/// # Parameters
///
/// - `path` - Directory path where the dataset will be stored.
/// - `dst` - Destination file path for the dataset.
/// - `expected_sha256` - Expected SHA256 hash for the dataset file.
///
/// # Returns
///
/// - `(need_download, need_overwrite)` - Flags indicating whether to download and
///   whether an existing file should be overwritten.
///
/// # Errors
///
/// - `DatasetError::StdIoError` - Returned when creating the directory fails or when
///   file I/O operations fail during hash verification.
pub fn prepare_download_dir(
    path: &Path,
    dst: &Path,
    expected_sha256: &str,
) -> Result<(bool, bool), DatasetError> {
    let mut need_download = true;
    let mut need_overwrite = false;

    if !path.exists() {
        std::fs::create_dir_all(path).map_err(|e| DatasetError::StdIoError(e))?;
    }

    if dst.exists() {
        if file_sha256_matches(dst, expected_sha256)? {
            need_download = false;
        } else {
            need_overwrite = true;
        }
    }

    Ok((need_download, need_overwrite))
}

/// Error type used by dataset loading utilities.
///
/// # Variants
///
/// - `DownloadError` - The download step failed (network, invalid URL, or downloader configuration).
/// - `ValidationError` - Downloaded file content failed integrity validation (SHA256 mismatch).
/// - `UnzipError` - Extracting a zip archive failed.
/// - `StdIoError` - A standard I/O operation failed (reading directories, opening/removing files, etc.).
/// - `DataFormatError` - The dataset content was not in the expected format.
/// - `TempFileError` - Creating a temporary directory failed.
#[derive(Debug)]
pub enum DatasetError {
    DownloadError(downloader::Error),
    ValidationError(String),
    UnzipError(ZipError),
    StdIoError(std::io::Error),
    DataFormatError(String),
    TempFileError(std::io::Error),
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::DownloadError(e) => write!(f, "Download error: {}", e),
            DatasetError::ValidationError(e) => write!(f, "Validation error: {}", e),
            DatasetError::UnzipError(e) => write!(f, "Unzip error: {}", e),
            DatasetError::StdIoError(e) => write!(f, "Std IO error: {}", e),
            DatasetError::DataFormatError(e) => write!(f, "Data format error: {}", e),
            DatasetError::TempFileError(e) => write!(f, "Temp file error: {}", e),
        }
    }
}

impl std::error::Error for DatasetError {}

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
