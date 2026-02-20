//! A collection of classic machine learning datasets for Rust.
//!
//! This crate provides easy access to popular datasets commonly used in machine learning
//! and data science education and research. Each dataset module includes functions to load
//! the data along with utilities for splitting into training and testing sets.
//!
//! # Available Datasets
//!
//! - **Boston Housing**: Regression dataset for predicting house prices
//! - **Diabetes**: Regression dataset for predicting disease progression
//! - **Iris**: Classification dataset for flower species identification
//! - **Titanic**: Classification dataset for survival prediction
//! - **Wine Quality**: Classification/regression dataset for wine quality assessment
//!
//! # Example
//!
//! ```rust
//! use rustyml_dataset::iris::load_iris;
//!
//! let download_dir = "./downloads"; // you need to create this directory manually beforehand
//!
//! let (features, labels) = load_iris(download_dir).unwrap();
//! assert_eq!(features.shape(), &[150, 4]);
//! assert_eq!(labels.len(), 150);
//!
//! // clean up: remove the downloaded files if they exist
//! if let Ok(entries) = std::fs::read_dir(download_dir) {
//!     for entry in entries.flatten() {
//!         let _ = std::fs::remove_file(entry.path());
//!     }
//! }
//! ```

use zip::result::ZipError;
use downloader::downloader::Builder;
use downloader::Download;
use std::path::Path;
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

/// Error type used by dataset loading utilities.
///
/// # Variants
///
/// - `DownloadError` - The download step failed (network, invalid URL, or downloader configuration).
/// - `UnzipError` - Extracting a zip archive failed.
/// - `StdIoError` - A standard I/O operation failed (reading directories, opening/removing files, etc.).
/// - `DataFormatError` - The dataset content was not in the expected format.
#[derive(Debug)]
pub enum DatasetError {
    DownloadError(downloader::Error),
    UnzipError(ZipError),
    StdIoError(std::io::Error),
    DataFormatError(String),
    TempFileError(std::io::Error),
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::DownloadError(e) => write!(f, "Download error: {}", e),
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
/// Contains a dataset for predicting diabetes disease progression one year
/// after baseline, using ten baseline variables including age, sex, BMI,
/// and six blood serum measurements.
pub mod diabetes;

/// Iris flower dataset module.
///
/// Contains the classic Iris dataset for classifying iris flowers into
/// three species (setosa, versicolor, virginica) based on sepal and petal
/// measurements.
pub mod iris;

/// Internal module for raw data storage.
mod raw_data;

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