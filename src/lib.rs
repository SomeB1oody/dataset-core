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
//! let (features, labels) = load_iris();
//! println!("Loaded {} samples", features.len());
//! ```

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

pub fn download_to(url: &str, local_path: &str) -> Result<(), downloader::Error> {
    use downloader::downloader::Builder;
    use downloader::Download;
    use std::path::Path;

    let data = Download::new(url);

    let mut dl = Builder::default()
        .connect_timeout(std::time::Duration::from_secs(10))
        .download_folder(Path::new(local_path))
        .build()?;

    let response = dl.download(&[data])?;

    for r in response { r?; };

    Ok(())
}

pub fn unzip(file_path: std::fs::File, extract_path: &str) -> Result<(), zip::result::ZipError> {
    use zip::ZipArchive;
    
    ZipArchive::new(file_path)?.extract(extract_path)?;
    
    Ok(())
}

#[derive(Debug)]
pub enum DatasetError {
    DownloadError(downloader::Error),
    UnzipError(zip::result::ZipError),
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::DownloadError(e) => write!(f, "Download error: {}", e),
            DatasetError::UnzipError(e) => write!(f, "Unzip error: {}", e),
        }
    }
}

impl std::error::Error for DatasetError {}