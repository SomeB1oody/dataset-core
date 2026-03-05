use std::fs::{remove_file, rename, File};
use std::io::Read;
use std::path::Path;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;
use crate::{create_temp_dir, download_to, DatasetError, unzip, file_sha256_matches, prepare_download_dir};

/// A static variable to store the Red Wine Quality dataset.
///
/// This variable is of type `OnceLock`, which ensures thread-safe, one-time initialization
/// of its contents. It contains a tuple of:
///
/// - `Array2<f64>`: A 2-dimensional array representing the numerical features of the dataset
/// (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
/// - `Array1<f64>`: A 1-dimensional array containing the corresponding targets (wine quality scores, from 0 to 10)
///
/// The `OnceLock` ensures that the dataset is initialized only once and is then immutable
/// for the lifetime of the program.
static RED_WINE_DATA: OnceLock<(Array2<f64>, Array1<f64>)> = OnceLock::new();

/// A static variable to store the White Wine Quality dataset.
///
/// This variable is of type `OnceLock`, which ensures thread-safe, one-time initialization
/// of its contents. It contains a tuple of:
///
/// - `Array2<f64>`: A 2-dimensional array representing the numerical features of the dataset
/// (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
/// - `Array1<f64>`: A 1-dimensional array containing the corresponding targets (wine quality scores, from 0 to 10)
///
/// The `OnceLock` ensures that the dataset is initialized only once and is then immutable
/// for the lifetime of the program.
static WHITE_WINE_DATA: OnceLock<(Array2<f64>, Array1<f64>)> = OnceLock::new();

/// A static string slice containing the URL for the Wine Quality dataset.
///
/// # Citation
///
/// P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. "Wine Quality," UCI Machine Learning Repository, 2009. \[Online\]. Available: <https://doi.org/10.24432/C56S3T>.
///
/// # About Dataset
///
/// The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
///
/// Features:
///   - fixed acidity
///   - volatile acidity
///   - citric acid
///   - residual sugar
///   - chlorides
///   - free sulfur dioxide
///   - total sulfur dioxide
///   - density
///   - pH
///   - sulphates
///   - alcohol
///
/// Targets:
/// - quality (score between 0 and 10)
///
/// See more information at <https://archive.ics.uci.edu/dataset/186/wine+quality>
const WINE_QUALITY_URL: &str = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip";

/// The prefix for temporary files created during dataset download and extraction.
const WINE_QUALITY_TEMP_FILE_PREFIX: &str = ".tmp-wine-quality-";

/// The filename of the zip archive containing the Wine Quality datasets.
const WINE_QUALITY_ZIP_FILENAME: &str = "wine+quality.zip";

/// The red wine file of the CSV files inside the zip archive.
const RED_WINE_QUALITY_FILENAME: &str = "winequality-red.csv";

/// The white wine file of the CSV files inside the zip archive.
const WHITE_WINE_QUALITY_FILENAME: &str = "winequality-white.csv";

/// The number of samples in white wine quality datasets.
const WHITE_WINE_QUALITY_SAMPLE_SIZE: usize = 4898;

/// The number of samples in red wine quality datasets.
const RED_WINE_QUALITY_SAMPLE_SIZE: usize = 1599;

/// The number of features in the Wine Quality datasets.
const WINE_QUALITY_NUM_FEATURES: usize = 11;

/// The SHA256 hash of the white wine quality dataset.
const WHITE_WINE_QUALITY_SHA256: &str = "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836";

/// The SHA256 hash of the red wine quality dataset.
const RED_WINE_QUALITY_SHA256: &str = "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e";

/// Downloads and stores a Wine Quality CSV file if needed.
///
/// This internal helper function handles the download, extraction, and validation
/// of a specific Wine Quality CSV file (either red or white wine data). It downloads
/// the UCI Wine Quality zip archive, extracts the specified CSV file, validates it
/// against the expected SHA256 hash, and moves it to the target location.
///
/// # Parameters
///
/// - `storage_dir` - Directory where the dataset files will be stored
/// - `dst_file` - Target path where the CSV file should be placed
/// - `csv_filename` - Name of the CSV file to extract from the zip archive
/// - `expected_sha256` - Expected SHA256 hash for file validation
/// - `need_download` - Whether download is needed (if false, function returns early)
/// - `need_overwrite` - Whether to overwrite the existing file if it exists
///
/// # Returns
///
/// Returns `Ok(())` if the operation succeeds, or returns early if `need_download` is false.
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Creating temporary directory fails
/// - Download fails due to network issues or invalid URL
/// - Unzipping fails or the archive is corrupted
/// - SHA256 validation fails (file corruption or wrong data)
/// - File I/O operations fail (removing existing file, moving extracted file)
fn ensure_wine_quality_csv(
    storage_dir: &Path,
    dst_file: &Path,
    csv_filename: &str,
    expected_sha256: &str,
    need_download: bool,
    need_overwrite: bool,
) -> Result<(), DatasetError> {
    if !need_download {
        return Ok(());
    }

    // temporary directory to store the downloaded zip file
    let temp_dir = create_temp_dir(storage_dir, WINE_QUALITY_TEMP_FILE_PREFIX)?;
    let path_temp = temp_dir.path();

    // download the zip file and extract it to the temporary directory
    download_to(WINE_QUALITY_URL, path_temp)?;
    unzip(&path_temp.join(WINE_QUALITY_ZIP_FILENAME), path_temp)?;

    // move the extracted file to the original directory
    let src_file = path_temp.join(csv_filename);

    if !file_sha256_matches(src_file.as_path(), expected_sha256)? {
        return Err(DatasetError::ValidationError(format!(
            "{} SHA256 validation failed",
            csv_filename
        )));
    }
    if need_overwrite {
        remove_file(dst_file).map_err(|e| DatasetError::StdIoError(e))?;
    }
    rename(&src_file, dst_file).map_err(|e| DatasetError::StdIoError(e))?;

    Ok(())
}

/// Internal function to download, extract, and parse the Red Wine Quality dataset.
///
/// This function downloads the UCI Wine Quality archive, extracts the CSV file,
/// moves it into `storage_path`, then parses the dataset into `ndarray` arrays.
///
/// # Parameters
///
/// - `path` - Directory where the dataset files will be stored. The directory is created if missing.
///
/// # Returns
///
/// Returns a `(features, targets)` tuple:
/// - features shape: `(1599, 11)`
/// - targets length: `1599`
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Download fails due to network issues or invalid URL
/// - Unzipping fails or the archive is corrupted
/// - File I/O fails (creating directories, moving/removing files, reading CSV)
/// - Data format is invalid (wrong number of columns, unparseable values)
/// - Parsed dataset size does not match the expected number of samples
fn parse_red_wine_data(path: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
    let path = Path::new(path);
    let dst_red = path.join(RED_WINE_QUALITY_FILENAME);
    let (need_download, need_overwrite) =
        prepare_download_dir(path, &dst_red, RED_WINE_QUALITY_SHA256)?;
    ensure_wine_quality_csv(
        path,
        &dst_red,
        RED_WINE_QUALITY_FILENAME,
        RED_WINE_QUALITY_SHA256,
        need_download,
        need_overwrite,
    )?;

    let mut red_wine_file = File::open(dst_red).map_err(|e| DatasetError::StdIoError(e))?;
    let mut red_wine_data = String::new();
    red_wine_file.read_to_string(&mut red_wine_data).map_err(|e| DatasetError::StdIoError(e))?;

    parse_wine_data_to_array(red_wine_data, RED_WINE_QUALITY_SAMPLE_SIZE)
}

/// Internal function to download, extract, and parse the White Wine Quality dataset.
///
/// This function downloads the UCI Wine Quality archive, extracts the CSV file,
/// moves it into `storage_path`, then parses the dataset into `ndarray` arrays.
///
/// # Parameters
///
/// - `path` - Directory where the dataset files will be stored. The directory is created if missing.
///
/// # Returns
///
/// Returns a `(features, targets)` tuple:
/// - features shape: `(4898, 11)`
/// - targets length: `4898`
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Download fails due to network issues or invalid URL
/// - Unzipping fails or the archive is corrupted
/// - File I/O fails (creating directories, moving/removing files, reading CSV)
/// - Data format is invalid (wrong number of columns, unparseable values)
/// - Parsed dataset size does not match the expected number of samples
fn parse_white_wine_data(path: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
    let path = Path::new(path);
    let dst_white = path.join(WHITE_WINE_QUALITY_FILENAME);
    let (need_download, need_overwrite) =
        prepare_download_dir(path, &dst_white, WHITE_WINE_QUALITY_SHA256)?;
    ensure_wine_quality_csv(
        path,
        &dst_white,
        WHITE_WINE_QUALITY_FILENAME,
        WHITE_WINE_QUALITY_SHA256,
        need_download,
        need_overwrite,
    )?;

    let mut white_wine_file = File::open(dst_white).map_err(|e| DatasetError::StdIoError(e))?;
    let mut white_wine_data = String::new();
    white_wine_file.read_to_string(&mut white_wine_data).map_err(|e| DatasetError::StdIoError(e))?;

    parse_wine_data_to_array(white_wine_data, WHITE_WINE_QUALITY_SAMPLE_SIZE)
}

/// Parses a single Wine Quality CSV (red or white) into `(features, targets)`.
///
/// The CSV is expected to be `;`-separated with a **header row**, followed by data rows.
/// Each data row must contain:
/// - 11 feature columns (all parseable as `f64`)
/// - 1 target column (`quality`, parseable as `f64`)
///
/// # Parameters
///
/// - `data` - Full CSV file contents as a string.
/// - `n_samples` - Expected number of samples (rows excluding the header).
///
/// # Returns
///
/// - `Array2<f64>` - Feature matrix with shape `(n_samples, 11)`.
/// - `Array1<f64>` - Target vector with length `n_samples`.
///
/// # Errors
///
/// Returns `DatasetError::DataFormatError` if:
/// - Any row has an unexpected number of columns
/// - Any feature/target value fails to parse as `f64`
/// - The final number of parsed values does not match the expected shape
fn parse_wine_data_to_array(data: String, n_samples: usize) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
    let lines: Vec<&str> = data.trim().lines().collect();

    let mut features_array = Vec::with_capacity(n_samples * WINE_QUALITY_NUM_FEATURES);
    let mut target_array = Vec::with_capacity(n_samples);

    for line in &lines[1..] {
        if line.is_empty() { continue; }
        let cols: Vec<&str> = line.split(';').collect();

        if cols.len() != WINE_QUALITY_NUM_FEATURES + 1 {
            return Err(DatasetError::DataFormatError(format!(
                "Invalid wine quality data format: expected {} columns, found {} at line {}"
                , WINE_QUALITY_NUM_FEATURES + 1
                , cols.len()
                , line
            )))
        }

        for i in 0..WINE_QUALITY_NUM_FEATURES {
            features_array.push(cols[i].parse::<f64>().map_err(
                |e| DatasetError::DataFormatError(
                    format!("Failed to parse Wine Quality dataset features {} at line {}: {}", i, line, e)))?);
        }

        target_array.push(cols[11].parse::<f64>().map_err(
            |e| DatasetError::DataFormatError(
                format!("Failed to parse Wine Quality target at line {}: {}", line, e))
        )?);
    }

    if features_array.len() != n_samples * WINE_QUALITY_NUM_FEATURES {
        return Err(DatasetError::DataFormatError(format!(
            "Expected {} * {} elements in features, got {}", n_samples, 
            WINE_QUALITY_NUM_FEATURES, 
            features_array.len()
        )))
    }
    if target_array.len() != n_samples {
        return Err(DatasetError::DataFormatError(format!(
            "Expected {} elements in target, got {}", n_samples, target_array.len()
        )))
    }

    let features_array =
        Array2::from_shape_vec((n_samples, WINE_QUALITY_NUM_FEATURES), features_array)
            .map_err(
                |e| DatasetError::DataFormatError(
                    format!("Failed to create features array: {}", e)
                )
            )?;
    let target_array = Array1::from_vec(target_array);

    Ok((features_array, target_array))
}

/// Loads the Red Wine Quality dataset with automatic caching.
///
/// This function returns references to cached data, which incurs no additional
/// memory allocation after the first load. If you need owned data that you can
/// modify, prefer [`load_red_wine_quality_owned()`].
///
/// # About Dataset
///
/// The dataset contains physicochemical properties of Portuguese "Vinho Verde"
/// red wine samples and a quality score for each sample.
///
/// Features (11 total, all `f64`):
///   - fixed acidity
///   - volatile acidity
///   - citric acid
///   - residual sugar
///   - chlorides
///   - free sulfur dioxide
///   - total sulfur dioxide
///   - density
///   - pH
///   - sulphates
///   - alcohol
///
/// Target:
///   - quality (score between 0 and 10, stored as `f64`)
///
/// See more information at <https://archive.ics.uci.edu/dataset/186/wine+quality>
///
/// # Parameters
///
/// - `storage_path` - Directory path where the dataset files will be stored
///
/// # Returns
///
/// - `&Array2<f64>` - Static reference to feature matrix with shape `(1599, 11)`
/// - `&Array1<f64>` - Static reference to target vector with length `1599`
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Download fails due to network issues
/// - File extraction or I/O operations fail
/// - Data format is invalid (wrong number of columns, unparseable values)
/// - Dataset size doesn't match expected dimensions (1599 samples, 11 features)
///
/// # Example
/// ```rust, no_run
/// use rustyml_dataset::wine_quality::load_red_wine_quality;
///
/// let download_dir = "./downloads"; // the code will create this directory if it doesn't exist
///
/// let (features, targets) = load_red_wine_quality(download_dir).unwrap();
///
/// // Use the feature matrix for machine learning
/// assert_eq!(features.shape(), &[1599, 11]);  // 1599 samples, 11 features
/// assert_eq!(targets.len(), 1599); // 1599 samples
///
/// // clean up: remove the downloaded files if they exist
/// if let Ok(entries) = std::fs::read_dir(download_dir) {
///     for entry in entries.flatten() {
///         let _ = std::fs::remove_file(entry.path());
///     }
/// }
/// ```
pub fn load_red_wine_quality(storage_path: &str) -> Result<(&Array2<f64>, &Array1<f64>), DatasetError> {
    // if already initialized
    if let Some(cache) = RED_WINE_DATA.get() {
        return Ok((&cache.0, &cache.1));
    }

    // if not, initialize then store
    let red_wine_arrays = parse_red_wine_data(storage_path)?;
    // Try to set the value. If another thread already set it, that's fine - just use the existing value
    let _ = RED_WINE_DATA.set(red_wine_arrays);

    let cache = RED_WINE_DATA
        .get()
        .expect("RED_WINE_DATA should be initialized after set");
    Ok((&cache.0, &cache.1))
}

/// Loads the Red Wine Quality dataset and returns owned copies.
///
/// Use this function when you need owned data that can be modified.
/// For read-only access, prefer [`load_red_wine_quality()`] which returns references.
///
/// # About Dataset
///
/// The dataset contains physicochemical properties of Portuguese "Vinho Verde"
/// red wine samples and a quality score for each sample.
///
/// Features (11 total, all `f64`):
///   - fixed acidity
///   - volatile acidity
///   - citric acid
///   - residual sugar
///   - chlorides
///   - free sulfur dioxide
///   - total sulfur dioxide
///   - density
///   - pH
///   - sulphates
///   - alcohol
///
/// Target:
///   - quality (score between 0 and 10, stored as `f64`)
///
/// See more information at <https://archive.ics.uci.edu/dataset/186/wine+quality>
///
/// # Parameters
///
/// - `storage_path` - Directory path where the dataset files will be stored
///
/// # Returns
///
/// - `Array2<f64>` - Owned feature matrix with shape `(1599, 11)`
/// - `Array1<f64>` - Owned target vector with length `1599`
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Download fails due to network issues
/// - File extraction or I/O operations fail
/// - Data format is invalid (wrong number of columns, unparseable values)
/// - Dataset size doesn't match expected dimensions (1599 samples, 11 features)
///
/// # Performance
///
/// This function creates owned copies by cloning the cached data, which incurs
/// additional memory allocation. If you only need read-only access, use
/// [`load_red_wine_quality()`] instead for better performance.
///
/// # Examples
/// ```rust, no_run
/// use rustyml_dataset::wine_quality::load_red_wine_quality_owned;
///
/// let download_dir = "./downloads"; // the code will create this directory if it doesn't exist
///
/// let (mut features, targets) = load_red_wine_quality_owned(download_dir).unwrap();
///
/// // Use the feature matrix for machine learning
/// assert_eq!(features.shape(), &[1599, 11]);  // 1599 samples, 11 features
/// assert_eq!(targets.len(), 1599); // 1599 samples
///
/// // Example: Modify feature values (not possible with references)
/// features[[0, 0]] = 10.0;
///
/// // clean up: remove the downloaded files if they exist
/// if let Ok(entries) = std::fs::read_dir(download_dir) {
///     for entry in entries.flatten() {
///         let _ = std::fs::remove_file(entry.path());
///     }
/// }
/// ```
pub fn load_red_wine_quality_owned(storage_path: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
    let (red_wine_features, red_wine_targets) = load_red_wine_quality(storage_path)?;
    Ok((red_wine_features.clone(), red_wine_targets.clone()))
}

/// Loads the White Wine Quality dataset with automatic caching.
///
/// This function returns references to cached data, which incurs no additional
/// memory allocation after the first load. If you need owned data that you can
/// modify, prefer [`load_white_wine_quality_owned()`].
///
/// # About Dataset
///
/// The dataset contains physicochemical properties of Portuguese "Vinho Verde"
/// white wine samples and a quality score for each sample.
///
/// Features (11 total, all `f64`):
///   - fixed acidity
///   - volatile acidity
///   - citric acid
///   - residual sugar
///   - chlorides
///   - free sulfur dioxide
///   - total sulfur dioxide
///   - density
///   - pH
///   - sulphates
///   - alcohol
///
/// Target:
///   - quality (score between 0 and 10, stored as `f64`)
///
/// See more information at <https://archive.ics.uci.edu/dataset/186/wine+quality>
///
/// # Parameters
///
/// - `storage_path` - Directory path where the dataset files will be stored
///
/// # Returns
///
/// - `&Array2<f64>` - Static reference to feature matrix with shape `(4898, 11)`
/// - `&Array1<f64>` - Static reference to target vector with length `4898`
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Download fails due to network issues
/// - File extraction or I/O operations fail
/// - Data format is invalid (wrong number of columns, unparseable values)
/// - Dataset size doesn't match expected dimensions (4898 samples, 11 features)
///
/// # Example
/// ```rust, no_run
/// use rustyml_dataset::wine_quality::load_white_wine_quality;
///
/// let download_dir = "./downloads"; // the code will create this directory if it doesn't exist
///
/// let (features, targets) = load_white_wine_quality(download_dir).unwrap();
///
/// // Use the feature matrix for machine learning
/// assert_eq!(features.shape(), &[4898, 11]);  // 4898 samples, 11 features
/// assert_eq!(targets.len(), 4898); // 4898 samples
///
/// // clean up: remove the downloaded files if they exist
/// if let Ok(entries) = std::fs::read_dir(download_dir) {
///     for entry in entries.flatten() {
///         let _ = std::fs::remove_file(entry.path());
///     }
/// }
/// ```
pub fn load_white_wine_quality(storage_path: &str) -> Result<(&Array2<f64>, &Array1<f64>), DatasetError> {
    // if already initialized
    if let Some(cache) = WHITE_WINE_DATA.get() {
        return Ok((&cache.0, &cache.1));
    }

    // if not, initialize then store
    let white_wine_arrays = parse_white_wine_data(storage_path)?;
    // Try to set the value. If another thread already set it, that's fine - just use the existing value
    let _ = WHITE_WINE_DATA.set(white_wine_arrays);

    let cache = WHITE_WINE_DATA
        .get()
        .expect("WHITE_WINE_DATA should be initialized after set");
    Ok((&cache.0, &cache.1))
}

/// Loads the White Wine Quality dataset and returns owned copies.
///
/// Use this function when you need owned data that can be modified.
/// For read-only access, prefer [`load_white_wine_quality()`] which returns references.
///
/// # Parameters
///
/// - `storage_path` - Directory path where the dataset files will be stored
///
/// # Returns
///
/// - `Array2<f64>` - Owned feature matrix with shape `(4898, 11)`
/// - `Array1<f64>` - Owned target vector with length `4898`
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Download fails due to network issues
/// - File extraction or I/O operations fail
/// - Data format is invalid (wrong number of columns, unparseable values)
/// - Dataset size doesn't match expected dimensions (4898 samples, 11 features)
///
/// # Performance
///
/// This function creates owned copies by cloning the cached data, which incurs
/// additional memory allocation. If you only need read-only access, use
/// [`load_white_wine_quality()`] instead for better performance.
///
/// # Examples
/// ```rust, no_run
/// use rustyml_dataset::wine_quality::load_white_wine_quality_owned;
///
/// let download_dir = "./downloads"; // the code will create this directory if it doesn't exist
///
/// let (mut features, targets) = load_white_wine_quality_owned(download_dir).unwrap();
///
/// // Use the feature matrix for machine learning
/// assert_eq!(features.shape(), &[4898, 11]);  // 4898 samples, 11 features
/// assert_eq!(targets.len(), 4898); // 4898 samples
///
/// // Example: Modify feature values (not possible with references)
/// features[[0, 0]] = 10.0;
///
/// // clean up: remove the downloaded files if they exist
/// if let Ok(entries) = std::fs::read_dir(download_dir) {
///     for entry in entries.flatten() {
///         let _ = std::fs::remove_file(entry.path());
///     }
/// }
/// ```
pub fn load_white_wine_quality_owned(storage_path: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
    let (white_wine_features, white_wine_targets) = load_white_wine_quality(storage_path)?;
    Ok((white_wine_features.clone(), white_wine_targets.clone()))
}
