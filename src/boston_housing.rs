use std::fs::{rename, File};
use std::io::Read;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;
use crate::{DatasetError, create_temp_dir, download_to, unzip};
use std::path::Path;

/// A static variable to store the Boston Housing dataset.
///
/// This variable is of type [`OnceLock`], which ensures thread-safe, one-time initialization
/// of its contents. It contains a tuple of:
///
/// - `Array2<f64>`: A 2-dimensional array representing the numerical features of the dataset.
/// - `Array1<f64>`: A 1-dimensional array containing the corresponding target values (median home values in $1000s).
///
/// The `OnceLock` ensures that the dataset is initialized only once and is then immutable
/// for the lifetime of the program.
static BOSTON_HOUSING_DATA: OnceLock<(Array2<f64>, Array1<f64>)> = OnceLock::new();

/// A static string slice containing the URL for the Boston Housing dataset.
///
/// # About Dataset
///
/// The Boston Housing Dataset is a derived from information collected by the U.S. Census Service concerning housing in the area of Boston MA.
///
/// Features:
/// - CRIM - per capita crime rate by town
/// - ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
/// - INDUS - proportion of non-retail business acres per town.
/// - CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
/// - NOX - nitric oxides concentration (parts per 10 million)
/// - RM - average number of rooms per dwelling
/// - AGE - proportion of owner-occupied units built prior to 1940
/// - DIS - weighted distances to five Boston employment centres
/// - RAD - index of accessibility to radial highways
/// - TAX - full-value property-tax rate per $10,000
/// - PTRATIO - pupil-teacher ratio by town
/// - B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
/// - LSTAT - % lower status of the population
///
/// Targets:
/// - MEDV - Median value of owner-occupied homes in $1000's
pub const BOSTON_HOUSING_DATA_URL: &str = "https://gist.github.com/nnbphuong/def91b5553736764e8e08f6255390f37/archive/373a856a3c9c1119e34b344de9230ae2ea89569d.zip";

/// The prefix for temporary files used during dataset download and extraction
const BOSTON_HOUSING_TEMP_FILE_PREFIX: &str = ".tmp-boston-housing-";

/// The downloaded zip file name
const BOSTON_HOUSING_ZIP_FILENAME: &str = "373a856a3c9c1119e34b344de9230ae2ea89569d.zip";

/// The folder where the file is located inside after extraction
const BOSTON_HOUSING_UNZIP_FOLDER: &str = "def91b5553736764e8e08f6255390f37-373a856a3c9c1119e34b344de9230ae2ea89569d";

/// The name of the file inside the extracted folder
const BOSTON_HOUSING_FILENAME: &str = "BostonHousing.csv";

/// The number of samples in the dataset
const BOSTON_HOUSING_SAMPLE_SIZE: usize = 506;

/// The number of features in the dataset
const BOSTON_HOUSING_NUM_FEATURES: usize = 13;

/// Internal function to download, parse, and validate the Boston Housing dataset.
///
/// This function downloads the dataset archive into a temporary directory under `path`,
/// extracts it, moves the CSV file into `path`, and parses it into ndarray arrays.
///
/// # Parameters
///
/// - `path` - Directory path where the dataset will be stored.
///
/// # Returns
///
/// - `Array2<f64>` - Feature matrix with shape `(506, 13)`.
/// - `Array1<f64>` - Target vector with shape `(506,)` (MEDV in $1000s).
///
/// # Errors
///
/// - `DatasetError` - Returned if the download fails, extraction fails, I/O fails, or the CSV
///   content is not in the expected format/dimensions.
fn load_boston_housing_internal(path: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
    let path = Path::new(path);
    if !path.exists() {
        std::fs::create_dir_all(path).map_err(|e| DatasetError::StdIoError(e))?;
    }
    // temporary directory to store the downloaded zip file
    let temp_dir = create_temp_dir(path, BOSTON_HOUSING_TEMP_FILE_PREFIX)?;
    let path_temp = temp_dir.path();
    // download and extract boston housing dataset
    download_to(BOSTON_HOUSING_DATA_URL, path_temp)?;
    unzip(&path_temp.join(BOSTON_HOUSING_ZIP_FILENAME), path_temp)?;
    // move boston_housing.csv out of the temporary directory
    let src = path_temp.join(BOSTON_HOUSING_UNZIP_FOLDER).join( BOSTON_HOUSING_FILENAME);
    let dst = path.join(BOSTON_HOUSING_FILENAME);
    // cover the existing file (if any) with the new one
    if dst.exists() {
        std::fs::remove_file(&dst).map_err(|e| DatasetError::StdIoError(e))?;
    }
    rename(src, &dst).map_err(|e| DatasetError::StdIoError(e))?;

    let mut file = File::open(dst).map_err(|e| DatasetError::StdIoError(e))?;
    let mut data = String::new();
    file.read_to_string(&mut data).map_err(|e| DatasetError::StdIoError(e))?;

    let mut features = Vec::with_capacity(BOSTON_HOUSING_SAMPLE_SIZE * BOSTON_HOUSING_NUM_FEATURES);
    let mut targets = Vec::with_capacity(BOSTON_HOUSING_SAMPLE_SIZE);

    let lines: Vec<&str> = data.trim().lines().collect();

    for line in &lines[1..] {
        if line.is_empty() { continue; }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < BOSTON_HOUSING_NUM_FEATURES + 1 {
            return Err(DatasetError::DataFormatError(
                format!("Number of columns should be at least {}, got {} at line {}",
                        BOSTON_HOUSING_NUM_FEATURES + 1,
                        cols.len(),
                        line)
            ))
        }
        // Features are columns 0-12 (13 features)
        for i in 0..BOSTON_HOUSING_NUM_FEATURES {
            features.push(cols[i].parse::<f64>().map_err(
                |e| DatasetError::DataFormatError(
                format!("Failed to parse Boston Housing dataset features {} at line {}: {}", i, line, e)
            ))?);
        }

        // Target is column 13 (MEDV)
        targets.push(cols[13].parse::<f64>().map_err(
            |e| DatasetError::DataFormatError(
            format!("Failed to parse Boston Housing dataset target at line {}: {}", line, e)
        ))?);
    }
    
    if features.len() != BOSTON_HOUSING_SAMPLE_SIZE * BOSTON_HOUSING_NUM_FEATURES {
        return Err(DatasetError::DataFormatError(
            format!("Expected {} * {} elements in features, got {}", BOSTON_HOUSING_SAMPLE_SIZE,
                    BOSTON_HOUSING_NUM_FEATURES,
                    features.len())
        ))
    }
    if targets.len() != BOSTON_HOUSING_SAMPLE_SIZE {
        return Err(DatasetError::DataFormatError(
            format!("Expected {} elements in targets, got {}", BOSTON_HOUSING_SAMPLE_SIZE, targets.len())
        ))
    }

    let features_array = Array2::from_shape_vec((BOSTON_HOUSING_SAMPLE_SIZE, BOSTON_HOUSING_NUM_FEATURES), features)
        .map_err(|e| DatasetError::DataFormatError(
            format!("Failed to create features array: {}", e)
        ))?;
    let targets_array = Array1::from_vec(targets);

    Ok((features_array, targets_array))
}

/// Load the Boston Housing dataset with automatic caching.
///
/// This function returns references to the cached dataset.
/// If you need owned data, prefer [`load_boston_housing_owned`] which returns owned copies.
///
/// # About Dataset
///
/// The Boston Housing Dataset is a derived from information collected by the U.S. Census Service concerning housing in the area of Boston MA.
///
/// Features:
/// - CRIM - per capita crime rate by town
/// - ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
/// - INDUS - proportion of non-retail business acres per town.
/// - CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
/// - NOX - nitric oxides concentration (parts per 10 million)
/// - RM - average number of rooms per dwelling
/// - AGE - proportion of owner-occupied units built prior to 1940
/// - DIS - weighted distances to five Boston employment centres
/// - RAD - index of accessibility to radial highways
/// - TAX - full-value property-tax rate per $10,000
/// - PTRATIO - pupil-teacher ratio by town
/// - B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
/// - LSTAT - % lower status of the population
///
/// Targets:
/// - MEDV - Median value of owner-occupied homes in $1000's
///
/// # Parameters
///
/// - `storage_path` - Directory path where the dataset will be stored.
///
/// # Returns
///
/// - `&Array2<f64>` - Feature matrix with shape `(506, 13)`.
/// - `&Array1<f64>` - Target vector with shape `(506,)` (MEDV in $1000s).
///
/// # Errors
///
/// - `DatasetError` - Returned if the download fails, extraction fails, I/O fails, or the CSV
///   content is not in the expected format/dimensions.
///
/// # Examples
/// ```rust, no_run
/// use rustyml_dataset::boston_housing::load_boston_housing;
///
/// let download_dir = "./downloads"; // the code will create the directory if it doesn't exist
/// let (features, targets) = load_boston_housing(download_dir).unwrap();
///
/// assert_eq!(features.shape(), &[506, 13]);
/// assert_eq!(targets.len(), 506);
///
/// // clean up: remove the downloaded files
/// if let Ok(entries) = std::fs::read_dir(download_dir) {
///     for entry in entries.flatten() {
///         let _ = std::fs::remove_file(entry.path());
///     }
/// }
/// ```
pub fn load_boston_housing(storage_path: &str) -> Result<(&Array2<f64>, &Array1<f64>), DatasetError> {
    // if already initialized
    if let Some(cache) = BOSTON_HOUSING_DATA.get() {
        return Ok((&cache.0, &cache.1));
    }
    // if not, initialize then store
    let (features, targets) = load_boston_housing_internal(storage_path)?;

    // Try to set the value. If another thread already set it, that's fine - just use the existing value
    let _ = BOSTON_HOUSING_DATA.set((features, targets));

    let cache = BOSTON_HOUSING_DATA
        .get()
        .expect("BOSTON_HOUSING_DATA should be initialized after set");
    Ok((&cache.0, &cache.1))
}

/// Load the Boston Housing dataset and return owned copies.
///
/// Use this function when you need owned arrays that can be modified.
/// For read-only access, prefer [`load_boston_housing`] which returns references.
///
/// # About Dataset
///
/// The Boston Housing Dataset is a derived from information collected by the U.S. Census Service concerning housing in the area of Boston MA.
///
/// Features:
/// - CRIM - per capita crime rate by town
/// - ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
/// - INDUS - proportion of non-retail business acres per town.
/// - CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
/// - NOX - nitric oxides concentration (parts per 10 million)
/// - RM - average number of rooms per dwelling
/// - AGE - proportion of owner-occupied units built prior to 1940
/// - DIS - weighted distances to five Boston employment centres
/// - RAD - index of accessibility to radial highways
/// - TAX - full-value property-tax rate per $10,000
/// - PTRATIO - pupil-teacher ratio by town
/// - B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
/// - LSTAT - % lower status of the population
///
/// Targets:
/// - MEDV - Median value of owner-occupied homes in $1000's
///
/// # Parameters
///
/// - `storage_path` - Directory path where the dataset will be stored.
///
/// # Returns
///
/// - `Array2<f64>` - Owned feature matrix with shape `(506, 13)`.
/// - `Array1<f64>` - Owned target vector with shape `(506,)` (MEDV in $1000s).
///
/// # Errors
///
/// - `DatasetError` - Returned if the download fails, extraction fails, I/O fails, or the CSV
///   content is not in the expected format/dimensions.
///
/// # Performance
///
/// This function returns owned arrays and may allocate additional memory.
/// If you only need read-only access, prefer [`load_boston_housing`].
///
/// # Examples
/// ```rust, no_run
/// use rustyml_dataset::boston_housing::load_boston_housing_owned;
///
/// let download_dir = "./downloads"; // the code will create the directory if it doesn't exist
/// let (mut features, mut targets) = load_boston_housing_owned(download_dir).unwrap();
///
/// assert_eq!(features.shape(), &[506, 13]);
/// assert_eq!(targets.len(), 506);
///
/// // Owned arrays can be modified.
/// features[[0, 0]] = 0.1;
/// targets[0] = 25.5;
///
/// // clean up: remove the downloaded files
/// if let Ok(entries) = std::fs::read_dir(download_dir) {
///     for entry in entries.flatten() {
///         let _ = std::fs::remove_file(entry.path());
///     }
/// }
/// ```
pub fn load_boston_housing_owned(storage_path: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
    let (features, targets) = load_boston_housing(storage_path)?;
    Ok((features.clone(), targets.clone()))
}
