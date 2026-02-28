use std::fs::{remove_file, rename};
use std::path::Path;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;
use crate::{download_to, DatasetError, create_temp_dir, file_sha256_matches};
use std::fs::File;
use std::io::Read;

/// A static variable to store the Iris dataset.
///
/// This variable is of type `OnceLock`, which ensures thread-safe, one-time initialization
/// of its contents. It contains a tuple of:
///
/// - `Array2<f64>`: A 2-dimensional array representing the numerical features of the dataset(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age).
/// - `Array1<f64>`: A 1-dimensional array containing the corresponding labels (1 for tested positive, 0 for negative).
///
/// The `OnceLock` ensures that the dataset is initialized only once and is then immutable
/// for the lifetime of the program.
static DIABETES_DATA: OnceLock<(Array2<f64>, Array1<f64>)> = OnceLock::new();

/// A static string slice containing the URL for the Diabetes dataset.
///
/// # About Dataset
///
/// This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.
///
/// Features:
/// - Pregnancies: Number of times pregnant
/// - Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance tests
/// - BloodPressure: Diastolic blood pressure (mm Hg)
/// - SkinThickness: Triceps skin fold thickness (mm)
/// - Insulin: 2-Hour serum insulin (mu U/ml)
/// - BMI: Body mass index (weight in kg/(height in m)^2)
/// - DiabetesPedigreeFunction: Diabetes pedigree function
/// - Age: Age (years)
///
/// Labels:
/// - Outcome: Class variable (0 or 1)
///
/// See more information at <https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data>
pub const DIABETES_DATA_URL: &str = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv";

/// A static string slice containing the name of the Diabetes dataset file.
const DIABETES_FILENAME: &str = "diabetes.csv";

/// The number of samples in the Diabetes dataset.
const DIABETES_SAMPLE_SIZE: usize = 768;

/// The number of features in the Diabetes dataset.
const DIABETES_NUM_FEATURES: usize = 8;

/// The prefix for temporary files created during dataset download and parsing.
const DIABETES_TEMP_FILE_PREFIX: &str = ".tmp-diabetes-";

/// The SHA256 hash of the Diabetes dataset file.
const DIABETES_SHA256: &str = "698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72";

/// Downloads, parses, and validates the Diabetes dataset.
///
/// This internal function downloads the dataset CSV into a temporary directory under `path`,
/// moves it to `path/diabetes.csv`, then parses the file into ndarray arrays.
///
/// # Parameters
///
/// - `path` - Directory path where the dataset will be stored
///
/// # Returns
///
/// - `Array2<f64>` - Feature matrix with shape (768, 8)
/// - `Array1<f64>` - Label vector with shape (768,), where values are 0.0 or 1.0
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Download fails due to network issues
/// - Temporary directory creation fails
/// - File move, read, or other I/O operations fail
/// - Data format is invalid (wrong number of columns, unparseable values)
/// - Dataset size doesn't match expected dimensions (768 samples, 8 features)
fn load_diabetes_internal(path: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
    let path = Path::new(path);
    let dst = path.join(DIABETES_FILENAME);
    let mut need_download = true;
    let mut need_overwrite = false;
    // create directory if it doesn't exist
    if !path.exists() {
        std::fs::create_dir_all(path).map_err(|e| DatasetError::StdIoError(e))?;
    } else {
        // check if the file exists and matches the expected SHA256 hash
        if dst.exists() {
            if file_sha256_matches(dst.as_path(), DIABETES_SHA256)? {
                need_download = false;
            } else {
                // if file exists but hash doesn't match, overwrite it
                need_overwrite = true;
            }
        }
    }
    if need_download {
        // temporary directory
        let temp_dir = create_temp_dir(path, DIABETES_TEMP_FILE_PREFIX)?;
        let path_temp = temp_dir.path();
        // download file to temporary directory
        download_to(DIABETES_DATA_URL, path_temp)?;
        // move downloaded file to final location
        let src = path_temp.join(DIABETES_FILENAME);
        // check if the file matches the expected SHA256 hash
        if !file_sha256_matches(src.as_path(), DIABETES_SHA256)? {
            return Err(DatasetError::ValidationError(format!("{} SHA256 validation failed", DIABETES_FILENAME)));
        }
        if need_overwrite {
            remove_file(&dst).map_err(|e| DatasetError::StdIoError(e))?;
        }
        rename(src, &dst).map_err(|e| DatasetError::StdIoError(e))?;
    }

    let mut file = File::open(dst).map_err(|e| DatasetError::StdIoError(e))?;
    let mut data = String::new();
    file.read_to_string(&mut data).map_err(|e| DatasetError::StdIoError(e))?;
    let lines: Vec<&str> = data.trim().lines().collect();

    let mut features = Vec::with_capacity(DIABETES_SAMPLE_SIZE * DIABETES_NUM_FEATURES);
    let mut labels = Vec::with_capacity(DIABETES_SAMPLE_SIZE);

    // Process lines as data (skip header)
    for line in &lines[1..] {
        if line.is_empty() { continue; }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() != DIABETES_NUM_FEATURES + 1 {
            return Err(DatasetError::DataFormatError(
                format!("Expected {} columns, got {} at line {}", 
                        DIABETES_NUM_FEATURES + 1, 
                        cols.len(), 
                        line
                )
            ));
        }
        for i in 0..DIABETES_NUM_FEATURES {
            features.push(cols[i].parse::<f64>().map_err(
                |e| DatasetError::DataFormatError(
                        format!("Failed to parse Diabetes dataset features {} at line {}: {}", i, line, e)
                    ))?);
        }

        labels.push(cols[8].parse::<f64>().map_err(
            |e| DatasetError::DataFormatError(
                    format!("Failed to parse Diabetes label at line {}: {}", line, e)
                )
        )?);
    }
    if features.len() != DIABETES_SAMPLE_SIZE * DIABETES_NUM_FEATURES {
        return Err(DatasetError::DataFormatError(
            format!("Expected {} * {} elements in features, got {} ", 
                    DIABETES_SAMPLE_SIZE, 
                    DIABETES_NUM_FEATURES, 
                    features.len()
            )
        ));
    }
    if labels.len() != DIABETES_SAMPLE_SIZE {
        return Err(DatasetError::DataFormatError(
            format!("Expected {} elements in labels, got {} ", DIABETES_SAMPLE_SIZE, labels.len())
        ));
    }
    let features_array = Array2::from_shape_vec((DIABETES_SAMPLE_SIZE, DIABETES_NUM_FEATURES), features)
        .map_err(|e| DatasetError::DataFormatError(
            format!("Failed to create feature array: {}", e))
        )?;
    let labels_array = Array1::from_vec(labels);

    Ok((features_array, labels_array))
}

/// Loads the Diabetes dataset with automatic caching.
/// 
/// This function returns references to the cached data.
/// If you need owned data, prefer [`load_diabetes_owned()`] which returns owned copies.
///
/// # About Dataset
///
/// This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.
///
/// Features:
/// - Pregnancies: Number of times pregnant
/// - Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance tests
/// - BloodPressure: Diastolic blood pressure (mm Hg)
/// - SkinThickness: Triceps skin fold thickness (mm)
/// - Insulin: 2-Hour serum insulin (mu U/ml)
/// - BMI: Body mass index (weight in kg/(height in m)^2)
/// - DiabetesPedigreeFunction: Diabetes pedigree function
/// - Age: Age (years)
///
/// Labels:
/// - Outcome: Class variable (0 or 1)
///
/// See more information at <https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data>
///
/// # Parameters
///
/// - `storage_path` - Directory path where the dataset will be stored
///
/// # Returns
///
/// - `&Array2<f64>` - Static reference to the feature matrix with shape (768, 8)
/// - `&Array1<f64>` - Static reference to the labels vector with shape (768,)
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Download fails due to network issues
/// - Temporary directory creation fails
/// - File move, read, or other I/O operations fail
/// - Data format is invalid (wrong number of columns, unparseable values)
/// - Dataset size doesn't match expected dimensions (768 samples, 8 features)
///
/// # Examples
/// ```rust, no_run
/// use rustyml_dataset::diabetes::load_diabetes;
///
/// let download_dir = "./downloads"; // the code will create the directory if it doesn't exist
///
/// let (features, labels) = load_diabetes(download_dir).unwrap();
/// assert_eq!(features.shape(), &[768, 8]);
/// assert_eq!(labels.len(), 768);
///
/// // clean up: remove the downloaded files if they exist
/// if let Ok(entries) = std::fs::read_dir(download_dir) {
///     for entry in entries.flatten() {
///         let _ = std::fs::remove_file(entry.path());
///     }
/// }
/// ```
pub fn load_diabetes(storage_path: &str) -> Result<(&Array2<f64>, &Array1<f64>), DatasetError> {
    // if already initialized
    if let Some(cache) = DIABETES_DATA.get() {
        return Ok((&cache.0, &cache.1));
    }
    // if not, initialize then store
    let (features, labels) = load_diabetes_internal(storage_path)?;

    // Try to set the value. If another thread already set it, that's fine - just use the existing value
    let _ = DIABETES_DATA.set((features, labels));

    let cache = DIABETES_DATA
        .get()
        .expect("DIABETES_DATA should be initialized after set");
    Ok((&cache.0, &cache.1))
}

/// Loads the Diabetes dataset and returns owned copies.
///
/// Use this function when you need owned data that can be modified.
/// For read-only access, prefer [`load_diabetes()`] which returns references.
///
/// # About Dataset
///
/// This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.
///
/// Features:
/// - Pregnancies: Number of times pregnant
/// - Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance tests
/// - BloodPressure: Diastolic blood pressure (mm Hg)
/// - SkinThickness: Triceps skin fold thickness (mm)
/// - Insulin: 2-Hour serum insulin (mu U/ml)
/// - BMI: Body mass index (weight in kg/(height in m)^2)
/// - DiabetesPedigreeFunction: Diabetes pedigree function
/// - Age: Age (years)
///
/// Labels:
/// - Outcome: Class variable (0 or 1)
///
/// See more information at <https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data>
///
/// # Parameters
///
/// - `storage_path` - Directory path where the dataset will be stored
///
/// # Returns
///
/// - `Array2<f64>` - Owned feature matrix with shape (768, 8)
/// - `Array1<f64>` - Owned labels vector with shape (768,)
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Download fails due to network issues
/// - Temporary directory creation fails
/// - File move, read, or other I/O operations fail
/// - Data format is invalid (wrong number of columns, unparseable values)
/// - Dataset size doesn't match expected dimensions (768 samples, 8 features)
///
/// # Performance
///
/// This function creates owned copies by cloning the cached data, which incurs additional
/// memory allocation. If you only need read-only access, use [`load_diabetes()`] instead.
///
/// # Examples
/// ```rust, no_run
/// use rustyml_dataset::diabetes::load_diabetes_owned;
///
/// let download_dir = "./downloads"; // the code will create the directory if it doesn't exist
///
/// let (mut features, mut labels) = load_diabetes_owned(download_dir).unwrap();
///
/// assert_eq!(features.shape(), &[768, 8]);
/// assert_eq!(labels.len(), 768);
///
/// // Example: Modify feature values (not possible with references)
/// features[[0, 0]] = 10.0;
/// labels[0] = 1.0;
///
/// // clean up: remove the downloaded files
/// if let Ok(entries) = std::fs::read_dir(download_dir) {
///     for entry in entries.flatten() {
///         let _ = std::fs::remove_file(entry.path());
///     }
/// }
/// ```
pub fn load_diabetes_owned(storage_path: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
    let (features, labels) = load_diabetes(storage_path)?;
    Ok((features.clone(), labels.clone()))
}
