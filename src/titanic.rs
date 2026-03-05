use std::fs::{remove_file, rename, File};
use std::io::Read;
use std::path::Path;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;
use crate::{DatasetError, download_to, file_sha256_matches, prepare_download_dir};

/// A static variable to store the Titanic dataset.
///
/// This variable is of type `OnceLock`, which ensures thread-safe, one-time initialization
/// of its contents. It contains a tuple of:
///
/// - `Array2<String>` - String features matrix with shape `(891, 5)`, columns: `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked` (empty string if missing in source)
/// - `Array2<f64>` - Numeric features matrix with shape `(891, 6)`, columns: `PassengerId`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare` (`NaN` if missing in source)
/// - `Array1<f64>` - Label vector with shape `(891,)`, values are `Survived` (`0.0`/`1.0`, `NaN` if missing in source)
///
/// The `OnceLock` ensures that the dataset is initialized only once and is then immutable
/// for the lifetime of the program.
static TITANIC_DATA: OnceLock<(
    Array2<String>,
    Array2<f64>,
    Array1<f64>,
)> = OnceLock::new();

/// A static string slice containing the URL of the Titanic dataset
///
/// # About Dataset
///
/// On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg.
/// Unfortunately, there weren’t enough lifeboats for everyone on board, resulting in the death of 1502 out of 2224 passengers and crew.
/// While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
///
/// Features:
/// - PassengerId - Passenger ID
/// - Pclass - Ticket class: 1 = 1st, 2 = 2nd, 3 = 3rd
/// - Name - Name of the Passenger
/// - Sex - Gender of the Passenger: male or female
/// - Age - Age in Years
/// - SibSp - No. of siblings / spouses aboard the Titanic
/// - Parch - No. of parents / children aboard the Titanic
/// - Ticket - Ticket number
/// - Fare - Passenger fare
/// - Cabin - Cabin number
/// - Embarked - Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton
///
/// Missing values:
/// - Numeric fields are parsed as `NaN` when missing.
/// - String fields are parsed as empty strings when missing.
///
/// Labels:
/// - Survived - Whether the passenger survived: 0 = No, 1 = Yes
///
/// See more information at <https://www.kaggle.com/c/titanic/data>.
const TITANIC_DATA_URL: &str = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv";

/// The prefix for temporary files created during dataset download and parsing.
const TITANIC_TEMP_FILE_PREFIX: &str = ".tmp-titanic-";

/// A static string slice containing the name of the Titanic dataset file.
const TITANIC_FILENAME: &str = "titanic.csv";

/// The number of samples in the Titanic dataset.
const TITANIC_SAMPLE_SIZE: usize = 891;

/// The number of string features in the Titanic dataset.
const TITANIC_NUM_STRING_FEATURES: usize = 5;

/// The number of numeric features in the Titanic dataset.
const TITANIC_NUM_NUMERIC_FEATURES: usize = 6;

/// The SHA256 hash of the Titanic dataset file.
const TITANIC_SHA256: &str = "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7";

/// Parses a CSV line, correctly handling quoted fields that may contain commas.
///
/// This function splits a CSV line by commas, but treats commas inside double quotes
/// as part of the field content rather than field separators.
///
/// # Parameters
///
/// - `line` - A line from a CSV file
///
/// # Returns
///
/// A vector of strings, one for each field. Quoted fields have their quotes removed.
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current_field = String::new();
    let mut inside_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                inside_quotes = !inside_quotes;
            }
            ',' if !inside_quotes => {
                fields.push(current_field.clone());
                current_field.clear();
            }
            _ => {
                current_field.push(ch);
            }
        }
    }
    // Push the last field
    fields.push(current_field);

    fields
}

/// Downloads, parses, and validates the Titanic dataset.
///
/// This internal function downloads the dataset CSV into a temporary directory under `path`,
/// moves it to `path/titanic.csv`, then parses the file into ndarray arrays.
///
/// # Parameters
///
/// - `path` - Directory path where the dataset will be stored
///
/// # Returns
///
/// - `Array2<String>` - String features matrix with shape `(891, 5)`, columns: `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked` (empty string if missing in source)
/// - `Array2<f64>` - Numeric features matrix with shape `(891, 6)`, columns: `PassengerId`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare` (`NaN` if missing in source)
/// - `Array1<f64>` - Label vector with shape `(891,)`, values are `Survived` (`0.0`/`1.0`, `NaN` if missing in source)
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Download fails due to network issues
/// - Temporary directory creation fails
/// - File move, read, or other I/O operations fail
/// - Data format is invalid (wrong number of columns, unparseable values)
/// - Dataset size doesn't match expected dimensions (891 samples)
fn load_titanic_internal(path: &str) -> Result<(Array2<String>, Array2<f64>, Array1<f64>), DatasetError> {
    // the path the user wants dataset to be stored in
    let path = Path::new(path);
    let dst = path.join(TITANIC_FILENAME);
    let (need_download, need_overwrite) =
        prepare_download_dir(path, &dst, TITANIC_SHA256)?;
    if need_download {
        // temporary directory to store the downloaded zip file
        let temp_dir = crate::create_temp_dir(path, TITANIC_TEMP_FILE_PREFIX)?;
        let path_temp = temp_dir.path();
        // download and extract titanic dataset
        download_to(TITANIC_DATA_URL, path_temp)?;
        // move downloaded file to final location
        let src = path_temp.join(TITANIC_FILENAME);
        if !file_sha256_matches(src.as_path(), TITANIC_SHA256)? {
            return Err(DatasetError::ValidationError(
                format!("{} SHA256 validation failed", TITANIC_SHA256)
            ));
        }
        if need_overwrite {
            remove_file(&dst).map_err(|e| DatasetError::StdIoError(e))?;
        }
        rename(src, &dst).map_err(|e| DatasetError::StdIoError(e))?;
    }

    let mut data = String::new();
    let mut raw_data = File::open(dst).map_err(|e| DatasetError::StdIoError(e))?;
    raw_data.read_to_string(&mut data).map_err(|e| DatasetError::StdIoError(e))?;

    let lines: Vec<&str> = data.trim().lines().collect();

    let mut string_features = Vec::with_capacity(TITANIC_SAMPLE_SIZE * 5);
    let mut numeric_features = Vec::with_capacity(TITANIC_SAMPLE_SIZE * 6);
    let mut labels = Vec::with_capacity(TITANIC_SAMPLE_SIZE);

    // Process lines as data (skip header)
    for line in &lines[1..] {
        if line.is_empty() { continue; }
        let cols = parse_csv_line(line);
        if cols.len() != TITANIC_NUM_STRING_FEATURES + TITANIC_NUM_NUMERIC_FEATURES + 1 {
            return Err(DatasetError::DataFormatError(
                format!("Expected {} columns, got {} at line: {}", 
                        TITANIC_NUM_STRING_FEATURES + TITANIC_NUM_NUMERIC_FEATURES + 1, 
                        cols.len(), 
                        line)
            ));
        }

        // Parse Survived (label) - index 1
        labels.push(
            if cols[1].is_empty() {
                f64::NAN
            } else {
                cols[1].parse::<f64>().map_err(
                    |e| DatasetError::DataFormatError(
                        format!("Failed to parse Survived at line {}: {}", line, e)
                    )
                )?
            }
        );

        // Parse PassengerId - index 0
        numeric_features.push(
            if cols[0].is_empty() {
                f64::NAN
            } else {
                cols[0].parse::<f64>().map_err(
                    |e| DatasetError::DataFormatError(
                        format!("Failed to parse PassengerId at line {}: {}", line, e)
                    )
                )?
            }
        );

        // Parse Pclass - index 2
        numeric_features.push(
            if cols[2].is_empty() {
                f64::NAN
            } else {
                cols[2].parse::<f64>().map_err(
                    |e| DatasetError::DataFormatError(
                        format!("Failed to parse Pclass at line {}: {}", line, e)
                    )
                )?
            }
        );

        // Parse Age - index 5
        numeric_features.push(
            if cols[5].is_empty() {
                f64::NAN
            } else {
                cols[5].parse::<f64>().map_err(
                    |e| DatasetError::DataFormatError(
                        format!("Failed to parse Age at line {}: {}", line, e)
                    )
                )?
            }
        );

        // Parse SibSp - index 6
        numeric_features.push(
            if cols[6].is_empty() {
                f64::NAN
            } else {
                cols[6].parse::<f64>().map_err(
                    |e| DatasetError::DataFormatError(
                        format!("Failed to parse SibSp at line {}: {}", line, e)
                    )
                )?
            }
        );

        // Parse Parch - index 7
        numeric_features.push(
            if cols[7].is_empty() {
                f64::NAN
            } else {
                cols[7].parse::<f64>().map_err(
                    |e| DatasetError::DataFormatError(
                        format!("Failed to parse Parch at line {}: {}", line, e)
                    )
                )?
            }
        );

        // Parse Fare - index 9
        numeric_features.push(
            if cols[9].is_empty() {
                f64::NAN
            } else {
                cols[9].parse::<f64>().map_err(
                    |e| DatasetError::DataFormatError(
                        format!("Failed to parse Fare at line {}: {}", line, e)
                    )
                )?
            }
        );

        // String features: Name, Sex, Ticket, Cabin, Embarked
        string_features.push(cols[3].clone()); // Name - index 3
        string_features.push(cols[4].clone()); // Sex - index 4
        string_features.push(cols[8].clone()); // Ticket - index 8
        string_features.push(cols[10].clone()); // Cabin - index 10
        string_features.push(cols[11].clone()); // Embarked - index 11
    }

    if labels.len() != TITANIC_SAMPLE_SIZE {
        return Err(DatasetError::DataFormatError(
            format!("Expected {} rows, got {}", TITANIC_SAMPLE_SIZE, labels.len())
        ));
    }
    if numeric_features.len() != TITANIC_SAMPLE_SIZE * TITANIC_NUM_NUMERIC_FEATURES {
        return Err(DatasetError::DataFormatError(
            format!("Expected {} * {} elements in numeric features, got {}", 
                    TITANIC_SAMPLE_SIZE, 
                    TITANIC_NUM_NUMERIC_FEATURES, 
                    numeric_features.len()
            )
        ));
    }
    if string_features.len() != TITANIC_SAMPLE_SIZE * TITANIC_NUM_STRING_FEATURES {
        return Err(DatasetError::DataFormatError(
            format!("Expected {} * {} elements in string features, got {}", 
                    TITANIC_SAMPLE_SIZE, 
                    TITANIC_NUM_STRING_FEATURES, 
                    string_features.len()
            )
        ));
    }

    let string_array = Array2::from_shape_vec((TITANIC_SAMPLE_SIZE, TITANIC_NUM_STRING_FEATURES), string_features)
        .map_err(|e| DatasetError::DataFormatError(
            format!("Failed to create string feature array: {}", e)
        ))?;

    let numeric_array = Array2::from_shape_vec((TITANIC_SAMPLE_SIZE, TITANIC_NUM_NUMERIC_FEATURES), numeric_features)
        .map_err(|e| DatasetError::DataFormatError(
            format!("Failed to create numeric feature array: {}", e)
        ))?;

    let labels_array = Array1::from_vec(labels);

    Ok((string_array, numeric_array, labels_array))
}

/// Loads the Titanic dataset with automatic caching.
///
/// This function returns references to the cached data stored in TITANIC_DATA.
/// On the first call, it downloads and parses the dataset, then memoizes it. Subsequent
/// calls are fast and allocation-free.
///
/// If you need owned data that you can modify, prefer [`load_titanic_owned()`].
///
/// # About Dataset
///
/// The Titanic dataset contains passenger information and a binary survival label.
/// It is commonly used for binary classification and feature engineering exercises.
///
/// String features (shape `(891, 5)`), in column order:
/// - `Name`
/// - `Sex`
/// - `Ticket`
/// - `Cabin`
/// - `Embarked`
///
/// Numeric features (shape `(891, 6)`), in column order (may be `NaN` if missing in the source):
/// - `PassengerId`
/// - `Pclass`
/// - `Age`
/// - `SibSp`
/// - `Parch`
/// - `Fare`
///
/// Labels (shape `(891,)`):
/// - `Survived` (`0.0` or `1.0`; `NaN` if missing in source)
///
/// See more information at <https://www.kaggle.com/c/titanic/data>.
///
/// # Parameters
///
/// - `storage_path` - Directory path where the dataset file (`titanic.csv`) will be stored.
///
/// # Returns
///
/// - `&Array2<String>` - String feature matrix with shape `(891, 5)` (empty string if missing in source)
/// - `&Array2<f64>` - Numeric feature matrix with shape `(891, 6)` (`NaN` if missing in source)
/// - `&Array1<f64>` - Labels vector with shape `(891,)` (`0.0`/`1.0`, `NaN` if missing in source)
///
/// # Errors
///
/// Returns [`DatasetError`] if:
/// - Download fails due to network issues or invalid URL
/// - Temporary directory creation fails
/// - File move, read, or other I/O operations fail
/// - Data format is invalid (wrong number of columns, unparseable values)
/// - Dataset size doesn't match expected dimensions (891 samples)
///
/// # Examples
///
/// ```rust, no_run
/// use rustyml_dataset::titanic::load_titanic;
///
/// let download_dir = "./downloads"; // the code will create the directory if it doesn't exist
///
/// let (string_features, num_features, labels) = load_titanic(download_dir).unwrap();
///
/// assert_eq!(string_features.shape(), &[891, 5]); // 891 samples, 5 features
/// assert_eq!(num_features.shape(), &[891, 6]); // 891 samples, 6 features
/// assert_eq!(labels.len(), 891); // 891 samples
///
/// // clean up: remove the downloaded files
/// if let Ok(entries) = std::fs::read_dir(download_dir) {
///     for entry in entries.flatten() {
///         let _ = std::fs::remove_file(entry.path());
///     }
/// }
/// ```
pub fn load_titanic(storage_path: &str) -> Result<
    (&Array2<String>, &Array2<f64>, &Array1<f64>)
    , DatasetError> {
    // if already initialized
    if let Some(cache) = TITANIC_DATA.get() {
        return Ok((&cache.0, &cache.1, &cache.2));
    }
    // if not, initialize then store
    let (string_features,
        num_features,
        labels) = load_titanic_internal(storage_path)?;

    // Try to set value. If another thread already set it, that's fine - just use the existing value
    let _ = TITANIC_DATA.set((string_features, num_features, labels));
    let cache = TITANIC_DATA
        .get()
        .expect("TITANIC_DATA should be initialized after set");
    Ok((&cache.0, &cache.1, &cache.2))
}

/// Loads the Titanic dataset and returns owned copies.
///
/// Use this function when you need owned data that can be modified independently.
/// For read-only access with zero extra allocation, prefer [`load_titanic()`] which returns
/// references to cached arrays.
///
/// # About Dataset
///
/// The Titanic dataset contains passenger information and a binary survival label.
/// It is commonly used for binary classification and feature engineering exercises.
///
/// String features (shape `(891, 5)`), in column order:
/// - `Name`
/// - `Sex`
/// - `Ticket`
/// - `Cabin`
/// - `Embarked`
///
/// Numeric features (shape `(891, 6)`), in column order (may be `NaN` if missing in the source):
/// - `PassengerId`
/// - `Pclass`
/// - `Age`
/// - `SibSp`
/// - `Parch`
/// - `Fare`
///
/// Labels (shape `(891,)`):
/// - `Survived` (`0.0` or `1.0`; `NaN` if missing in source)
///
/// See more information at <https://www.kaggle.com/c/titanic/data>.
///
/// # Parameters
///
/// - `storage_path` - Directory path where the dataset file (`titanic.csv`) will be stored.
///
/// # Returns
///
/// - `Array2<String>` - Owned string feature matrix with shape `(891, 5)` (empty string if missing in source)
/// - `Array2<f64>` - Owned numeric feature matrix with shape `(891, 6)` (`NaN` if missing in source)
/// - `Array1<f64>` - Owned labels vector with shape `(891,)` (`0.0`/`1.0`, `NaN` if missing in source)
///
/// # Errors
///
/// Returns [`DatasetError`] if:
/// - Download fails due to network issues or invalid URL
/// - Temporary directory creation fails
/// - File move, read, or other I/O operations fail
/// - Data format is invalid (wrong number of columns, unparseable values)
/// - Dataset size doesn't match expected dimensions (891 samples)
///
/// # Performance
///
/// This function clones the cached arrays, which incurs additional memory allocation.
/// If you don't need mutation, use [`load_titanic()`] for better performance.
///
/// # Examples
/// ```rust, no_run
/// use rustyml_dataset::titanic::load_titanic_owned;
///
/// let download_dir = "./downloads"; // the code will create the directory if it doesn't exist
///
/// let (mut string_features, mut num_features, mut labels) = load_titanic_owned(download_dir).unwrap();
///
/// assert_eq!(string_features.nrows(), 891); // 891 samples
/// assert_eq!(string_features.ncols(), 5); // 5 features
/// assert_eq!(num_features.nrows(), 891); // 891 samples
/// assert_eq!(num_features.ncols(), 6); // 6 features
/// assert_eq!(labels.len(), 891); // 891 samples
///
/// // modify the data (not possible with references)
/// num_features.mapv_inplace(|x| {
///     if x.is_nan() { 0.0 } else { x }
/// });
///
/// // clean up: remove the downloaded files
/// if let Ok(entries) = std::fs::read_dir(download_dir) {
///     for entry in entries.flatten() {
///         let _ = std::fs::remove_file(entry.path());
///     }
/// }
/// ```
pub fn load_titanic_owned(storage_path: &str) -> Result<
    (Array2<String>, Array2<f64>, Array1<f64>)
    , DatasetError> {
    let (string_features,
        num_features,
        labels) = load_titanic(storage_path)?;

    Ok((string_features.clone(), num_features.clone(), labels.clone()))
}
