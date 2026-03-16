use crate::{DatasetError, download_to, file_sha256_matches, prepare_download_dir};
use ndarray::{Array1, Array2};
use std::fs::{File, remove_file, rename};
use csv::ReaderBuilder;
use std::path::Path;
use std::sync::OnceLock;

/// The URL for the Titanic dataset.
const TITANIC_DATA_URL: &str =
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv";

/// The prefix for temporary files created during dataset download and parsing.
const TITANIC_TEMP_FILE_PREFIX: &str = ".tmp-titanic-";

/// The name of the Titanic dataset file.
const TITANIC_FILENAME: &str = "titanic.csv";

/// The number of samples in the Titanic dataset.
const TITANIC_SAMPLE_SIZE: usize = 891;

/// The number of string features in the Titanic dataset.
const TITANIC_NUM_STRING_FEATURES: usize = 5;

/// The number of numeric features in the Titanic dataset.
const TITANIC_NUM_NUMERIC_FEATURES: usize = 6;

/// The SHA256 hash of the Titanic dataset file.
const TITANIC_SHA256: &str = "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7";
const TITANIC_DATASET_NAME: &str = "titanic";

/// A struct representing the Titanic dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// On April 15, 1912, during her maiden voyage, the widely considered "unsinkable" RMS Titanic sank after colliding with an iceberg.
/// Unfortunately, there weren't enough lifeboats for everyone on board, resulting in the death of 1502 out of 2224 passengers and crew.
/// While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
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
/// Missing values:
/// - Numeric fields are parsed as `NaN` when missing.
/// - String fields are parsed as empty strings when missing.
///
/// See more information at <https://www.kaggle.com/c/titanic/data>.
///
/// # Fields
///
/// - `storage_path` - Directory path where the dataset will be stored.
/// - `data` - Cached data as a tuple of `Array2<String>`, `Array2<f64>` and `Array1<f64>`. (`OnceLock` is used to ensure thread-safety)
///
/// # Example
/// ```rust
/// use rustyml_dataset::titanic::Titanic;
///
/// let download_dir = "./titanic"; // the code will create the directory if it doesn't exist
///
/// let dataset = Titanic::new(download_dir);
/// let (string_features, numeric_features) = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (string_features, numeric_features, labels) = dataset.data().unwrap(); // this is also a way to get all data
/// // you can use `.to_owned()` to get owned copies of the data
/// let mut string_features_owned = string_features.to_owned();
/// let mut numeric_features_owned = numeric_features.to_owned();
/// let mut labels_owned = labels.to_owned();
///
/// // Example: Modify feature values
/// numeric_features_owned[[0, 0]] = 1.0;
/// labels_owned[0] = 1.0;
///
/// assert_eq!(string_features.shape(), &[891, 5]);
/// assert_eq!(numeric_features.shape(), &[891, 6]);
/// assert_eq!(labels.len(), 891);
///
/// // clean up: remove the downloaded files
/// std::fs::remove_dir_all(download_dir).unwrap();
/// ```
pub struct Titanic {
    storage_path: String,
    data: OnceLock<(Array2<String>, Array2<f64>, Array1<f64>)>,
}

impl Titanic {
    /// Create a new Titanic instance without loading data.
    ///
    /// The dataset will be loaded lazily when you first call any data accessor method.
    /// This is a lightweight operation that only stores the storage path.
    ///
    /// # Parameters
    ///
    /// - `storage_path` - Directory path where the dataset will be stored.
    ///
    /// # Returns
    ///
    /// - `Self` - `Titanic` instance ready for lazy loading.
    pub fn new(storage_path: &str) -> Self {
        Titanic {
            storage_path: storage_path.to_string(),
            data: OnceLock::new(),
        }
    }

    /// Internal function to load the dataset from disk or download it.
    ///
    /// This function is called automatically by the accessor methods.
    fn load_data_internal(
        path: &str,
    ) -> Result<(Array2<String>, Array2<f64>, Array1<f64>), DatasetError> {
        // the path the user wants dataset to be stored in
        let path = Path::new(path);
        let dst = path.join(TITANIC_FILENAME);
        let (need_download, need_overwrite) = prepare_download_dir(path, &dst, TITANIC_SHA256)?;
        if need_download {
            // temporary directory to store the downloaded zip file
            let temp_dir = crate::create_temp_dir(path, TITANIC_TEMP_FILE_PREFIX)?;
            let path_temp = temp_dir.path();
            // download and extract titanic dataset
            download_to(TITANIC_DATA_URL, path_temp)?;
            // move downloaded file to final location
            let src = path_temp.join(TITANIC_FILENAME);
            if !file_sha256_matches(src.as_path(), TITANIC_SHA256)? {
                return Err(DatasetError::sha256_validation_failed(TITANIC_FILENAME));
            }
            if need_overwrite {
                remove_file(&dst).map_err(DatasetError::io)?;
            }
            rename(src, &dst).map_err(DatasetError::io)?;
        }

        let file = File::open(&dst).map_err(DatasetError::io)?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        let mut string_features = Vec::with_capacity(TITANIC_SAMPLE_SIZE * TITANIC_NUM_STRING_FEATURES);
        let mut numeric_features = Vec::with_capacity(TITANIC_SAMPLE_SIZE * TITANIC_NUM_NUMERIC_FEATURES);
        let mut labels = Vec::with_capacity(TITANIC_SAMPLE_SIZE);

        // CSV columns: PassengerId(0), Survived(1), Pclass(2), Name(3), Sex(4), Age(5), SibSp(6), Parch(7), Ticket(8), Fare(9), Cabin(10), Embarked(11)

        for result in rdr.records() {
            let record = result.map_err(|e| {
                DatasetError::data_format(format!(
                    "[{}] failed to read CSV record: {}",
                    TITANIC_DATASET_NAME, e
                ))
            })?;

            if record.len() != TITANIC_NUM_STRING_FEATURES + TITANIC_NUM_NUMERIC_FEATURES + 1 {
                return Err(DatasetError::invalid_column_count(
                    TITANIC_DATASET_NAME,
                    TITANIC_NUM_STRING_FEATURES + TITANIC_NUM_NUMERIC_FEATURES + 1,
                    record.len(),
                    &format!("{:?}", record),
                ));
            }

            // Helper closure: parse a numeric field, returning NaN for empty strings
            let parse_numeric = |index: usize, field_name: &str| -> Result<f64, DatasetError> {
                let val = record[index].trim();
                if val.is_empty() {
                    Ok(f64::NAN)
                } else {
                    val.parse::<f64>().map_err(|e| {
                        DatasetError::parse_failed(
                            TITANIC_DATASET_NAME,
                            field_name,
                            &format!("{:?}", record),
                            e,
                        )
                    })
                }
            };

            // Label: Survived (index 1)
            labels.push(parse_numeric(1, "survived")?);

            // Numeric features: PassengerId(0), Pclass(2), Age(5), SibSp(6), Parch(7), Fare(9)
            numeric_features.push(parse_numeric(0, "passenger_id")?);
            numeric_features.push(parse_numeric(2, "pclass")?);
            numeric_features.push(parse_numeric(5, "age")?);
            numeric_features.push(parse_numeric(6, "sib_sp")?);
            numeric_features.push(parse_numeric(7, "parch")?);
            numeric_features.push(parse_numeric(9, "fare")?);

            // String features: Name(3), Sex(4), Ticket(8), Cabin(10), Embarked(11)
            string_features.push(record[3].to_string());
            string_features.push(record[4].to_string());
            string_features.push(record[8].to_string());
            string_features.push(record[10].to_string());
            string_features.push(record[11].to_string());
        }

        if labels.len() != TITANIC_SAMPLE_SIZE {
            return Err(DatasetError::length_mismatch(
                TITANIC_DATASET_NAME,
                "labels",
                TITANIC_SAMPLE_SIZE,
                labels.len(),
            ));
        }
        if numeric_features.len() != TITANIC_SAMPLE_SIZE * TITANIC_NUM_NUMERIC_FEATURES {
            return Err(DatasetError::length_mismatch(
                TITANIC_DATASET_NAME,
                "numeric_features",
                TITANIC_SAMPLE_SIZE * TITANIC_NUM_NUMERIC_FEATURES,
                numeric_features.len(),
            ));
        }
        if string_features.len() != TITANIC_SAMPLE_SIZE * TITANIC_NUM_STRING_FEATURES {
            return Err(DatasetError::length_mismatch(
                TITANIC_DATASET_NAME,
                "string_features",
                TITANIC_SAMPLE_SIZE * TITANIC_NUM_STRING_FEATURES,
                string_features.len(),
            ));
        }

        let string_array = Array2::from_shape_vec(
            (TITANIC_SAMPLE_SIZE, TITANIC_NUM_STRING_FEATURES),
            string_features,
        )
            .map_err(|e| DatasetError::array_shape_error(TITANIC_DATASET_NAME, "string_features", e))?;

        let numeric_array = Array2::from_shape_vec(
            (TITANIC_SAMPLE_SIZE, TITANIC_NUM_NUMERIC_FEATURES),
            numeric_features,
        )
            .map_err(|e| {
                DatasetError::array_shape_error(TITANIC_DATASET_NAME, "numeric_features", e)
            })?;

        let labels_array = Array1::from_vec(labels);

        Ok((string_array, numeric_array, labels_array))
    }

    /// Internal helper to ensure data is loaded and return a reference.
    fn load_data(&self) -> Result<&(Array2<String>, Array2<f64>, Array1<f64>), DatasetError> {
        // if already initialized
        if let Some(cache) = self.data.get() {
            return Ok(cache);
        }
        // if not, initialize then store
        let (string_features, numeric_features, labels) =
            Self::load_data_internal(&self.storage_path)?;

        // Try to set the value. If another thread already set it, that's fine - just use the existing value
        let _ = self.data.set((string_features, numeric_features, labels));

        let cache = self
            .data
            .get()
            .expect("TITANIC_DATA should be initialized after set");
        Ok(cache)
    }

    /// Get a reference to both string and numeric feature matrices.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<String>` - Reference to string feature matrix with shape `(891, 5)` containing:
    ///     - `Name`
    ///     - `Sex`
    ///     - `Ticket`
    ///     - `Cabin`
    ///     - `Embarked`
    ///
    ///   (empty string if missing in source)
    ///
    /// - `&Array2<f64>` - Reference to numeric feature matrix with shape `(891, 6)` containing:
    ///     - `PassengerId`
    ///     - `Pclass`
    ///     - `Age`
    ///     - `SibSp`
    ///     - `Parch`
    ///     - `Fare`
    ///
    ///   (`NaN` if missing in source)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (891 samples)
    pub fn features(&self) -> Result<(&Array2<String>, &Array2<f64>), DatasetError> {
        let data = self.load_data()?;
        Ok((&data.0, &data.1))
    }

    /// Get a reference to the label vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<f64>` - Reference to label vector with shape `(891,)` containing `Survived` values (`0.0` or `1.0`, `NaN` if missing in source)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (891 samples)
    pub fn labels(&self) -> Result<&Array1<f64>, DatasetError> {
        Ok(&self.load_data()?.2)
    }

    /// Get string features, numeric features and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<String>` - Reference to string feature matrix with shape `(891, 5)` containing:
    ///     - `Name`
    ///     - `Sex`
    ///     - `Ticket`
    ///     - `Cabin`
    ///     - `Embarked`
    /// - `&Array2<f64>` - Reference to numeric feature matrix with shape `(891, 6)` containing:
    ///     - `PassengerId`
    ///     - `Pclass`
    ///     - `Age`
    ///     - `SibSp`
    ///     - `Parch`
    ///     - `Fare`
    /// - `&Array1<f64>` - Reference to label vector with shape `(891,)` containing `Survived` values
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (891 samples)
    pub fn data(&self) -> Result<(&Array2<String>, &Array2<f64>, &Array1<f64>), DatasetError> {
        let data = self.load_data()?;
        Ok((&data.0, &data.1, &data.2))
    }
}