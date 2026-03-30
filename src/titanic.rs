use crate::{DatasetError, download_to, file_sha256_matches, prepare_download_dir};
use ndarray::{Array1, Array2};
use std::fs::{File, remove_file, rename};
use csv::ReaderBuilder;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Type alias for Titanic dataset: (string features, numeric features, labels)
type TitanicData = (Array2<String>, Array2<f64>, Array1<f64>);

/// The URL for the Titanic dataset.
const TITANIC_DATA_URL: &str = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv";

/// The prefix for temporary files created during dataset download and parsing.
const TITANIC_TEMP_FILE_PREFIX: &str = ".tmp-titanic-";

/// The name of the Titanic dataset file.
const TITANIC_FILENAME: &str = "titanic.csv";


/// The SHA256 hash of the Titanic dataset file.
const TITANIC_SHA256: &str = "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7";

/// The name of the dataset
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
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The `OnceLock` ensures thread-safe lazy initialization.
///
/// # Fields
///
/// - `storage_dir` - Directory where the dataset will be stored.
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
/// // clean up: remove the downloaded files (dispensable)
/// std::fs::remove_dir_all(download_dir).unwrap();
/// ```
#[derive(Clone)]
pub struct Titanic {
    storage_dir: String,
    data: OnceLock<TitanicData>,
}

impl std::fmt::Debug for Titanic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Titanic")
            .field("storage_dir", &self.storage_dir)
            .field("data_loaded", &self.data.get().is_some())
            .finish()
    }
}

impl Titanic {
    /// Create a new Titanic instance without loading data.
    ///
    /// The dataset will be loaded lazily when you first call any data accessor method.
    /// This is a lightweight operation that only stores the storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory where the dataset will be stored.
    ///
    /// # Returns
    ///
    /// - `Self` - `Titanic` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Titanic {
            storage_dir: storage_dir.to_string(),
            data: OnceLock::new(),
        }
    }

    /// Downloads the Titanic dataset if needed.
    ///
    /// This function handles downloading the dataset file,
    /// performing SHA256 validation to ensure data integrity.
    ///
    /// # Returns
    ///
    /// - `PathBuf` - Path to the downloaded dataset file
    fn download_dataset(dir: &str) -> Result<PathBuf, DatasetError> {
        let dir = Path::new(dir);
        let dst = dir.join(TITANIC_FILENAME);
        let (need_download, need_overwrite) = prepare_download_dir(dir, &dst, TITANIC_SHA256)?;
        if need_download {
            // temporary directory to store the downloaded zip file
            let temp_dir = crate::create_temp_dir(dir, TITANIC_TEMP_FILE_PREFIX)?;
            let dir_temp = temp_dir.path();
            // download and extract titanic dataset
            download_to(TITANIC_DATA_URL, dir_temp)?;
            // move downloaded file to final location
            let src = dir_temp.join(TITANIC_FILENAME);
            if !file_sha256_matches(src.as_path(), TITANIC_SHA256)? {
                // clean up temporary directory
                drop(temp_dir);
                return Err(DatasetError::sha256_validation_failed(
                    TITANIC_DATASET_NAME,
                    TITANIC_FILENAME,
                ));
            }
            if need_overwrite {
                remove_file(&dst)?;
            }
            rename(src, &dst)?;
        }

        Ok(dst)
    }

    /// Parses the Titanic dataset from the CSV file.
    ///
    /// This function reads and parses the dataset file, converting it into
    /// string features, numeric features, and label arrays.
    ///
    /// # Parameters
    ///
    /// - `file_path` - Path to the dataset file
    fn parse_dataset(file_path: PathBuf) -> Result<TitanicData, DatasetError> {
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        let mut string_features = Vec::new();
        let mut numeric_features = Vec::new();
        let mut labels = Vec::new();

        // CSV columns: PassengerId(0), Survived(1), Pclass(2), Name(3), Sex(4), Age(5), SibSp(6), Parch(7), Ticket(8), Fare(9), Cabin(10), Embarked(11)
        // Numeric indices: [0, 2, 5, 6, 7, 9]
        // String indices: [3, 4, 8, 10, 11]
        // Label index: 1
        let numeric_indices = vec![0, 2, 5, 6, 7, 9];
        let string_indices = vec![3, 4, 8, 10, 11];
        let label_index = 1;

        let mut num_string_features: Option<usize> = None;
        let mut num_numeric_features: Option<usize> = None;

        for (idx, result) in rdr.records().enumerate() {
            let record = result.map_err(|e| {
                DatasetError::csv_read_error(TITANIC_DATASET_NAME, e)
            })?;
            let line_num = idx + 2; // +1 for 0-indexed, +1 for header

            // Infer number of features from the first row
            if num_string_features.is_none() {
                if record.len() < 12 {
                    return Err(DatasetError::invalid_column_count(
                        TITANIC_DATASET_NAME,
                        12,
                        record.len(),
                        line_num,
                        &format!("{:?}", record),
                    ));
                }
                num_string_features = Some(string_indices.len());
                num_numeric_features = Some(numeric_indices.len());
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
                            line_num,
                            &format!("{:?}", record),
                            e,
                        )
                    })
                }
            };

            // Label: Survived (index 1)
            labels.push(parse_numeric(label_index, "survived")?);

            // Numeric features: PassengerId(0), Pclass(2), Age(5), SibSp(6), Parch(7), Fare(9)
            for (i, &col_idx) in numeric_indices.iter().enumerate() {
                let field_name = match i {
                    0 => "passenger_id",
                    1 => "pclass",
                    2 => "age",
                    3 => "sib_sp",
                    4 => "parch",
                    5 => "fare",
                    _ => "numeric_feature",
                };
                numeric_features.push(parse_numeric(col_idx, field_name)?);
            }

            // String features: Name(3), Sex(4), Ticket(8), Cabin(10), Embarked(11)
            for &col_idx in string_indices.iter() {
                string_features.push(record[col_idx].to_string());
            }
        }

        // Verify the dataset is not empty
        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(TITANIC_DATASET_NAME));
        }

        let n_string_features = num_string_features.unwrap();
        let n_numeric_features = num_numeric_features.unwrap();

        let string_array = Array2::from_shape_vec(
            (n_samples, n_string_features),
            string_features,
        )
            .map_err(|e| DatasetError::array_shape_error(TITANIC_DATASET_NAME, "string_features", e))?;

        let numeric_array = Array2::from_shape_vec(
            (n_samples, n_numeric_features),
            numeric_features,
        )
            .map_err(|e| {
                DatasetError::array_shape_error(TITANIC_DATASET_NAME, "numeric_features", e)
            })?;

        let labels_array = Array1::from_vec(labels);

        Ok((string_array, numeric_array, labels_array))
    }

    /// Internal function to load the dataset from disk or download it.
    ///
    /// This function is called automatically by the accessor methods.
    /// It first downloads the dataset if needed, then parses it.
    fn load_data_internal(dir: &str) -> Result<TitanicData, DatasetError> {
        let file_path = Self::download_dataset(dir)?;
        Self::parse_dataset(file_path)
    }

    /// Internal helper to ensure data is loaded and return a reference.
    fn load_data(&self) -> Result<&TitanicData, DatasetError> {
        // if already initialized
        if let Some(cache) = self.data.get() {
            return Ok(cache);
        }
        // if not, initialize then store
        let (string_features, numeric_features, labels) =
            Self::load_data_internal(&self.storage_dir)?;

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