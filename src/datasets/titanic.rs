use crate::{Dataset, DatasetError, download_dataset_with, download_to};
use ndarray::{Array1, Array2};
use std::fs::File;
use csv::ReaderBuilder;

/// Type alias for Titanic dataset: (string features, numeric features, labels)
type TitanicData = (Array2<String>, Array2<f64>, Array1<f64>);

/// The URL for the Titanic dataset.
const TITANIC_DATA_URL: &str = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv";

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
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```rust
/// use rustyml_dataset::datasets::titanic::Titanic;
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
#[derive(Debug)]
pub struct Titanic {
    dataset: Dataset<TitanicData>,
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
            dataset: Dataset::new(storage_dir),
        }
    }

    /// Download and parse the Titanic dataset.
    fn load_data(dir: &str) -> Result<TitanicData, DatasetError> {
        // Download and unzip the dataset
        let file_path = download_dataset_with(
            dir,
            TITANIC_FILENAME,
            TITANIC_DATASET_NAME,
            Some(TITANIC_SHA256),
            |temp_path| {
                download_to(TITANIC_DATA_URL, temp_path)?;
                Ok(temp_path.join(TITANIC_FILENAME))
            },
        )?;

        // Parse the file
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
        let data = self.dataset.load(Self::load_data)?;
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
        Ok(&self.dataset.load(Self::load_data)?.2)
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
        let data = self.dataset.load(Self::load_data)?;
        Ok((&data.0, &data.1, &data.2))
    }
}
