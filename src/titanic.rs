use std::fs::{remove_file, rename, File};
use std::io::Read;
use std::path::Path;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;
use crate::{DatasetError, download_to, file_sha256_matches, prepare_download_dir};

/// The URL for the Titanic dataset.
const TITANIC_DATA_URL: &str = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv";

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

    /// Internal function to load the dataset from disk or download it.
    ///
    /// This function is called automatically by the accessor methods.
    fn load_data_internal(path: &str) -> Result<(Array2<String>, Array2<f64>, Array1<f64>), DatasetError> {
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
            let cols = Self::parse_csv_line(line);
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

    /// Internal helper to ensure data is loaded and return a reference.
    fn load_data(&self) -> Result<&(Array2<String>, Array2<f64>, Array1<f64>), DatasetError> {
        // if already initialized
        if let Some(cache) = self.data.get() {
            return Ok(cache);
        }
        // if not, initialize then store
        let (string_features, numeric_features, labels) = Self::load_data_internal(&self.storage_path)?;

        // Try to set the value. If another thread already set it, that's fine - just use the existing value
        let _ = self.data.set((string_features, numeric_features, labels));

        let cache = self.data
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