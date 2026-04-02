use crate::{Dataset, DatasetError, download_dataset_with, download_to, unzip};
use ndarray::{Array1, Array2};
use std::fs::File;
use csv::ReaderBuilder;
use std::path::PathBuf;

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

/// The filename of the zip archive containing the Wine Quality datasets.
const WINE_QUALITY_ZIP_FILENAME: &str = "wine+quality.zip";

/// The red wine file of the CSV files inside the zip archive.
const RED_WINE_QUALITY_FILENAME: &str = "winequality-red.csv";

/// The white wine file of the CSV files inside the zip archive.
const WHITE_WINE_QUALITY_FILENAME: &str = "winequality-white.csv";

/// The SHA256 hash of the white wine quality dataset.
const WHITE_WINE_QUALITY_SHA256: &str = "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836";

/// The SHA256 hash of the red wine quality dataset.
const RED_WINE_QUALITY_SHA256: &str = "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e";

/// Parses a wine quality dataset from the CSV file.
///
/// This function reads and parses the dataset file, converting it into
/// feature and target arrays.
///
/// # Parameters
///
/// - `file_path` - Path to the dataset file
/// - `dataset_name` - Name of the dataset for error messages
fn parse_wine_quality_dataset(
    file_path: PathBuf,
    dataset_name: &str,
) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
    let file = File::open(&file_path)?;
    parse_wine_data_to_array(dataset_name, file)
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
/// - `dataset_name` - Name of the dataset for error messages.
/// - `reader` - CSV file reader.
///
/// # Returns
///
/// - `Array2<f64>` - Feature matrix with shape `(n_samples, 11)`.
/// - `Array1<f64>` - Target vector with length `n_samples`.
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Any row has an unexpected number of columns
/// - Any feature/target value fails to parse as `f64`
/// - The final number of parsed values does not match the expected shape
fn parse_wine_data_to_array<R: std::io::Read>(
    dataset_name: &str,
    reader: R,
) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
    let mut rdr = ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(true)
        .from_reader(reader);

    let mut features_array = Vec::new();
    let mut target_array = Vec::new();
    let mut num_features: Option<usize> = None;

    for (idx, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| {
            DatasetError::csv_read_error(dataset_name, e)
        })?;
        let line_num = idx + 2; // +1 for 0-indexed, +1 for header

        if num_features.is_none() {
            if record.len() < 2 {
                return Err(DatasetError::invalid_column_count(
                    dataset_name,
                    2,
                    record.len(),
                    line_num,
                    &format!("{:?}", record),
                ));
            }
            num_features = Some(record.len() - 1);
        }

        let n_features = num_features.unwrap();
        if record.len() != n_features + 1 {
            return Err(DatasetError::invalid_column_count(
                dataset_name,
                n_features + 1,
                record.len(),
                line_num,
                &format!("{:?}", record),
            ));
        }

        for i in 0..n_features {
            let field = format!("feature[{i}]");
            features_array.push(
                record[i]
                    .parse::<f64>()
                    .map_err(|e| DatasetError::parse_failed(dataset_name, &field, line_num, &format!("{:?}", record), e))?,
            );
        }

        target_array.push(
            record[n_features]
                .parse::<f64>()
                .map_err(|e| DatasetError::parse_failed(dataset_name, "target", line_num, &format!("{:?}", record), e))?,
        );
    }

    let n_samples = target_array.len();
    if n_samples == 0 {
        return Err(DatasetError::empty_dataset(dataset_name));
    }

    let n_features = num_features.unwrap();
    let features_array =
        Array2::from_shape_vec((n_samples, n_features), features_array)
            .map_err(|e| DatasetError::array_shape_error(dataset_name, "features", e))?;
    let target_array = Array1::from_vec(target_array);

    Ok((features_array, target_array))
}

/// A struct representing the Red Wine Quality dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
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
/// Targets:
/// - quality (score between 0 and 10, stored as `f64`)
///
/// See more information at <https://archive.ics.uci.edu/dataset/186/wine+quality>
///
/// # Citation
///
/// P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. "Wine Quality," UCI Machine Learning Repository, 2009. \[Online\]. Available: <https://doi.org/10.24432/C56S3T>.
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```rust
/// use rustyml_dataset::wine_quality::RedWineQuality;
///
/// let download_dir = "./red_wine"; // the code will create the directory if it doesn't exist
///
/// let dataset = RedWineQuality::new(download_dir);
/// let features = dataset.features().unwrap();
/// let targets = dataset.targets().unwrap();
///
/// let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
/// // you can use `.to_owned()` to get owned copies of the data
/// let mut features_owned = features.to_owned();
/// let mut targets_owned = targets.to_owned();
///
/// // Example: Modify feature values
/// features_owned[[0, 0]] = 10.0;
/// targets_owned[0] = 7.0;
///
/// assert_eq!(features.shape(), &[1599, 11]);
/// assert_eq!(targets.len(), 1599);
///
/// // clean up: remove the downloaded files
/// std::fs::remove_dir_all(download_dir).unwrap();
/// ```
pub struct RedWineQuality {
    dataset: Dataset<(Array2<f64>, Array1<f64>)>,
}

impl std::fmt::Debug for RedWineQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedWineQuality")
            .field("storage_dir", &self.dataset.storage_dir())
            .field("data_loaded", &self.dataset.is_loaded())
            .finish()
    }
}

impl RedWineQuality {
    /// Create a new RedWineQuality instance without loading data.
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
    /// - `Self` - `RedWineQuality` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        RedWineQuality {
            dataset: Dataset::new(storage_dir),
        }
    }

    /// Download and parse the dataset from the storage directory.
    fn load_data_internal(dir: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
        let file_path = download_dataset_with(
            dir,
            RED_WINE_QUALITY_FILENAME,
            "red_wine_quality",
            Some(RED_WINE_QUALITY_SHA256),
            |temp_path| {
                download_to(WINE_QUALITY_URL, temp_path)?;
                unzip(&temp_path.join(WINE_QUALITY_ZIP_FILENAME), temp_path)?;
                Ok(temp_path.join(RED_WINE_QUALITY_FILENAME))
            },
        )?;
        parse_wine_quality_dataset(file_path, "red_wine_quality")
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(1599, 11)` containing:
    ///     - fixed acidity
    ///     - volatile acidity
    ///     - citric acid
    ///     - residual sugar
    ///     - chlorides
    ///     - free sulfur dioxide
    ///     - total sulfur dioxide
    ///     - density
    ///     - pH
    ///     - sulphates
    ///     - alcohol
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (1599 samples, 11 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load(Self::load_data_internal)?.0)
    }

    /// Get a reference to the target vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<f64>` - Reference to target vector with shape `(1599,)` containing quality scores (0-10)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (1599 samples)
    pub fn targets(&self) -> Result<&Array1<f64>, DatasetError> {
        Ok(&self.dataset.load(Self::load_data_internal)?.1)
    }

    /// Get both features and targets as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(1599, 11)` containing:
    ///     - fixed acidity
    ///     - volatile acidity
    ///     - citric acid
    ///     - residual sugar
    ///     - chlorides
    ///     - free sulfur dioxide
    ///     - total sulfur dioxide
    ///     - density
    ///     - pH
    ///     - sulphates
    ///     - alcohol
    /// - `&Array1<f64>` - Reference to target vector with shape `(1599,)` containing quality scores (0-10)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (1599 samples, 11 features)
    pub fn data(&self) -> Result<(&Array2<f64>, &Array1<f64>), DatasetError> {
        let data = self.dataset.load(Self::load_data_internal)?;
        Ok((&data.0, &data.1))
    }
}

/// A struct representing the White Wine Quality dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
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
/// Targets:
/// - quality (score between 0 and 10, stored as `f64`)
///
/// See more information at <https://archive.ics.uci.edu/dataset/186/wine+quality>
///
/// # Citation
///
/// P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. "Wine Quality," UCI Machine Learning Repository, 2009. \[Online\]. Available: <https://doi.org/10.24432/C56S3T>.
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```rust
/// use rustyml_dataset::wine_quality::WhiteWineQuality;
///
/// let download_dir = "./white_wine"; // the code will create the directory if it doesn't exist
///
/// let dataset = WhiteWineQuality::new(download_dir);
/// let features = dataset.features().unwrap();
/// let targets = dataset.targets().unwrap();
///
/// let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
/// // you can use `.to_owned()` to get owned copies of the data
/// let mut features_owned = features.to_owned();
/// let mut targets_owned = targets.to_owned();
///
/// // Example: Modify feature values
/// features_owned[[0, 0]] = 10.0;
/// targets_owned[0] = 7.0;
///
/// assert_eq!(features.shape(), &[4898, 11]);
/// assert_eq!(targets.len(), 4898);
///
/// // clean up: remove the downloaded files (dispensable)
/// std::fs::remove_dir_all(download_dir).unwrap();
/// ```
pub struct WhiteWineQuality {
    dataset: Dataset<(Array2<f64>, Array1<f64>)>,
}

impl std::fmt::Debug for WhiteWineQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhiteWineQuality")
            .field("storage_dir", &self.dataset.storage_dir())
            .field("data_loaded", &self.dataset.is_loaded())
            .finish()
    }
}

impl WhiteWineQuality {
    /// Create a new WhiteWineQuality instance without loading data.
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
    /// - `Self` - `WhiteWineQuality` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        WhiteWineQuality {
            dataset: Dataset::new(storage_dir),
        }
    }

    /// Download and parse the dataset from the storage directory.
    fn load_data_internal(dir: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
        let file_path = download_dataset_with(
            dir,
            WHITE_WINE_QUALITY_FILENAME,
            "white_wine_quality",
            Some(WHITE_WINE_QUALITY_SHA256),
            |temp_path| {
                download_to(WINE_QUALITY_URL, temp_path)?;
                unzip(&temp_path.join(WINE_QUALITY_ZIP_FILENAME), temp_path)?;
                Ok(temp_path.join(WHITE_WINE_QUALITY_FILENAME))
            },
        )?;
        parse_wine_quality_dataset(file_path, "white_wine_quality")
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(4898, 11)` containing:
    ///     - fixed acidity
    ///     - volatile acidity
    ///     - citric acid
    ///     - residual sugar
    ///     - chlorides
    ///     - free sulfur dioxide
    ///     - total sulfur dioxide
    ///     - density
    ///     - pH
    ///     - sulphates
    ///     - alcohol
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (4898 samples, 11 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load(Self::load_data_internal)?.0)
    }

    /// Get a reference to the target vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<f64>` - Reference to target vector with shape `(4898,)` containing quality scores (0-10)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (4898 samples)
    pub fn targets(&self) -> Result<&Array1<f64>, DatasetError> {
        Ok(&self.dataset.load(Self::load_data_internal)?.1)
    }

    /// Get both features and targets as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(4898, 11)` containing:
    ///     - fixed acidity
    ///     - volatile acidity
    ///     - citric acid
    ///     - residual sugar
    ///     - chlorides
    ///     - free sulfur dioxide
    ///     - total sulfur dioxide
    ///     - density
    ///     - pH
    ///     - sulphates
    ///     - alcohol
    /// - `&Array1<f64>` - Reference to target vector with shape `(4898,)` containing quality scores (0-10)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (4898 samples, 11 features)
    pub fn data(&self) -> Result<(&Array2<f64>, &Array1<f64>), DatasetError> {
        let data = self.dataset.load(Self::load_data_internal)?;
        Ok((&data.0, &data.1))
    }
}
