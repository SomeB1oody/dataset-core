use crate::{
    DatasetError, create_temp_dir, download_to, file_sha256_matches, prepare_download_dir, unzip,
};
use ndarray::{Array1, Array2};
use std::fs::{File, remove_file, rename};
use csv::ReaderBuilder;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The URL for the Boston Housing dataset.
const BOSTON_HOUSING_DATA_URL: &str = "https://gist.github.com/nnbphuong/def91b5553736764e8e08f6255390f37/archive/373a856a3c9c1119e34b344de9230ae2ea89569d.zip";

/// The prefix for temporary files used during dataset download and extraction
const BOSTON_HOUSING_TEMP_FILE_PREFIX: &str = ".tmp-boston-housing-";

/// The downloaded zip file name
const BOSTON_HOUSING_ZIP_FILENAME: &str = "373a856a3c9c1119e34b344de9230ae2ea89569d.zip";

/// The folder where the file is located inside after extraction
const BOSTON_HOUSING_UNZIP_FOLDER: &str = "def91b5553736764e8e08f6255390f37-373a856a3c9c1119e34b344de9230ae2ea89569d";

/// The name of the file inside the extracted folder
const BOSTON_HOUSING_FILENAME: &str = "BostonHousing.csv";


/// The SHA256 hash of the dataset file
const BOSTON_HOUSING_SHA256: &str = "c9aef7e921f2b44d4e7a234aea24f478186d5d457c3758035864b083ac8e7451";

/// The name of the dataset
const BOSTON_HOUSING_DATASET_NAME: &str = "boston_housing";

/// A struct representing the Boston Housing dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The Boston Housing Dataset is derived from information collected by the U.S. Census Service
/// concerning housing in the area of Boston MA.
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
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The `OnceLock` ensures thread-safe lazy initialization.
///
/// # Fields
///
/// - `storage_dir` - Directory where the dataset will be stored.
/// - `data` - Cached data as a tuple of references to `Array2<f64>` and `Array1<f64>`. (`OnceLock` is used to ensure thread-safety)
///
/// # Example
/// ```rust
/// use rustyml_dataset::boston_housing::BostonHousing;
///
/// let download_dir = "./boston_housing"; // the code will create the directory if it doesn't exist
///
/// let dataset = BostonHousing::new(download_dir);
/// let features = dataset.features().unwrap();
/// let targets = dataset.targets().unwrap();
///
/// let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
/// // you can use `.to_owned()` to get owned copies of the data
/// let mut features_owned = features.to_owned();
/// let mut targets_owned = targets.to_owned();
///
/// // Example: Modify feature values
/// features_owned[[0, 0]] = 0.1;
/// targets_owned[0] = 25.5;
///
/// assert_eq!(features.shape(), &[506, 13]);
/// assert_eq!(targets.len(), 506);
///
/// // clean up: remove the downloaded files (dispensable)
/// std::fs::remove_dir_all(download_dir).unwrap();
/// ```
#[derive(Clone)]
pub struct BostonHousing {
    storage_dir: String,
    data: OnceLock<(Array2<f64>, Array1<f64>)>,
}

impl std::fmt::Debug for BostonHousing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BostonHousing")
            .field("storage_dir", &self.storage_dir)
            .field("data_loaded", &self.data.get().is_some())
            .finish()
    }
}

impl BostonHousing {
    /// Create a new BostonHousing instance without loading data.
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
    /// - `Self` - `BostonHousing` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        BostonHousing {
            storage_dir: storage_dir.to_string(),
            data: OnceLock::new(),
        }
    }

    /// Downloads the Boston Housing dataset if needed.
    ///
    /// This function handles downloading and extracting the dataset file,
    /// performing SHA256 validation to ensure data integrity.
    ///
    /// # Returns
    ///
    /// - `PathBuf` - Path to the downloaded dataset file
    fn download_dataset(dir: &str) -> Result<PathBuf, DatasetError> {
        let dir = Path::new(dir);
        let dst = dir.join(BOSTON_HOUSING_FILENAME);
        let (need_download, need_overwrite) = prepare_download_dir(dir, &dst, BOSTON_HOUSING_SHA256)?;

        // download and extract boston housing dataset if needed
        if need_download {
            // temporary directory to store the downloaded zip file
            let temp_dir = create_temp_dir(dir, BOSTON_HOUSING_TEMP_FILE_PREFIX)?;
            let dir_temp = temp_dir.path();
            // download and extract boston housing dataset
            download_to(BOSTON_HOUSING_DATA_URL, dir_temp)?;
            unzip(&dir_temp.join(BOSTON_HOUSING_ZIP_FILENAME), dir_temp)?;
            let src = dir_temp
                .join(BOSTON_HOUSING_UNZIP_FOLDER)
                .join(BOSTON_HOUSING_FILENAME);
            // check if the file exists and matches the expected SHA256 hash
            if !file_sha256_matches(src.as_path(), BOSTON_HOUSING_SHA256)? {
                // clean up temporary directory
                drop(temp_dir);
                return Err(DatasetError::sha256_validation_failed(
                    BOSTON_HOUSING_DATASET_NAME,
                    BOSTON_HOUSING_FILENAME,
                ));
            }
            if need_overwrite {
                remove_file(&dst)?;
            }
            // move boston_housing.csv out of the temporary directory
            rename(src, &dst)?;
        }

        Ok(dst)
    }

    /// Parses the Boston Housing dataset from the CSV file.
    ///
    /// This function reads and parses the dataset file, converting it into
    /// feature and target arrays.
    ///
    /// # Parameters
    ///
    /// - `file_path` - Path to the dataset file
    fn parse_dataset(file_path: PathBuf) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        let mut features = Vec::new();
        let mut targets = Vec::new();
        let mut num_features: Option<usize> = None;

        for (idx, result) in rdr.records().enumerate() {
            let record = result.map_err(|e| {
                DatasetError::csv_read_error(BOSTON_HOUSING_DATASET_NAME, e)
            })?;
            let line_num = idx + 2; // +1 for 0-indexed, +1 for header

            // Infer number of features from the first row
            if num_features.is_none() {
                if record.len() < 2 {
                    return Err(DatasetError::invalid_column_count(
                        BOSTON_HOUSING_DATASET_NAME,
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
                    BOSTON_HOUSING_DATASET_NAME,
                    n_features + 1,
                    record.len(),
                    line_num,
                    &format!("{:?}", record),
                ));
            }

            // Features are all columns except the last one
            for i in 0..n_features {
                let field = format!("feature[{i}]");
                features.push(record[i].parse::<f64>().map_err(|e| {
                    DatasetError::parse_failed(
                        BOSTON_HOUSING_DATASET_NAME,
                        &field,
                        line_num,
                        &format!("{:?}", record),
                        e,
                    )
                })?);
            }

            // Target is the last column
            targets.push(record[n_features].parse::<f64>().map_err(|e| {
                DatasetError::parse_failed(
                    BOSTON_HOUSING_DATASET_NAME,
                    "target",
                    line_num,
                    &format!("{:?}", record),
                    e,
                )
            })?);
        }

        // Verify the dataset is not empty
        let n_samples = targets.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(BOSTON_HOUSING_DATASET_NAME));
        }

        let n_features = num_features.unwrap();
        let features_array = Array2::from_shape_vec(
            (n_samples, n_features),
            features,
        )
            .map_err(|e| DatasetError::array_shape_error(BOSTON_HOUSING_DATASET_NAME, "features", e))?;
        let targets_array = Array1::from_vec(targets);

        Ok((features_array, targets_array))
    }

    /// Internal function to load the dataset from disk or download it.
    ///
    /// This function is called automatically by the accessor methods.
    /// It first downloads the dataset if needed, then parses it.
    fn load_data_internal(dir: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
        let file_path = Self::download_dataset(dir)?;
        Self::parse_dataset(file_path)
    }

    /// Internal helper to ensure data is loaded and return a reference.
    fn load_data(&self) -> Result<&(Array2<f64>, Array1<f64>), DatasetError> {
        // if already initialized
        if let Some(cache) = self.data.get() {
            return Ok(cache);
        }
        // if not, initialize then store
        let (features, targets) = Self::load_data_internal(&self.storage_dir)?;

        // Try to set the value. If another thread already set it, that's fine - just use the existing value
        let _ = self.data.set((features, targets));

        let cache = self
            .data
            .get()
            .expect("BOSTON_HOUSING_DATA should be initialized after set");
        Ok(cache)
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(506, 13)` containing:
    ///     - CRIM - per capita crime rate by town
    ///     - ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    ///     - INDUS - proportion of non-retail business acres per town
    ///     - CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    ///     - NOX - nitric oxides concentration (parts per 10 million)
    ///     - RM - average number of rooms per dwelling
    ///     - AGE - proportion of owner-occupied units built prior to 1940
    ///     - DIS - weighted distances to five Boston employment centres
    ///     - RAD - index of accessibility to radial highways
    ///     - TAX - full-value property-tax rate per $10,000
    ///     - PTRATIO - pupil-teacher ratio by town
    ///     - B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    ///     - LSTAT - % lower status of the population
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (506 samples, 13 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.load_data()?.0)
    }

    /// Get a reference to the target vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<f64>` - Reference to target vector with shape `(506,)` containing median value of owner-occupied homes in $1000's (MEDV)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (506 samples)
    pub fn targets(&self) -> Result<&Array1<f64>, DatasetError> {
        Ok(&self.load_data()?.1)
    }

    /// Get both features and targets as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(506, 13)` containing:
    ///     - CRIM - per capita crime rate by town
    ///     - ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    ///     - INDUS - proportion of non-retail business acres per town
    ///     - CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    ///     - NOX - nitric oxides concentration (parts per 10 million)
    ///     - RM - average number of rooms per dwelling
    ///     - AGE - proportion of owner-occupied units built prior to 1940
    ///     - DIS - weighted distances to five Boston employment centres
    ///     - RAD - index of accessibility to radial highways
    ///     - TAX - full-value property-tax rate per $10,000
    ///     - PTRATIO - pupil-teacher ratio by town
    ///     - B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    ///     - LSTAT - % lower status of the population
    /// - `&Array1<f64>` - Reference to target vector with shape `(506,)` containing median value of owner-occupied homes in $1000's (MEDV)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (506 samples, 13 features)
    pub fn data(&self) -> Result<(&Array2<f64>, &Array1<f64>), DatasetError> {
        let data = self.load_data()?;
        Ok((&data.0, &data.1))
    }
}
