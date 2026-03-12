use crate::{
    DatasetError, create_temp_dir, download_to, file_sha256_matches, prepare_download_dir, unzip,
};
use ndarray::{Array1, Array2};
use std::fs::{File, remove_file, rename};
use csv::ReaderBuilder;
use std::path::Path;
use std::sync::OnceLock;

/// The URL for the Boston Housing dataset.
const BOSTON_HOUSING_DATA_URL: &str = "https://gist.github.com/nnbphuong/def91b5553736764e8e08f6255390f37/archive/373a856a3c9c1119e34b344de9230ae2ea89569d.zip";

/// The prefix for temporary files used during dataset download and extraction
const BOSTON_HOUSING_TEMP_FILE_PREFIX: &str = ".tmp-boston-housing-";

/// The downloaded zip file name
const BOSTON_HOUSING_ZIP_FILENAME: &str = "373a856a3c9c1119e34b344de9230ae2ea89569d.zip";

/// The folder where the file is located inside after extraction
const BOSTON_HOUSING_UNZIP_FOLDER: &str =
    "def91b5553736764e8e08f6255390f37-373a856a3c9c1119e34b344de9230ae2ea89569d";

/// The name of the file inside the extracted folder
const BOSTON_HOUSING_FILENAME: &str = "BostonHousing.csv";

/// The number of samples in the dataset
const BOSTON_HOUSING_SAMPLE_SIZE: usize = 506;

/// The number of features in the dataset
const BOSTON_HOUSING_NUM_FEATURES: usize = 13;

/// The SHA256 hash of the dataset file
const BOSTON_HOUSING_SHA256: &str =
    "c9aef7e921f2b44d4e7a234aea24f478186d5d457c3758035864b083ac8e7451";
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
/// # Fields
///
/// - `storage_path` - Directory path where the dataset will be stored.
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
/// // clean up: remove the downloaded files
/// std::fs::remove_dir_all(download_dir).unwrap();
/// ```
pub struct BostonHousing {
    storage_path: String,
    data: OnceLock<(Array2<f64>, Array1<f64>)>,
}

impl BostonHousing {
    /// Create a new BostonHousing instance without loading data.
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
    /// - `Self` - `BostonHousing` instance ready for lazy loading.
    pub fn new(storage_path: &str) -> Self {
        BostonHousing {
            storage_path: storage_path.to_string(),
            data: OnceLock::new(),
        }
    }

    /// Internal function to load the dataset from disk or download it.
    ///
    /// This function is called automatically by the accessor methods.
    fn load_data_internal(path: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
        let path = Path::new(path);
        let dst = path.join(BOSTON_HOUSING_FILENAME);
        let (need_download, need_overwrite) =
            prepare_download_dir(path, &dst, BOSTON_HOUSING_SHA256)?;

        // download and extract boston housing dataset if needed
        if need_download {
            // temporary directory to store the downloaded zip file
            let temp_dir = create_temp_dir(path, BOSTON_HOUSING_TEMP_FILE_PREFIX)?;
            let path_temp = temp_dir.path();
            // download and extract boston housing dataset
            download_to(BOSTON_HOUSING_DATA_URL, path_temp)?;
            unzip(&path_temp.join(BOSTON_HOUSING_ZIP_FILENAME), path_temp)?;
            let src = path_temp
                .join(BOSTON_HOUSING_UNZIP_FOLDER)
                .join(BOSTON_HOUSING_FILENAME);
            // check if the file exists and matches the expected SHA256 hash
            if !file_sha256_matches(src.as_path(), BOSTON_HOUSING_SHA256)? {
                return Err(DatasetError::sha256_validation_failed(
                    BOSTON_HOUSING_FILENAME,
                ));
            }
            if need_overwrite {
                remove_file(&dst).map_err(DatasetError::io)?;
            }
            // move boston_housing.csv out of the temporary directory
            rename(src, &dst).map_err(DatasetError::io)?;
        }

        let file = File::open(&dst).map_err(DatasetError::io)?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        let mut features =
            Vec::with_capacity(BOSTON_HOUSING_SAMPLE_SIZE * BOSTON_HOUSING_NUM_FEATURES);
        let mut targets = Vec::with_capacity(BOSTON_HOUSING_SAMPLE_SIZE);

        for result in rdr.records() {
            let record = result.map_err(|e| {
                DatasetError::data_format(format!(
                    "[{}] failed to read CSV record: {}",
                    BOSTON_HOUSING_DATASET_NAME, e
                ))
            })?;

            if record.len() < BOSTON_HOUSING_NUM_FEATURES + 1 {
                return Err(DatasetError::insufficient_column_count(
                    BOSTON_HOUSING_DATASET_NAME,
                    BOSTON_HOUSING_NUM_FEATURES + 1,
                    record.len(),
                    &format!("{:?}", record),
                ));
            }

            // Features are columns 0-12 (13 features)
            for i in 0..BOSTON_HOUSING_NUM_FEATURES {
                let field = format!("feature[{i}]");
                features.push(record[i].parse::<f64>().map_err(|e| {
                    DatasetError::parse_failed(
                        BOSTON_HOUSING_DATASET_NAME,
                        &field,
                        &format!("{:?}", record),
                        e,
                    )
                })?);
            }

            // Target is column 13 (MEDV)
            targets.push(record[13].parse::<f64>().map_err(|e| {
                DatasetError::parse_failed(
                    BOSTON_HOUSING_DATASET_NAME,
                    "target",
                    &format!("{:?}", record),
                    e,
                )
            })?);
        }

        if features.len() != BOSTON_HOUSING_SAMPLE_SIZE * BOSTON_HOUSING_NUM_FEATURES {
            return Err(DatasetError::length_mismatch(
                BOSTON_HOUSING_DATASET_NAME,
                "features",
                BOSTON_HOUSING_SAMPLE_SIZE * BOSTON_HOUSING_NUM_FEATURES,
                features.len(),
            ));
        }
        if targets.len() != BOSTON_HOUSING_SAMPLE_SIZE {
            return Err(DatasetError::length_mismatch(
                BOSTON_HOUSING_DATASET_NAME,
                "targets",
                BOSTON_HOUSING_SAMPLE_SIZE,
                targets.len(),
            ));
        }

        let features_array = Array2::from_shape_vec(
            (BOSTON_HOUSING_SAMPLE_SIZE, BOSTON_HOUSING_NUM_FEATURES),
            features,
        )
            .map_err(|e| DatasetError::array_shape_error(BOSTON_HOUSING_DATASET_NAME, "features", e))?;
        let targets_array = Array1::from_vec(targets);

        Ok((features_array, targets_array))
    }

    /// Internal helper to ensure data is loaded and return a reference.
    fn load_data(&self) -> Result<&(Array2<f64>, Array1<f64>), DatasetError> {
        // if already initialized
        if let Some(cache) = self.data.get() {
            return Ok(cache);
        }
        // if not, initialize then store
        let (features, targets) = Self::load_data_internal(&self.storage_path)?;

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
