//! Boston Housing dataset.
//!
//! Housing data for suburbs of Boston, collected from U.S. Census-derived
//! information and commonly used as a regression benchmark.
//!
//! **Features (13):**
//! - `CRIM` - per capita crime rate by town
//! - `ZN` - proportion of residential land zoned for lots over 25,000 sq.ft.
//! - `INDUS` - proportion of non-retail business acres per town
//! - `CHAS` - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
//! - `NOX` - nitric oxides concentration (parts per 10 million)
//! - `RM` - average number of rooms per dwelling
//! - `AGE` - proportion of owner-occupied units built prior to 1940
//! - `DIS` - weighted distances to five Boston employment centres
//! - `RAD` - index of accessibility to radial highways
//! - `TAX` - full-value property-tax rate per $10,000
//! - `PTRATIO` - pupil-teacher ratio by town
//! - `B` - 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town
//! - `LSTAT` - percentage of lower-status population
//!
//! **Target:** `MEDV` - median value of owner-occupied homes in $1000s
//!
//! **Samples:** 506
//! **Application:** Regression / housing value prediction
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5C88K>

use crate::{Dataset, DatasetError, acquire_dataset, download_to};
use csv::ReaderBuilder;
use ndarray::{Array1, Array2};
use std::fs::File;

/// The URL for the Boston Housing dataset.
const BOSTON_HOUSING_DATA_URL: &str =
    "https://github.com/selva86/datasets/raw/master/BostonHousing.csv";

/// The name of the file inside the extracted folder
const BOSTON_HOUSING_FILENAME: &str = "BostonHousing.csv";

/// The SHA256 hash of the dataset file
const BOSTON_HOUSING_SHA256: &str =
    "ab16ba38fbbbbcc69fe930aab1293104f1442c8279c130d9eba03dd864bef675";

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
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```rust
/// use dataset_core::datasets::boston_housing::BostonHousing;
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
#[derive(Debug)]
pub struct BostonHousing {
    dataset: Dataset<(Array2<f64>, Array1<f64>)>,
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
            dataset: Dataset::new(storage_dir),
        }
    }

    /// Acquire and parse the Boston Housing dataset.
    fn load_data(dir: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
        // Prepare the dataset file
        let file_path = acquire_dataset(
            dir,
            BOSTON_HOUSING_FILENAME,
            BOSTON_HOUSING_DATASET_NAME,
            Some(BOSTON_HOUSING_SHA256),
            |temp_path| {
                download_to(BOSTON_HOUSING_DATA_URL, temp_path, None)?;
                Ok(temp_path.join(BOSTON_HOUSING_FILENAME))
            },
        )?;

        // Parse the file
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

        let mut features = Vec::new();
        let mut targets = Vec::new();
        let mut num_features: Option<usize> = None;

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(BOSTON_HOUSING_DATASET_NAME, e))?;
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
                features.push(record[i].parse::<f64>().map_err(|e| {
                    let field = format!("feature[{i}]");
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
        let features_array =
            Array2::from_shape_vec((n_samples, n_features), features).map_err(|e| {
                DatasetError::array_shape_error(BOSTON_HOUSING_DATASET_NAME, "features", e)
            })?;
        let targets_array = Array1::from_vec(targets);

        Ok((features_array, targets_array))
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
        Ok(&self.dataset.load(Self::load_data)?.0)
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
        Ok(&self.dataset.load(Self::load_data)?.1)
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
        let data = self.dataset.load(Self::load_data)?;
        Ok((&data.0, &data.1))
    }
}
