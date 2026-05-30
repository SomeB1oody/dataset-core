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

use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to};
use ndarray::{Array1, Array2};
use serde::Deserialize;
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

/// Type alias for the Boston Housing dataset: (features, targets).
type BostonHousingData = (Array2<f64>, Array1<f64>);

/// One CSV record of the Boston Housing dataset: 13 `f64` feature columns
/// followed by the `medv` target.
///
/// Fields are declared in CSV column order and deserialized **positionally**
/// (the loader disables csv's header handling), so this struct is independent
/// of the exact header spelling.
#[derive(Deserialize)]
struct BostonHousingRecord {
    crim: f64,
    zn: f64,
    indus: f64,
    chas: f64,
    nox: f64,
    rm: f64,
    age: f64,
    dis: f64,
    rad: f64,
    tax: f64,
    ptratio: f64,
    b: f64,
    lstat: f64,
    medv: f64,
}

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
/// ```no_run
/// use dataset_ml::boston_housing::BostonHousing;
///
/// let download_dir = "./boston_housing"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = BostonHousing::new(download_dir);
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
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place (no clone, no reload — the change stays cached).
/// if let Some((features, _targets)) = dataset.get_data_mut() {
///     features[[0, 0]] = 0.1;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable — the next access reloads from the cached file.
/// let (owned_features, owned_targets) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[506, 13]);
/// assert_eq!(owned_targets.len(), 506);
///
/// // `into_data()` also returns owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_features, owned_targets) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[506, 13]);
/// assert_eq!(owned_targets.len(), 506);
/// ```
#[derive(Debug)]
pub struct BostonHousing {
    dataset: Dataset<BostonHousingData>,
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
    fn load_data(dir: &str) -> Result<BostonHousingData, DatasetError> {
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

        // Stream the cached file through csv, deserializing one record at a time.
        // `has_headers(false)` makes csv deserialize into the named struct
        // *positionally* (by column order) rather than by header name, keeping
        // parsing independent of the exact header spelling. We skip the header
        // row ourselves with `.skip(1)`.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features = Vec::new();
        let mut targets = Vec::new();

        for result in rdr.deserialize::<BostonHousingRecord>().skip(1) {
            let BostonHousingRecord {
                crim,
                zn,
                indus,
                chas,
                nox,
                rm,
                age,
                dis,
                rad,
                tax,
                ptratio,
                b,
                lstat,
                medv,
            } = result.map_err(|e| DatasetError::csv_read_error(BOSTON_HOUSING_DATASET_NAME, e))?;

            // Features are every column except the last; the target is `medv`.
            features.extend_from_slice(&[
                crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat,
            ]);
            targets.push(medv);
        }

        // Verify the dataset is not empty
        let n_samples = targets.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(BOSTON_HOUSING_DATASET_NAME));
        }

        // Boston Housing has a fixed schema of 13 numeric features per sample.
        let features_array = Array2::from_shape_vec((n_samples, 13), features).map_err(|e| {
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
    /// - `&BostonHousingData` - reference to the cached `(features, targets)`
    ///   tuple: feature matrix with shape `(506, 13)` (CRIM, ZN, INDUS, CHAS, NOX,
    ///   RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT) and target vector with shape
    ///   `(506,)` (MEDV, median home value in $1000's).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (506 samples, 13 features)
    pub fn data(&self) -> Result<&BostonHousingData, DatasetError> {
        self.dataset.load(Self::load_data)
    }

    /// Get both features and targets as references **without** triggering loading.
    ///
    /// Unlike [`BostonHousing::data`], which loads the dataset on first call, this
    /// never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&BostonHousingData)` - reference to the cached `(features, targets)`
    ///   tuple (feature matrix `(506, 13)`, target vector `(506,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&BostonHousingData> {
        self.dataset.get()
    }

    /// Get mutable references to features and targets for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize features,
    /// rescale targets) with no `to_owned()` clone and without removing them from
    /// the cache: the changes persist, so later [`BostonHousing::features`],
    /// [`BostonHousing::data`], or [`BostonHousing::get_data`] calls observe them.
    ///
    /// Like [`BostonHousing::get_data`], this does **not** trigger loading: it
    /// returns `None` if the dataset has not been loaded. Call a loading accessor
    /// (e.g. [`BostonHousing::data`]) first if you need to ensure the data is
    /// present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut BostonHousingData)` - mutable reference to the cached
    ///   `(features, targets)` tuple (feature matrix `(506, 13)`, target vector
    ///   `(506,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut BostonHousingData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and targets.
    ///
    /// Unlike [`BostonHousing::data`], which borrows the cached data, this moves it
    /// out and returns owned arrays directly — no `to_owned()` clone needed. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`BostonHousing::take_data`] instead — it takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<f64>)` - owned feature matrix with shape `(506, 13)`
    ///   and owned target vector with shape `(506,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<BostonHousingData, DatasetError> {
        self.dataset.load(Self::load_data)?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and targets out of the dataset, leaving it reusable.
    ///
    /// Like [`BostonHousing::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out, resetting the instance to its
    /// unloaded state: the next accessor call (e.g. [`BostonHousing::features`] or
    /// [`BostonHousing::data`]) loads the dataset again.
    ///
    /// Use [`BostonHousing::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<f64>)` - owned feature matrix with shape `(506, 13)`
    ///   and owned target vector with shape `(506,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<BostonHousingData, DatasetError> {
        self.dataset.load(Self::load_data)?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}
