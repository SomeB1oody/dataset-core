//! Boston Housing dataset.
//!
//! This dataset holds housing data for Boston suburbs, drawn from U.S.
//! Census-derived information. Researchers commonly use it as a regression
//! benchmark.
//!
//! **Features (13):**
//! - `CRIM` - per capita crime rate by town
//! - `ZN` - proportion of residential land zoned for lots over 25,000 sq.ft.
//! - `INDUS` - proportion of non-retail business acres per town
//! - `CHAS` - Charles River dummy variable (1 if tract bounds river, 0 otherwise)
//! - `NOX` - nitric oxides concentration (parts per 10 million)
//! - `RM` - average number of rooms per dwelling
//! - `AGE` - proportion of owner-occupied units built before 1940
//! - `DIS` - weighted distances to five Boston employment centers
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

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
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

/// This struct represents the Boston Housing dataset and loads data lazily.
///
/// You do not load the dataset until you call one of the data accessor
/// methods. After that, the dataset caches the data for later calls.
///
/// # About Dataset
///
/// The U.S. Census Service collected the information behind the Boston Housing
/// Dataset. It describes housing in the Boston, MA area.
///
/// # Feature columns
///
/// The 13 feature columns, by 0-based column index in the feature matrix:
///
/// | Columns | Attributes | Unit |
/// |---------|------------|------|
/// | `0`     | `CRIM` (per capita crime rate by town)                                            |                       |
/// | `1`     | `ZN` (proportion of residential land zoned for lots over 25,000 sq.ft.)           | sq.ft.                |
/// | `2`     | `INDUS` (proportion of non-retail business acres per town)                        |                       |
/// | `3`     | `CHAS` (Charles River dummy variable: 1 if tract bounds river, 0 otherwise)       |                       |
/// | `4`     | `NOX` (nitric oxides concentration)                                               | parts per 10 million  |
/// | `5`     | `RM` (average number of rooms per dwelling)                                        |                       |
/// | `6`     | `AGE` (proportion of owner-occupied units built before 1940)                      |                       |
/// | `7`     | `DIS` (weighted distances to five Boston employment centers)                      |                       |
/// | `8`     | `RAD` (index of accessibility to radial highways)                                 |                       |
/// | `9`     | `TAX` (full-value property-tax rate)                                              | per $10,000           |
/// | `10`    | `PTRATIO` (pupil-teacher ratio by town)                                           |                       |
/// | `11`    | `B` (1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town)     |                       |
/// | `12`    | `LSTAT` (% lower status of the population)                                        |                       |
///
/// # Targets
///
/// - `MEDV` - median value of owner-occupied homes in $1000's
///
/// # Citation
///
/// D. Harrison and D. L. Rubinfeld, "Hedonic prices and the demand for clean
/// air," Journal of Environmental Economics and Management, vol. 5, no. 1,
/// pp. 81–102, 1978. Dataset via UCI Machine Learning Repository,
/// <https://doi.org/10.24432/C5C88K>.
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because every field
/// does. This makes it safe to share across threads. The internal [`Dataset`]
/// keeps lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::boston_housing::BostonHousing;
///
/// let download_dir = "./boston_housing"; // the code creates the directory if it does not exist
///
/// let mut dataset = BostonHousing::new(download_dir);
/// let features = dataset.features().unwrap();
/// let targets = dataset.targets().unwrap();
///
/// let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
/// assert_eq!(features.shape(), &[506, 13]);
/// assert_eq!(targets.len(), 506);
///
/// // `get_data()` borrows the cached arrays without reloading. `get_data_mut()`
/// // edits them in place: no clone, no reload, and the change stays cached.
/// // Prefer this over cloning with `.to_owned()` when you only need to tweak
/// // values.
/// if let Some((features, targets)) = dataset.get_data_mut() {
///     features[[0, 0]] = 0.1;
///     targets[0] = 25.5;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable. The next access reloads from the cached file.
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
    dataset: Dataset<BostonHousingData, DatasetError>,
}

impl BostonHousing {
    /// Create a new BostonHousing instance without loading data.
    ///
    /// The dataset loads lazily, on your first call to a data accessor method.
    /// This function only stores the storage directory.
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
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Boston Housing dataset.
    fn load_data(dir: &str) -> Result<BostonHousingData, DatasetError> {
        // Prepare the dataset file.
        let file_path = acquire_dataset(
            dir,
            BOSTON_HOUSING_FILENAME,
            BOSTON_HOUSING_DATASET_NAME,
            Some(BOSTON_HOUSING_SHA256),
            |temp_path| {
                download_to_with_retries(
                    BOSTON_HOUSING_DATA_URL,
                    temp_path,
                    None,
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(BOSTON_HOUSING_FILENAME))
            },
        )?;

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

            // Features are every column except the last. The target is `medv`.
            features.extend_from_slice(&[
                crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat,
            ]);
            targets.push(medv);
        }

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
    /// This method triggers lazy loading on first call. Later calls return the
    /// cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(506, 13)` containing:
    ///     - CRIM - per capita crime rate by town
    ///     - ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    ///     - INDUS - proportion of non-retail business acres per town
    ///     - CHAS - Charles River dummy variable (1 if tract bounds river, 0 otherwise)
    ///     - NOX - nitric oxides concentration (parts per 10 million)
    ///     - RM - average number of rooms per dwelling
    ///     - AGE - proportion of owner-occupied units built before 1940
    ///     - DIS - weighted distances to five Boston employment centers
    ///     - RAD - index of accessibility to radial highways
    ///     - TAX - full-value property-tax rate per $10,000
    ///     - PTRATIO - pupil-teacher ratio by town
    ///     - B - 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town
    ///     - LSTAT - % lower status of the population
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match expected dimensions (506 samples, 13 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the target vector.
    ///
    /// This method triggers lazy loading on first call. Later calls return the
    /// cached data instantly.
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
    /// - Dataset size does not match expected dimensions (506 samples)
    pub fn targets(&self) -> Result<&Array1<f64>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and targets as references.
    ///
    /// This method triggers lazy loading on first call. Later calls return the
    /// cached data instantly.
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
    /// - Dataset size does not match expected dimensions (506 samples, 13 features)
    pub fn data(&self) -> Result<&BostonHousingData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and targets as references **without** triggering loading.
    ///
    /// Unlike [`BostonHousing::data`], which loads the dataset on first call, this
    /// never runs the loader. If the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. If the data is already cached
    /// and you want to avoid the download and parse cost, use this method.
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
    /// This lets you change the cached arrays directly (e.g. normalize features,
    /// rescale targets), with no `to_owned()` clone. The cache keeps the change: it
    /// does not remove the arrays. Later calls to [`BostonHousing::features`],
    /// [`BostonHousing::data`], or [`BostonHousing::get_data`] observe the change.
    ///
    /// Like [`BostonHousing::get_data`], this does **not** trigger loading: it
    /// returns `None` if the dataset has not been loaded. If you need to make sure
    /// the data is present, call a loading accessor first (e.g.
    /// [`BostonHousing::data`]).
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
    /// out and returns owned arrays directly. It needs no `to_owned()` clone. If
    /// the dataset is not loaded yet, this call loads it.
    ///
    /// This consumes `self`, so you cannot use the instance afterward. If you want
    /// owned data but need to keep using the instance, use
    /// [`BostonHousing::take_data`] instead. It takes `&mut self` and leaves the
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
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and targets out of the dataset, leaving it reusable.
    ///
    /// Like [`BostonHousing::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. Unlike that method, it takes `&mut self` instead of
    /// consuming the instance. It moves the cached data out and resets the
    /// instance to its unloaded state. The next accessor call (e.g.
    /// [`BostonHousing::features`] or [`BostonHousing::data`]) loads the dataset
    /// again.
    ///
    /// If you are done with the instance, use [`BostonHousing::into_data`] instead.
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
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(BostonHousing, BostonHousingData, "boston_housing");
