//! California Housing dataset.
//!
//! This dataset has median house values for California districts (block groups),
//! derived from the 1990 U.S. census. It is a common regression benchmark and a
//! modern replacement for the deprecated Boston Housing dataset.
//!
//! This loader reproduces the **scikit-learn** `fetch_california_housing` feature
//! set. Instead of exposing the raw census columns, it derives the same eight
//! per-district features that sklearn uses. The source file is the widely
//! mirrored `housing.csv` from Géron's *Hands-On Machine Learning*. Its raw
//! columns are `longitude`, `latitude`, `housing_median_age`, `total_rooms`,
//! `total_bedrooms`, `population`, `households`, `median_income`,
//! `median_house_value`, and `ocean_proximity`. This loader combines those
//! columns into the sklearn features below.
//!
//! **Features (8):** in sklearn column order
//! - `MedInc` - median income in block group (tens of thousands of USD)
//! - `HouseAge` - median house age in block group
//! - `AveRooms` - average number of rooms per household (`total_rooms / households`)
//! - `AveBedrms` - average number of bedrooms per household (`total_bedrooms / households`)
//! - `Population` - block group population
//! - `AveOccup` - average household occupancy (`population / households`)
//! - `Latitude` - block group latitude
//! - `Longitude` - block group longitude
//!
//! **Target:** `MedHouseVal` - median house value, in units of $100,000
//!   (`median_house_value / 100000`), matching sklearn.
//!
//! **Samples:** 20,640
//! **Application:** Regression / median house value prediction
//!
//! **Missing values:** Géron's file omits `total_bedrooms` from 207 rows on
//! purpose, to teach imputation. Those rows yield `NaN` in `AveBedrms`. Sklearn's
//! complete upstream source has no missing values.
//!
//! **Source:** Pace, R. Kelley and Ronald Barry (1997), "Sparse Spatial
//! Autoregressions," *Statistics and Probability Letters*. Distributed via
//! Géron's *Hands-On Machine Learning* repository.

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::fs::File;

/// The URL for the California Housing dataset.
///
/// # Citation
///
/// R. Kelley Pace and Ronald Barry. "Sparse Spatial Autoregressions,"
/// Statistics and Probability Letters, 33 (1997) 291-297.
const CALIFORNIA_HOUSING_DATA_URL: &str =
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv";

/// The name of the California Housing dataset file.
const CALIFORNIA_HOUSING_FILENAME: &str = "california_housing.csv";

/// The SHA256 hash of the California Housing dataset file.
const CALIFORNIA_HOUSING_SHA256: &str =
    "8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e";

/// The name of the dataset.
const CALIFORNIA_HOUSING_DATASET_NAME: &str = "california_housing";

/// The number of derived (sklearn) features per sample.
const N_FEATURES: usize = 8;

/// The divisor sklearn applies to `median_house_value` to produce a target in
/// units of $100,000.
const TARGET_SCALE: f64 = 100_000.0;

/// Type alias for the California Housing dataset: (features, targets).
type CaliforniaHousingData = (Array2<f64>, Array1<f64>);

/// This struct represents one CSV record of the California Housing dataset. Its
/// fields follow the source column order: `longitude`, `latitude`,
/// `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`,
/// `households`, `median_income`, `median_house_value`, `ocean_proximity`.
///
/// `total_bedrooms` is `Option<f64>` because 207 rows leave it empty, and those
/// rows become `NaN` in the derived `AveBedrms`. The struct keeps
/// `ocean_proximity` only to consume its column positionally. The sklearn feature
/// set does not use it. The struct declares its fields in CSV column order. The
/// loader disables csv's header handling, so csv deserializes the fields
/// **positionally**. This design makes the struct independent of the exact
/// header spelling.
#[derive(Deserialize)]
struct HousingRecord {
    longitude: f64,
    latitude: f64,
    housing_median_age: f64,
    total_rooms: f64,
    total_bedrooms: Option<f64>,
    population: f64,
    households: f64,
    median_income: f64,
    median_house_value: f64,
    /// This field is not part of the sklearn feature set. It exists only to
    /// consume the final CSV column. The loader never reads it.
    #[allow(dead_code)]
    ocean_proximity: String,
}

/// This struct represents the California Housing dataset and loads it lazily.
///
/// The dataset loads only when you call a data accessor method. Later calls
/// return the cached data without loading again.
///
/// # About Dataset
///
/// This dataset describes the houses in a California district (block group). It
/// also has summary statistics about those houses, based on the 1990 census. The
/// target is the median house value for the district. This loader reproduces
/// scikit-learn's `fetch_california_housing` feature set. It derives eight
/// per-district features from the raw census columns.
///
/// # Feature columns
///
/// The eight features reproduce scikit-learn's `fetch_california_housing` set.
/// The loader derives them per district from the raw census columns. The three
/// per-household ratios are `AveRooms = total_rooms / households`,
/// `AveBedrms = total_bedrooms / households`, and
/// `AveOccup = population / households`. A missing `total_bedrooms` yields `NaN`
/// in `AveBedrms`. By 0-based column index in the feature matrix:
///
/// | Columns | Attributes   | Unit                    |
/// |---------|--------------|-------------------------|
/// | `0`     | `MedInc`     | tens of thousands of USD |
/// | `1`     | `HouseAge`   |                         |
/// | `2`     | `AveRooms`   |                         |
/// | `3`     | `AveBedrms`  |                         |
/// | `4`     | `Population` |                         |
/// | `5`     | `AveOccup`   |                         |
/// | `6`     | `Latitude`   | degrees                 |
/// | `7`     | `Longitude`  | degrees                 |
///
/// # Targets
///
/// - `MedHouseVal` - median house value in units of $100,000
///
/// Missing values: the source file has 207 rows with a missing `total_bedrooms`
/// value. These rows yield `NaN` in the derived `AveBedrms` feature.
///
/// See more information at <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html>
///
/// # Citation
///
/// R. Kelley Pace and Ronald Barry. "Sparse Spatial Autoregressions,"
/// Statistics and Probability Letters, 33 (1997) 291-297.
///
/// # Thread Safety
///
/// All fields of this struct implement `Send` and `Sync`, so the struct implements
/// them too. This makes the struct safe to share across threads. The internal
/// [`Dataset`] makes sure initialization is lazy and thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::CaliforniaHousing;
///
/// let download_dir = "./california_housing"; // the code creates the directory if missing
///
/// let mut dataset = CaliforniaHousing::new(download_dir);
/// let features = dataset.features().unwrap();
/// let targets = dataset.targets().unwrap();
///
/// let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
/// assert_eq!(features.shape(), &[20640, 8]);
/// assert_eq!(targets.len(), 20640);
///
/// // `get_data()` borrows the cached arrays and does not reload them.
/// // `get_data_mut()` edits the arrays in place. It makes no clone, and the
/// // change stays in the cache. Prefer `get_data_mut()` over `.to_owned()` when
/// // you only need to change values.
/// if let Some((features, targets)) = dataset.get_data_mut() {
///     features[[0, 0]] = 5.0;
///     targets[0] = 4.5;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out with no `.to_owned()` clone. It leaves
/// // the instance reusable. The next access reloads the data from the cached
/// // file.
/// let (owned_features, owned_targets) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[20640, 8]);
/// assert_eq!(owned_targets.len(), 20640);
///
/// // `into_data()` also returns owned arrays with no clone. But it consumes the
/// // instance. Use it when you are done with the dataset.
/// let (owned_features, owned_targets) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[20640, 8]);
/// assert_eq!(owned_targets.len(), 20640);
/// ```
#[derive(Debug)]
pub struct CaliforniaHousing {
    dataset: Dataset<CaliforniaHousingData, DatasetError>,
}

impl CaliforniaHousing {
    /// Create a new CaliforniaHousing instance without loading data.
    ///
    /// The dataset loads on the first call to a data accessor method. This function
    /// only stores the storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - the directory that stores the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - a `CaliforniaHousing` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        CaliforniaHousing {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the California Housing dataset.
    fn load_data(dir: &str) -> Result<CaliforniaHousingData, DatasetError> {
        // Prepare the dataset file
        let file_path = acquire_dataset(
            dir,
            CALIFORNIA_HOUSING_FILENAME,
            CALIFORNIA_HOUSING_DATASET_NAME,
            Some(CALIFORNIA_HOUSING_SHA256),
            |temp_path| {
                download_to_with_retries(
                    CALIFORNIA_HOUSING_DATA_URL,
                    temp_path,
                    Some(CALIFORNIA_HOUSING_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(CALIFORNIA_HOUSING_FILENAME))
            },
        )?;

        // csv deserializes into the struct. The file has a header row, so skip it.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features = Vec::new();
        let mut targets = Vec::new();

        for result in rdr.deserialize::<HousingRecord>().skip(1) {
            let HousingRecord {
                longitude,
                latitude,
                housing_median_age,
                total_rooms,
                total_bedrooms,
                population,
                households,
                median_income,
                median_house_value,
                ocean_proximity: _,
            } = result
                .map_err(|e| DatasetError::csv_read_error(CALIFORNIA_HOUSING_DATASET_NAME, e))?;

            // Derive sklearn's eight features. `households >= 1` throughout the
            // dataset, so the per-household ratios never divide by zero. A missing
            // `total_bedrooms` propagates to `NaN` in `AveBedrms`.
            features.extend_from_slice(&[
                median_income,                                       // MedInc
                housing_median_age,                                  // HouseAge
                total_rooms / households,                            // AveRooms
                total_bedrooms.map_or(f64::NAN, |b| b / households), // AveBedrms
                population,                                          // Population
                population / households,                             // AveOccup
                latitude,                                            // Latitude
                longitude,                                           // Longitude
            ]);

            // Target, scaled to units of $100,000 as sklearn does.
            targets.push(median_house_value / TARGET_SCALE);
        }

        let n_samples = targets.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(CALIFORNIA_HOUSING_DATASET_NAME));
        }

        // California Housing has a fixed schema of 8 derived features per sample.
        let features_array =
            Array2::from_shape_vec((n_samples, N_FEATURES), features).map_err(|e| {
                DatasetError::array_shape_error(CALIFORNIA_HOUSING_DATASET_NAME, "features", e)
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
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(20640, 8)`
    ///   containing the sklearn features (`MedInc`, `HouseAge`, `AveRooms`,
    ///   `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`).
    ///   `AveBedrms` is `NaN` for the 207 rows with a missing `total_bedrooms`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match expected dimensions (20640 samples, 8 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the target vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<f64>` - Reference to target vector with shape `(20640,)`
    ///   containing median house values in units of $100,000.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match expected dimensions (20640 samples)
    pub fn targets(&self) -> Result<&Array1<f64>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and targets as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&CaliforniaHousingData` - reference to the cached `(features, targets)`
    ///   tuple: the feature matrix has shape `(20640, 8)` and the target vector
    ///   has shape `(20640,)` (median house value in units of $100,000).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match expected dimensions (20640 samples, 8 features)
    pub fn data(&self) -> Result<&CaliforniaHousingData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and targets as references **without** triggering loading.
    ///
    /// Unlike [`CaliforniaHousing::data`], this method never runs the loader. If
    /// the dataset has not loaded yet, it returns `None` instead of downloading
    /// and parsing the data. Use this method when you want the data only if it is
    /// already cached. This choice avoids the download and parse cost.
    ///
    /// # Returns
    ///
    /// - `Some(&CaliforniaHousingData)` - reference to the cached `(features,
    ///   targets)` tuple (feature matrix `(20640, 8)`, target vector `(20640,)`),
    ///   if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&CaliforniaHousingData> {
        self.dataset.get()
    }

    /// Get mutable references to features and targets for **in-place** editing.
    ///
    /// This lets you change the cached arrays directly (for example, normalize
    /// features, or impute the missing `AveBedrms` values) with no `to_owned()`
    /// clone. The changes stay in the cache: later calls to
    /// [`CaliforniaHousing::features`], [`CaliforniaHousing::data`], or
    /// [`CaliforniaHousing::get_data`] see them.
    ///
    /// Like [`CaliforniaHousing::get_data`], this method does **not** trigger
    /// loading. It returns `None` if the dataset has not loaded. If you need to
    /// make sure the data is present, call a loading accessor first (for example,
    /// [`CaliforniaHousing::data`]).
    ///
    /// # Returns
    ///
    /// - `Some(&mut CaliforniaHousingData)` - a mutable reference to the cached
    ///   `(features, targets)` tuple (feature matrix `(20640, 8)`, target vector
    ///   `(20640,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut CaliforniaHousingData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and targets.
    ///
    /// Unlike [`CaliforniaHousing::data`], which borrows the cached data, this
    /// method moves the data out and returns owned arrays. It needs no
    /// `to_owned()` clone. The dataset loads on the first access if it has not
    /// loaded yet.
    ///
    /// This **consumes** `self`, so you cannot use the instance afterward. If you
    /// want owned data but need to keep using the instance, use
    /// [`CaliforniaHousing::take_data`] instead. It takes `&mut self` and leaves
    /// the instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<f64>)` - an owned feature matrix with shape
    ///   `(20640, 8)` and an owned target vector with shape `(20640,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<CaliforniaHousingData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and targets out of the dataset. Leave the dataset
    /// reusable.
    ///
    /// Like [`CaliforniaHousing::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out. This resets the instance to its
    /// unloaded state. The next accessor call (for example,
    /// [`CaliforniaHousing::features`] or [`CaliforniaHousing::data`]) loads the
    /// dataset again.
    ///
    /// If you are done with the instance, use [`CaliforniaHousing::into_data`]
    /// instead.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<f64>)` - an owned feature matrix with shape
    ///   `(20640, 8)` and an owned target vector with shape `(20640,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<CaliforniaHousingData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(
    CaliforniaHousing,
    CaliforniaHousingData,
    "california_housing"
);
