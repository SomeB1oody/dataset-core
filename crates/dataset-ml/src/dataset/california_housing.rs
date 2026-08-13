//! California Housing dataset.
//!
//! This dataset has median house values for California districts (block groups),
//! derived from the 1990 U.S. census. It is a common regression benchmark and a
//! replacement for the deprecated Boston Housing dataset.
//!
//! This loader reproduces the **scikit-learn** `fetch_california_housing` feature
//! set. Instead of exposing the raw census columns, it derives the same eight
//! per-district features that sklearn uses. The source file is the widely
//! mirrored `housing.csv` from Géron's *Hands-On Machine Learning*. Its raw
//! columns are `longitude`, `latitude`, `housing_median_age`, `total_rooms`,
//! `total_bedrooms`, `population`, `households`, `median_income`,
//! `median_house_value`, and `ocean_proximity`. This loader combines those
//! columns into the sklearn columns below.
//!
//! **Columns (9):** in sklearn column order
//!
//! | Name          | Type      | Description                                                     |
//! |---------------|-----------|---------------------------------------------------------------------|
//! | `MedInc`      | `Numeric` | median income in the block group, in tens of thousands of USD   |
//! | `HouseAge`    | `Numeric` | median house age in the block group                             |
//! | `AveRooms`    | `Numeric` | average rooms per household, `total_rooms / households`          |
//! | `AveBedrms`   | `Numeric` | average bedrooms per household, `total_bedrooms / households`    |
//! | `Population`  | `Numeric` | block group population                                          |
//! | `AveOccup`    | `Numeric` | average household occupancy, `population / households`           |
//! | `Latitude`    | `Numeric` | block group latitude in degrees                                 |
//! | `Longitude`   | `Numeric` | block group longitude in degrees                                |
//! | `MedHouseVal` | `Numeric` | median house value in units of $100,000                         |
//!
//! The source designates the first eight columns as the inputs
//! ([`CaliforniaHousing::FEATURE_NAMES`](crate::CaliforniaHousing::FEATURE_NAMES)) and `MedHouseVal` as the label
//! ([`CaliforniaHousing::TARGET`](crate::CaliforniaHousing::TARGET)). The target is `median_house_value / 100000`,
//! which matches sklearn.
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
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
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

/// Number of samples.
const N_SAMPLES: usize = 20_640;

/// The divisor sklearn applies to `median_house_value` to produce a target in
/// units of $100,000.
const TARGET_SCALE: f64 = 100_000.0;

/// This struct represents one CSV record of the California Housing dataset. Its
/// fields follow the source column order: `longitude`, `latitude`,
/// `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`,
/// `households`, `median_income`, `median_house_value`, `ocean_proximity`.
///
/// `total_bedrooms` is `Option<f64>` because 207 rows leave it empty, and those
/// rows become `NaN` in the derived `AveBedrms`. The struct keeps
/// `ocean_proximity` only to consume its column positionally. The sklearn column
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
    /// This field is not part of the sklearn column set. It exists only to
    /// consume the final CSV column. The loader never reads it.
    #[allow(dead_code)]
    ocean_proximity: String,
}

/// A struct that represents the California Housing dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// This dataset describes the houses in a California district (block group). It
/// also has summary statistics about those houses, based on the 1990 census. The
/// target is the median house value for the district. This loader reproduces
/// scikit-learn's `fetch_california_housing` column set. It derives eight
/// per-district features from the raw census columns.
///
/// # Columns
///
/// The eight features reproduce scikit-learn's `fetch_california_housing` set.
/// The loader derives them per district from the raw census columns. A missing
/// `total_bedrooms` yields `NaN` in `AveBedrms`. The columns keep sklearn column
/// order:
///
/// | Name          | Type      | Description                                                     |
/// |---------------|-----------|---------------------------------------------------------------------|
/// | `MedInc`      | `Numeric` | median income in the block group, in tens of thousands of USD   |
/// | `HouseAge`    | `Numeric` | median house age in the block group                             |
/// | `AveRooms`    | `Numeric` | average rooms per household, `total_rooms / households`          |
/// | `AveBedrms`   | `Numeric` | average bedrooms per household, `total_bedrooms / households`    |
/// | `Population`  | `Numeric` | block group population                                          |
/// | `AveOccup`    | `Numeric` | average household occupancy, `population / households`           |
/// | `Latitude`    | `Numeric` | block group latitude in degrees                                 |
/// | `Longitude`   | `Numeric` | block group longitude in degrees                                |
/// | `MedHouseVal` | `Numeric` | median house value in units of $100,000                         |
///
/// The source designates the first eight columns as the inputs
/// ([`CaliforniaHousing::FEATURE_NAMES`]) and `MedHouseVal` as the label
/// ([`CaliforniaHousing::TARGET`]).
///
/// Missing values: the source file has 207 rows with a missing `total_bedrooms`
/// value. These rows yield `NaN` in the derived `AveBedrms` column.
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
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::CaliforniaHousing;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./california_housing";
///
/// let mut dataset = CaliforniaHousing::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 20640);
/// assert_eq!(table.n_columns(), 9);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&CaliforniaHousing::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[20640, 8]);
///
/// // Reach one column by name.
/// let target = table.column(CaliforniaHousing::TARGET).unwrap().as_numeric().unwrap();
/// assert_eq!(target.len(), 20640);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("MedInc") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 5.0;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 20640);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 20640);
/// ```
#[derive(Debug)]
pub struct CaliforniaHousing {
    dataset: Dataset<Table, DatasetError>,
}

impl CaliforniaHousing {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; 8] = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "MedHouseVal";

    /// Create a new CaliforniaHousing instance without loading data.
    ///
    /// The dataset loads lazily, on your first call to a data accessor method.
    /// This is a lightweight operation that only stores the storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - The directory that stores the dataset.
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
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
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

        // The csv crate deserializes into the struct. The file has a header row,
        // so the loader skips it.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut med_inc = Vec::with_capacity(N_SAMPLES);
        let mut house_age = Vec::with_capacity(N_SAMPLES);
        let mut ave_rooms = Vec::with_capacity(N_SAMPLES);
        let mut ave_bedrms = Vec::with_capacity(N_SAMPLES);
        let mut population_column = Vec::with_capacity(N_SAMPLES);
        let mut ave_occup = Vec::with_capacity(N_SAMPLES);
        let mut latitude_column = Vec::with_capacity(N_SAMPLES);
        let mut longitude_column = Vec::with_capacity(N_SAMPLES);
        let mut med_house_val = Vec::with_capacity(N_SAMPLES);

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

            // `households >= 1` throughout the dataset, so the per-household
            // ratios never divide by zero. A missing `total_bedrooms` propagates
            // to `NaN` in `AveBedrms`.
            med_inc.push(median_income);
            house_age.push(housing_median_age);
            ave_rooms.push(total_rooms / households);
            ave_bedrms.push(total_bedrooms.map_or(f64::NAN, |b| b / households));
            population_column.push(population);
            ave_occup.push(population / households);
            latitude_column.push(latitude);
            longitude_column.push(longitude);

            // Target, scaled to units of $100,000 as sklearn does.
            med_house_val.push(median_house_value / TARGET_SCALE);
        }

        Table::new(
            CALIFORNIA_HOUSING_DATASET_NAME,
            vec![
                Column::new(
                    Self::FEATURE_NAMES[0],
                    ColumnData::Numeric(Array1::from_vec(med_inc)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[1],
                    ColumnData::Numeric(Array1::from_vec(house_age)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[2],
                    ColumnData::Numeric(Array1::from_vec(ave_rooms)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[3],
                    ColumnData::Numeric(Array1::from_vec(ave_bedrms)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[4],
                    ColumnData::Numeric(Array1::from_vec(population_column)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[5],
                    ColumnData::Numeric(Array1::from_vec(ave_occup)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[6],
                    ColumnData::Numeric(Array1::from_vec(latitude_column)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[7],
                    ColumnData::Numeric(Array1::from_vec(longitude_column)),
                ),
                Column::new(
                    Self::TARGET,
                    ColumnData::Numeric(Array1::from_vec(med_house_val)),
                ),
            ],
        )
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 20640 samples and 9
    ///   columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`CaliforniaHousing::data`], this method never runs the loader. If
    /// the data has not loaded yet, it returns `None` instead of downloading and
    /// parsing it.
    ///
    /// # Returns
    ///
    /// - `Some(&Table)` - reference to the cached table, if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&Table> {
        self.dataset.get()
    }

    /// Get a mutable reference to the parsed table for **in-place** editing.
    ///
    /// This needs no clone, and it does not remove the data from the cache. The
    /// changes stay in the cache. Later calls to [`CaliforniaHousing::data`] or
    /// [`CaliforniaHousing::get_data`] see them.
    ///
    /// Like [`CaliforniaHousing::get_data`], this does **not** trigger loading.
    ///
    /// # Returns
    ///
    /// - `Some(&mut Table)` - mutable reference to the cached table, if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut Table> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return the **owned** table.
    ///
    /// This **consumes** `self`. If you want owned data but need to keep using
    /// the instance, use [`CaliforniaHousing::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 20640 samples and 9 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, or parsing).
    pub fn into_data(self) -> Result<Table, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take the **owned** table out of the dataset. This leaves the instance
    /// reusable.
    ///
    /// This resets the instance to its unloaded state. The next accessor call
    /// loads the dataset again.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 20640 samples and 9 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, or parsing).
    pub fn take_data(&mut self) -> Result<Table, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(CaliforniaHousing, "california_housing");
