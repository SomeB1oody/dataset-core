//! Bike Sharing dataset, hourly subset (`hour.csv`).
//!
//! One record per hour of the years 2011 and 2012, with the rental counts of
//! the Capital Bikeshare system in Washington, D.C.
//!
//! **Columns (16):**
//!
//! | Name         | Type      | Description                     |
//! |--------------|-----------|----------------------------------|
//! | `dteday`     | `String`  | calendar date as `YYYY-MM-DD`, from `2011-01-01` to `2012-12-31` |
//! | `season`     | `Numeric` | `1` = winter, `2` = spring, `3` = summer, `4` = fall |
//! | `yr`         | `Numeric` | `0` = 2011, `1` = 2012          |
//! | `mnth`       | `Numeric` | month, `1` to `12`              |
//! | `hr`         | `Numeric` | hour, `0` to `23`               |
//! | `holiday`    | `Numeric` | `1` on a holiday, else `0`      |
//! | `weekday`    | `Numeric` | `0` = Sunday to `6` = Saturday  |
//! | `workingday` | `Numeric` | `1` on a day that is neither a weekend nor a holiday, else `0` |
//! | `weathersit` | `Numeric` | `1` = clear, `2` = mist, `3` = light rain or snow, `4` = heavy rain or snow |
//! | `temp`       | `Numeric` | temperature in Celsius, divided by 41 |
//! | `atemp`      | `Numeric` | apparent temperature in Celsius, divided by 50 |
//! | `hum`        | `Numeric` | humidity, divided by 100        |
//! | `windspeed`  | `Numeric` | wind speed, divided by 67       |
//! | `casual`     | `Numeric` | rentals by users without a membership |
//! | `registered` | `Numeric` | rentals by members              |
//! | `cnt`        | `Numeric` | total rentals, the sum of `casual` and `registered` |
//!
//! The source designates the twelve weather and calendar columns as the
//! inputs ([`BikeSharingHourly::FEATURE_NAMES`](crate::BikeSharingHourly::FEATURE_NAMES)) and `casual`, `registered`,
//! and `cnt` as the labels ([`BikeSharingHourly::TARGET_NAMES`](crate::BikeSharingHourly::TARGET_NAMES)).
//!
//! **Samples:** 17,379
//! **Application:** Regression / hourly demand forecasting
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5W894>

use super::{acquire_bike_csv, parse_bike_data};
use crate::table::Table;
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError};

/// The name of the file inside the archive that this loader uses.
const BIKE_HOURLY_SOURCE_FILENAME: &str = "hour.csv";

/// The name of the final cached hourly Bike Sharing dataset file.
const BIKE_HOURLY_FILENAME: &str = "bike_sharing_hourly.csv";

/// The SHA256 hash of the cached hourly Bike Sharing dataset file (`hour.csv`).
const BIKE_HOURLY_SHA256: &str = "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f";

/// The name of the dataset.
const BIKE_HOURLY_DATASET_NAME: &str = "bike_sharing_hourly";

/// Number of samples in the hourly subset.
const N_SAMPLES: usize = 17_379;

/// Number of features per sample.
const N_FEATURES: usize = 12;

/// Number of target columns.
const N_TARGETS: usize = 3;

/// A struct that represents the hourly Bike Sharing dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the
/// first load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The hourly Bike Sharing subset holds one record per hour of the years 2011
/// and 2012, from the Capital Bikeshare system in Washington, D.C. Each record
/// joins the rental counts of that hour with the calendar attributes and the
/// weather readings. The task is to predict the rental count. Every record
/// carries its date, and the rows stay in chronological order. The subset
/// therefore also suits time-series work, such as a split by time.
///
/// # Columns
///
/// | Name         | Type      | Description                     |
/// |--------------|-----------|----------------------------------|
/// | `dteday`     | `String`  | calendar date as `YYYY-MM-DD`, from `2011-01-01` to `2012-12-31` |
/// | `season`     | `Numeric` | `1` = winter, `2` = spring, `3` = summer, `4` = fall |
/// | `yr`         | `Numeric` | `0` = 2011, `1` = 2012          |
/// | `mnth`       | `Numeric` | month, `1` to `12`              |
/// | `hr`         | `Numeric` | hour, `0` to `23`               |
/// | `holiday`    | `Numeric` | `1` on a holiday, else `0`      |
/// | `weekday`    | `Numeric` | `0` = Sunday to `6` = Saturday  |
/// | `workingday` | `Numeric` | `1` on a day that is neither a weekend nor a holiday, else `0` |
/// | `weathersit` | `Numeric` | `1` = clear, `2` = mist, `3` = light rain or snow, `4` = heavy rain or snow |
/// | `temp`       | `Numeric` | temperature in Celsius, divided by 41 |
/// | `atemp`      | `Numeric` | apparent temperature in Celsius, divided by 50 |
/// | `hum`        | `Numeric` | humidity, divided by 100        |
/// | `windspeed`  | `Numeric` | wind speed, divided by 67       |
/// | `casual`     | `Numeric` | rentals by users without a membership |
/// | `registered` | `Numeric` | rentals by members              |
/// | `cnt`        | `Numeric` | total rentals, the sum of `casual` and `registered` |
///
/// The source designates the twelve weather and calendar columns as the
/// inputs ([`BikeSharingHourly::FEATURE_NAMES`]) and `casual`, `registered`,
/// and `cnt` as the labels ([`BikeSharingHourly::TARGET_NAMES`]).
///
/// # Dates
///
/// The subset holds 17,379 of the 17,544 hours of the two years, because the
/// source omits the hours with no rental activity. As a result, the gap between
/// two neighboring rows is not always one hour. One `dteday` value repeats once
/// per recorded hour of that day.
///
/// The date does not repeat the `yr` and `mnth` features, which are integer
/// codes. Keep the date to recover the exact day, or to join the data with
/// another source.
///
/// # Features
///
/// The source normalizes `temp`, `atemp`, `hum`, and `windspeed` to `[0, 1]`.
/// To read a value in its physical unit, multiply it by the divisor in the
/// table.
///
/// The `weathersit` code `4` is rare: 3 of the 17,379 records carry it.
///
/// The archive's own `Readme.txt` maps `season` to `1:springer, 2:summer,
/// 3:fall, 4:winter`. The dates in the data disagree: season `1` runs from
/// December 21 to March 20, which is winter in the northern hemisphere. The
/// table above follows the dates, as the UCI web page does.
///
/// # Targets
///
/// The three target columns make a multi-output regression target, like
/// [`Linnerud`](crate::Linnerud). Most published work predicts `cnt` alone. For
/// that task, take the `cnt` column alone.
///
/// `cnt` is the exact sum of `casual` and `registered` in every record. Never
/// train on `casual` or `registered` as a feature for `cnt`. That leaks the
/// answer.
///
/// Missing values: none. No field is empty in this subset.
///
/// See more information at <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>.
///
/// # Citation
///
/// Fanaee-T, H. (2013). Bike Sharing \[Dataset\]. UCI Machine Learning
/// Repository. <https://doi.org/10.24432/C5W894>
///
/// Fanaee-T, H. & Gama, J. (2013). "Event labeling combining ensemble detectors
/// and background knowledge." *Progress in Artificial Intelligence*, 2, 113-127.
/// <https://doi.org/10.1007/s13748-013-0040-3>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::BikeSharingHourly;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./bike_sharing";
///
/// let mut dataset = BikeSharingHourly::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 17379);
/// assert_eq!(table.n_columns(), 16);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&BikeSharingHourly::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[17379, 12]);
///
/// // Reach the total rental count by name.
/// let cnt = table.column("cnt").unwrap().as_numeric().unwrap();
/// assert_eq!(cnt.len(), 17379);
///
/// // Reach the date column by name.
/// let dates = table.column("dteday").unwrap().as_string().unwrap();
/// assert_eq!(dates[0], "2011-01-01");
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("temp") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] *= 41.0;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 17379);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 17379);
/// ```
#[derive(Debug)]
pub struct BikeSharingHourly {
    dataset: Dataset<Table, DatasetError>,
}

impl BikeSharingHourly {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
    ];

    /// The columns the source designates as the labels, in source order.
    pub const TARGET_NAMES: [&'static str; N_TARGETS] = ["casual", "registered", "cnt"];

    /// Create a new BikeSharingHourly instance without loading data.
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
    /// - `Self` - a `BikeSharingHourly` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        BikeSharingHourly {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the hourly Bike Sharing dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_bike_csv(
            dir,
            BIKE_HOURLY_FILENAME,
            BIKE_HOURLY_DATASET_NAME,
            BIKE_HOURLY_SHA256,
            BIKE_HOURLY_SOURCE_FILENAME,
        )?;

        parse_bike_data(
            BIKE_HOURLY_DATASET_NAME,
            &file_path,
            &Self::FEATURE_NAMES,
            N_SAMPLES,
        )
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 17,379 samples and 16
    ///   columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`BikeSharingHourly::data`], this method never runs the loader. If
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
    /// changes stay in the cache. Later calls to [`BikeSharingHourly::data`] or
    /// [`BikeSharingHourly::get_data`] see them.
    ///
    /// Like [`BikeSharingHourly::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`BikeSharingHourly::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 17,379 samples and 16 columns.
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
    /// - `Table` - the owned table of 17,379 samples and 16 columns.
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

impl_ml_dataset!(BikeSharingHourly, "bike_sharing_hourly");
