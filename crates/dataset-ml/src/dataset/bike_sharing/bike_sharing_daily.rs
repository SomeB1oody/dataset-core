//! Bike Sharing dataset, daily subset (`day.csv`).
//!
//! One record per day of the years 2011 and 2012, with the rental counts of the
//! Capital Bikeshare system in Washington, D.C.
//!
//! **Columns (15):**
//!
//! | Name         | Type      | Description                     |
//! |--------------|-----------|----------------------------------|
//! | `dteday`     | `String`  | calendar date as `YYYY-MM-DD`, from `2011-01-01` to `2012-12-31` |
//! | `season`     | `Numeric` | `1` = winter, `2` = spring, `3` = summer, `4` = fall |
//! | `yr`         | `Numeric` | `0` = 2011, `1` = 2012          |
//! | `mnth`       | `Numeric` | month, `1` to `12`              |
//! | `holiday`    | `Numeric` | `1` on a holiday, else `0`      |
//! | `weekday`    | `Numeric` | `0` = Sunday to `6` = Saturday  |
//! | `workingday` | `Numeric` | `1` on a day that is neither a weekend nor a holiday, else `0` |
//! | `weathersit` | `Numeric` | `1` = clear, `2` = mist, `3` = light rain or snow |
//! | `temp`       | `Numeric` | temperature in Celsius, divided by 41 |
//! | `atemp`      | `Numeric` | apparent temperature in Celsius, divided by 50 |
//! | `hum`        | `Numeric` | humidity, divided by 100        |
//! | `windspeed`  | `Numeric` | wind speed, divided by 67       |
//! | `casual`     | `Numeric` | rentals by users without a membership |
//! | `registered` | `Numeric` | rentals by members              |
//! | `cnt`        | `Numeric` | total rentals, the sum of `casual` and `registered` |
//!
//! The source designates the eleven weather and calendar columns as the
//! inputs ([`BikeSharingDaily::FEATURE_NAMES`](crate::BikeSharingDaily::FEATURE_NAMES)) and `casual`, `registered`,
//! and `cnt` as the labels ([`BikeSharingDaily::TARGET_NAMES`](crate::BikeSharingDaily::TARGET_NAMES)).
//!
//! **Samples:** 731
//! **Application:** Regression / daily demand forecasting
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5W894>

use super::{acquire_bike_csv, parse_bike_data};
use crate::table::Table;
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError};

/// The name of the file inside the archive that this loader uses.
const BIKE_DAILY_SOURCE_FILENAME: &str = "day.csv";

/// The name of the final cached daily Bike Sharing dataset file.
const BIKE_DAILY_FILENAME: &str = "bike_sharing_daily.csv";

/// The SHA256 hash of the cached daily Bike Sharing dataset file (`day.csv`).
const BIKE_DAILY_SHA256: &str = "a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be";

/// The name of the dataset.
const BIKE_DAILY_DATASET_NAME: &str = "bike_sharing_daily";

/// Number of samples in the daily subset.
const N_SAMPLES: usize = 731;

/// Number of features per sample.
const N_FEATURES: usize = 11;

/// Number of target columns.
const N_TARGETS: usize = 3;

/// A struct that represents the daily Bike Sharing dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the
/// first load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The daily Bike Sharing subset holds one record per day of the years 2011 and
/// 2012, from the Capital Bikeshare system in Washington, D.C. It aggregates
/// the same rental log as
/// [`BikeSharingHourly`](super::bike_sharing_hourly::BikeSharingHourly), with
/// the hour column dropped. Each daily count is the exact sum of the hourly
/// counts of that date, and each daily weather reading is their mean. The task
/// is to predict the daily rental count. Its two full years of consecutive days
/// suit a split by time.
///
/// # Columns
///
/// | Name         | Type      | Description                     |
/// |--------------|-----------|----------------------------------|
/// | `dteday`     | `String`  | calendar date as `YYYY-MM-DD`, from `2011-01-01` to `2012-12-31` |
/// | `season`     | `Numeric` | `1` = winter, `2` = spring, `3` = summer, `4` = fall |
/// | `yr`         | `Numeric` | `0` = 2011, `1` = 2012          |
/// | `mnth`       | `Numeric` | month, `1` to `12`              |
/// | `holiday`    | `Numeric` | `1` on a holiday, else `0`      |
/// | `weekday`    | `Numeric` | `0` = Sunday to `6` = Saturday  |
/// | `workingday` | `Numeric` | `1` on a day that is neither a weekend nor a holiday, else `0` |
/// | `weathersit` | `Numeric` | `1` = clear, `2` = mist, `3` = light rain or snow |
/// | `temp`       | `Numeric` | temperature in Celsius, divided by 41 |
/// | `atemp`      | `Numeric` | apparent temperature in Celsius, divided by 50 |
/// | `hum`        | `Numeric` | humidity, divided by 100        |
/// | `windspeed`  | `Numeric` | wind speed, divided by 67       |
/// | `casual`     | `Numeric` | rentals by users without a membership |
/// | `registered` | `Numeric` | rentals by members              |
/// | `cnt`        | `Numeric` | total rentals, the sum of `casual` and `registered` |
///
/// The source designates the eleven weather and calendar columns as the
/// inputs ([`BikeSharingDaily::FEATURE_NAMES`]) and `casual`, `registered`,
/// and `cnt` as the labels ([`BikeSharingDaily::TARGET_NAMES`]).
///
/// # Dates
///
/// The subset covers all 731 days of the two years, in chronological order and
/// with no gaps. Each `dteday` value appears once.
///
/// # Features
///
/// The source normalizes `temp`, `atemp`, `hum`, and `windspeed` to `[0, 1]`.
/// They hold the daily mean of the hourly readings. To read a value in its
/// physical unit, multiply it by the divisor in the table.
///
/// The `weathersit` scale also defines a code `4` for heavy rain or snow. No
/// day carries it. The hourly subset does hold that code, on 3 of its records.
/// Those 3 hours fall on `2011-01-26`, `2012-01-09`, and `2012-01-21`, days
/// that carry the daily codes `3`, `2`, and `2`.
///
/// The daily subset has no `hr` column, so its features are the 12 of the
/// hourly subset less that one. Every other column keeps the same name and the
/// same meaning.
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
/// use dataset_ml::BikeSharingDaily;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./bike_sharing";
///
/// let mut dataset = BikeSharingDaily::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 731);
/// assert_eq!(table.n_columns(), 15);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&BikeSharingDaily::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[731, 11]);
///
/// // Reach the total rental count by name.
/// let cnt = table.column("cnt").unwrap().as_numeric().unwrap();
/// assert_eq!(cnt.len(), 731);
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
/// assert_eq!(owned.n_samples(), 731);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 731);
/// ```
#[derive(Debug)]
pub struct BikeSharingDaily {
    dataset: Dataset<Table, DatasetError>,
}

impl BikeSharingDaily {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "season",
        "yr",
        "mnth",
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

    /// Create a new BikeSharingDaily instance without loading data.
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
    /// - `Self` - a `BikeSharingDaily` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        BikeSharingDaily {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the daily Bike Sharing dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_bike_csv(
            dir,
            BIKE_DAILY_FILENAME,
            BIKE_DAILY_DATASET_NAME,
            BIKE_DAILY_SHA256,
            BIKE_DAILY_SOURCE_FILENAME,
        )?;

        parse_bike_data(
            BIKE_DAILY_DATASET_NAME,
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
    /// - `&Table` - reference to the cached table of 731 samples and 15
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
    /// Unlike [`BikeSharingDaily::data`], this method never runs the loader. If
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
    /// changes stay in the cache. Later calls to [`BikeSharingDaily::data`] or
    /// [`BikeSharingDaily::get_data`] see them.
    ///
    /// Like [`BikeSharingDaily::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`BikeSharingDaily::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 731 samples and 15 columns.
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
    /// - `Table` - the owned table of 731 samples and 15 columns.
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

impl_ml_dataset!(BikeSharingDaily, "bike_sharing_daily");
