//! Bike Sharing dataset, hourly subset (`hour.csv`).
//!
//! One record per hour of the years 2011 and 2012, with the rental counts of
//! the Capital Bikeshare system in Washington, D.C.
//!
//! **Dates:** `dteday`, from `2011-01-01` to `2012-12-31`, as `YYYY-MM-DD`.
//!
//! **Features (12, all numeric):** `season`, `yr`, `mnth`, `hr`, `holiday`,
//! `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`
//!
//! **Targets (3):** `casual`, `registered`, `cnt`
//!
//! **Samples:** 17,379
//! **Application:** Regression / hourly demand forecasting
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5W894>

use super::{BikeSharingData, acquire_bike_csv, parse_bike_data};
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError};
use ndarray::{Array1, Array2};

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
/// # Dates
///
/// `dteday` (shape `(17379,)`). The `Array1<String>` holds the calendar date of
/// each record as `YYYY-MM-DD`, from `2011-01-01` to `2012-12-31`. The subset
/// holds 17,379 of the 17,544 hours of the two years, because the source omits
/// the hours with no rental activity. As a result, the gap between two
/// neighboring rows is not always one hour.
///
/// The date does not repeat the `yr` and `mnth` features, which are integer
/// codes. Keep the date to recover the exact day, or to join the data with
/// another source.
///
/// # Feature columns
///
/// Features (`Array2<f64>`), by 0-based column:
///
/// | Column | Attribute    | Domain                                          |
/// |--------|--------------|-------------------------------------------------|
/// | `0`    | `season`     | `1` = winter, `2` = spring, `3` = summer, `4` = fall |
/// | `1`    | `yr`         | `0` = 2011, `1` = 2012                          |
/// | `2`    | `mnth`       | `1` to `12`                                     |
/// | `3`    | `hr`         | `0` to `23`                                     |
/// | `4`    | `holiday`    | `1` on a holiday, else `0`                      |
/// | `5`    | `weekday`    | `0` = Sunday to `6` = Saturday                  |
/// | `6`    | `workingday` | `1` on a day that is neither a weekend nor a holiday, else `0` |
/// | `7`    | `weathersit` | `1` = clear, `2` = mist, `3` = light rain or snow, `4` = heavy rain or snow |
/// | `8`    | `temp`       | temperature in Celsius, divided by 41           |
/// | `9`    | `atemp`      | apparent temperature in Celsius, divided by 50  |
/// | `10`   | `hum`        | humidity, divided by 100                        |
/// | `11`   | `windspeed`  | wind speed, divided by 67                       |
///
/// The source normalizes the last four columns to `[0, 1]`. To read a value in
/// its physical unit, multiply it by the divisor in the table.
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
/// Targets (`Array2<f64>`), by 0-based column:
///
/// | Column | Attribute    | Description                     |
/// |--------|--------------|---------------------------------|
/// | `0`    | `casual`     | rentals by users without a membership |
/// | `1`    | `registered` | rentals by members              |
/// | `2`    | `cnt`        | total rentals, the sum of `casual` and `registered` |
///
/// This is a multi-output regression target, like [`Linnerud`](crate::Linnerud).
/// Most published work predicts `cnt` alone. For that task, take the third
/// column with `targets.column(2)`, which copies nothing.
///
/// Column `2` is the exact sum of columns `0` and `1` in every record. Never
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
/// let features = dataset.features().unwrap();
/// let targets = dataset.targets().unwrap();
/// let dates = dataset.dates().unwrap();
///
/// // data() also returns all data at once
/// let (dates, features, targets) = dataset.data().unwrap();
/// assert_eq!(dates.len(), 17379);
/// assert_eq!(features.shape(), &[17379, 12]);
/// assert_eq!(targets.shape(), &[17379, 3]);
///
/// // The total rental count `cnt` is the third target column.
/// let cnt = targets.column(2);
/// assert_eq!(cnt.len(), 17379);
///
/// // `get_data_mut()` edits the arrays in place. This needs no clone and no
/// // reload. The change stays cached. If you only need to change values,
/// // prefer this method over `.to_owned()`.
/// if let Some((_dates, features, _targets)) = dataset.get_data_mut() {
///     features[[0, 0]] = 2.0;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out with no `to_owned()` clone. This
/// // leaves the instance reusable. The next access reloads the data from the
/// // cached file.
/// let (owned_dates, owned_features, owned_targets) = dataset.take_data().unwrap();
/// assert_eq!(owned_dates.len(), 17379);
/// assert_eq!(owned_features.shape(), &[17379, 12]);
/// assert_eq!(owned_targets.shape(), &[17379, 3]);
///
/// // `into_data()` also returns the owned arrays with no clone, but it
/// // consumes the instance. If you are done with the dataset, use it.
/// let (owned_dates, owned_features, owned_targets) = dataset.into_data().unwrap();
/// assert_eq!(owned_dates.len(), 17379);
/// assert_eq!(owned_features.shape(), &[17379, 12]);
/// assert_eq!(owned_targets.shape(), &[17379, 3]);
/// ```
#[derive(Debug)]
pub struct BikeSharingHourly {
    dataset: Dataset<BikeSharingData, DatasetError>,
}

impl BikeSharingHourly {
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
    fn load_data(dir: &str) -> Result<BikeSharingData, DatasetError> {
        let file_path = acquire_bike_csv(
            dir,
            BIKE_HOURLY_FILENAME,
            BIKE_HOURLY_DATASET_NAME,
            BIKE_HOURLY_SHA256,
            BIKE_HOURLY_SOURCE_FILENAME,
        )?;

        parse_bike_data(BIKE_HOURLY_DATASET_NAME, &file_path, N_FEATURES, N_SAMPLES)
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(17379, 12)` containing:
    ///     - `season`
    ///     - `yr`
    ///     - `mnth`
    ///     - `hr`
    ///     - `holiday`
    ///     - `weekday`
    ///     - `workingday`
    ///     - `weathersit`
    ///     - `temp`
    ///     - `atemp`
    ///     - `hum`
    ///     - `windspeed`
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (17,379 samples)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get a reference to the target matrix.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to target matrix with shape `(17379, 3)` containing
    ///   `casual`, `registered`, and `cnt`. Column `2` (`cnt`) is the sum of columns
    ///   `0` and `1`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (17,379 samples)
    pub fn targets(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.2)
    }

    /// Get a reference to the date vector.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array1<String>` - Reference to date vector with shape `(17379,)`. Each
    ///   entry is the calendar date of one record, as `YYYY-MM-DD`. The dates run
    ///   from `2011-01-01` to `2012-12-31` in chronological order, and one date
    ///   repeats for each recorded hour of that day.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (17,379 samples)
    pub fn dates(&self) -> Result<&Array1<String>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get dates, features, and targets as references.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&BikeSharingData` - reference to the cached `(dates, features, targets)`
    ///   tuple: date vector `(17379,)`, feature matrix `(17379, 12)`, and target
    ///   matrix `(17379, 3)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (17,379 samples)
    pub fn data(&self) -> Result<&BikeSharingData, DatasetError> {
        self.dataset.load()
    }

    /// Get dates, features, and targets as references **without** triggering
    /// loading.
    ///
    /// Unlike [`BikeSharingHourly::data`], this method never runs the loader. If
    /// the data has not loaded yet, it returns `None` instead of downloading and
    /// parsing it. Use this method when you want the data only if it is already
    /// cached. This skips the cost of a download and a parse.
    ///
    /// # Returns
    ///
    /// - `Some(&BikeSharingData)` - reference to the cached `(dates, features,
    ///   targets)` tuple (`(17379,)`, `(17379, 12)`, `(17379, 3)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&BikeSharingData> {
        self.dataset.get()
    }

    /// Get mutable references to dates, features, and targets for **in-place**
    /// editing.
    ///
    /// This lets you change the cached arrays directly. For example, you can scale
    /// the normalized weather columns back to their physical units. This needs no
    /// `.to_owned()` clone, and it does not remove the data from the cache. The
    /// changes stay in the cache. Later calls to [`BikeSharingHourly::features`],
    /// [`BikeSharingHourly::data`], or [`BikeSharingHourly::get_data`] see the
    /// changes.
    ///
    /// Like [`BikeSharingHourly::get_data`], this does **not** trigger loading. It
    /// returns `None` if the dataset has not loaded yet. If you need the data to be
    /// present, call a loading accessor first, for example
    /// [`BikeSharingHourly::data`].
    ///
    /// # Returns
    ///
    /// - `Some(&mut BikeSharingData)` - mutable reference to the cached `(dates,
    ///   features, targets)` tuple (`(17379,)`, `(17379, 12)`, `(17379, 3)`), if
    ///   loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut BikeSharingData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** dates, features, and targets.
    ///
    /// Unlike [`BikeSharingHourly::data`], which borrows the cached data, this
    /// moves the data out and returns owned arrays directly. It needs no
    /// `to_owned()` clone. If the dataset has not loaded yet, the first access
    /// loads it.
    ///
    /// This **consumes** `self`. After the call, you cannot use the instance again.
    /// If you want owned data but need to keep using the instance, use
    /// [`BikeSharingHourly::take_data`] instead. It takes `&mut self` and leaves
    /// the instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array2<f64>, Array2<f64>)` - owned date vector
    ///   `(17379,)`, owned feature matrix `(17379, 12)`, and owned target matrix
    ///   `(17379, 3)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<BikeSharingData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** dates, features, and targets out of the dataset. This
    /// leaves the instance reusable.
    ///
    /// Like [`BikeSharingHourly::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. Instead of consuming the instance, it takes `&mut self`
    /// and moves the cached data out. This resets the instance to its unloaded
    /// state. The next accessor call, for example [`BikeSharingHourly::features`]
    /// or [`BikeSharingHourly::data`], loads the dataset again.
    ///
    /// If you are done with the instance, use [`BikeSharingHourly::into_data`]
    /// instead.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array2<f64>, Array2<f64>)` - owned date vector
    ///   `(17379,)`, owned feature matrix `(17379, 12)`, and owned target matrix
    ///   `(17379, 3)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<BikeSharingData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(BikeSharingHourly, BikeSharingData, "bike_sharing_hourly");
