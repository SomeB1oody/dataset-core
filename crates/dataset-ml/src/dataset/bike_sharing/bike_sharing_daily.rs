//! Bike Sharing dataset, daily subset (`day.csv`).
//!
//! One record per day of the years 2011 and 2012, with the rental counts of the
//! Capital Bikeshare system in Washington, D.C.
//!
//! **Dates:** `dteday`, from `2011-01-01` to `2012-12-31`, as `YYYY-MM-DD`.
//!
//! **Features (11, all numeric):** `season`, `yr`, `mnth`, `holiday`,
//! `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`
//!
//! **Targets (3):** `casual`, `registered`, `cnt`
//!
//! **Samples:** 731
//! **Application:** Regression / daily demand forecasting
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5W894>

use super::{BikeSharingData, acquire_bike_csv, parse_bike_data};
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError};
use ndarray::{Array1, Array2};

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
/// is to predict the daily rental count. At 731 samples it is small enough for
/// a quick experiment, and its two full years of consecutive days suit a split
/// by time.
///
/// # Dates
///
/// `dteday` (shape `(731,)`). The `Array1<String>` holds the calendar date of
/// each record as `YYYY-MM-DD`, from `2011-01-01` to `2012-12-31`. The subset
/// covers all 731 days of the two years, in chronological order and with no
/// gaps. Each date appears once.
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
/// | `3`    | `holiday`    | `1` on a holiday, else `0`                      |
/// | `4`    | `weekday`    | `0` = Sunday to `6` = Saturday                  |
/// | `5`    | `workingday` | `1` on a day that is neither a weekend nor a holiday, else `0` |
/// | `6`    | `weathersit` | `1` = clear, `2` = mist, `3` = light rain or snow |
/// | `7`    | `temp`       | temperature in Celsius, divided by 41           |
/// | `8`    | `atemp`      | apparent temperature in Celsius, divided by 50  |
/// | `9`    | `hum`        | humidity, divided by 100                        |
/// | `10`   | `windspeed`  | wind speed, divided by 67                       |
///
/// The last four columns arrive normalized to `[0, 1]` in the source. They hold
/// the daily mean of the hourly readings. To read a value in its physical unit,
/// multiply it by the divisor in the table.
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
/// December 21 to March 20, which is winter in the northern hemisphere. This
/// loader documents the mapping that the dates support, which is also the
/// mapping the UCI web page lists.
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
/// use dataset_ml::BikeSharingDaily;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./bike_sharing";
///
/// let mut dataset = BikeSharingDaily::new(download_dir);
/// let features = dataset.features().unwrap();
/// let targets = dataset.targets().unwrap();
/// let dates = dataset.dates().unwrap();
///
/// // data() also returns all data at once
/// let (dates, features, targets) = dataset.data().unwrap();
/// assert_eq!(dates.len(), 731);
/// assert_eq!(features.shape(), &[731, 11]);
/// assert_eq!(targets.shape(), &[731, 3]);
///
/// // The total rental count `cnt` is the third target column.
/// let cnt = targets.column(2);
/// assert_eq!(cnt.len(), 731);
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
/// assert_eq!(owned_dates.len(), 731);
/// assert_eq!(owned_features.shape(), &[731, 11]);
/// assert_eq!(owned_targets.shape(), &[731, 3]);
///
/// // `into_data()` also returns the owned arrays with no clone, but it
/// // consumes the instance. If you are done with the dataset, use it.
/// let (owned_dates, owned_features, owned_targets) = dataset.into_data().unwrap();
/// assert_eq!(owned_dates.len(), 731);
/// assert_eq!(owned_features.shape(), &[731, 11]);
/// assert_eq!(owned_targets.shape(), &[731, 3]);
/// ```
#[derive(Debug)]
pub struct BikeSharingDaily {
    dataset: Dataset<BikeSharingData, DatasetError>,
}

impl BikeSharingDaily {
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
    fn load_data(dir: &str) -> Result<BikeSharingData, DatasetError> {
        let file_path = acquire_bike_csv(
            dir,
            BIKE_DAILY_FILENAME,
            BIKE_DAILY_DATASET_NAME,
            BIKE_DAILY_SHA256,
            BIKE_DAILY_SOURCE_FILENAME,
        )?;

        parse_bike_data(BIKE_DAILY_DATASET_NAME, &file_path, N_FEATURES, N_SAMPLES)
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(731, 11)` containing:
    ///     - `season`
    ///     - `yr`
    ///     - `mnth`
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
    /// - Dataset size does not match the expected dimensions (731 samples)
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
    /// - `&Array2<f64>` - Reference to target matrix with shape `(731, 3)` containing
    ///   `casual`, `registered`, and `cnt`. Column `2` (`cnt`) is the sum of columns
    ///   `0` and `1`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (731 samples)
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
    /// - `&Array1<String>` - Reference to date vector with shape `(731,)`. Each
    ///   entry is the calendar date of one record, as `YYYY-MM-DD`. The dates run
    ///   from `2011-01-01` to `2012-12-31` in chronological order, one per day,
    ///   with no gaps.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (731 samples)
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
    ///   tuple: date vector `(731,)`, feature matrix `(731, 11)`, and target matrix
    ///   `(731, 3)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (731 samples)
    pub fn data(&self) -> Result<&BikeSharingData, DatasetError> {
        self.dataset.load()
    }

    /// Get dates, features, and targets as references **without** triggering
    /// loading.
    ///
    /// Unlike [`BikeSharingDaily::data`], this method never runs the loader. If
    /// the data has not loaded yet, it returns `None` instead of downloading and
    /// parsing it. Use this method when you want the data only if it is already
    /// cached. This skips the cost of a download and a parse.
    ///
    /// # Returns
    ///
    /// - `Some(&BikeSharingData)` - reference to the cached `(dates, features,
    ///   targets)` tuple (`(731,)`, `(731, 11)`, `(731, 3)`), if loaded.
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
    /// changes stay in the cache. Later calls to [`BikeSharingDaily::features`],
    /// [`BikeSharingDaily::data`], or [`BikeSharingDaily::get_data`] see the
    /// changes.
    ///
    /// Like [`BikeSharingDaily::get_data`], this does **not** trigger loading. It
    /// returns `None` if the dataset has not loaded yet. If you need the data to be
    /// present, call a loading accessor first, for example
    /// [`BikeSharingDaily::data`].
    ///
    /// # Returns
    ///
    /// - `Some(&mut BikeSharingData)` - mutable reference to the cached `(dates,
    ///   features, targets)` tuple (`(731,)`, `(731, 11)`, `(731, 3)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut BikeSharingData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** dates, features, and targets.
    ///
    /// Unlike [`BikeSharingDaily::data`], which borrows the cached data, this
    /// moves the data out and returns owned arrays directly. It needs no
    /// `to_owned()` clone. If the dataset has not loaded yet, the first access
    /// loads it.
    ///
    /// This **consumes** `self`. After the call, you cannot use the instance again.
    /// If you want owned data but need to keep using the instance, use
    /// [`BikeSharingDaily::take_data`] instead. It takes `&mut self` and leaves
    /// the instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array2<f64>, Array2<f64>)` - owned date vector `(731,)`,
    ///   owned feature matrix `(731, 11)`, and owned target matrix `(731, 3)`.
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
    /// Like [`BikeSharingDaily::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. Instead of consuming the instance, it takes `&mut self`
    /// and moves the cached data out. This resets the instance to its unloaded
    /// state. The next accessor call, for example [`BikeSharingDaily::features`]
    /// or [`BikeSharingDaily::data`], loads the dataset again.
    ///
    /// If you are done with the instance, use [`BikeSharingDaily::into_data`]
    /// instead.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array2<f64>, Array2<f64>)` - owned date vector `(731,)`,
    ///   owned feature matrix `(731, 11)`, and owned target matrix `(731, 3)`.
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

impl_ml_dataset!(BikeSharingDaily, BikeSharingData, "bike_sharing_daily");
