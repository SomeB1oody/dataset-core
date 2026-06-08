//! Linnerud dataset (scikit-learn `load_linnerud`).
//!
//! A small multi-output regression dataset collected by Dr. A. C. Linnerud at the
//! North Carolina State University. Three exercise variables and three
//! physiological variables were measured on 20 middle-aged men in a fitness club.
//! The task is to predict the three physiological measurements from the three
//! exercise measurements (multi-output regression).
//!
//! This loader reproduces scikit-learn's `load_linnerud()` output: the **features**
//! are the three exercise variables and the **targets** are the three physiological
//! variables (so both are `Array2<f64>` with shape `(20, 3)`). The two underlying
//! files are the whitespace-separated `linnerud_exercise.csv` and
//! `linnerud_physiological.csv` distributed with scikit-learn.
//!
//! **Features (3):** the exercise variables, in scikit-learn column order
//! - `Chins` - number of chin-ups
//! - `Situps` - number of sit-ups
//! - `Jumps` - number of jumping jacks
//!
//! **Targets (3):** the physiological variables, in scikit-learn column order
//! - `Weight` - body weight
//! - `Waist` - waist circumference
//! - `Pulse` - resting pulse
//!
//! **Samples:** 20
//! **Application:** Multi-output regression / fitness modeling
//!
//! **Source:** Tenenhaus, M. (1998), *La régression PLS: théorie et pratique*,
//! Paris: Editions Technip; distributed with scikit-learn as
//! `linnerud_exercise.csv` and `linnerud_physiological.csv`.

use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to};
use ndarray::Array2;
use std::path::Path;

/// The URL for the Linnerud exercise (feature) file distributed with scikit-learn.
const LINNERUD_EXERCISE_URL: &str = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/linnerud_exercise.csv";

/// The URL for the Linnerud physiological (target) file distributed with scikit-learn.
const LINNERUD_PHYSIOLOGICAL_URL: &str = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/linnerud_physiological.csv";

/// The name the Linnerud exercise (feature) file is cached under.
const LINNERUD_EXERCISE_FILENAME: &str = "linnerud_exercise.csv";

/// The name the Linnerud physiological (target) file is cached under.
const LINNERUD_PHYSIOLOGICAL_FILENAME: &str = "linnerud_physiological.csv";

/// The SHA256 hash of the Linnerud exercise (feature) file.
const LINNERUD_EXERCISE_SHA256: &str =
    "cb8d8c24937643fa2459682efb86c5e667bcd6dd93109eef81964d9e9f11bf8c";

/// The SHA256 hash of the Linnerud physiological (target) file.
const LINNERUD_PHYSIOLOGICAL_SHA256: &str =
    "2bf7e05c1cd7d0adf0eca1e456941f624bed0a4fc96694d60d0ff7853ec5fcf7";

/// The name of the dataset.
const LINNERUD_DATASET_NAME: &str = "linnerud";

/// The number of columns in each of the two files (exercise: 3, physiological: 3).
const N_COLUMNS: usize = 3;

/// Type alias for the Linnerud dataset: (exercise features, physiological targets).
type LinnerudData = (Array2<f64>, Array2<f64>);

/// Parse one of the Linnerud whitespace-separated files into an `Array2<f64>`.
///
/// The files have a single header row (column names) followed by 20 data rows,
/// each holding exactly [`N_COLUMNS`] whitespace-separated numeric values. The
/// header is skipped and every data row is split on arbitrary whitespace.
fn parse_linnerud_file(file_path: &Path, array_name: &str) -> Result<Array2<f64>, DatasetError> {
    let content = std::fs::read_to_string(file_path)?;

    let mut values: Vec<f64> = Vec::new();
    let mut n_rows = 0usize;

    // `enumerate` gives 0-based indices; the header is line 1, so data starts at
    // index 1 and its 1-based line number is `idx + 1`.
    for (idx, line) in content.lines().enumerate().skip(1) {
        let line_num = idx + 1;
        if line.trim().is_empty() {
            continue;
        }

        let mut count = 0usize;
        for token in line.split_whitespace() {
            let value: f64 = token.parse().map_err(|e| {
                DatasetError::parse_failed(LINNERUD_DATASET_NAME, array_name, line_num, e)
            })?;
            values.push(value);
            count += 1;
        }

        if count != N_COLUMNS {
            return Err(DatasetError::invalid_column_count(
                LINNERUD_DATASET_NAME,
                N_COLUMNS,
                count,
                line_num,
            ));
        }
        n_rows += 1;
    }

    if n_rows == 0 {
        return Err(DatasetError::empty_dataset(LINNERUD_DATASET_NAME));
    }

    Array2::from_shape_vec((n_rows, N_COLUMNS), values)
        .map_err(|e| DatasetError::array_shape_error(LINNERUD_DATASET_NAME, array_name, e))
}

/// A struct representing the Linnerud dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The Linnerud dataset records three exercise variables and three physiological
/// variables measured on 20 middle-aged men in a fitness club. This loader
/// reproduces scikit-learn's `load_linnerud()` output: the features are the three
/// exercise variables and the targets are the three physiological variables, so
/// both are `Array2<f64>` with shape `(20, 3)` (multi-output regression).
///
/// Features (the exercise variables, in scikit-learn column order):
/// - `Chins` - number of chin-ups
/// - `Situps` - number of sit-ups
/// - `Jumps` - number of jumping jacks
///
/// Targets (the physiological variables, in scikit-learn column order):
/// - `Weight` - body weight
/// - `Waist` - waist circumference
/// - `Pulse` - resting pulse
///
/// See more information at <https://scikit-learn.org/stable/datasets/toy_dataset.html#linnerrud-dataset>
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::linnerud::Linnerud;
///
/// let download_dir = "./linnerud"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = Linnerud::new(download_dir);
/// let features = dataset.features().unwrap();
/// let targets = dataset.targets().unwrap();
///
/// let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
/// assert_eq!(features.shape(), &[20, 3]);
/// assert_eq!(targets.shape(), &[20, 3]);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((features, targets)) = dataset.get_data_mut() {
///     features[[0, 0]] = 6.0;
///     targets[[0, 0]] = 190.0;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable — the next access reloads from the cached file.
/// let (owned_features, owned_targets) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[20, 3]);
/// assert_eq!(owned_targets.shape(), &[20, 3]);
///
/// // `into_data()` also returns owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_features, owned_targets) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[20, 3]);
/// assert_eq!(owned_targets.shape(), &[20, 3]);
/// ```
#[derive(Debug)]
pub struct Linnerud {
    dataset: Dataset<LinnerudData, DatasetError>,
}

impl Linnerud {
    /// Create a new Linnerud instance without loading data.
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
    /// - `Self` - `Linnerud` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Linnerud {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Linnerud dataset.
    fn load_data(dir: &str) -> Result<LinnerudData, DatasetError> {
        // The exercise and physiological measurements live in two separate files,
        // each acquired (and SHA-256 verified) independently.
        let exercise_path = acquire_dataset(
            dir,
            LINNERUD_EXERCISE_FILENAME,
            LINNERUD_DATASET_NAME,
            Some(LINNERUD_EXERCISE_SHA256),
            |temp_path| {
                download_to(
                    LINNERUD_EXERCISE_URL,
                    temp_path,
                    Some(LINNERUD_EXERCISE_FILENAME),
                )?;
                Ok(temp_path.join(LINNERUD_EXERCISE_FILENAME))
            },
        )?;

        let physiological_path = acquire_dataset(
            dir,
            LINNERUD_PHYSIOLOGICAL_FILENAME,
            LINNERUD_DATASET_NAME,
            Some(LINNERUD_PHYSIOLOGICAL_SHA256),
            |temp_path| {
                download_to(
                    LINNERUD_PHYSIOLOGICAL_URL,
                    temp_path,
                    Some(LINNERUD_PHYSIOLOGICAL_FILENAME),
                )?;
                Ok(temp_path.join(LINNERUD_PHYSIOLOGICAL_FILENAME))
            },
        )?;

        let features = parse_linnerud_file(&exercise_path, "features")?;
        let targets = parse_linnerud_file(&physiological_path, "targets")?;

        // The two files must describe the same 20 men, so their row counts must match.
        if features.nrows() != targets.nrows() {
            return Err(DatasetError::length_mismatch(
                LINNERUD_DATASET_NAME,
                "targets",
                features.nrows(),
                targets.nrows(),
            ));
        }

        Ok((features, targets))
    }

    /// Get a reference to the feature matrix (the exercise variables).
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to the feature matrix with shape `(20, 3)`
    ///   containing the exercise variables (`Chins`, `Situps`, `Jumps`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (20 samples, 3 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the target matrix (the physiological variables).
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to the target matrix with shape `(20, 3)`
    ///   containing the physiological variables (`Weight`, `Waist`, `Pulse`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (20 samples, 3 targets)
    pub fn targets(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and targets as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&LinnerudData` - reference to the cached `(features, targets)` tuple:
    ///   the exercise feature matrix with shape `(20, 3)` (`Chins`, `Situps`,
    ///   `Jumps`) and the physiological target matrix with shape `(20, 3)`
    ///   (`Weight`, `Waist`, `Pulse`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (20 samples, 3 features, 3 targets)
    pub fn data(&self) -> Result<&LinnerudData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and targets as references **without** triggering loading.
    ///
    /// Unlike [`Linnerud::data`], which loads the dataset on first call, this
    /// never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&LinnerudData)` - reference to the cached `(features, targets)` tuple
    ///   (feature matrix `(20, 3)`, target matrix `(20, 3)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&LinnerudData> {
        self.dataset.get()
    }

    /// Get mutable references to features and targets for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. re-scale features,
    /// clip outliers) with no `to_owned()` clone and without removing them from
    /// the cache: the changes persist, so later [`Linnerud::features`],
    /// [`Linnerud::data`], or [`Linnerud::get_data`] calls observe them.
    ///
    /// Like [`Linnerud::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call a loading accessor (e.g.
    /// [`Linnerud::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut LinnerudData)` - mutable reference to the cached
    ///   `(features, targets)` tuple (feature matrix `(20, 3)`, target matrix
    ///   `(20, 3)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut LinnerudData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and targets.
    ///
    /// Unlike [`Linnerud::data`], which borrows the cached data, this moves it out
    /// and returns owned arrays directly — no `to_owned()` clone needed. The dataset
    /// is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`Linnerud::take_data`] instead — it takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array2<f64>)` - owned feature matrix with shape `(20, 3)`
    ///   and owned target matrix with shape `(20, 3)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<LinnerudData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and targets out of the dataset, leaving it reusable.
    ///
    /// Like [`Linnerud::into_data`], this returns owned arrays with no `to_owned()`
    /// clone. But instead of consuming the instance, it takes `&mut self` and moves
    /// the cached data out, resetting the instance to its unloaded state: the next
    /// accessor call (e.g. [`Linnerud::features`] or [`Linnerud::data`]) loads the
    /// dataset again.
    ///
    /// Use [`Linnerud::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array2<f64>)` - owned feature matrix with shape `(20, 3)`
    ///   and owned target matrix with shape `(20, 3)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<LinnerudData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}
