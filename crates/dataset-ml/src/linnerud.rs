//! Linnerud dataset (scikit-learn `load_linnerud`).
//!
//! Dr. A. C. Linnerud collected this small multi-output regression dataset at
//! North Carolina State University. He measured three exercise variables and
//! three physiological variables on 20 middle-aged men in a fitness club. The
//! task is to predict the three physiological measurements from the three
//! exercise measurements (multi-output regression).
//!
//! This loader reproduces scikit-learn's `load_linnerud()` output. The
//! **features** are the three exercise variables, and the **targets** are the
//! three physiological variables, so both are `Array2<f64>` with shape
//! `(20, 3)`. The two underlying files are the whitespace-separated
//! `linnerud_exercise.csv` and `linnerud_physiological.csv`, distributed with
//! scikit-learn.
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
//! Paris: Editions Technip. Distributed with scikit-learn as
//! `linnerud_exercise.csv` and `linnerud_physiological.csv`.

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array2;
use std::path::Path;

/// The URL for the Linnerud exercise (feature) file distributed with scikit-learn.
const LINNERUD_EXERCISE_URL: &str = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/linnerud_exercise.csv";

/// The URL for the Linnerud physiological (target) file distributed with scikit-learn.
const LINNERUD_PHYSIOLOGICAL_URL: &str = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/linnerud_physiological.csv";

/// The cache filename for the Linnerud exercise (feature) file.
const LINNERUD_EXERCISE_FILENAME: &str = "linnerud_exercise.csv";

/// The cache filename for the Linnerud physiological (target) file.
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
/// function skips the header and splits every data row on arbitrary whitespace.
fn parse_linnerud_file(file_path: &Path, array_name: &str) -> Result<Array2<f64>, DatasetError> {
    let content = std::fs::read_to_string(file_path)?;

    let mut values: Vec<f64> = Vec::new();
    let mut n_rows = 0usize;

    // `enumerate` gives 0-based indices. The header is line 1, so data starts at
    // index 1. Its 1-based line number is `idx + 1`.
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

/// This struct represents the Linnerud dataset. It loads data lazily: the
/// dataset does not load until you call a data accessor method. Once loaded, the
/// data stays cached for later accesses.
///
/// # About Dataset
///
/// The Linnerud dataset records three exercise variables and three physiological
/// variables, measured on 20 middle-aged men in a fitness club. This loader
/// reproduces scikit-learn's `load_linnerud()` output. The features are the
/// three exercise variables, and the targets are the three physiological
/// variables. Both are `Array2<f64>` with shape `(20, 3)` (multi-output
/// regression).
///
/// # Feature columns
///
/// The exercise variables, in scikit-learn column order. By 0-based column index
/// in the feature matrix:
///
/// | Columns | Attributes | Unit  |
/// |---------|------------|-------|
/// | `0`     | `Chins`    | count |
/// | `1`     | `Situps`   | count |
/// | `2`     | `Jumps`    | count |
///
/// # Target columns
///
/// The physiological variables, in scikit-learn column order. By 0-based column
/// index in the target matrix:
///
/// | Columns | Attributes | Unit |
/// |---------|------------|------|
/// | `0`     | `Weight`   |      |
/// | `1`     | `Waist`    |      |
/// | `2`     | `Pulse`    |      |
///
/// See more information at <https://scikit-learn.org/stable/datasets/toy_dataset.html#linnerrud-dataset>
///
/// # Citation
///
/// M. Tenenhaus, *La régression PLS: théorie et pratique*. Paris: Editions
/// Technip, 1998. Distributed with scikit-learn as `linnerud_exercise.csv` and
/// `linnerud_physiological.csv`.
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` because all its fields implement them.
/// This makes it safe to share the struct across threads. The internal
/// [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::linnerud::Linnerud;
///
/// let download_dir = "./linnerud"; // the code creates the directory if it does not exist
///
/// let mut dataset = Linnerud::new(download_dir);
/// let features = dataset.features().unwrap();
/// let targets = dataset.targets().unwrap();
///
/// let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
/// assert_eq!(features.shape(), &[20, 3]);
/// assert_eq!(targets.shape(), &[20, 3]);
///
/// // `get_data()` borrows the cached arrays without reloading. `get_data_mut()`
/// // edits them in place, with no clone and no reload. The change stays cached.
/// // Prefer this method over `.to_owned()` when you only need to change values.
/// if let Some((features, targets)) = dataset.get_data_mut() {
///     features[[0, 0]] = 6.0;
///     targets[[0, 0]] = 190.0;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out with no `to_owned()` clone. It leaves
/// // the instance reusable. The next access reloads data from the cached file.
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
    /// The dataset does not load immediately. It loads the first time you call a
    /// data accessor method. This call is lightweight: it only stores the storage
    /// directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory that holds the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - `Linnerud` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Linnerud {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Linnerud dataset.
    fn load_data(dir: &str) -> Result<LinnerudData, DatasetError> {
        // The exercise and physiological measurements live in two separate files,
        // each acquired (and SHA-256 verified) independently.
        let exercise_path = acquire_dataset(
            dir,
            LINNERUD_EXERCISE_FILENAME,
            LINNERUD_DATASET_NAME,
            Some(LINNERUD_EXERCISE_SHA256),
            |temp_path| {
                download_to_with_retries(
                    LINNERUD_EXERCISE_URL,
                    temp_path,
                    Some(LINNERUD_EXERCISE_FILENAME),
                    DOWNLOAD_RETRIES,
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
                download_to_with_retries(
                    LINNERUD_PHYSIOLOGICAL_URL,
                    temp_path,
                    Some(LINNERUD_PHYSIOLOGICAL_FILENAME),
                    DOWNLOAD_RETRIES,
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
    /// This method loads the dataset lazily on the first call. Later calls return
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
    /// - Dataset size does not match the expected dimensions (20 samples, 3 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the target matrix (the physiological variables).
    ///
    /// This method loads the dataset lazily on the first call. Later calls return
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
    /// - Dataset size does not match the expected dimensions (20 samples, 3 targets)
    pub fn targets(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and targets as references.
    ///
    /// This method loads the dataset lazily on the first call. Later calls return
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
    /// - Dataset size does not match the expected dimensions (20 samples, 3 features, 3 targets)
    pub fn data(&self) -> Result<&LinnerudData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and targets as references **without** triggering loading.
    ///
    /// Unlike [`Linnerud::data`], which loads the dataset on the first call, this
    /// method never runs the loader. If the data has not loaded yet, this method
    /// returns `None` instead of downloading and parsing it. Use this method only
    /// when you want data that is already cached. This avoids the download and
    /// parse cost if the dataset is not cached yet.
    ///
    /// # Returns
    ///
    /// - `Some(&LinnerudData)` - reference to the cached `(features, targets)` tuple
    ///   (feature matrix `(20, 3)`, target matrix `(20, 3)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&LinnerudData> {
        self.dataset.get()
    }

    /// Get mutable references to features and targets for **in-place** editing.
    ///
    /// This method lets you change the cached arrays in place (for example, to
    /// re-scale features or clip outliers). It needs no `to_owned()` clone, and
    /// it does not remove the data from the cache. The changes persist, so later
    /// calls to [`Linnerud::features`], [`Linnerud::data`], or
    /// [`Linnerud::get_data`] see them.
    ///
    /// Like [`Linnerud::get_data`], this method does **not** trigger loading. It
    /// returns `None` if the dataset has not loaded yet. If you need the data to
    /// be present, call a loading accessor first, for example [`Linnerud::data`].
    ///
    /// # Returns
    ///
    /// - `Some(&mut LinnerudData)` - mutable reference to the cached
    ///   `(features, targets)` tuple (feature matrix `(20, 3)`, target matrix
    ///   `(20, 3)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut LinnerudData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and targets.
    ///
    /// Unlike [`Linnerud::data`], which borrows the cached data, this method moves
    /// the data out and returns owned arrays directly, with no `to_owned()` clone
    /// needed. The dataset loads on the first access if it has not loaded yet.
    ///
    /// This method **consumes** `self`, so you cannot use the instance afterward.
    /// If you want owned data but need to keep using the instance, use
    /// [`Linnerud::take_data`] instead. That method takes `&mut self` and leaves
    /// the instance reusable.
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

    /// Take **owned** features and targets out of the dataset. This leaves it reusable.
    ///
    /// Like [`Linnerud::into_data`], this method returns owned arrays with no
    /// `to_owned()` clone. Instead of consuming the instance, it takes `&mut self`
    /// and moves the cached data out. This resets the instance to its unloaded
    /// state. The next accessor call, for example [`Linnerud::features`] or
    /// [`Linnerud::data`], loads the dataset again.
    ///
    /// If you are done with the instance, use [`Linnerud::into_data`] instead.
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

impl_ml_dataset!(Linnerud, LinnerudData, "linnerud");
