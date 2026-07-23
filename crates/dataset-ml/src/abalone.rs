//! Abalone dataset.
//!
//! Physical measurements of abalone (a marine snail), used to predict the age of
//! the animal. The age in years is the number of `rings` plus 1.5; counting the
//! rings through a microscope is a slow, tedious task, so the goal is to predict
//! it from easier physical measurements. This loader exposes `rings` itself as
//! the regression target.
//!
//! **Features (8, mixed):**
//! - String feature (1): `sex` — `M` (male), `F` (female), or `I` (infant)
//! - Numeric features (7): `length`, `diameter`, `height` (mm); `whole_weight`,
//!   `shucked_weight`, `viscera_weight`, `shell_weight` (grams)
//!
//! **Target:** `rings` — integer ring count (age in years is `rings + 1.5`),
//! exposed as an `f64` regression target.
//!
//! **Samples:** 4,177
//! **Application:** Regression / age prediction
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C55C7W>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::{Array1, Array2};
use std::fs::File;

/// Type alias for Abalone dataset: (string features, numeric features, targets).
type AbaloneData = (Array2<String>, Array2<f64>, Array1<f64>);

/// The URL for the Abalone dataset (the `abalone.data` file).
const ABALONE_DATA_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data";

/// The name of the cached Abalone dataset file.
const ABALONE_FILENAME: &str = "abalone.csv";

/// The SHA256 hash of the cached Abalone dataset file (`abalone.data`'s bytes).
const ABALONE_SHA256: &str = "de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f";

/// The name of the dataset.
const ABALONE_DATASET_NAME: &str = "abalone";

/// Number of samples.
const N_SAMPLES: usize = 4_177;

/// Number of categorical (string) features.
const N_STRING_FEATURES: usize = 1;

/// Number of numeric features.
const N_NUMERIC_FEATURES: usize = 7;

/// Number of columns per record (8 features + 1 target).
const N_COLUMNS: usize = 9;

/// Source column index of the target (`rings`). The target is the **last** column.
const TARGET_COLUMN: usize = 8;

/// Categorical feature columns, as `(source column index, name)`, in output order.
const STRING_COLUMNS: [(usize, &str); N_STRING_FEATURES] = [(0, "sex")];

/// Numeric feature columns, as `(source column index, name)`, in output order.
const NUMERIC_COLUMNS: [(usize, &str); N_NUMERIC_FEATURES] = [
    (1, "length"),
    (2, "diameter"),
    (3, "height"),
    (4, "whole_weight"),
    (5, "shucked_weight"),
    (6, "viscera_weight"),
    (7, "shell_weight"),
];

/// A struct representing the Abalone dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The age of an abalone is determined by cutting the shell through the cone,
/// staining it, and counting the number of rings through a microscope — a boring
/// and time-consuming task. The goal is to predict the age from other
/// measurements that are easier to obtain. This loader exposes `rings` (the raw
/// ring count) as the regression target; the actual age in years is `rings + 1.5`.
/// It is a mixed-type **regression** dataset — the only categorical feature is
/// `sex` (`M`/`F`/`I`), the remaining seven are continuous measurements.
///
/// # Feature columns
///
/// Features are split across two matrices: a `(4177, 1)` string matrix and a
/// `(4177, 7)` numeric `f64` matrix.
///
/// String features (`Array2<String>`), by 0-based column:
///
/// | Column | Attribute | Values                                 |
/// |--------|-----------|----------------------------------------|
/// | `0`    | `sex`     | `M` (male), `F` (female), `I` (infant) |
///
/// Numeric features (`Array2<f64>`), by 0-based column:
///
/// | Column | Attribute         | Unit  |
/// |--------|-------------------|-------|
/// | `0`    | `length`          | mm    |
/// | `1`    | `diameter`        | mm    |
/// | `2`    | `height`          | mm    |
/// | `3`    | `whole_weight`    | grams |
/// | `4`    | `shucked_weight`  | grams |
/// | `5`    | `viscera_weight`  | grams |
/// | `6`    | `shell_weight`    | grams |
///
/// # Targets
///
/// - `rings` (shape `(4177,)`): the `Array1<f64>` regression target, the integer
///   ring count (`1`–`29`) stored as `f64`. The age in years is `rings + 1.5`.
///
/// The dataset has no missing values.
///
/// See more information at <https://archive.ics.uci.edu/dataset/1/abalone>.
///
/// # Citation
///
/// Nash, W., Sellers, T., Talbot, S., Cawthorn, A., & Ford, W. (1994). Abalone
/// \[Dataset\]. UCI Machine Learning Repository. <https://doi.org/10.24432/C55C7W>
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::abalone::Abalone;
///
/// let download_dir = "./abalone"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = Abalone::new(download_dir);
/// let (string_features, numeric_features) = dataset.features().unwrap();
/// let targets = dataset.targets().unwrap();
///
/// let (string_features, numeric_features, targets) = dataset.data().unwrap(); // this is also a way to get all data
/// assert_eq!(string_features.shape(), &[4177, 1]);
/// assert_eq!(numeric_features.shape(), &[4177, 7]);
/// assert_eq!(targets.len(), 4177);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((_strings, numerics, targets)) = dataset.get_data_mut() {
///     numerics[[0, 0]] = 0.5;
///     targets[0] = 10.0;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out (no `to_owned()` clone) and leaves
/// // the instance reusable — the next access reloads from the cached file.
/// let (owned_strings, owned_numerics, owned_targets) = dataset.take_data().unwrap();
/// assert_eq!(owned_strings.shape(), &[4177, 1]);
/// assert_eq!(owned_numerics.shape(), &[4177, 7]);
/// assert_eq!(owned_targets.len(), 4177);
///
/// // `into_data()` also returns the owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_strings, owned_numerics, owned_targets) = dataset.into_data().unwrap();
/// assert_eq!(owned_strings.shape(), &[4177, 1]);
/// assert_eq!(owned_numerics.shape(), &[4177, 7]);
/// assert_eq!(owned_targets.len(), 4177);
/// ```
#[derive(Debug)]
pub struct Abalone {
    dataset: Dataset<AbaloneData, DatasetError>,
}

impl Abalone {
    /// Create a new Abalone instance without loading data.
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
    /// - `Self` - `Abalone` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Abalone {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Abalone dataset.
    fn load_data(dir: &str) -> Result<AbaloneData, DatasetError> {
        // Prepare the dataset file. The source file is `abalone.data`; cache it
        // under `abalone.csv`.
        let file_path = acquire_dataset(
            dir,
            ABALONE_FILENAME,
            ABALONE_DATASET_NAME,
            Some(ABALONE_SHA256),
            |temp_path| {
                download_to_with_retries(
                    ABALONE_DATA_URL,
                    temp_path,
                    Some(ABALONE_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(ABALONE_FILENAME))
            },
        )?;

        // The source is plain comma-separated with no header and no missing values.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut string_features: Vec<String> = Vec::with_capacity(N_SAMPLES * N_STRING_FEATURES);
        let mut numeric_features: Vec<f64> = Vec::with_capacity(N_SAMPLES * N_NUMERIC_FEATURES);
        let mut targets: Vec<f64> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(ABALONE_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            // Skip blank lines defensively (e.g. a trailing newline).
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    ABALONE_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // Categorical features (only `sex`), kept verbatim.
            for &(col, name) in STRING_COLUMNS.iter() {
                let value = &record[col];
                if value.is_empty() {
                    return Err(DatasetError::invalid_value(
                        ABALONE_DATASET_NAME,
                        name,
                        value,
                        line_num,
                    ));
                }
                string_features.push(value.to_string());
            }

            // Numeric features.
            for &(col, name) in NUMERIC_COLUMNS.iter() {
                let value: f64 = record[col].parse().map_err(|e| {
                    DatasetError::parse_failed(ABALONE_DATASET_NAME, name, line_num, e)
                })?;
                numeric_features.push(value);
            }

            // Regression target (`rings`).
            let target: f64 = record[TARGET_COLUMN].parse().map_err(|e| {
                DatasetError::parse_failed(ABALONE_DATASET_NAME, "rings", line_num, e)
            })?;
            targets.push(target);
        }

        let n_samples = targets.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(ABALONE_DATASET_NAME));
        }

        let string_array = Array2::from_shape_vec((n_samples, N_STRING_FEATURES), string_features)
            .map_err(|e| {
                DatasetError::array_shape_error(ABALONE_DATASET_NAME, "string_features", e)
            })?;

        let numeric_array =
            Array2::from_shape_vec((n_samples, N_NUMERIC_FEATURES), numeric_features).map_err(
                |e| DatasetError::array_shape_error(ABALONE_DATASET_NAME, "numeric_features", e),
            )?;

        let targets_array = Array1::from_vec(targets);

        Ok((string_array, numeric_array, targets_array))
    }

    /// Get a reference to both string and numeric feature matrices.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<String>` - Reference to the string feature matrix with shape
    ///   `(4177, 1)` containing `sex` (`M`, `F`, or `I`).
    /// - `&Array2<f64>` - Reference to the numeric feature matrix with shape
    ///   `(4177, 7)` containing `length`, `diameter`, `height`, `whole_weight`,
    ///   `shucked_weight`, `viscera_weight`, `shell_weight`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (4,177 samples)
    pub fn features(&self) -> Result<(&Array2<String>, &Array2<f64>), DatasetError> {
        let data = self.dataset.load()?;
        Ok((&data.0, &data.1))
    }

    /// Get a reference to the regression target vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<f64>` - Reference to the target vector with shape `(4177,)`
    ///   containing `rings` (the age in years is `rings + 1.5`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (4,177 samples)
    pub fn targets(&self) -> Result<&Array1<f64>, DatasetError> {
        Ok(&self.dataset.load()?.2)
    }

    /// Get string features, numeric features and targets as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&AbaloneData` - reference to the cached `(string features, numeric
    ///   features, targets)` tuple: string feature matrix `(4177, 1)`, numeric
    ///   feature matrix `(4177, 7)`, and target vector `(4177,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (4,177 samples)
    pub fn data(&self) -> Result<&AbaloneData, DatasetError> {
        self.dataset.load()
    }

    /// Get string features, numeric features and targets as references
    /// **without** triggering loading.
    ///
    /// Unlike [`Abalone::data`], which loads the dataset on first call, this never
    /// runs the loader: if the data has not been loaded yet, it returns `None`
    /// instead of downloading and parsing. Use it when you only want the data if
    /// it is already cached and want to avoid paying the download/parse cost
    /// otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&AbaloneData)` - reference to the cached `(string features, numeric
    ///   features, targets)` tuple (`(4177, 1)`, `(4177, 7)`, `(4177,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&AbaloneData> {
        self.dataset.get()
    }

    /// Get mutable references to string features, numeric features, and targets
    /// for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. encode `sex`,
    /// normalize measurements) with no `to_owned()` clone and without removing them
    /// from the cache: the changes persist, so later [`Abalone::features`],
    /// [`Abalone::data`], or [`Abalone::get_data`] calls observe them.
    ///
    /// Like [`Abalone::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call a loading accessor (e.g.
    /// [`Abalone::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut AbaloneData)` - mutable reference to the cached `(string
    ///   features, numeric features, targets)` tuple (`(4177, 1)`, `(4177, 7)`,
    ///   `(4177,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut AbaloneData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** string features, numeric features,
    /// and targets.
    ///
    /// Unlike [`Abalone::data`], which borrows the cached data, this moves it out
    /// and returns owned arrays directly — no `to_owned()` clone needed. The dataset
    /// is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`Abalone::take_data`] instead — it takes `&mut self` and leaves the instance
    /// reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array2<f64>, Array1<f64>)` - owned string feature matrix
    ///   `(4177, 1)`, owned numeric feature matrix `(4177, 7)`, and owned target
    ///   vector `(4177,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<AbaloneData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** string features, numeric features, and targets out of the
    /// dataset, leaving it reusable.
    ///
    /// Like [`Abalone::into_data`], this returns owned arrays with no `to_owned()`
    /// clone. But instead of consuming the instance, it takes `&mut self` and moves
    /// the cached data out, resetting the instance to its unloaded state: the next
    /// accessor call (e.g. [`Abalone::features`] or [`Abalone::data`]) loads the
    /// dataset again.
    ///
    /// Use [`Abalone::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array2<f64>, Array1<f64>)` - owned string feature matrix
    ///   `(4177, 1)`, owned numeric feature matrix `(4177, 7)`, and owned target
    ///   vector `(4177,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<AbaloneData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(Abalone, AbaloneData, "abalone");
