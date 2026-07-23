//! Car Evaluation dataset.
//!
//! Derived from a simple hierarchical decision model, this dataset evaluates cars
//! according to six categorical attributes describing price and technical
//! characteristics. The task is to predict a car's overall acceptability. Like
//! [`crate::mushroom`], it is **all-categorical** — every feature is a string
//! code, so there is no numeric feature matrix.
//!
//! **Features (6, all categorical):**
//! - `buying` — buying price: `vhigh`, `high`, `med`, `low`
//! - `maint` — maintenance price: `vhigh`, `high`, `med`, `low`
//! - `doors` — number of doors: `2`, `3`, `4`, `5more`
//! - `persons` — passenger capacity: `2`, `4`, `more`
//! - `lug_boot` — luggage boot size: `small`, `med`, `big`
//! - `safety` — estimated safety: `low`, `med`, `high`
//!
//! **Target:** `class` — one of `unacc`, `acc`, `good`, `vgood`
//!
//! **Samples:** 1,728 (the full cartesian product of the six attributes)
//! **Application:** Multi-class classification / car acceptability
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5JP48>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::{Array1, Array2};
use std::fs::File;

/// Type alias for Car Evaluation dataset: (categorical features, labels).
type CarEvaluationData = (Array2<String>, Array1<String>);

/// The URL for the Car Evaluation dataset (the `car.data` file).
const CAR_EVALUATION_DATA_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data";

/// The name of the cached Car Evaluation dataset file.
const CAR_EVALUATION_FILENAME: &str = "car_evaluation.csv";

/// The SHA256 hash of the cached Car Evaluation dataset file (`car.data`'s bytes).
const CAR_EVALUATION_SHA256: &str =
    "b703a9ac69f11e64ce8c223c0a40de4d2e9d769f7fb20be5f8f2e8a619893d83";

/// The name of the dataset.
const CAR_EVALUATION_DATASET_NAME: &str = "car_evaluation";

/// Number of samples.
const N_SAMPLES: usize = 1_728;

/// Number of categorical features.
const N_FEATURES: usize = 6;

/// Number of columns per record (6 features + 1 label).
const N_COLUMNS: usize = 7;

/// Source column index of the label (`class`). The label is the **last** column.
const LABEL_COLUMN: usize = 6;

/// Categorical feature columns, as `(source column index, name)`, in output order.
/// All 6 features precede the trailing `class` label column.
const FEATURE_COLUMNS: [(usize, &str); N_FEATURES] = [
    (0, "buying"),
    (1, "maint"),
    (2, "doors"),
    (3, "persons"),
    (4, "lug_boot"),
    (5, "safety"),
];

/// A struct representing the Car Evaluation dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The Car Evaluation dataset was derived from a simple hierarchical decision
/// model originally developed for the demonstration of DEX (an expert system
/// shell for multi-attribute decision making). It evaluates cars according to a
/// concept structure relating the overall acceptability (`class`) to price
/// (`buying`, `maint`) and technical characteristics (`doors`, `persons`,
/// `lug_boot`, `safety`). The dataset enumerates the full cartesian product of
/// the six attributes' levels, giving 1,728 records with no missing values. It is
/// useful for testing constructive induction and structure discovery methods.
///
/// # Feature columns
///
/// All 6 features are categorical, stored as string codes in one `(1728, 6)`
/// `Array2<String>` matrix (there is no numeric matrix). By 0-based column:
///
/// | Column | Attribute  | Values                        |
/// |--------|------------|-------------------------------|
/// | `0`    | `buying`   | `vhigh`, `high`, `med`, `low` |
/// | `1`    | `maint`    | `vhigh`, `high`, `med`, `low` |
/// | `2`    | `doors`    | `2`, `3`, `4`, `5more`        |
/// | `3`    | `persons`  | `2`, `4`, `more`              |
/// | `4`    | `lug_boot` | `small`, `med`, `big`         |
/// | `5`    | `safety`   | `low`, `med`, `high`          |
///
/// # Labels
///
/// - `class` (shape `(1728,)`): the `Array1<String>` is kept verbatim, each entry
///   being one of `unacc` (unacceptable), `acc` (acceptable), `good`, or `vgood`
///   (very good).
///
/// See more information at <https://archive.ics.uci.edu/dataset/19/car+evaluation>.
///
/// # Citation
///
/// Bohanec, M. (1988). Car Evaluation \[Dataset\]. UCI Machine Learning
/// Repository. <https://doi.org/10.24432/C5JP48>
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::car_evaluation::CarEvaluation;
///
/// let download_dir = "./car_evaluation"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = CarEvaluation::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap(); // this is also a way to get all data
/// assert_eq!(features.shape(), &[1728, 6]);
/// assert_eq!(labels.len(), 1728);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = "low".to_string();
///     labels[0] = "acc".to_string();
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out (no `to_owned()` clone) and leaves
/// // the instance reusable — the next access reloads from the cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[1728, 6]);
/// assert_eq!(owned_labels.len(), 1728);
///
/// // `into_data()` also returns the owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[1728, 6]);
/// assert_eq!(owned_labels.len(), 1728);
/// ```
#[derive(Debug)]
pub struct CarEvaluation {
    dataset: Dataset<CarEvaluationData, DatasetError>,
}

impl CarEvaluation {
    /// Create a new CarEvaluation instance without loading data.
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
    /// - `Self` - `CarEvaluation` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        CarEvaluation {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Car Evaluation dataset.
    fn load_data(dir: &str) -> Result<CarEvaluationData, DatasetError> {
        // Prepare the dataset file. The source file is `car.data`; cache it under
        // `car_evaluation.csv`.
        let file_path = acquire_dataset(
            dir,
            CAR_EVALUATION_FILENAME,
            CAR_EVALUATION_DATASET_NAME,
            Some(CAR_EVALUATION_SHA256),
            |temp_path| {
                download_to_with_retries(
                    CAR_EVALUATION_DATA_URL,
                    temp_path,
                    Some(CAR_EVALUATION_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(CAR_EVALUATION_FILENAME))
            },
        )?;

        // The source is plain comma-separated with no header. There are no missing
        // values.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features: Vec<String> = Vec::with_capacity(N_SAMPLES * N_FEATURES);
        let mut labels: Vec<String> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(CAR_EVALUATION_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            // Skip blank lines defensively (e.g. a trailing newline).
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    CAR_EVALUATION_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // Categorical features, kept verbatim.
            for &(col, name) in FEATURE_COLUMNS.iter() {
                let value = &record[col];
                if value.is_empty() {
                    return Err(DatasetError::invalid_value(
                        CAR_EVALUATION_DATASET_NAME,
                        name,
                        value,
                        line_num,
                    ));
                }
                features.push(value.to_string());
            }

            // Label, kept verbatim (`unacc`, `acc`, `good`, or `vgood`).
            let label = &record[LABEL_COLUMN];
            if label.is_empty() {
                return Err(DatasetError::invalid_value(
                    CAR_EVALUATION_DATASET_NAME,
                    "class",
                    label,
                    line_num,
                ));
            }
            labels.push(label.to_string());
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(CAR_EVALUATION_DATASET_NAME));
        }

        let features_array =
            Array2::from_shape_vec((n_samples, N_FEATURES), features).map_err(|e| {
                DatasetError::array_shape_error(CAR_EVALUATION_DATASET_NAME, "features", e)
            })?;

        let labels_array = Array1::from_vec(labels);

        Ok((features_array, labels_array))
    }

    /// Get a reference to the categorical feature matrix.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<String>` - Reference to the categorical feature matrix with shape
    ///   `(1728, 6)`. Each value is a string code (`buying`, `maint`, `doors`,
    ///   `persons`, `lug_boot`, `safety`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (1,728 samples)
    pub fn features(&self) -> Result<&Array2<String>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the label vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<String>` - Reference to label vector with shape `(1728,)` containing `class` values (`unacc`, `acc`, `good`, `vgood`)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (1,728 samples)
    pub fn labels(&self) -> Result<&Array1<String>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get features and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&CarEvaluationData` - reference to the cached `(features, labels)` tuple:
    ///   the categorical feature matrix `(1728, 6)` and the label vector `(1728,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (1,728 samples)
    pub fn data(&self) -> Result<&CarEvaluationData, DatasetError> {
        self.dataset.load()
    }

    /// Get features and labels as references **without** triggering loading.
    ///
    /// Unlike [`CarEvaluation::data`], which loads the dataset on first call, this
    /// never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&CarEvaluationData)` - reference to the cached `(features, labels)`
    ///   tuple (`(1728, 6)`, `(1728,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&CarEvaluationData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. encode categorical
    /// features) with no `to_owned()` clone and without removing them from the
    /// cache: the changes persist, so later [`CarEvaluation::features`],
    /// [`CarEvaluation::data`], or [`CarEvaluation::get_data`] calls observe them.
    ///
    /// Like [`CarEvaluation::get_data`], this does **not** trigger loading: it
    /// returns `None` if the dataset has not been loaded. Call a loading accessor
    /// (e.g. [`CarEvaluation::data`]) first if you need to ensure the data is
    /// present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut CarEvaluationData)` - mutable reference to the cached
    ///   `(features, labels)` tuple (`(1728, 6)`, `(1728,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut CarEvaluationData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`CarEvaluation::data`], which borrows the cached data, this moves it
    /// out and returns owned arrays directly — no `to_owned()` clone needed. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`CarEvaluation::take_data`] instead — it takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array1<String>)` - owned categorical feature matrix
    ///   `(1728, 6)` and owned label vector `(1728,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<CarEvaluationData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`CarEvaluation::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out, resetting the instance to its
    /// unloaded state: the next accessor call (e.g. [`CarEvaluation::features`] or
    /// [`CarEvaluation::data`]) loads the dataset again.
    ///
    /// Use [`CarEvaluation::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array1<String>)` - owned categorical feature matrix
    ///   `(1728, 6)` and owned label vector `(1728,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<CarEvaluationData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(CarEvaluation, CarEvaluationData, "car_evaluation");
