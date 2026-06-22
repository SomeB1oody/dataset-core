//! Adult / Census Income dataset.
//!
//! Census records extracted from the 1994 US Census database (Barry Becker),
//! used to predict whether a person's income exceeds $50K/year. This loader uses
//! the canonical `adult.data` training partition (32,561 records); the separate
//! `adult.test` partition is not bundled (it carries a non-data header line and
//! trailing periods on its labels).
//!
//! **Features (14, mixed):**
//! - String features (8): `workclass`, `education`, `marital-status`,
//!   `occupation`, `relationship`, `race`, `sex`, `native-country`
//! - Numeric features (6): `age`, `fnlwgt`, `education-num`, `capital-gain`,
//!   `capital-loss`, `hours-per-week`
//!
//! **Target:** `income` — binary label kept verbatim (`<=50K` or `>50K`)
//!
//! **Samples:** 32,561
//! **Application:** Binary classification / income prediction
//!
//! **Source:** UCI Machine Learning Repository
//! <https://archive.ics.uci.edu/dataset/2/adult>

use csv::{ReaderBuilder, Trim};
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to};
use ndarray::{Array1, Array2};
use std::fs::File;

/// Type alias for Adult dataset: (string features, numeric features, labels).
type AdultData = (Array2<String>, Array2<f64>, Array1<String>);

/// The URL for the Adult dataset (the `adult.data` training partition).
const ADULT_DATA_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data";

/// The name of the cached Adult dataset file.
const ADULT_FILENAME: &str = "adult.csv";

/// The SHA256 hash of the cached Adult dataset file (`adult.data`'s bytes).
const ADULT_SHA256: &str = "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d";

/// The name of the dataset.
const ADULT_DATASET_NAME: &str = "adult";

/// Number of samples in the `adult.data` partition.
const N_SAMPLES: usize = 32_561;

/// Number of categorical (string) features.
const N_STRING_FEATURES: usize = 8;

/// Number of numeric features.
const N_NUMERIC_FEATURES: usize = 6;

/// Number of columns per record (14 features + 1 label).
const N_COLUMNS: usize = 15;

/// Source column index of the label (`income`).
const LABEL_COLUMN: usize = 14;

/// Categorical feature columns, as `(source column index, name)`, in output order.
const STRING_COLUMNS: [(usize, &str); N_STRING_FEATURES] = [
    (1, "workclass"),
    (3, "education"),
    (5, "marital-status"),
    (6, "occupation"),
    (7, "relationship"),
    (8, "race"),
    (9, "sex"),
    (13, "native-country"),
];

/// Numeric feature columns, as `(source column index, name)`, in output order.
const NUMERIC_COLUMNS: [(usize, &str); N_NUMERIC_FEATURES] = [
    (0, "age"),
    (2, "fnlwgt"),
    (4, "education-num"),
    (10, "capital-gain"),
    (11, "capital-loss"),
    (12, "hours-per-week"),
];

/// The token marking a missing categorical value in the source.
const MISSING_TOKEN: &str = "?";

/// A struct representing the Adult / Census Income dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The Adult dataset (also called "Census Income") was extracted by Barry Becker
/// from the 1994 US Census database. The prediction task is to determine whether a
/// person earns over $50,000 a year from 14 demographic and employment attributes.
/// It is a standard benchmark for mixed categorical/numeric classification.
///
/// # Feature columns
///
/// Features are split across two matrices: a `(32561, 8)` string matrix and a
/// `(32561, 6)` numeric `f64` matrix.
///
/// String features (`Array2<String>`), by 0-based column:
///
/// | Column | Attribute        |
/// |--------|------------------|
/// | `0`    | `workclass`      |
/// | `1`    | `education`      |
/// | `2`    | `marital-status` |
/// | `3`    | `occupation`     |
/// | `4`    | `relationship`   |
/// | `5`    | `race`           |
/// | `6`    | `sex`            |
/// | `7`    | `native-country` |
///
/// Numeric features (`Array2<f64>`), by 0-based column:
///
/// | Column | Attribute        | Unit          |
/// |--------|------------------|---------------|
/// | `0`    | `age`            | years         |
/// | `1`    | `fnlwgt`         | sampling weight |
/// | `2`    | `education-num`  |               |
/// | `3`    | `capital-gain`   | USD           |
/// | `4`    | `capital-loss`   | USD           |
/// | `5`    | `hours-per-week` | hours         |
///
/// # Labels
///
/// - `income` (shape `(32561,)`): the `Array1<String>` is kept verbatim, each entry
///   being either `<=50K` or `>50K`.
///
/// Missing values:
/// - The source marks missing categorical values with `?` (in `workclass`,
///   `occupation`, and `native-country`); these are mapped to empty strings `""`.
/// - The numeric features have no missing values.
///
/// See more information at <https://archive.ics.uci.edu/dataset/2/adult>.
///
/// # Citation
///
/// Becker, B. & Kohavi, R. (1996). Adult \[Dataset\]. UCI Machine Learning
/// Repository. <https://doi.org/10.24432/C5XW20>
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::adult::Adult;
///
/// let download_dir = "./adult"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = Adult::new(download_dir);
/// let (string_features, numeric_features) = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (string_features, numeric_features, labels) = dataset.data().unwrap(); // this is also a way to get all data
/// assert_eq!(string_features.shape(), &[32561, 8]);
/// assert_eq!(numeric_features.shape(), &[32561, 6]);
/// assert_eq!(labels.len(), 32561);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((_strings, numerics, labels)) = dataset.get_data_mut() {
///     numerics[[0, 0]] = 99.0;
///     labels[0] = ">50K".to_string();
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out (no `to_owned()` clone) and leaves
/// // the instance reusable — the next access reloads from the cached file.
/// let (owned_strings, owned_numerics, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_strings.shape(), &[32561, 8]);
/// assert_eq!(owned_numerics.shape(), &[32561, 6]);
/// assert_eq!(owned_labels.len(), 32561);
///
/// // `into_data()` also returns the owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_strings, owned_numerics, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_strings.shape(), &[32561, 8]);
/// assert_eq!(owned_numerics.shape(), &[32561, 6]);
/// assert_eq!(owned_labels.len(), 32561);
/// ```
#[derive(Debug)]
pub struct Adult {
    dataset: Dataset<AdultData, DatasetError>,
}

impl Adult {
    /// Create a new Adult instance without loading data.
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
    /// - `Self` - `Adult` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Adult {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Adult dataset.
    fn load_data(dir: &str) -> Result<AdultData, DatasetError> {
        // Prepare the dataset file.
        let file_path = acquire_dataset(
            dir,
            ADULT_FILENAME,
            ADULT_DATASET_NAME,
            Some(ADULT_SHA256),
            |temp_path| {
                // The source file is `adult.data`; cache it under `adult.csv`.
                download_to(ADULT_DATA_URL, temp_path, Some(ADULT_FILENAME))?;
                Ok(temp_path.join(ADULT_FILENAME))
            },
        )?;

        // The source is comma-separated with a leading space after each comma
        // (e.g. `39, State-gov, ...`), so trim whitespace from every field.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(false)
            .trim(Trim::All)
            .from_reader(file);

        let mut string_features: Vec<String> = Vec::with_capacity(N_SAMPLES * N_STRING_FEATURES);
        let mut numeric_features: Vec<f64> = Vec::with_capacity(N_SAMPLES * N_NUMERIC_FEATURES);
        let mut labels: Vec<String> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record = result.map_err(|e| DatasetError::csv_read_error(ADULT_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            // Skip blank lines (the source file ends with a trailing newline).
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    ADULT_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // Categorical features, mapping the `?` missing token to an empty string.
            for &(col, _name) in STRING_COLUMNS.iter() {
                let value = &record[col];
                if value == MISSING_TOKEN {
                    string_features.push(String::new());
                } else {
                    string_features.push(value.to_string());
                }
            }

            // Numeric features.
            for &(col, name) in NUMERIC_COLUMNS.iter() {
                let value: f64 = record[col].parse().map_err(|e| {
                    DatasetError::parse_failed(ADULT_DATASET_NAME, name, line_num, e)
                })?;
                numeric_features.push(value);
            }

            // Label, kept verbatim (`<=50K` or `>50K`).
            let label = &record[LABEL_COLUMN];
            if label.is_empty() {
                return Err(DatasetError::invalid_value(
                    ADULT_DATASET_NAME,
                    "income",
                    label,
                    line_num,
                ));
            }
            labels.push(label.to_string());
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(ADULT_DATASET_NAME));
        }

        let string_array = Array2::from_shape_vec((n_samples, N_STRING_FEATURES), string_features)
            .map_err(|e| {
                DatasetError::array_shape_error(ADULT_DATASET_NAME, "string_features", e)
            })?;

        let numeric_array =
            Array2::from_shape_vec((n_samples, N_NUMERIC_FEATURES), numeric_features).map_err(
                |e| DatasetError::array_shape_error(ADULT_DATASET_NAME, "numeric_features", e),
            )?;

        let labels_array = Array1::from_vec(labels);

        Ok((string_array, numeric_array, labels_array))
    }

    /// Get a reference to both string and numeric feature matrices.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<String>` - Reference to string feature matrix with shape `(32561, 8)` containing:
    ///     - `workclass`
    ///     - `education`
    ///     - `marital-status`
    ///     - `occupation`
    ///     - `relationship`
    ///     - `race`
    ///     - `sex`
    ///     - `native-country`
    ///
    ///   (empty string if missing in source)
    ///
    /// - `&Array2<f64>` - Reference to numeric feature matrix with shape `(32561, 6)` containing:
    ///     - `age`
    ///     - `fnlwgt`
    ///     - `education-num`
    ///     - `capital-gain`
    ///     - `capital-loss`
    ///     - `hours-per-week`
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (32,561 samples)
    pub fn features(&self) -> Result<(&Array2<String>, &Array2<f64>), DatasetError> {
        let data = self.dataset.load()?;
        Ok((&data.0, &data.1))
    }

    /// Get a reference to the label vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<String>` - Reference to label vector with shape `(32561,)` containing `income` values (`<=50K` or `>50K`)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (32,561 samples)
    pub fn labels(&self) -> Result<&Array1<String>, DatasetError> {
        Ok(&self.dataset.load()?.2)
    }

    /// Get string features, numeric features and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&AdultData` - reference to the cached `(string features, numeric
    ///   features, labels)` tuple: string feature matrix `(32561, 8)`, numeric
    ///   feature matrix `(32561, 6)`, and label vector `(32561,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (32,561 samples)
    pub fn data(&self) -> Result<&AdultData, DatasetError> {
        self.dataset.load()
    }

    /// Get string features, numeric features and labels as references
    /// **without** triggering loading.
    ///
    /// Unlike [`Adult::data`], which loads the dataset on first call, this never
    /// runs the loader: if the data has not been loaded yet, it returns `None`
    /// instead of downloading and parsing. Use it when you only want the data if
    /// it is already cached and want to avoid paying the download/parse cost
    /// otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&AdultData)` - reference to the cached `(string features, numeric
    ///   features, labels)` tuple (`(32561, 8)`, `(32561, 6)`, `(32561,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&AdultData> {
        self.dataset.get()
    }

    /// Get mutable references to string features, numeric features, and labels
    /// for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. encode categorical
    /// features, normalize numeric features) with no `to_owned()` clone and without
    /// removing them from the cache: the changes persist, so later
    /// [`Adult::features`], [`Adult::data`], or [`Adult::get_data`] calls observe
    /// them.
    ///
    /// Like [`Adult::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call a loading accessor (e.g.
    /// [`Adult::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut AdultData)` - mutable reference to the cached `(string
    ///   features, numeric features, labels)` tuple (`(32561, 8)`, `(32561, 6)`,
    ///   `(32561,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut AdultData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** string features, numeric features,
    /// and labels.
    ///
    /// Unlike [`Adult::data`], which borrows the cached data, this moves it out
    /// and returns owned arrays directly — no `to_owned()` clone needed. The dataset
    /// is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use [`Adult::take_data`]
    /// instead — it takes `&mut self` and leaves the instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array2<f64>, Array1<String>)` - owned string feature matrix
    ///   `(32561, 8)`, owned numeric feature matrix `(32561, 6)`, and owned label vector
    ///   `(32561,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<AdultData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** string features, numeric features, and labels out of the
    /// dataset, leaving it reusable.
    ///
    /// Like [`Adult::into_data`], this returns owned arrays with no `to_owned()`
    /// clone. But instead of consuming the instance, it takes `&mut self` and moves
    /// the cached data out, resetting the instance to its unloaded state: the next
    /// accessor call (e.g. [`Adult::features`] or [`Adult::data`]) loads the
    /// dataset again.
    ///
    /// Use [`Adult::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array2<f64>, Array1<String>)` - owned string feature matrix
    ///   `(32561, 8)`, owned numeric feature matrix `(32561, 6)`, and owned label vector
    ///   `(32561,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<AdultData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}
