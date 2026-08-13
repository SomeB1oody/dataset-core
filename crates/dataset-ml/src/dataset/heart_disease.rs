//! Heart Disease (Cleveland) dataset.
//!
//! The dataset holds clinical records from the Cleveland Clinic Foundation,
//! collected by Robert Detrano. The task is to predict the presence of heart
//! disease in a patient. This loader uses the canonical
//! `processed.cleveland.data` partition. This is the 14-column subset that
//! virtually all published experiments on this database use (303 patients,
//! 13 features plus the diagnosis).
//!
//! **Columns (14):**
//!
//! | Name       | Type      | Description                                                 |
//! |------------|-----------|---------------------------------------------------------------|
//! | `age`      | `Numeric` | age in years                                                |
//! | `sex`      | `Numeric` | `1` = male, `0` = female                                    |
//! | `cp`       | `Numeric` | chest pain type (`1`–`4`)                                   |
//! | `trestbps` | `Numeric` | resting blood pressure (mm Hg)                              |
//! | `chol`     | `Numeric` | serum cholesterol (mg/dl)                                   |
//! | `fbs`      | `Numeric` | fasting blood sugar > 120 mg/dl (`1` = true, `0` = false)    |
//! | `restecg`  | `Numeric` | resting electrocardiographic results (`0`, `1`, `2`)        |
//! | `thalach`  | `Numeric` | maximum heart rate achieved                                 |
//! | `exang`    | `Numeric` | exercise-induced angina (`1` = yes, `0` = no)                |
//! | `oldpeak`  | `Numeric` | ST depression induced by exercise relative to rest          |
//! | `slope`    | `Numeric` | slope of the peak exercise ST segment (`1`–`3`)              |
//! | `ca`       | `Numeric` | number of major vessels (`0`–`3`) colored by fluoroscopy     |
//! | `thal`     | `Numeric` | `3` = normal, `6` = fixed defect, `7` = reversible defect    |
//! | `num`      | `Integer` | diagnosis, `0` (absence) through `4` (increasing presence)  |
//!
//! The source designates the 13 clinical measurements as the inputs
//! ([`HeartDisease::FEATURE_NAMES`](crate::HeartDisease::FEATURE_NAMES)) and `num` as the label
//! ([`HeartDisease::TARGET`](crate::HeartDisease::TARGET)).
//!
//! Users commonly binarize `num` to absence (`0`) versus presence (`> 0`).
//!
//! **Samples:** 303
//! **Application:** (Multi-class) classification / heart-disease diagnosis
//!
//! **Missing values:** the source marks them with `?`. There are 4 in `ca` and
//! 2 in `thal`, over 6 affected patients. The loader stores them as `NaN`.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C52P4X>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
use std::fs::File;

use csv::ReaderBuilder;

/// The URL for the Heart Disease dataset (the `processed.cleveland.data` file).
const HEART_DISEASE_DATA_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data";

/// The name of the cached Heart Disease dataset file.
const HEART_DISEASE_FILENAME: &str = "heart_disease.csv";

/// The SHA256 hash of the cached Heart Disease dataset file (`processed.cleveland.data`'s bytes).
const HEART_DISEASE_SHA256: &str =
    "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8";

/// The name of the dataset.
const HEART_DISEASE_DATASET_NAME: &str = "heart_disease";

/// Number of samples.
const N_SAMPLES: usize = 303;

/// Number of numeric features.
const N_FEATURES: usize = 13;

/// Number of columns per record (13 features + 1 target).
const N_COLUMNS: usize = 14;

/// Source column index of the target (`num`). The target is the **last** column.
const TARGET_COLUMN: usize = 13;

/// Numeric feature columns, as `(source column index, name)`, in output order.
const FEATURE_COLUMNS: [(usize, &str); N_FEATURES] = [
    (0, "age"),
    (1, "sex"),
    (2, "cp"),
    (3, "trestbps"),
    (4, "chol"),
    (5, "fbs"),
    (6, "restecg"),
    (7, "thalach"),
    (8, "exang"),
    (9, "oldpeak"),
    (10, "slope"),
    (11, "ca"),
    (12, "thal"),
];

/// The token marking a missing value in the source (only in `ca` and `thal`).
const MISSING_TOKEN: &str = "?";

/// A struct that represents the Heart Disease (Cleveland) dataset with lazy
/// loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// This database contains 76 attributes, but all published experiments use a
/// subset of 14 of them: the `processed.cleveland.data` file used here. The
/// "goal" field (`num`) refers to the presence of heart disease in the
/// patient. It is an integer value from `0` (no presence) to `4`. Most
/// experiments simply try to distinguish presence (values `1`, `2`, `3`, `4`)
/// from absence (value `0`). The data comes from the Cleveland Clinic
/// Foundation. Robert Detrano supplied it.
///
/// # Columns
///
/// Several features are integer-coded categoricals that the source stores as
/// numbers. The loader keeps the numbers verbatim.
///
/// | Name       | Type      | Description                                                 |
/// |------------|-----------|---------------------------------------------------------------|
/// | `age`      | `Numeric` | age in years                                                |
/// | `sex`      | `Numeric` | `1` = male, `0` = female                                    |
/// | `cp`       | `Numeric` | chest pain type (`1`–`4`)                                   |
/// | `trestbps` | `Numeric` | resting blood pressure (mm Hg)                              |
/// | `chol`     | `Numeric` | serum cholesterol (mg/dl)                                   |
/// | `fbs`      | `Numeric` | fasting blood sugar > 120 mg/dl (`1`/`0`)                    |
/// | `restecg`  | `Numeric` | resting ECG results (`0`, `1`, `2`)                          |
/// | `thalach`  | `Numeric` | maximum heart rate achieved                                 |
/// | `exang`    | `Numeric` | exercise-induced angina (`1`/`0`)                            |
/// | `oldpeak`  | `Numeric` | ST depression induced by exercise relative to rest          |
/// | `slope`    | `Numeric` | slope of the peak exercise ST segment (`1`–`3`)              |
/// | `ca`       | `Numeric` | number of major vessels (`0`–`3`) colored by fluoroscopy     |
/// | `thal`     | `Numeric` | `3` = normal, `6` = fixed defect, `7` = reversible defect    |
/// | `num`      | `Integer` | diagnosis, `0` (absence) through `4` (increasing presence)  |
///
/// The source designates the 13 clinical measurements as the inputs
/// ([`HeartDisease::FEATURE_NAMES`]) and `num` as the label
/// ([`HeartDisease::TARGET`]).
///
/// Users commonly binarize `num` to absence (`0`) versus presence (`> 0`).
///
/// Missing values: the source marks them with `?`. There are 4 in `ca` and 2 in
/// `thal`, over 6 affected patients. The loader stores them as `NaN`.
///
/// See more information at <https://archive.ics.uci.edu/dataset/45/heart+disease>.
///
/// # Citation
///
/// Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease
/// \[Dataset\]. UCI Machine Learning Repository. <https://doi.org/10.24432/C52P4X>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::HeartDisease;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./heart_disease";
///
/// let mut dataset = HeartDisease::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 303);
/// assert_eq!(table.n_columns(), 14);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&HeartDisease::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[303, 13]);
///
/// // Reach the label column by name.
/// let num = table.column(HeartDisease::TARGET).unwrap().as_integer().unwrap();
/// assert_eq!(num.len(), 303);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("age") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 60.0;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 303);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 303);
/// ```
#[derive(Debug)]
pub struct HeartDisease {
    dataset: Dataset<Table, DatasetError>,
}

impl HeartDisease {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "num";

    /// Create a new HeartDisease instance without loading data.
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
    /// - `Self` - a `HeartDisease` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        HeartDisease {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Heart Disease dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        // Prepare the dataset file. The source file is `processed.cleveland.data`.
        // The code caches it as `heart_disease.csv`.
        let file_path = acquire_dataset(
            dir,
            HEART_DISEASE_FILENAME,
            HEART_DISEASE_DATASET_NAME,
            Some(HEART_DISEASE_SHA256),
            |temp_path| {
                download_to_with_retries(
                    HEART_DISEASE_DATA_URL,
                    temp_path,
                    Some(HEART_DISEASE_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(HEART_DISEASE_FILENAME))
            },
        )?;

        // The source is plain comma-separated with no header.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features: Vec<Vec<f64>> = (0..N_FEATURES)
            .map(|_| Vec::with_capacity(N_SAMPLES))
            .collect();
        let mut labels: Vec<i64> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(HEART_DISEASE_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            // Skip blank lines defensively (for example, a trailing newline).
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    HEART_DISEASE_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // Numeric features, mapping the `?` missing token to NaN.
            for (out, &(col, name)) in FEATURE_COLUMNS.iter().enumerate() {
                let raw = &record[col];
                if raw == MISSING_TOKEN {
                    features[out].push(f64::NAN);
                } else {
                    let value: f64 = raw.parse().map_err(|e| {
                        DatasetError::parse_failed(HEART_DISEASE_DATASET_NAME, name, line_num, e)
                    })?;
                    features[out].push(value);
                }
            }

            // Target, an integer diagnosis in 0..=4. The source stores it as a
            // float-formatted integer in some related files, but the Cleveland
            // partition stores a plain integer.
            let target: u8 = record[TARGET_COLUMN].parse().map_err(|e| {
                DatasetError::parse_failed(HEART_DISEASE_DATASET_NAME, "num", line_num, e)
            })?;
            labels.push(i64::from(target));
        }

        let mut columns: Vec<Column> = Vec::with_capacity(N_COLUMNS);
        for (&(_, name), values) in FEATURE_COLUMNS.iter().zip(features) {
            columns.push(Column::new(
                name,
                ColumnData::Numeric(Array1::from_vec(values)),
            ));
        }
        columns.push(Column::new(
            Self::TARGET,
            ColumnData::Integer(Array1::from_vec(labels)),
        ));

        Table::new(HEART_DISEASE_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 303 samples and 14 columns.
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
    /// Unlike [`HeartDisease::data`], this method never runs the loader. If the
    /// data has not loaded yet, it returns `None` instead of downloading and
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
    /// changes stay in the cache. Later calls to [`HeartDisease::data`] or
    /// [`HeartDisease::get_data`] see them.
    ///
    /// Like [`HeartDisease::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`HeartDisease::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 303 samples and 14 columns.
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
    /// - `Table` - the owned table of 303 samples and 14 columns.
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

impl_ml_dataset!(HeartDisease, "heart_disease");
