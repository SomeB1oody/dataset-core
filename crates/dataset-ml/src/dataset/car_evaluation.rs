//! Car Evaluation dataset.
//!
//! A simple hierarchical decision model produced this dataset. It evaluates cars
//! using six categorical attributes that describe price and technical
//! characteristics. The task is to predict a car's overall acceptability.
//!
//! **Columns (7):**
//!
//! | Name       | Type      | Description                                   |
//! |------------|-----------|-------------------------------------------------|
//! | `buying`   | `String`  | buying price: `vhigh`, `high`, `med`, `low`   |
//! | `maint`    | `String`  | maintenance price: `vhigh`, `high`, `med`, `low` |
//! | `doors`    | `String`  | number of doors: `2`, `3`, `4`, `5more`       |
//! | `persons`  | `String`  | passenger capacity: `2`, `4`, `more`          |
//! | `lug_boot` | `String`  | luggage boot size: `small`, `med`, `big`      |
//! | `safety`   | `String`  | estimated safety: `low`, `med`, `high`        |
//! | `class`    | `String`  | acceptability: `unacc`, `acc`, `good`, `vgood` |
//!
//! The source designates the six attributes as the inputs
//! ([`CarEvaluation::FEATURE_NAMES`](crate::CarEvaluation::FEATURE_NAMES)) and `class` as the label
//! ([`CarEvaluation::TARGET`](crate::CarEvaluation::TARGET)).
//!
//! **Samples:** 1,728 (the full cartesian product of the six attributes)
//! **Application:** Multi-class classification / car acceptability
//!
//! **Missing values:** none.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5JP48>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
use std::fs::File;

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

/// A struct that represents the Car Evaluation dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// A simple hierarchical decision model is the source of the Car Evaluation
/// dataset. Developers built the model to show DEX, an expert system shell for
/// multi-attribute decision making. The dataset uses a concept structure that
/// relates overall acceptability (`class`) to price (`buying`, `maint`) and
/// technical characteristics (`doors`, `persons`, `lug_boot`, `safety`). The
/// dataset enumerates the full cartesian product of the six attributes' levels.
/// It has 1,728 records with no missing values. Researchers use it to test
/// constructive induction and structure discovery methods.
///
/// # Columns
///
/// | Name       | Type      | Description                                   |
/// |------------|-----------|-------------------------------------------------|
/// | `buying`   | `String`  | buying price: `vhigh`, `high`, `med`, `low`   |
/// | `maint`    | `String`  | maintenance price: `vhigh`, `high`, `med`, `low` |
/// | `doors`    | `String`  | number of doors: `2`, `3`, `4`, `5more`       |
/// | `persons`  | `String`  | passenger capacity: `2`, `4`, `more`          |
/// | `lug_boot` | `String`  | luggage boot size: `small`, `med`, `big`      |
/// | `safety`   | `String`  | estimated safety: `low`, `med`, `high`        |
/// | `class`    | `String`  | `unacc`, `acc`, `good`, or `vgood`            |
///
/// The loader keeps every value as the source spells it. The source
/// designates the six attributes as the inputs
/// ([`CarEvaluation::FEATURE_NAMES`]) and `class` as the label
/// ([`CarEvaluation::TARGET`]).
///
/// Missing values: none.
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
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::CarEvaluation;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./car_evaluation";
///
/// let mut dataset = CarEvaluation::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 1728);
/// assert_eq!(table.n_columns(), 7);
///
/// // Every feature is a string, so reach each one by name.
/// let buying = table.column("buying").unwrap().as_string().unwrap();
/// assert_eq!(buying.len(), 1728);
///
/// // Reach the label column by name.
/// let class = table.column(CarEvaluation::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(class.len(), 1728);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("buying") {
///         if let dataset_ml::ColumnData::String(values) = column.data_mut() {
///             values[0] = "low".to_string();
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 1728);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 1728);
/// ```
#[derive(Debug)]
pub struct CarEvaluation {
    dataset: Dataset<Table, DatasetError>,
}

impl CarEvaluation {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] =
        ["buying", "maint", "doors", "persons", "lug_boot", "safety"];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "class";

    /// Create a new CarEvaluation instance without loading data.
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
    /// - `Self` - a `CarEvaluation` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        CarEvaluation {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Car Evaluation dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        // The source file is `car.data`. The loader caches it as
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

        let mut features: Vec<Vec<String>> = FEATURE_COLUMNS
            .iter()
            .map(|_| Vec::with_capacity(N_SAMPLES))
            .collect();
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
            for (values, &(col, name)) in features.iter_mut().zip(FEATURE_COLUMNS.iter()) {
                let value = &record[col];
                if value.is_empty() {
                    return Err(DatasetError::invalid_value(
                        CAR_EVALUATION_DATASET_NAME,
                        name,
                        value,
                        line_num,
                    ));
                }
                values.push(value.to_string());
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

        // The columns follow the source order: the 6 features, then `class`.
        let mut columns = Vec::with_capacity(N_COLUMNS);
        for (values, &(_col, name)) in features.into_iter().zip(FEATURE_COLUMNS.iter()) {
            columns.push(Column::new(
                name,
                ColumnData::String(Array1::from_vec(values)),
            ));
        }
        columns.push(Column::new(
            Self::TARGET,
            ColumnData::String(Array1::from_vec(labels)),
        ));

        Table::new(CAR_EVALUATION_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 1,728 samples and 7
    ///   columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, an empty value)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`CarEvaluation::data`], this method never runs the loader. If the
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
    /// changes stay in the cache. Later calls to [`CarEvaluation::data`] or
    /// [`CarEvaluation::get_data`] see them.
    ///
    /// Like [`CarEvaluation::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`CarEvaluation::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 1,728 samples and 7 columns.
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
    /// - `Table` - the owned table of 1,728 samples and 7 columns.
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

impl_ml_dataset!(CarEvaluation, "car_evaluation");
