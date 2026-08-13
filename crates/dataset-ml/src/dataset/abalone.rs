//! Abalone dataset.
//!
//! Physical measurements of abalone (a marine snail), used to predict the age of
//! the animal. The age in years is the number of `rings` plus 1.5. Counting the
//! rings through a microscope is a slow, tedious task. The goal is to predict the
//! age from easier physical measurements instead. The `rings` column is the
//! regression target.
//!
//! **Columns (9):**
//!
//! | Name             | Type      | Description                            |
//! |------------------|-----------|-----------------------------------------|
//! | `sex`            | `String`  | `M` (male), `F` (female), `I` (infant) |
//! | `length`         | `Numeric` | longest shell measurement in mm        |
//! | `diameter`       | `Numeric` | diameter across the length in mm       |
//! | `height`         | `Numeric` | height with meat in the shell in mm    |
//! | `whole_weight`   | `Numeric` | weight of the whole abalone in grams   |
//! | `shucked_weight` | `Numeric` | weight of the meat in grams            |
//! | `viscera_weight` | `Numeric` | gut weight after bleeding in grams     |
//! | `shell_weight`   | `Numeric` | weight of the dried shell in grams     |
//! | `rings`          | `Numeric` | ring count from 1 to 29, age is `rings + 1.5` |
//!
//! The source designates `sex` and the seven measurements as the inputs
//! ([`Abalone::FEATURE_NAMES`](crate::Abalone::FEATURE_NAMES)) and `rings` as the label ([`Abalone::TARGET`](crate::Abalone::TARGET)).
//!
//! **Samples:** 4,177
//! **Application:** Regression / age prediction
//!
//! **Missing values:** none.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C55C7W>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
use std::fs::File;

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

/// A struct that represents the Abalone dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// You find the age of an abalone by cutting the shell through the cone,
/// staining it, and counting the rings through a microscope. This is a boring
/// and time-consuming task. The goal is to predict the age instead, from other
/// measurements that are easier to get. The `rings` column holds the raw ring
/// count, and it is the regression target. The actual age in years is
/// `rings + 1.5`. The only categorical column is `sex` (`M`/`F`/`I`). The other
/// seven features are continuous measurements.
///
/// # Columns
///
/// | Name             | Type      | Description                            |
/// |------------------|-----------|-----------------------------------------|
/// | `sex`            | `String`  | `M` (male), `F` (female), `I` (infant) |
/// | `length`         | `Numeric` | longest shell measurement in mm        |
/// | `diameter`       | `Numeric` | diameter across the length in mm       |
/// | `height`         | `Numeric` | height with meat in the shell in mm    |
/// | `whole_weight`   | `Numeric` | weight of the whole abalone in grams   |
/// | `shucked_weight` | `Numeric` | weight of the meat in grams            |
/// | `viscera_weight` | `Numeric` | gut weight after bleeding in grams     |
/// | `shell_weight`   | `Numeric` | weight of the dried shell in grams     |
/// | `rings`          | `Numeric` | ring count from 1 to 29, age is `rings + 1.5` |
///
/// The source designates `sex` and the seven measurements as the inputs
/// ([`Abalone::FEATURE_NAMES`]) and `rings` as the label ([`Abalone::TARGET`]).
///
/// Missing values: none.
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
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Abalone;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./abalone";
///
/// let mut dataset = Abalone::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 4177);
/// assert_eq!(table.n_columns(), 9);
///
/// // Ask for the numeric target column when you want it.
/// let rings = table.column(Abalone::TARGET).unwrap().as_numeric().unwrap();
/// assert_eq!(rings.len(), 4177);
///
/// // Reach the one string column by name.
/// let sex = table.column("sex").unwrap().as_string().unwrap();
/// assert_eq!(sex.len(), 4177);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("length") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 0.5;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 4177);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 4177);
/// ```
#[derive(Debug)]
pub struct Abalone {
    dataset: Dataset<Table, DatasetError>,
}

impl Abalone {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_COLUMNS - 1] = [
        "sex",
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "rings";

    /// Create a new Abalone instance without loading data.
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
    /// - `Self` - an `Abalone` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Abalone {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Abalone dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        // The source file is `abalone.data`. The loader caches it under
        // `abalone.csv`.
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

        let mut string_features: Vec<Vec<String>> = STRING_COLUMNS
            .iter()
            .map(|_| Vec::with_capacity(N_SAMPLES))
            .collect();
        let mut numeric_features: Vec<Vec<f64>> = NUMERIC_COLUMNS
            .iter()
            .map(|_| Vec::with_capacity(N_SAMPLES))
            .collect();
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
            for (values, &(col, name)) in string_features.iter_mut().zip(STRING_COLUMNS.iter()) {
                let value = &record[col];
                if value.is_empty() {
                    return Err(DatasetError::invalid_value(
                        ABALONE_DATASET_NAME,
                        name,
                        value,
                        line_num,
                    ));
                }
                values.push(value.to_string());
            }

            // Numeric features.
            for (values, &(col, name)) in numeric_features.iter_mut().zip(NUMERIC_COLUMNS.iter()) {
                let value: f64 = record[col].parse().map_err(|e| {
                    DatasetError::parse_failed(ABALONE_DATASET_NAME, name, line_num, e)
                })?;
                values.push(value);
            }

            // Regression target (`rings`).
            let target: f64 = record[TARGET_COLUMN].parse().map_err(|e| {
                DatasetError::parse_failed(ABALONE_DATASET_NAME, "rings", line_num, e)
            })?;
            targets.push(target);
        }

        // The columns follow the source order: `sex`, the 7 measurements, then
        // `rings`.
        let mut columns = Vec::with_capacity(N_COLUMNS);
        for (values, &(_col, name)) in string_features.into_iter().zip(STRING_COLUMNS.iter()) {
            columns.push(Column::new(
                name,
                ColumnData::String(Array1::from_vec(values)),
            ));
        }
        for (values, &(_col, name)) in numeric_features.into_iter().zip(NUMERIC_COLUMNS.iter()) {
            columns.push(Column::new(
                name,
                ColumnData::Numeric(Array1::from_vec(values)),
            ));
        }
        columns.push(Column::new(
            Self::TARGET,
            ColumnData::Numeric(Array1::from_vec(targets)),
        ));

        Table::new(ABALONE_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 4,177 samples and 9
    ///   columns.
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
    /// Unlike [`Abalone::data`], this method never runs the loader. If the data
    /// has not loaded yet, it returns `None` instead of downloading and parsing
    /// it.
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
    /// changes stay in the cache. Later calls to [`Abalone::data`] or
    /// [`Abalone::get_data`] see them.
    ///
    /// Like [`Abalone::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`Abalone::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 4,177 samples and 9 columns.
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
    /// - `Table` - the owned table of 4,177 samples and 9 columns.
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

impl_ml_dataset!(Abalone, "abalone");
