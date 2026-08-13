//! Linnerud dataset (scikit-learn `load_linnerud`).
//!
//! Dr. A. C. Linnerud collected this small multi-output regression dataset at
//! North Carolina State University. He measured three exercise variables and
//! three physiological variables on 20 middle-aged men in a fitness club. The
//! task is to predict the three physiological measurements from the three
//! exercise measurements (multi-output regression).
//!
//! This loader reproduces scikit-learn's `load_linnerud()` output. Scikit-learn
//! distributes the two underlying files: the whitespace-separated
//! `linnerud_exercise.csv` and `linnerud_physiological.csv`.
//!
//! **Columns (6):** in scikit-learn column order
//!
//! | Name     | Type      | Description             |
//! |----------|-----------|-------------------------|
//! | `Chins`  | `Numeric` | number of chin-ups      |
//! | `Situps` | `Numeric` | number of sit-ups       |
//! | `Jumps`  | `Numeric` | number of jumping jacks |
//! | `Weight` | `Numeric` | body weight             |
//! | `Waist`  | `Numeric` | waist circumference     |
//! | `Pulse`  | `Numeric` | resting pulse           |
//!
//! The source designates the three exercise measurements as the inputs
//! ([`Linnerud::FEATURE_NAMES`](crate::Linnerud::FEATURE_NAMES)) and the three physiological measurements as
//! the labels ([`Linnerud::TARGET_NAMES`](crate::Linnerud::TARGET_NAMES)).
//!
//! **Samples:** 20
//! **Application:** Multi-output regression / fitness modeling
//!
//! **Source:** Tenenhaus, M. (1998), *La régression PLS: théorie et pratique*,
//! Paris: Editions Technip. Distributed with scikit-learn as
//! `linnerud_exercise.csv` and `linnerud_physiological.csv`.

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
use std::path::Path;

/// The URL for the Linnerud exercise (feature) file that scikit-learn distributes.
const LINNERUD_EXERCISE_URL: &str = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/linnerud_exercise.csv";

/// The URL for the Linnerud physiological (target) file that scikit-learn distributes.
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

/// Parse one of the Linnerud whitespace-separated files into named columns.
///
/// The files have a single header row (column names) followed by 20 data rows,
/// each holding exactly [`N_COLUMNS`] whitespace-separated numeric values. The
/// function skips the header and splits every data row on arbitrary whitespace.
///
/// # Parameters
///
/// - `file_path` - The file to read.
/// - `array_name` - The name this file carries in error messages.
/// - `names` - The column names, in file column order.
fn parse_linnerud_file(
    file_path: &Path,
    array_name: &str,
    names: [&'static str; N_COLUMNS],
) -> Result<Vec<Column>, DatasetError> {
    let content = std::fs::read_to_string(file_path)?;

    let mut values: Vec<Vec<f64>> = vec![Vec::new(); N_COLUMNS];

    // `enumerate` gives 0-based indices. The header is line 1, so data starts at
    // index 1. Its 1-based line number is `idx + 1`.
    for (idx, line) in content.lines().enumerate().skip(1) {
        let line_num = idx + 1;
        if line.trim().is_empty() {
            continue;
        }

        let mut row: Vec<f64> = Vec::new();
        for token in line.split_whitespace() {
            let value: f64 = token.parse().map_err(|e| {
                DatasetError::parse_failed(LINNERUD_DATASET_NAME, array_name, line_num, e)
            })?;
            row.push(value);
        }

        if row.len() != N_COLUMNS {
            return Err(DatasetError::invalid_column_count(
                LINNERUD_DATASET_NAME,
                N_COLUMNS,
                row.len(),
                line_num,
            ));
        }

        for (column, value) in values.iter_mut().zip(row) {
            column.push(value);
        }
    }

    Ok(names
        .iter()
        .zip(values)
        .map(|(&name, column)| Column::new(name, ColumnData::Numeric(Array1::from_vec(column))))
        .collect())
}

/// A struct that represents the Linnerud dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The Linnerud dataset records three exercise variables and three physiological
/// variables, measured on 20 middle-aged men in a fitness club. This loader
/// reproduces scikit-learn's `load_linnerud()` output. Three target columns
/// make this a multi-output regression task.
///
/// # Columns
///
/// | Name     | Type      | Description             |
/// |----------|-----------|-------------------------|
/// | `Chins`  | `Numeric` | number of chin-ups      |
/// | `Situps` | `Numeric` | number of sit-ups       |
/// | `Jumps`  | `Numeric` | number of jumping jacks |
/// | `Weight` | `Numeric` | body weight             |
/// | `Waist`  | `Numeric` | waist circumference     |
/// | `Pulse`  | `Numeric` | resting pulse           |
///
/// The source designates the three exercise measurements as the inputs
/// ([`Linnerud::FEATURE_NAMES`]) and the three physiological measurements as
/// the labels ([`Linnerud::TARGET_NAMES`]).
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
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Linnerud;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./linnerud";
///
/// let mut dataset = Linnerud::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 20);
/// assert_eq!(table.n_columns(), 6);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&Linnerud::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[20, 3]);
///
/// // The three target columns form the multi-output target matrix.
/// let targets = table.numeric_matrix(&Linnerud::TARGET_NAMES).unwrap();
/// assert_eq!(targets.shape(), &[20, 3]);
///
/// // Reach one column by name.
/// let weight = table.column("Weight").unwrap().as_numeric().unwrap();
/// assert_eq!(weight.len(), 20);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("Chins") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 6.0;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 20);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 20);
/// ```
#[derive(Debug)]
pub struct Linnerud {
    dataset: Dataset<Table, DatasetError>,
}

impl Linnerud {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_COLUMNS] = ["Chins", "Situps", "Jumps"];

    /// The columns the source designates as the labels, in source order.
    pub const TARGET_NAMES: [&'static str; N_COLUMNS] = ["Weight", "Waist", "Pulse"];

    /// Create a new Linnerud instance without loading data.
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
    /// - `Self` - a `Linnerud` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Linnerud {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Linnerud dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        // The exercise and physiological measurements live in two separate files.
        // The loader gets and verifies each file independently with SHA-256.
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

        let mut columns = parse_linnerud_file(&exercise_path, "features", Self::FEATURE_NAMES)?;
        columns.extend(parse_linnerud_file(
            &physiological_path,
            "targets",
            Self::TARGET_NAMES,
        )?);

        Table::new(LINNERUD_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 20 samples and 6 columns.
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
    /// Unlike [`Linnerud::data`], this method never runs the loader. If the data
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
    /// changes stay in the cache. Later calls to [`Linnerud::data`] or
    /// [`Linnerud::get_data`] see them.
    ///
    /// Like [`Linnerud::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`Linnerud::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 20 samples and 6 columns.
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
    /// - `Table` - the owned table of 20 samples and 6 columns.
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

impl_ml_dataset!(Linnerud, "linnerud");
