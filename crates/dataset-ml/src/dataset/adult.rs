//! Adult / Census Income dataset.
//!
//! Barry Becker extracted these census records from the 1994 US Census
//! database. The task is to predict if a person's income is over $50K a
//! year. This loader uses the canonical `adult.data` training partition:
//! 32,561 records. This loader does not bundle the separate `adult.test`
//! partition. That partition has a non-data header line and trailing
//! periods on its labels.
//!
//! **Columns (15):**
//!
//! | Name             | Type      | Description                  |
//! |------------------|-----------|-------------------------------|
//! | `age`            | `Numeric` | age in years                 |
//! | `workclass`      | `String`  | employer type                |
//! | `fnlwgt`         | `Numeric` | census sampling weight       |
//! | `education`      | `String`  | highest education level      |
//! | `education-num`  | `Numeric` | education level as a number  |
//! | `marital-status` | `String`  | marital status               |
//! | `occupation`     | `String`  | occupation                   |
//! | `relationship`   | `String`  | role in the household         |
//! | `race`           | `String`  | race                         |
//! | `sex`            | `String`  | `Male` or `Female`            |
//! | `capital-gain`   | `Numeric` | capital gain in USD          |
//! | `capital-loss`   | `Numeric` | capital loss in USD          |
//! | `hours-per-week` | `Numeric` | worked hours per week        |
//! | `native-country` | `String`  | country of origin             |
//! | `income`         | `String`  | `<=50K` or `>50K`             |
//!
//! The source designates the 14 attributes as the inputs
//! ([`Adult::FEATURE_NAMES`](crate::Adult::FEATURE_NAMES)) and `income` as the label ([`Adult::TARGET`](crate::Adult::TARGET)).
//!
//! **Samples:** 32,561
//! **Application:** Binary classification / income prediction
//!
//! **Missing values:** the source marks a missing category with `?` in
//! `workclass`, `occupation`, and `native-country`. The loader stores these
//! values as empty strings. The numeric columns have no missing value.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://archive.ics.uci.edu/dataset/2/adult>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::{ReaderBuilder, Trim};
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
use std::fs::File;

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

/// Number of categorical columns.
const N_STRING_FEATURES: usize = 8;

/// Number of numeric columns.
const N_NUMERIC_FEATURES: usize = 6;

/// Number of feature columns.
const N_FEATURES: usize = N_STRING_FEATURES + N_NUMERIC_FEATURES;

/// Number of columns per record (14 features + 1 label).
const N_COLUMNS: usize = 15;

/// Source column index of the label (`income`).
const LABEL_COLUMN: usize = 14;

/// Categorical columns, as `(source column index, name)`.
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

/// Numeric columns, as `(source column index, name)`.
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

/// A struct that represents the Adult / Census Income dataset with lazy
/// loading.
///
/// The dataset loads only when you call a data accessor method. After the
/// first load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// Barry Becker extracted the Adult dataset (also called "Census Income") from
/// the 1994 US Census database. The prediction task is to determine whether a
/// person earns over $50,000 a year from 14 demographic and employment attributes.
/// It is a standard benchmark for mixed categorical/numeric classification.
///
/// # Columns
///
/// | Name             | Type      | Description                  |
/// |------------------|-----------|-------------------------------|
/// | `age`            | `Numeric` | age in years                 |
/// | `workclass`      | `String`  | employer type                |
/// | `fnlwgt`         | `Numeric` | census sampling weight       |
/// | `education`      | `String`  | highest education level      |
/// | `education-num`  | `Numeric` | education level as a number  |
/// | `marital-status` | `String`  | marital status               |
/// | `occupation`     | `String`  | occupation                   |
/// | `relationship`   | `String`  | role in the household         |
/// | `race`           | `String`  | race                         |
/// | `sex`            | `String`  | `Male` or `Female`            |
/// | `capital-gain`   | `Numeric` | capital gain in USD          |
/// | `capital-loss`   | `Numeric` | capital loss in USD          |
/// | `hours-per-week` | `Numeric` | worked hours per week        |
/// | `native-country` | `String`  | country of origin             |
/// | `income`         | `String`  | `<=50K` or `>50K`             |
///
/// The columns keep the source column order. The source designates the 14
/// attributes as the inputs ([`Adult::FEATURE_NAMES`]) and `income` as the
/// label ([`Adult::TARGET`]).
///
/// Missing values:
/// - The source marks a missing category with `?` in `workclass`, `occupation`,
///   and `native-country`. The loader stores these values as empty strings.
/// - The numeric columns have no missing value.
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
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Adult;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./adult";
///
/// let mut dataset = Adult::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 32561);
/// assert_eq!(table.n_columns(), 15);
///
/// // The 14 features mix types, so `numeric_matrix(&Adult::FEATURE_NAMES)`
/// // fails. Name the six numeric features instead.
/// let numeric = table
///     .numeric_matrix(&[
///         "age",
///         "fnlwgt",
///         "education-num",
///         "capital-gain",
///         "capital-loss",
///         "hours-per-week",
///     ])
///     .unwrap();
/// assert_eq!(numeric.shape(), &[32561, 6]);
///
/// // Reach one column by name.
/// let age = table.column("age").unwrap().as_numeric().unwrap();
/// assert_eq!(age.len(), 32561);
/// let income = table.column(Adult::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(income.len(), 32561);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("age") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 99.0;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 32561);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 32561);
/// ```
#[derive(Debug)]
pub struct Adult {
    dataset: Dataset<Table, DatasetError>,
}

impl Adult {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "income";

    /// Create a new Adult instance without loading data.
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
    /// - `Self` - an `Adult` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Adult {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Adult dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            ADULT_FILENAME,
            ADULT_DATASET_NAME,
            Some(ADULT_SHA256),
            |temp_path| {
                // The source file is `adult.data`, cached here as `adult.csv`.
                download_to_with_retries(
                    ADULT_DATA_URL,
                    temp_path,
                    Some(ADULT_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(ADULT_FILENAME))
            },
        )?;

        // The source is comma-separated with a leading space after each comma, for
        // example `39, State-gov, ...`. The `csv` reader trims whitespace from
        // every field.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(false)
            .trim(Trim::All)
            .from_reader(file);

        let mut string_values: Vec<Vec<String>> = STRING_COLUMNS
            .iter()
            .map(|_| Vec::with_capacity(N_SAMPLES))
            .collect();
        let mut numeric_values: Vec<Vec<f64>> = NUMERIC_COLUMNS
            .iter()
            .map(|_| Vec::with_capacity(N_SAMPLES))
            .collect();
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

            // Categorical columns. Maps the `?` missing token to an empty string.
            for (values, &(col, _name)) in string_values.iter_mut().zip(STRING_COLUMNS.iter()) {
                let value = &record[col];
                if value == MISSING_TOKEN {
                    values.push(String::new());
                } else {
                    values.push(value.to_string());
                }
            }

            // Numeric columns.
            for (values, &(col, name)) in numeric_values.iter_mut().zip(NUMERIC_COLUMNS.iter()) {
                let value: f64 = record[col].parse().map_err(|e| {
                    DatasetError::parse_failed(ADULT_DATASET_NAME, name, line_num, e)
                })?;
                values.push(value);
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

        // Each entry keeps its source column index. The sort then restores the
        // source column order.
        let mut columns: Vec<(usize, Column)> = Vec::with_capacity(N_COLUMNS);
        for (values, &(col, name)) in string_values.into_iter().zip(STRING_COLUMNS.iter()) {
            columns.push((
                col,
                Column::new(name, ColumnData::String(Array1::from_vec(values))),
            ));
        }
        for (values, &(col, name)) in numeric_values.into_iter().zip(NUMERIC_COLUMNS.iter()) {
            columns.push((
                col,
                Column::new(name, ColumnData::Numeric(Array1::from_vec(values))),
            ));
        }
        columns.push((
            LABEL_COLUMN,
            Column::new(Self::TARGET, ColumnData::String(Array1::from_vec(labels))),
        ));
        columns.sort_by_key(|entry| entry.0);

        Table::new(
            ADULT_DATASET_NAME,
            columns.into_iter().map(|(_, column)| column).collect(),
        )
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 32,561 samples and 15
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
    /// Unlike [`Adult::data`], this method never runs the loader. If the data has
    /// not loaded yet, it returns `None` instead of downloading and parsing it.
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
    /// changes stay in the cache. Later calls to [`Adult::data`] or
    /// [`Adult::get_data`] see them.
    ///
    /// Like [`Adult::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`Adult::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 32,561 samples and 15 columns.
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
    /// - `Table` - the owned table of 32,561 samples and 15 columns.
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

impl_ml_dataset!(Adult, "adult");
