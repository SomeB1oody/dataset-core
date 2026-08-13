//! Bank Marketing dataset.
//!
//! Direct marketing campaign records (phone calls) from a Portuguese bank. The
//! task is to predict if a client will subscribe to a term deposit. This
//! loader uses the full `bank-full.csv` partition: 45,211 records and 16
//! features. This is the classic version of the dataset.
//!
//! **Columns (17):**
//!
//! | Name        | Type      | Description                                     |
//! |-------------|-----------|--------------------------------------------------|
//! | `age`       | `Numeric` | age in years                                    |
//! | `job`       | `String`  | job type                                        |
//! | `marital`   | `String`  | `married`, `single`, or `divorced`              |
//! | `education` | `String`  | education level                                 |
//! | `default`   | `String`  | `yes` if the client has credit in default       |
//! | `balance`   | `Numeric` | average yearly balance in EUR, can be negative  |
//! | `housing`   | `String`  | `yes` if the client has a housing loan          |
//! | `loan`      | `String`  | `yes` if the client has a personal loan         |
//! | `contact`   | `String`  | contact communication type                      |
//! | `day`       | `Numeric` | day of the month of the last contact            |
//! | `month`     | `String`  | month of the last contact, such as `may`        |
//! | `duration`  | `Numeric` | duration of the last contact in seconds         |
//! | `campaign`  | `Numeric` | contacts made during this campaign              |
//! | `pdays`     | `Numeric` | days since the last contact of a previous campaign, `-1` for none |
//! | `previous`  | `Numeric` | contacts made before this campaign              |
//! | `poutcome`  | `String`  | outcome of the previous campaign                |
//! | `y`         | `String`  | `yes` or `no`, the term deposit subscription    |
//!
//! The source designates the 16 attributes as the inputs
//! ([`BankMarketing::FEATURE_NAMES`](crate::BankMarketing::FEATURE_NAMES)) and `y` as the label
//! ([`BankMarketing::TARGET`](crate::BankMarketing::TARGET)).
//!
//! **Samples:** 45,211
//! **Application:** Binary classification / term-deposit subscription prediction
//!
//! **Missing values:** `job`, `education`, `contact`, and `poutcome` use the
//! literal category `unknown`. The loader keeps `unknown` verbatim. The numeric
//! columns have no missing value.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://archive.ics.uci.edu/dataset/222/bank+marketing>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::Array1;
use std::fs::File;

/// The URL for the Bank Marketing dataset (the ZIP archive holding `bank-full.csv`).
const BANK_DATA_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip";

/// The filename used for the downloaded ZIP archive inside the temp directory.
const BANK_ZIP_FILENAME: &str = "bank.zip";

/// The name of the file inside the archive that this loader uses (the full set).
const BANK_SOURCE_FILENAME: &str = "bank-full.csv";

/// The name of the final cached Bank Marketing dataset file.
const BANK_FILENAME: &str = "bank_marketing.csv";

/// The SHA256 hash of the cached Bank Marketing dataset file (`bank-full.csv`).
const BANK_SHA256: &str = "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d";

/// The name of the dataset.
const BANK_DATASET_NAME: &str = "bank_marketing";

/// Number of samples in the `bank-full.csv` partition.
const N_SAMPLES: usize = 45_211;

/// Number of categorical columns.
const N_STRING_FEATURES: usize = 9;

/// Number of numeric columns.
const N_NUMERIC_FEATURES: usize = 7;

/// Number of feature columns.
const N_FEATURES: usize = N_STRING_FEATURES + N_NUMERIC_FEATURES;

/// Number of columns per record (16 features + 1 label).
const N_COLUMNS: usize = 17;

/// Source column index of the label (`y`).
const LABEL_COLUMN: usize = 16;

/// Categorical columns, as `(source column index, name)`.
const STRING_COLUMNS: [(usize, &str); N_STRING_FEATURES] = [
    (1, "job"),
    (2, "marital"),
    (3, "education"),
    (4, "default"),
    (6, "housing"),
    (7, "loan"),
    (8, "contact"),
    (10, "month"),
    (15, "poutcome"),
];

/// Numeric columns, as `(source column index, name)`.
const NUMERIC_COLUMNS: [(usize, &str); N_NUMERIC_FEATURES] = [
    (0, "age"),
    (5, "balance"),
    (9, "day"),
    (11, "duration"),
    (12, "campaign"),
    (13, "pdays"),
    (14, "previous"),
];

/// A struct that represents the Bank Marketing dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the
/// first load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The Bank Marketing dataset records direct marketing campaigns (phone calls) of a
/// Portuguese banking institution. The classification goal is to predict whether a
/// client will subscribe to a term deposit (`y`) from 16 client, contact, and campaign
/// attributes. It is a standard benchmark for mixed categorical/numeric, heavily
/// imbalanced binary classification.
///
/// # Columns
///
/// | Name        | Type      | Description                                     |
/// |-------------|-----------|--------------------------------------------------|
/// | `age`       | `Numeric` | age in years                                    |
/// | `job`       | `String`  | job type                                        |
/// | `marital`   | `String`  | `married`, `single`, or `divorced`              |
/// | `education` | `String`  | education level                                 |
/// | `default`   | `String`  | `yes` if the client has credit in default       |
/// | `balance`   | `Numeric` | average yearly balance in EUR, can be negative  |
/// | `housing`   | `String`  | `yes` if the client has a housing loan          |
/// | `loan`      | `String`  | `yes` if the client has a personal loan         |
/// | `contact`   | `String`  | contact communication type                      |
/// | `day`       | `Numeric` | day of the month of the last contact            |
/// | `month`     | `String`  | month of the last contact, such as `may`        |
/// | `duration`  | `Numeric` | duration of the last contact in seconds         |
/// | `campaign`  | `Numeric` | contacts made during this campaign              |
/// | `pdays`     | `Numeric` | days since the last contact of a previous campaign, `-1` for none |
/// | `previous`  | `Numeric` | contacts made before this campaign              |
/// | `poutcome`  | `String`  | outcome of the previous campaign                |
/// | `y`         | `String`  | `yes` or `no`, the term deposit subscription    |
///
/// The columns keep the source column order. The source designates the 16
/// attributes as the inputs ([`BankMarketing::FEATURE_NAMES`]) and `y` as the
/// label ([`BankMarketing::TARGET`]).
///
/// Missing values:
/// - Some categorical columns (`job`, `education`, `contact`, `poutcome`) use the
///   literal label `unknown`. This loader keeps `unknown` **verbatim** as a category
///   value, unlike loaders that map a missing token to an empty string. `unknown` is
///   a documented level: for `poutcome`, it means there was no previous campaign
///   contact. This is useful information, not a missing value.
/// - The numeric columns have no missing value. `pdays = -1` encodes "not
///   previously contacted".
///
/// See more information at <https://archive.ics.uci.edu/dataset/222/bank+marketing>.
///
/// # Citation
///
/// Moro, S., Rita, P. & Cortez, P. (2012). Bank Marketing \[Dataset\]. UCI Machine
/// Learning Repository. <https://doi.org/10.24432/C5K306>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::BankMarketing;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./bank_marketing";
///
/// let mut dataset = BankMarketing::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 45211);
/// assert_eq!(table.n_columns(), 17);
///
/// // The 16 features mix types, so `numeric_matrix(&BankMarketing::FEATURE_NAMES)`
/// // fails. Name the seven numeric features instead.
/// let numeric = table
///     .numeric_matrix(&[
///         "age", "balance", "day", "duration", "campaign", "pdays", "previous",
///     ])
///     .unwrap();
/// assert_eq!(numeric.shape(), &[45211, 7]);
///
/// // Reach one column by name.
/// let age = table.column("age").unwrap().as_numeric().unwrap();
/// assert_eq!(age.len(), 45211);
/// let y = table.column(BankMarketing::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(y.len(), 45211);
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
/// assert_eq!(owned.n_samples(), 45211);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 45211);
/// ```
#[derive(Debug)]
pub struct BankMarketing {
    dataset: Dataset<Table, DatasetError>,
}

impl BankMarketing {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "y";

    /// Create a new BankMarketing instance without loading data.
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
    /// - `Self` - a `BankMarketing` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        BankMarketing {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Bank Marketing dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        // This loader uses the full `bank-full.csv` partition, cached as
        // `bank_marketing.csv`.
        let file_path = acquire_dataset(
            dir,
            BANK_FILENAME,
            BANK_DATASET_NAME,
            Some(BANK_SHA256),
            |temp_path| {
                download_to_with_retries(
                    BANK_DATA_URL,
                    temp_path,
                    Some(BANK_ZIP_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                unzip(&temp_path.join(BANK_ZIP_FILENAME), temp_path)?;
                Ok(temp_path.join(BANK_SOURCE_FILENAME))
            },
        )?;

        // The source is semicolon-separated, with double-quoted string fields and a
        // header row. The `csv` crate strips the quotes, and `has_headers(true)`
        // skips the header.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new()
            .delimiter(b';')
            .has_headers(true)
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
            let record = result.map_err(|e| DatasetError::csv_read_error(BANK_DATASET_NAME, e))?;
            let line_num = idx + 2; // +1 for the header, +1 for 1-based lines

            // Skip blank lines, such as a trailing newline at the end of the file.
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    BANK_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // Categorical columns, kept verbatim (`unknown` is a documented level).
            for (values, &(col, _name)) in string_values.iter_mut().zip(STRING_COLUMNS.iter()) {
                values.push(record[col].to_string());
            }

            // Numeric columns (`balance` and `pdays` may be negative).
            for (values, &(col, name)) in numeric_values.iter_mut().zip(NUMERIC_COLUMNS.iter()) {
                let value: f64 = record[col].parse().map_err(|e| {
                    DatasetError::parse_failed(BANK_DATASET_NAME, name, line_num, e)
                })?;
                values.push(value);
            }

            // Label, kept verbatim (`yes` or `no`).
            let label = &record[LABEL_COLUMN];
            if label.is_empty() {
                return Err(DatasetError::invalid_value(
                    BANK_DATASET_NAME,
                    "y",
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
            BANK_DATASET_NAME,
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
    /// - `&Table` - reference to the cached table of 45,211 samples and 17
    ///   columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`BankMarketing::data`], this method never runs the loader. If the
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
    /// changes stay in the cache. Later calls to [`BankMarketing::data`] or
    /// [`BankMarketing::get_data`] see them.
    ///
    /// Like [`BankMarketing::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`BankMarketing::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 45,211 samples and 17 columns.
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
    /// - `Table` - the owned table of 45,211 samples and 17 columns.
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

impl_ml_dataset!(BankMarketing, "bank_marketing");
