//! Bank Marketing dataset.
//!
//! Direct marketing campaign records (phone calls) of a Portuguese banking
//! institution, used to predict whether a client will subscribe a term deposit.
//! This loader uses the full `bank-full.csv` partition (45,211 records, 16
//! features), the classic version of the dataset.
//!
//! **Features (16, mixed):**
//! - String features (9): `job`, `marital`, `education`, `default`, `housing`,
//!   `loan`, `contact`, `month`, `poutcome`
//! - Numeric features (7): `age`, `balance`, `day`, `duration`, `campaign`,
//!   `pdays`, `previous`
//!
//! **Target:** `y` — binary label kept verbatim (`yes` or `no`): has the client
//! subscribed a term deposit?
//!
//! **Samples:** 45,211
//! **Application:** Binary classification / term-deposit subscription prediction
//!
//! **Source:** UCI Machine Learning Repository
//! <https://archive.ics.uci.edu/dataset/222/bank+marketing>

use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to, unzip};
use ndarray::{Array1, Array2};
use std::fs::File;

/// Type alias for Bank Marketing dataset: (string features, numeric features, labels).
type BankMarketingData = (Array2<String>, Array2<f64>, Array1<String>);

/// The URL for the Bank Marketing dataset (the ZIP archive holding `bank-full.csv`).
const BANK_DATA_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip";

/// The name the downloaded ZIP archive is saved under inside the temp directory.
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

/// Number of categorical (string) features.
const N_STRING_FEATURES: usize = 9;

/// Number of numeric features.
const N_NUMERIC_FEATURES: usize = 7;

/// Number of columns per record (16 features + 1 label).
const N_COLUMNS: usize = 17;

/// Source column index of the label (`y`).
const LABEL_COLUMN: usize = 16;

/// Categorical feature columns, as `(source column index, name)`, in output order.
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

/// Numeric feature columns, as `(source column index, name)`, in output order.
const NUMERIC_COLUMNS: [(usize, &str); N_NUMERIC_FEATURES] = [
    (0, "age"),
    (5, "balance"),
    (9, "day"),
    (11, "duration"),
    (12, "campaign"),
    (13, "pdays"),
    (14, "previous"),
];

/// A struct representing the Bank Marketing dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The Bank Marketing dataset records direct marketing campaigns (phone calls) of a
/// Portuguese banking institution. The classification goal is to predict whether a
/// client will subscribe a term deposit (`y`) from 16 client, contact, and campaign
/// attributes. It is a standard benchmark for mixed categorical/numeric, heavily
/// imbalanced binary classification.
///
/// # Feature columns
///
/// Features are split across two matrices: a `(45211, 9)` string matrix and a
/// `(45211, 7)` numeric `f64` matrix.
///
/// String features (`Array2<String>`), by 0-based column:
///
/// | Column | Attribute   |
/// |--------|-------------|
/// | `0`    | `job`       |
/// | `1`    | `marital`   |
/// | `2`    | `education` |
/// | `3`    | `default`   |
/// | `4`    | `housing`   |
/// | `5`    | `loan`      |
/// | `6`    | `contact`   |
/// | `7`    | `month`     |
/// | `8`    | `poutcome`  |
///
/// Numeric features (`Array2<f64>`), by 0-based column:
///
/// | Column | Attribute  | Unit            |
/// |--------|------------|-----------------|
/// | `0`    | `age`      | years           |
/// | `1`    | `balance`  | EUR (can be < 0)|
/// | `2`    | `day`      | day of month    |
/// | `3`    | `duration` | seconds         |
/// | `4`    | `campaign` | contacts        |
/// | `5`    | `pdays`    | days (`-1` = not previously contacted) |
/// | `6`    | `previous` | contacts        |
///
/// # Labels
///
/// - `y` (shape `(45211,)`): the `Array1<String>` is kept verbatim, each entry being
///   either `yes` or `no` (whether the client subscribed a term deposit).
///
/// Missing values:
/// - Some categorical attributes (`job`, `education`, `contact`, `poutcome`) use the
///   literal label `unknown`. This loader keeps `unknown` **verbatim** as a category
///   value (unlike some datasets that map a missing token to an empty string), since
///   it is a documented level — in particular `poutcome = unknown` means there was no
///   previous campaign contact, which is informative.
/// - The numeric features have no missing values (`pdays = -1` encodes "not
///   previously contacted").
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
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::bank_marketing::BankMarketing;
///
/// let download_dir = "./bank_marketing"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = BankMarketing::new(download_dir);
/// let (string_features, numeric_features) = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (string_features, numeric_features, labels) = dataset.data().unwrap(); // this is also a way to get all data
/// assert_eq!(string_features.shape(), &[45211, 9]);
/// assert_eq!(numeric_features.shape(), &[45211, 7]);
/// assert_eq!(labels.len(), 45211);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((_strings, numerics, labels)) = dataset.get_data_mut() {
///     numerics[[0, 0]] = 99.0;
///     labels[0] = "yes".to_string();
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out (no `to_owned()` clone) and leaves
/// // the instance reusable — the next access reloads from the cached file.
/// let (owned_strings, owned_numerics, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_strings.shape(), &[45211, 9]);
/// assert_eq!(owned_numerics.shape(), &[45211, 7]);
/// assert_eq!(owned_labels.len(), 45211);
///
/// // `into_data()` also returns the owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_strings, owned_numerics, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_strings.shape(), &[45211, 9]);
/// assert_eq!(owned_numerics.shape(), &[45211, 7]);
/// assert_eq!(owned_labels.len(), 45211);
/// ```
#[derive(Debug)]
pub struct BankMarketing {
    dataset: Dataset<BankMarketingData, DatasetError>,
}

impl BankMarketing {
    /// Create a new Bank Marketing instance without loading data.
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
    /// - `Self` - `BankMarketing` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        BankMarketing {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Bank Marketing dataset.
    fn load_data(dir: &str) -> Result<BankMarketingData, DatasetError> {
        // Prepare the dataset file: download the ZIP, extract it, and use the
        // full `bank-full.csv` partition (cached under `bank_marketing.csv`).
        let file_path = acquire_dataset(
            dir,
            BANK_FILENAME,
            BANK_DATASET_NAME,
            Some(BANK_SHA256),
            |temp_path| {
                download_to(BANK_DATA_URL, temp_path, Some(BANK_ZIP_FILENAME))?;
                unzip(&temp_path.join(BANK_ZIP_FILENAME), temp_path)?;
                Ok(temp_path.join(BANK_SOURCE_FILENAME))
            },
        )?;

        // The source is semicolon-separated with double-quoted string fields and a
        // header row; csv strips the quotes and `has_headers(true)` skips the header.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new()
            .delimiter(b';')
            .has_headers(true)
            .from_reader(file);

        let mut string_features: Vec<String> = Vec::with_capacity(N_SAMPLES * N_STRING_FEATURES);
        let mut numeric_features: Vec<f64> = Vec::with_capacity(N_SAMPLES * N_NUMERIC_FEATURES);
        let mut labels: Vec<String> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record = result.map_err(|e| DatasetError::csv_read_error(BANK_DATASET_NAME, e))?;
            let line_num = idx + 2; // +1 for the header, +1 for 1-based lines

            // Skip blank lines defensively (e.g. a trailing newline).
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

            // Categorical features, kept verbatim (`unknown` is a documented level).
            for &(col, _name) in STRING_COLUMNS.iter() {
                string_features.push(record[col].to_string());
            }

            // Numeric features (`balance` and `pdays` may be negative).
            for &(col, name) in NUMERIC_COLUMNS.iter() {
                let value: f64 = record[col].parse().map_err(|e| {
                    DatasetError::parse_failed(BANK_DATASET_NAME, name, line_num, e)
                })?;
                numeric_features.push(value);
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

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(BANK_DATASET_NAME));
        }

        let string_array = Array2::from_shape_vec((n_samples, N_STRING_FEATURES), string_features)
            .map_err(|e| {
                DatasetError::array_shape_error(BANK_DATASET_NAME, "string_features", e)
            })?;

        let numeric_array =
            Array2::from_shape_vec((n_samples, N_NUMERIC_FEATURES), numeric_features).map_err(
                |e| DatasetError::array_shape_error(BANK_DATASET_NAME, "numeric_features", e),
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
    /// - `&Array2<String>` - Reference to string feature matrix with shape `(45211, 9)` containing:
    ///     - `job`
    ///     - `marital`
    ///     - `education`
    ///     - `default`
    ///     - `housing`
    ///     - `loan`
    ///     - `contact`
    ///     - `month`
    ///     - `poutcome`
    ///
    ///   (`unknown` kept verbatim where present in source)
    ///
    /// - `&Array2<f64>` - Reference to numeric feature matrix with shape `(45211, 7)` containing:
    ///     - `age`
    ///     - `balance`
    ///     - `day`
    ///     - `duration`
    ///     - `campaign`
    ///     - `pdays`
    ///     - `previous`
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (45,211 samples)
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
    /// - `&Array1<String>` - Reference to label vector with shape `(45211,)` containing `y` values (`yes` or `no`)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (45,211 samples)
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
    /// - `&BankMarketingData` - reference to the cached `(string features, numeric
    ///   features, labels)` tuple: string feature matrix `(45211, 9)`, numeric
    ///   feature matrix `(45211, 7)`, and label vector `(45211,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (45,211 samples)
    pub fn data(&self) -> Result<&BankMarketingData, DatasetError> {
        self.dataset.load()
    }

    /// Get string features, numeric features and labels as references
    /// **without** triggering loading.
    ///
    /// Unlike [`BankMarketing::data`], which loads the dataset on first call, this
    /// never runs the loader: if the data has not been loaded yet, it returns `None`
    /// instead of downloading and parsing. Use it when you only want the data if
    /// it is already cached and want to avoid paying the download/parse cost
    /// otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&BankMarketingData)` - reference to the cached `(string features,
    ///   numeric features, labels)` tuple (`(45211, 9)`, `(45211, 7)`, `(45211,)`),
    ///   if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&BankMarketingData> {
        self.dataset.get()
    }

    /// Get mutable references to string features, numeric features, and labels
    /// for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. encode categorical
    /// features, normalize numeric features) with no `to_owned()` clone and without
    /// removing them from the cache: the changes persist, so later
    /// [`BankMarketing::features`], [`BankMarketing::data`], or
    /// [`BankMarketing::get_data`] calls observe them.
    ///
    /// Like [`BankMarketing::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call a loading accessor (e.g.
    /// [`BankMarketing::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut BankMarketingData)` - mutable reference to the cached `(string
    ///   features, numeric features, labels)` tuple (`(45211, 9)`, `(45211, 7)`,
    ///   `(45211,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut BankMarketingData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** string features, numeric features,
    /// and labels.
    ///
    /// Unlike [`BankMarketing::data`], which borrows the cached data, this moves it
    /// out and returns owned arrays directly — no `to_owned()` clone needed. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`BankMarketing::take_data`] instead — it takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array2<f64>, Array1<String>)` - owned string feature matrix
    ///   `(45211, 9)`, owned numeric feature matrix `(45211, 7)`, and owned label vector
    ///   `(45211,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<BankMarketingData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** string features, numeric features, and labels out of the
    /// dataset, leaving it reusable.
    ///
    /// Like [`BankMarketing::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out, resetting the instance to its
    /// unloaded state: the next accessor call (e.g. [`BankMarketing::features`] or
    /// [`BankMarketing::data`]) loads the dataset again.
    ///
    /// Use [`BankMarketing::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array2<f64>, Array1<String>)` - owned string feature matrix
    ///   `(45211, 9)`, owned numeric feature matrix `(45211, 7)`, and owned label vector
    ///   `(45211,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<BankMarketingData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}
