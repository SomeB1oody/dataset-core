//! Banknote Authentication dataset.
//!
//! The dataset holds features extracted from images of genuine and forged
//! banknote-like specimens. Researchers digitized the images with an
//! industrial camera normally used for print inspection. They then used a
//! Wavelet Transform tool to derive four continuous statistics from each
//! image. The task is to predict the class of a specimen from those four
//! features.
//!
//! **Columns (5):**
//!
//! | Name       | Type      | Description                               |
//! |------------|-----------|--------------------------------------------|
//! | `variance` | `Numeric` | variance of the Wavelet-Transformed image |
//! | `skewness` | `Numeric` | skewness of the Wavelet-Transformed image |
//! | `curtosis` | `Numeric` | curtosis of the Wavelet-Transformed image |
//! | `entropy`  | `Numeric` | entropy of the image                      |
//! | `class`    | `Integer` | raw class code, `0` or `1`                |
//!
//! `curtosis` keeps the source's spelling. UCI names the attribute that way.
//!
//! The source designates the four statistics as the inputs
//! ([`BanknoteAuthentication::FEATURE_NAMES`](crate::BanknoteAuthentication::FEATURE_NAMES)) and `class` as the label
//! ([`BanknoteAuthentication::TARGET`](crate::BanknoteAuthentication::TARGET)).
//!
//! **Samples:** 1372 total (762 of class `0`, 610 of class `1`)
//! **Application:** Binary classification / banknote authentication
//!
//! **Missing values:** none.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C55P57>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::Array1;
use std::fs::File;

use csv::ReaderBuilder;

/// The URL for the Banknote Authentication dataset.
///
/// This is the UCI static package. It is a ZIP archive that contains a single
/// file, `data_banknote_authentication.txt`.
///
/// # Citation
///
/// V. Lohweg. "Banknote Authentication," UCI Machine Learning Repository,
/// \[Online\]. Available: <https://doi.org/10.24432/C55P57>
const BANKNOTE_AUTHENTICATION_DATA_URL: &str =
    "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip";

/// The filename used for the downloaded ZIP archive inside the temp directory.
const BANKNOTE_AUTHENTICATION_ZIP_FILENAME: &str = "banknote_authentication.zip";

/// The name of the only file inside the archive, holding all 1372 records.
const BANKNOTE_AUTHENTICATION_SOURCE_FILENAME: &str = "data_banknote_authentication.txt";

/// The name of the final cached Banknote Authentication dataset file.
const BANKNOTE_AUTHENTICATION_FILENAME: &str = "banknote_authentication.csv";

/// The SHA256 hash of the Banknote Authentication dataset file
/// (`data_banknote_authentication.txt`).
const BANKNOTE_AUTHENTICATION_SHA256: &str =
    "d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9";

/// The name of the dataset.
const BANKNOTE_AUTHENTICATION_DATASET_NAME: &str = "banknote_authentication";

/// Number of samples.
const N_SAMPLES: usize = 1372;

/// The number of numeric features per sample.
const N_FEATURES: usize = 4;

/// The number of columns per CSV record (4 features + 1 label).
const N_COLUMNS: usize = N_FEATURES + 1;

/// A struct that represents the Banknote Authentication dataset with lazy
/// loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// Researchers extracted the data from images of genuine and forged
/// banknote-like specimens. They digitized the images with an industrial
/// camera normally used for print inspection. This camera produced 400×400
/// pixel grayscale images at a resolution of about 660 dpi. Researchers then
/// used a Wavelet Transform tool to extract four continuous statistics from
/// each image. These statistics are the variance, skewness, curtosis, and
/// entropy of the transformed image. They cover 1372 specimens.
///
/// # Columns
///
/// | Name       | Type      | Description                               |
/// |------------|-----------|--------------------------------------------|
/// | `variance` | `Numeric` | variance of the Wavelet-Transformed image |
/// | `skewness` | `Numeric` | skewness of the Wavelet-Transformed image |
/// | `curtosis` | `Numeric` | curtosis of the Wavelet-Transformed image |
/// | `entropy`  | `Numeric` | entropy of the image                      |
/// | `class`    | `Integer` | raw class code, `0` or `1`                |
///
/// `curtosis` keeps the source's spelling. UCI names the attribute that way.
///
/// The source designates the four statistics as the inputs
/// ([`BanknoteAuthentication::FEATURE_NAMES`]) and `class` as the label
/// ([`BanknoteAuthentication::TARGET`]).
///
/// UCI does not document which `class` code marks a genuine note and which one
/// marks a forged note. The loader keeps the code verbatim.
///
/// Missing values: none.
///
/// See more information at
/// <https://archive.ics.uci.edu/dataset/267/banknote+authentication>.
///
/// # Citation
///
/// V. Lohweg. "Banknote Authentication," UCI Machine Learning Repository,
/// \[Online\]. Available: <https://doi.org/10.24432/C55P57>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::BanknoteAuthentication;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./banknote_authentication";
///
/// let mut dataset = BanknoteAuthentication::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 1372);
/// assert_eq!(table.n_columns(), 5);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&BanknoteAuthentication::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[1372, 4]);
///
/// // Reach one column by name.
/// let class = table.column(BanknoteAuthentication::TARGET).unwrap().as_integer().unwrap();
/// assert_eq!(class.len(), 1372);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("variance") {
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
/// assert_eq!(owned.n_samples(), 1372);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 1372);
/// ```
#[derive(Debug)]
pub struct BanknoteAuthentication {
    dataset: Dataset<Table, DatasetError>,
}

impl BanknoteAuthentication {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] =
        ["variance", "skewness", "curtosis", "entropy"];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "class";

    /// Create a new BanknoteAuthentication instance without loading data.
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
    /// - `Self` - a `BanknoteAuthentication` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        BanknoteAuthentication {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Banknote Authentication dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            BANKNOTE_AUTHENTICATION_FILENAME,
            BANKNOTE_AUTHENTICATION_DATASET_NAME,
            Some(BANKNOTE_AUTHENTICATION_SHA256),
            |temp_path| {
                download_to_with_retries(
                    BANKNOTE_AUTHENTICATION_DATA_URL,
                    temp_path,
                    Some(BANKNOTE_AUTHENTICATION_ZIP_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                unzip(
                    &temp_path.join(BANKNOTE_AUTHENTICATION_ZIP_FILENAME),
                    temp_path,
                )?;
                Ok(temp_path.join(BANKNOTE_AUTHENTICATION_SOURCE_FILENAME))
            },
        )?;

        // The source is plain comma-separated with no header: every line is a
        // record of 4 numeric features followed by the class code.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features: Vec<Vec<f64>> = (0..N_FEATURES)
            .map(|_| Vec::with_capacity(N_SAMPLES))
            .collect();
        let mut labels: Vec<i64> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record = result.map_err(|e| {
                DatasetError::csv_read_error(BANKNOTE_AUTHENTICATION_DATASET_NAME, e)
            })?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            // Skip blank lines defensively (e.g. a trailing newline).
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    BANKNOTE_AUTHENTICATION_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // 4 numeric features.
            for (col, name) in Self::FEATURE_NAMES.iter().enumerate() {
                let value: f64 = record[col].trim().parse().map_err(|e| {
                    DatasetError::parse_failed(
                        BANKNOTE_AUTHENTICATION_DATASET_NAME,
                        name,
                        line_num,
                        e,
                    )
                })?;
                features[col].push(value);
            }

            // Label, kept as the raw `0`/`1` code the source records.
            let raw_label = record[N_FEATURES].trim();
            let label: u8 = raw_label.parse().map_err(|e| {
                DatasetError::parse_failed(
                    BANKNOTE_AUTHENTICATION_DATASET_NAME,
                    "class",
                    line_num,
                    e,
                )
            })?;
            if label > 1 {
                return Err(DatasetError::invalid_value(
                    BANKNOTE_AUTHENTICATION_DATASET_NAME,
                    "class",
                    raw_label,
                    line_num,
                ));
            }
            labels.push(i64::from(label));
        }

        let mut columns: Vec<Column> = Vec::with_capacity(N_COLUMNS);
        for (name, values) in Self::FEATURE_NAMES.into_iter().zip(features) {
            columns.push(Column::new(
                name,
                ColumnData::Numeric(Array1::from_vec(values)),
            ));
        }
        columns.push(Column::new(
            Self::TARGET,
            ColumnData::Integer(Array1::from_vec(labels)),
        ));

        Table::new(BANKNOTE_AUTHENTICATION_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 1372 samples and 5 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`BanknoteAuthentication::data`], this method never runs the
    /// loader. If the data has not loaded yet, it returns `None` instead of
    /// downloading and parsing it.
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
    /// changes stay in the cache. Later calls to
    /// [`BanknoteAuthentication::data`] or [`BanknoteAuthentication::get_data`]
    /// see them.
    ///
    /// Like [`BanknoteAuthentication::get_data`], this does **not** trigger
    /// loading.
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
    /// the instance, use [`BanknoteAuthentication::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 1372 samples and 5 columns.
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
    /// - `Table` - the owned table of 1372 samples and 5 columns.
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

impl_ml_dataset!(BanknoteAuthentication, "banknote_authentication");
