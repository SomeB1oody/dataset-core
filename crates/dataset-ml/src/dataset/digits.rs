//! Optical Recognition of Handwritten Digits dataset.
//!
//! This dataset provides the digits data for multi-class classification,
//! identical to the one bundled with scikit-learn as `load_digits`. Each sample
//! is an 8×8 image of a handwritten digit, flattened into 64 integer pixel
//! intensities in the range `0..=16`. The task is to recognize which digit
//! (`0`–`9`) the image shows.
//!
//! This reproduces scikit-learn's `load_digits` output: scikit-learn uses the
//! **test** partition (`optdigits.tes`) of the UCI archive, which holds exactly
//! 1797 samples.
//!
//! **Columns (65):** the 64 pixels of the image, flattened in row-major order,
//! then the digit.
//!
//! | Name                        | Type      | Description                        |
//! |-----------------------------|-----------|-------------------------------------|
//! | `pixel_0_0` … `pixel_0_7`   | `Numeric` | row 0 pixel intensities (`0..=16`) |
//! | `pixel_1_0` … `pixel_1_7`   | `Numeric` | row 1 pixel intensities (`0..=16`) |
//! | `pixel_2_0` … `pixel_2_7`   | `Numeric` | row 2 pixel intensities (`0..=16`) |
//! | `pixel_3_0` … `pixel_3_7`   | `Numeric` | row 3 pixel intensities (`0..=16`) |
//! | `pixel_4_0` … `pixel_4_7`   | `Numeric` | row 4 pixel intensities (`0..=16`) |
//! | `pixel_5_0` … `pixel_5_7`   | `Numeric` | row 5 pixel intensities (`0..=16`) |
//! | `pixel_6_0` … `pixel_6_7`   | `Numeric` | row 6 pixel intensities (`0..=16`) |
//! | `pixel_7_0` … `pixel_7_7`   | `Numeric` | row 7 pixel intensities (`0..=16`) |
//! | `digit`                     | `Integer` | the handwritten digit, `0`–`9`     |
//!
//! The source designates the 64 pixel columns as the inputs
//! ([`Digits::FEATURE_NAMES`](crate::Digits::FEATURE_NAMES)) and `digit` as the label ([`Digits::TARGET`](crate::Digits::TARGET)).
//!
//! **Samples:** 1797 total (roughly 180 per digit class)
//! **Application:** Multi-class classification / handwritten digit recognition
//!
//! **Missing values:** none.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C50P49>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::Array1;
use std::fs::File;

/// The URL for the Optical Recognition of Handwritten Digits dataset.
///
/// This is the UCI static package. It is a ZIP archive with several files. The
/// loader uses only the `optdigits.tes` test partition, which matches
/// scikit-learn.
///
/// # Citation
///
/// E. Alpaydin and C. Kaynak. "Optical Recognition of Handwritten Digits," UCI
/// Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C50P49>
const DIGITS_DATA_URL: &str =
    "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip";

/// The loader saves the downloaded ZIP archive under this name inside the temp
/// directory.
const DIGITS_ZIP_FILENAME: &str = "optdigits.zip";

/// The name of the file inside the archive that scikit-learn's `load_digits` uses
/// (the test partition, 1797 samples).
const DIGITS_SOURCE_FILENAME: &str = "optdigits.tes";

/// The name of the final cached Digits dataset file.
const DIGITS_FILENAME: &str = "digits.csv";

/// The SHA256 hash of the Digits dataset file (`optdigits.tes`).
const DIGITS_SHA256: &str = "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8";

/// The name of the dataset
const DIGITS_DATASET_NAME: &str = "digits";

/// The number of pixel features per sample (an 8×8 image flattened to 64 values).
const N_FEATURES: usize = 64;

/// The number of columns per CSV record (64 pixels + 1 label).
const N_COLUMNS: usize = N_FEATURES + 1;

/// A struct that represents the Digits dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The Optical Recognition of Handwritten Digits dataset contains 8×8 grayscale
/// images of handwritten digits. The source flattens each image into 64 pixel
/// intensities in the range `0..=16`. The target is the digit (`0`–`9`) that the
/// image shows.
///
/// This is the same data scikit-learn exposes through `load_digits`: it uses the
/// test partition (`optdigits.tes`) of the UCI archive, with 1797 samples.
///
/// # Columns
///
/// The 64 pixel columns hold the 8×8 image, flattened in row-major order. The
/// name of the pixel of row `r` and column `c` is `pixel_r_c`.
///
/// | Name                        | Type      | Description                        |
/// |-----------------------------|-----------|-------------------------------------|
/// | `pixel_0_0` … `pixel_0_7`   | `Numeric` | row 0 pixel intensities (`0..=16`) |
/// | `pixel_1_0` … `pixel_1_7`   | `Numeric` | row 1 pixel intensities (`0..=16`) |
/// | `pixel_2_0` … `pixel_2_7`   | `Numeric` | row 2 pixel intensities (`0..=16`) |
/// | `pixel_3_0` … `pixel_3_7`   | `Numeric` | row 3 pixel intensities (`0..=16`) |
/// | `pixel_4_0` … `pixel_4_7`   | `Numeric` | row 4 pixel intensities (`0..=16`) |
/// | `pixel_5_0` … `pixel_5_7`   | `Numeric` | row 5 pixel intensities (`0..=16`) |
/// | `pixel_6_0` … `pixel_6_7`   | `Numeric` | row 6 pixel intensities (`0..=16`) |
/// | `pixel_7_0` … `pixel_7_7`   | `Numeric` | row 7 pixel intensities (`0..=16`) |
/// | `digit`                     | `Integer` | the handwritten digit, `0`–`9`     |
///
/// The source designates the 64 pixel columns as the inputs
/// ([`Digits::FEATURE_NAMES`]) and `digit` as the label ([`Digits::TARGET`]).
///
/// Missing values: none.
///
/// See more information at
/// <https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits>
///
/// # Citation
///
/// E. Alpaydin and C. Kaynak. "Optical Recognition of Handwritten Digits," UCI
/// Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C50P49>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Digits;
///
/// let download_dir = "./digits"; // the loader creates the directory if it does not exist
///
/// let mut dataset = Digits::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 1797);
/// assert_eq!(table.n_columns(), 65);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&Digits::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[1797, 64]);
///
/// // Reach one column by name.
/// let digit = table.column(Digits::TARGET).unwrap().as_integer().unwrap();
/// assert_eq!(digit.len(), 1797);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("pixel_0_0") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 5.0;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 1797);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 1797);
/// ```
#[derive(Debug)]
pub struct Digits {
    dataset: Dataset<Table, DatasetError>,
}

impl Digits {
    /// The columns the source designates as the model inputs, in source order.
    /// The name of the pixel of row `r` and column `c` of the 8×8 image is
    /// `pixel_r_c`.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "pixel_0_0",
        "pixel_0_1",
        "pixel_0_2",
        "pixel_0_3",
        "pixel_0_4",
        "pixel_0_5",
        "pixel_0_6",
        "pixel_0_7",
        "pixel_1_0",
        "pixel_1_1",
        "pixel_1_2",
        "pixel_1_3",
        "pixel_1_4",
        "pixel_1_5",
        "pixel_1_6",
        "pixel_1_7",
        "pixel_2_0",
        "pixel_2_1",
        "pixel_2_2",
        "pixel_2_3",
        "pixel_2_4",
        "pixel_2_5",
        "pixel_2_6",
        "pixel_2_7",
        "pixel_3_0",
        "pixel_3_1",
        "pixel_3_2",
        "pixel_3_3",
        "pixel_3_4",
        "pixel_3_5",
        "pixel_3_6",
        "pixel_3_7",
        "pixel_4_0",
        "pixel_4_1",
        "pixel_4_2",
        "pixel_4_3",
        "pixel_4_4",
        "pixel_4_5",
        "pixel_4_6",
        "pixel_4_7",
        "pixel_5_0",
        "pixel_5_1",
        "pixel_5_2",
        "pixel_5_3",
        "pixel_5_4",
        "pixel_5_5",
        "pixel_5_6",
        "pixel_5_7",
        "pixel_6_0",
        "pixel_6_1",
        "pixel_6_2",
        "pixel_6_3",
        "pixel_6_4",
        "pixel_6_5",
        "pixel_6_6",
        "pixel_6_7",
        "pixel_7_0",
        "pixel_7_1",
        "pixel_7_2",
        "pixel_7_3",
        "pixel_7_4",
        "pixel_7_5",
        "pixel_7_6",
        "pixel_7_7",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "digit";

    /// Create a new Digits instance without loading data.
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
    /// - `Self` - a `Digits` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Digits {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Digits dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        // Prepare the dataset file: download the UCI ZIP package, extract it, and
        // return the `optdigits.tes` test partition (which scikit-learn uses).
        let file_path = acquire_dataset(
            dir,
            DIGITS_FILENAME,
            DIGITS_DATASET_NAME,
            Some(DIGITS_SHA256),
            |temp_path| {
                download_to_with_retries(
                    DIGITS_DATA_URL,
                    temp_path,
                    Some(DIGITS_ZIP_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                unzip(&temp_path.join(DIGITS_ZIP_FILENAME), temp_path)?;
                Ok(temp_path.join(DIGITS_SOURCE_FILENAME))
            },
        )?;

        // `optdigits.tes` is a headerless comma-separated file: every line is a
        // record of 64 pixel values followed by the digit label.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features: Vec<Vec<f64>> = vec![Vec::new(); N_FEATURES];
        let mut labels: Vec<i64> = Vec::new();

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(DIGITS_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    DIGITS_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            for (col, field) in record.iter().take(N_FEATURES).enumerate() {
                let value: f64 = field.trim().parse().map_err(|e| {
                    DatasetError::parse_failed(
                        DIGITS_DATASET_NAME,
                        &format!("pixel_{}_{}", col / 8, col % 8),
                        line_num,
                        e,
                    )
                })?;
                features[col].push(value);
            }

            let raw_label = record[N_FEATURES].trim();
            let label: u8 = raw_label.parse().map_err(|e| {
                DatasetError::parse_failed(DIGITS_DATASET_NAME, "digit", line_num, e)
            })?;
            if label > 9 {
                return Err(DatasetError::invalid_value(
                    DIGITS_DATASET_NAME,
                    "digit",
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

        Table::new(DIGITS_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 1797 samples and 65
    ///   columns.
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
    /// Unlike [`Digits::data`], this method never runs the loader. If the data
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
    /// changes stay in the cache. Later calls to [`Digits::data`] or
    /// [`Digits::get_data`] see them.
    ///
    /// Like [`Digits::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`Digits::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 1797 samples and 65 columns.
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
    /// - `Table` - the owned table of 1797 samples and 65 columns.
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

impl_ml_dataset!(Digits, "digits");
