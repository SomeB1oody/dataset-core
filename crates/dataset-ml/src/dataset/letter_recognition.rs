//! Letter Recognition dataset.
//!
//! This dataset has black-and-white rectangular pixel displays of the 26 capital
//! letters of the English alphabet. It uses 20 different fonts, and random
//! distortion of each produces 20,000 unique stimuli. The dataset reduces each
//! stimulus to 16 primitive numerical attributes: statistical moments and edge
//! counts. It scales each attribute to an integer value in the range `0..=15`.
//! The task is to identify which capital letter (`A`–`Z`) a display shows.
//!
//! **Columns (17):** the source puts the letter first, then the 16 attributes.
//!
//! | Name    | Type      | Description                          |
//! |---------|-----------|----------------------------------------|
//! | `lettr` | `String`  | the capital letter, `A` through `Z`  |
//! | `x-box` | `Numeric` | horizontal position of box (`0..=15`)|
//! | `y-box` | `Numeric` | vertical position of box (`0..=15`)  |
//! | `width` | `Numeric` | width of box (`0..=15`)              |
//! | `high`  | `Numeric` | height of box (`0..=15`)             |
//! | `onpix` | `Numeric` | total number of "on" pixels (`0..=15`)|
//! | `x-bar` | `Numeric` | mean x of "on" pixels in box (`0..=15`)|
//! | `y-bar` | `Numeric` | mean y of "on" pixels in box (`0..=15`)|
//! | `x2bar` | `Numeric` | mean x variance (`0..=15`)           |
//! | `y2bar` | `Numeric` | mean y variance (`0..=15`)           |
//! | `xybar` | `Numeric` | mean x y correlation (`0..=15`)      |
//! | `x2ybr` | `Numeric` | mean of x * x * y (`0..=15`)         |
//! | `xy2br` | `Numeric` | mean of x * y * y (`0..=15`)         |
//! | `x-ege` | `Numeric` | mean edge count left to right (`0..=15`)|
//! | `xegvy` | `Numeric` | correlation of `x-ege` with y (`0..=15`)|
//! | `y-ege` | `Numeric` | mean edge count bottom to top (`0..=15`)|
//! | `yegvx` | `Numeric` | correlation of `y-ege` with x (`0..=15`)|
//!
//! The source designates the 16 attributes as the inputs
//! ([`LetterRecognition::FEATURE_NAMES`](crate::LetterRecognition::FEATURE_NAMES)) and `lettr` as the label
//! ([`LetterRecognition::TARGET`](crate::LetterRecognition::TARGET)). The `lettr` column holds one capital letter
//! per sample, as a one-character string.
//!
//! **Samples:** 20,000 total (about 734–813 per letter class)
//! **Application:** Multi-class classification / character recognition
//!
//! **Missing values:** none.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5ZP40>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::Array1;
use std::fs::File;

/// The URL for the Letter Recognition dataset.
///
/// This is the UCI static package. It is a ZIP archive with several files. The
/// loader uses only the `letter-recognition.data` file.
///
/// # Citation
///
/// D. Slate. "Letter Recognition," UCI Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C5ZP40>
const LETTER_RECOGNITION_DATA_URL: &str =
    "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip";

/// The name of the downloaded ZIP archive inside the temp directory.
const LETTER_RECOGNITION_ZIP_FILENAME: &str = "letter_recognition.zip";

/// The name of the file inside the archive that holds the 20,000 samples.
const LETTER_RECOGNITION_SOURCE_FILENAME: &str = "letter-recognition.data";

/// The name of the final cached Letter Recognition dataset file.
const LETTER_RECOGNITION_FILENAME: &str = "letter_recognition.csv";

/// The SHA256 hash of the Letter Recognition dataset file (`letter-recognition.data`).
const LETTER_RECOGNITION_SHA256: &str =
    "2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2";

/// The name of the dataset.
const LETTER_RECOGNITION_DATASET_NAME: &str = "letter_recognition";

/// Number of samples.
const N_SAMPLES: usize = 20_000;

/// The number of numeric features per sample.
const N_FEATURES: usize = 16;

/// The number of columns per CSV record (1 label + 16 features).
const N_COLUMNS: usize = N_FEATURES + 1;

/// Source column index of the label (`lettr`). The label is the **first** column.
const LABEL_COLUMN: usize = 0;

/// A struct that represents the Letter Recognition dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The goal is to identify each of many black-and-white rectangular pixel
/// displays as one of the 26 capital letters in the English alphabet. The
/// character images use 20 different fonts. Random distortion of each letter
/// within these fonts produces a file of 20,000 unique stimuli. A conversion
/// step turns each stimulus into 16 primitive numerical attributes (statistical
/// moments and edge counts). It scales the attributes to fit a range of integer
/// values from `0` through `15`.
///
/// # Columns
///
/// The source puts the letter first, then the 16 attributes.
///
/// | Name    | Type      | Description                          |
/// |---------|-----------|----------------------------------------|
/// | `lettr` | `String`  | the capital letter, `A` through `Z`  |
/// | `x-box` | `Numeric` | horizontal position of box (`0..=15`)|
/// | `y-box` | `Numeric` | vertical position of box (`0..=15`)  |
/// | `width` | `Numeric` | width of box (`0..=15`)              |
/// | `high`  | `Numeric` | height of box (`0..=15`)             |
/// | `onpix` | `Numeric` | total number of "on" pixels (`0..=15`)|
/// | `x-bar` | `Numeric` | mean x of "on" pixels in box (`0..=15`)|
/// | `y-bar` | `Numeric` | mean y of "on" pixels in box (`0..=15`)|
/// | `x2bar` | `Numeric` | mean x variance (`0..=15`)           |
/// | `y2bar` | `Numeric` | mean y variance (`0..=15`)           |
/// | `xybar` | `Numeric` | mean x y correlation (`0..=15`)      |
/// | `x2ybr` | `Numeric` | mean of x * x * y (`0..=15`)         |
/// | `xy2br` | `Numeric` | mean of x * y * y (`0..=15`)         |
/// | `x-ege` | `Numeric` | mean edge count left to right (`0..=15`)|
/// | `xegvy` | `Numeric` | correlation of `x-ege` with y (`0..=15`)|
/// | `y-ege` | `Numeric` | mean edge count bottom to top (`0..=15`)|
/// | `yegvx` | `Numeric` | correlation of `y-ege` with x (`0..=15`)|
///
/// The source designates the 16 attributes as the inputs
/// ([`LetterRecognition::FEATURE_NAMES`]) and `lettr` as the label
/// ([`LetterRecognition::TARGET`]).
///
/// The `lettr` column holds one capital letter per sample, as a one-character
/// string. To get a class index, use `(letter.as_bytes()[0] - b'A') as usize`.
///
/// Missing values: none.
///
/// See more information at
/// <https://archive.ics.uci.edu/dataset/59/letter+recognition>
///
/// # Citation
///
/// D. Slate. "Letter Recognition," UCI Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C5ZP40>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::LetterRecognition;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./letter_recognition";
///
/// let mut dataset = LetterRecognition::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 20000);
/// assert_eq!(table.n_columns(), 17);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&LetterRecognition::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[20000, 16]);
///
/// // Reach one column by name.
/// let lettr = table.column(LetterRecognition::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(lettr.len(), 20000);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("x-box") {
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
/// assert_eq!(owned.n_samples(), 20000);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 20000);
/// ```
#[derive(Debug)]
pub struct LetterRecognition {
    dataset: Dataset<Table, DatasetError>,
}

impl LetterRecognition {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
        "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "lettr";

    /// Create a new LetterRecognition instance without loading data.
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
    /// - `Self` - a `LetterRecognition` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        LetterRecognition {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Letter Recognition dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        // The closure downloads the UCI ZIP package, extracts it, and uses
        // `letter-recognition.data`, the only partition needed from the archive.
        let file_path = acquire_dataset(
            dir,
            LETTER_RECOGNITION_FILENAME,
            LETTER_RECOGNITION_DATASET_NAME,
            Some(LETTER_RECOGNITION_SHA256),
            |temp_path| {
                download_to_with_retries(
                    LETTER_RECOGNITION_DATA_URL,
                    temp_path,
                    Some(LETTER_RECOGNITION_ZIP_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                unzip(&temp_path.join(LETTER_RECOGNITION_ZIP_FILENAME), temp_path)?;
                Ok(temp_path.join(LETTER_RECOGNITION_SOURCE_FILENAME))
            },
        )?;

        // `letter-recognition.data` is a headerless comma-separated file: every line
        // is a record of the letter label followed by its 16 integer attributes.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features: Vec<Vec<f64>> = (0..N_FEATURES)
            .map(|_| Vec::with_capacity(N_SAMPLES))
            .collect();
        let mut labels: Vec<String> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record = result
                .map_err(|e| DatasetError::csv_read_error(LETTER_RECOGNITION_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    LETTER_RECOGNITION_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // The label is the capital letter itself. The loader keeps it verbatim.
            // It must be exactly one ASCII character in `A..=Z`.
            let raw_label = record[LABEL_COLUMN].trim();
            let mut label_chars = raw_label.chars();
            let label = match (label_chars.next(), label_chars.next()) {
                (Some(c), None) if c.is_ascii_uppercase() => c,
                _ => {
                    return Err(DatasetError::invalid_value(
                        LETTER_RECOGNITION_DATASET_NAME,
                        "letter",
                        raw_label,
                        line_num,
                    ));
                }
            };
            labels.push(label.to_string());

            // The 16 numeric attributes follow the leading label column.
            for (col, field) in record.iter().skip(LABEL_COLUMN + 1).enumerate() {
                let value: f64 = field.trim().parse().map_err(|e| {
                    DatasetError::parse_failed(
                        LETTER_RECOGNITION_DATASET_NAME,
                        Self::FEATURE_NAMES[col],
                        line_num,
                        e,
                    )
                })?;
                features[col].push(value);
            }
        }

        let mut columns: Vec<Column> = Vec::with_capacity(N_COLUMNS);
        columns.push(Column::new(
            Self::TARGET,
            ColumnData::String(Array1::from_vec(labels)),
        ));
        for (name, values) in Self::FEATURE_NAMES.into_iter().zip(features) {
            columns.push(Column::new(
                name,
                ColumnData::Numeric(Array1::from_vec(values)),
            ));
        }

        Table::new(LETTER_RECOGNITION_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 20,000 samples and 17
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
    /// Unlike [`LetterRecognition::data`], this method never runs the loader. If
    /// the data has not loaded yet, it returns `None` instead of downloading and
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
    /// changes stay in the cache. Later calls to [`LetterRecognition::data`] or
    /// [`LetterRecognition::get_data`] see them.
    ///
    /// Like [`LetterRecognition::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`LetterRecognition::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 20,000 samples and 17 columns.
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
    /// - `Table` - the owned table of 20,000 samples and 17 columns.
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

impl_ml_dataset!(LetterRecognition, "letter_recognition");
