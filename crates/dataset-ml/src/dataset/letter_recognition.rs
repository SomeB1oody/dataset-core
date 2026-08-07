//! Letter Recognition dataset.
//!
//! This dataset has black-and-white rectangular pixel displays of the 26 capital
//! letters of the English alphabet. It uses 20 different fonts, and random
//! distortion of each produces 20,000 unique stimuli. The dataset reduces each
//! stimulus to 16 primitive numerical attributes: statistical moments and edge
//! counts. It scales each attribute to an integer value in the range `0..=15`.
//! The task is to identify which capital letter (`A`–`Z`) a display shows.
//!
//! **Features (16, all numeric):** `x-box`, `y-box`, `width`, `high`, `onpix`,
//! `x-bar`, `y-bar`, `x2bar`, `y2bar`, `xybar`, `x2ybr`, `xy2br`, `x-ege`,
//! `xegvy`, `y-ege`, `yegvx`. Each is an integer in `0..=15` (stored as `f64`).
//!
//! **Target:** `lettr` - the capital letter, one of `A`–`Z` (stored as `char`).
//!
//! **Samples:** 20,000 total (about 734–813 per letter class)
//! **Application:** Multi-class classification / character recognition
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5ZP40>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::{Array1, Array2};
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

/// The names of the 16 numeric features, in source (and output) order. They follow
/// the leading `lettr` label column.
const FEATURE_NAMES: [&str; N_FEATURES] = [
    "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
    "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx",
];

/// Type alias for the Letter Recognition dataset: (features, labels).
type LetterRecognitionData = (Array2<f64>, Array1<char>);

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
/// # Feature columns
///
/// All 16 features are quantitative. The loader stores them in one
/// `(20000, 16)` `Array2<f64>` matrix. Each value is an integer in `0..=15`
/// (stored as `f64`). By 0-based column index:
///
/// | Column | Attribute | Description                       |
/// |--------|-----------|-----------------------------------|
/// | `0`    | `x-box`   | horizontal position of box        |
/// | `1`    | `y-box`   | vertical position of box          |
/// | `2`    | `width`   | width of box                      |
/// | `3`    | `high`    | height of box                     |
/// | `4`    | `onpix`   | total number of "on" pixels       |
/// | `5`    | `x-bar`   | mean x of "on" pixels in box      |
/// | `6`    | `y-bar`   | mean y of "on" pixels in box      |
/// | `7`    | `x2bar`   | mean x variance                   |
/// | `8`    | `y2bar`   | mean y variance                   |
/// | `9`    | `xybar`   | mean x y correlation              |
/// | `10`   | `x2ybr`   | mean of x * x * y                 |
/// | `11`   | `xy2br`   | mean of x * y * y                 |
/// | `12`   | `x-ege`   | mean edge count left to right     |
/// | `13`   | `xegvy`   | correlation of `x-ege` with y     |
/// | `14`   | `y-ege`   | mean edge count bottom to top     |
/// | `15`   | `yegvx`   | correlation of `y-ege` with x     |
///
/// # Labels
///
/// - `lettr` (shape `(20000,)`): the capital letter itself, one of `A`–`Z`.
///
/// This is the crate's first `Array1<char>` target. A class that is exactly one
/// letter is naturally a `char`, so the loader stores the letter verbatim. There
/// is no lookup table and no encoding step to undo. `labels[i]` *is* the answer,
/// and comparisons like `labels[i] == 'Q'` read as the task does. If you need an
/// index, encoding one is a one-liner: `(c as u8 - b'A') as usize`.
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
/// let download_dir = "./letter_recognition"; // the loader creates the directory if it does not exist
///
/// let mut dataset = LetterRecognition::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap();
/// assert_eq!(features.shape(), &[20000, 16]);
/// assert_eq!(labels.len(), 20000);
///
/// // `get_data()` borrows the cached arrays without a reload. `get_data_mut()`
/// // edits the arrays in place. This needs no clone and no reload. The change
/// // stays cached. If you only need to change values, prefer this method over
/// // `.to_owned()`.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 5.0;
///     labels[0] = 'A';
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out with no `to_owned()` clone. This
/// // leaves the instance reusable. The next access reloads the data from the
/// // cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[20000, 16]);
/// assert_eq!(owned_labels.len(), 20000);
///
/// // `into_data()` also returns the owned arrays with no clone, but it
/// // consumes the instance. If you are done with the dataset, use it.
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[20000, 16]);
/// assert_eq!(owned_labels.len(), 20000);
/// ```
#[derive(Debug)]
pub struct LetterRecognition {
    dataset: Dataset<LetterRecognitionData, DatasetError>,
}

impl LetterRecognition {
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
    fn load_data(dir: &str) -> Result<LetterRecognitionData, DatasetError> {
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

        let mut features: Vec<f64> = Vec::with_capacity(N_SAMPLES * N_FEATURES);
        let mut labels: Vec<char> = Vec::with_capacity(N_SAMPLES);

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
            labels.push(label);

            // The 16 numeric attributes follow the leading label column.
            for (col, field) in record.iter().skip(LABEL_COLUMN + 1).enumerate() {
                let value: f64 = field.trim().parse().map_err(|e| {
                    DatasetError::parse_failed(
                        LETTER_RECOGNITION_DATASET_NAME,
                        FEATURE_NAMES[col],
                        line_num,
                        e,
                    )
                })?;
                features.push(value);
            }
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(LETTER_RECOGNITION_DATASET_NAME));
        }

        // Letter Recognition has a fixed schema of 16 numeric features per sample.
        let features_array =
            Array2::from_shape_vec((n_samples, N_FEATURES), features).map_err(|e| {
                DatasetError::array_shape_error(LETTER_RECOGNITION_DATASET_NAME, "features", e)
            })?;
        let labels_array = Array1::from_vec(labels);

        Ok((features_array, labels_array))
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(20000, 16)`. It
    ///   contains the 16 primitive attributes (`x-box` … `yegvx`, each an integer
    ///   in `0..=15`) extracted from each letter image.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match the expected dimensions (20,000 samples, 16 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the label vector.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array1<char>` - Reference to label vector with shape `(20000,)`,
    ///   containing the letter classes (`A`–`Z`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match the expected dimensions (20,000 samples)
    pub fn labels(&self) -> Result<&Array1<char>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and labels as references.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&LetterRecognitionData` - reference to the cached `(features, labels)`
    ///   tuple: the feature matrix has shape `(20000, 16)` and the label vector has
    ///   shape `(20000,)` containing the letter classes (`A`–`Z`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match the expected dimensions (20,000 samples, 16 features)
    pub fn data(&self) -> Result<&LetterRecognitionData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and labels as references **without** triggering loading.
    ///
    /// Unlike [`LetterRecognition::data`], this method never runs the loader. If
    /// the dataset has not loaded yet, it returns `None` instead of downloading
    /// and parsing the data. If you want the data only when it is already cached,
    /// use this method. This choice avoids the download and parse cost.
    ///
    /// # Returns
    ///
    /// - `Some(&LetterRecognitionData)` - reference to the cached `(features, labels)`
    ///   tuple (feature matrix `(20000, 16)`, label vector `(20000,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&LetterRecognitionData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This lets you change the cached arrays directly (for example, normalize
    /// features, or replace label values) with no `to_owned()` clone. The changes
    /// stay in the cache: later calls to [`LetterRecognition::features`],
    /// [`LetterRecognition::data`], or [`LetterRecognition::get_data`] see them.
    ///
    /// Like [`LetterRecognition::get_data`], this method does **not** trigger
    /// loading. If the dataset has not loaded, it returns `None`. If you need to
    /// make sure the data is present, call a loading accessor first (for example,
    /// [`LetterRecognition::data`]).
    ///
    /// # Returns
    ///
    /// - `Some(&mut LetterRecognitionData)` - a mutable reference to the cached
    ///   `(features, labels)` tuple (feature matrix `(20000, 16)`, label vector
    ///   `(20000,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut LetterRecognitionData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`LetterRecognition::data`], which borrows the cached data, this
    /// method moves the data out and returns owned arrays. It needs no
    /// `to_owned()` clone. If it has not loaded yet, the dataset loads on the
    /// first access.
    ///
    /// This **consumes** `self`, so you cannot use the instance afterward. If you
    /// want owned data but need to keep using the instance, use
    /// [`LetterRecognition::take_data`] instead. It takes `&mut self` and leaves
    /// the instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<char>)` - an owned feature matrix with shape
    ///   `(20000, 16)` and an owned label vector with shape `(20000,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn into_data(self) -> Result<LetterRecognitionData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and labels out of the dataset. This leaves the
    /// instance reusable.
    ///
    /// Like [`LetterRecognition::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out. This resets the instance to its
    /// unloaded state. The next accessor call (for example,
    /// [`LetterRecognition::features`] or [`LetterRecognition::data`]) loads the
    /// dataset again.
    ///
    /// If you are done with the instance, use [`LetterRecognition::into_data`]
    /// instead.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<char>)` - an owned feature matrix with shape
    ///   `(20000, 16)` and an owned label vector with shape `(20000,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<LetterRecognitionData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(
    LetterRecognition,
    LetterRecognitionData,
    "letter_recognition"
);
