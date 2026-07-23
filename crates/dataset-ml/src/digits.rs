//! Optical Recognition of Handwritten Digits dataset.
//!
//! The classic digits dataset for multi-class classification, identical to the
//! one bundled with scikit-learn as `load_digits`. Each sample is an 8×8 image of
//! a handwritten digit, flattened into 64 integer pixel intensities in the range
//! `0..=16`. The task is to recognise which digit (`0`–`9`) the image shows.
//!
//! This reproduces scikit-learn's `load_digits` output: scikit-learn uses the
//! **test** partition (`optdigits.tes`) of the UCI archive, which holds exactly
//! 1797 samples.
//!
//! **Features (64):** `pixel_0_0` … `pixel_7_7` - the 8×8 image flattened in
//! row-major order, each an integer pixel intensity in `0..=16` (stored as `f64`).
//!
//! **Target:** `digit` - the handwritten digit, one of `0`–`9` (stored as `u8`).
//!
//! **Samples:** 1797 total (roughly 180 per digit class)
//! **Application:** Multi-class classification / handwritten digit recognition
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C50P49>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::{Array1, Array2};
use std::fs::File;

/// The URL for the Optical Recognition of Handwritten Digits dataset.
///
/// This is the UCI static package; it is a ZIP archive containing several files,
/// of which only the `optdigits.tes` test partition is used (matching scikit-learn).
///
/// # Citation
///
/// E. Alpaydin and C. Kaynak. "Optical Recognition of Handwritten Digits," UCI
/// Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C50P49>
const DIGITS_DATA_URL: &str =
    "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip";

/// The name the downloaded ZIP archive is saved under inside the temp directory.
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

/// Type alias for the Digits dataset: (features, labels).
type DigitsData = (Array2<f64>, Array1<u8>);

/// A struct representing the Digits dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The Optical Recognition of Handwritten Digits dataset contains 8×8 grayscale
/// images of handwritten digits. Each image is flattened into 64 pixel intensities
/// in the range `0..=16`, and the target is the digit (`0`–`9`) the image depicts.
///
/// This is the same data scikit-learn exposes through `load_digits`: it uses the
/// test partition (`optdigits.tes`) of the UCI archive, with 1797 samples.
///
/// # Feature columns
///
/// The 64 features are the pixels of an 8×8 grayscale image, flattened in
/// row-major order. Each pixel holds an integer intensity in `0..=16` stored as
/// `f64`. By 0-based column index:
///
/// | Columns   | Attributes                                  | Unit                 |
/// |-----------|---------------------------------------------|----------------------|
/// | `0..=7`   | row 0 pixels (`pixel_0_0` .. `pixel_0_7`)   | intensity (`0..=16`) |
/// | `8..=15`  | row 1 pixels (`pixel_1_0` .. `pixel_1_7`)   | intensity (`0..=16`) |
/// | `16..=23` | row 2 pixels (`pixel_2_0` .. `pixel_2_7`)   | intensity (`0..=16`) |
/// | `24..=31` | row 3 pixels (`pixel_3_0` .. `pixel_3_7`)   | intensity (`0..=16`) |
/// | `32..=39` | row 4 pixels (`pixel_4_0` .. `pixel_4_7`)   | intensity (`0..=16`) |
/// | `40..=47` | row 5 pixels (`pixel_5_0` .. `pixel_5_7`)   | intensity (`0..=16`) |
/// | `48..=55` | row 6 pixels (`pixel_6_0` .. `pixel_6_7`)   | intensity (`0..=16`) |
/// | `56..=63` | row 7 pixels (`pixel_7_0` .. `pixel_7_7`)   | intensity (`0..=16`) |
///
/// # Labels
///
/// - digit (in `u8`): `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
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
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::digits::Digits;
///
/// let download_dir = "./digits"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = Digits::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
/// assert_eq!(features.shape(), &[1797, 64]);
/// assert_eq!(labels.len(), 1797);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 5.0;
///     labels[0] = 7;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable — the next access reloads from the cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[1797, 64]);
/// assert_eq!(owned_labels.len(), 1797);
///
/// // `into_data()` also returns owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[1797, 64]);
/// assert_eq!(owned_labels.len(), 1797);
/// ```
#[derive(Debug)]
pub struct Digits {
    dataset: Dataset<DigitsData, DatasetError>,
}

impl Digits {
    /// Create a new Digits instance without loading data.
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
    /// - `Self` - `Digits` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Digits {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Digits dataset.
    fn load_data(dir: &str) -> Result<DigitsData, DatasetError> {
        // Prepare the dataset file: download the UCI ZIP package, extract it, and
        // surface the `optdigits.tes` test partition (which scikit-learn uses).
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

        let mut features = Vec::new();
        let mut labels = Vec::new();

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
                features.push(value);
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
            labels.push(label);
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(DIGITS_DATASET_NAME));
        }

        // Digits has a fixed schema of 64 numeric pixel features per sample.
        let features_array = Array2::from_shape_vec((n_samples, N_FEATURES), features)
            .map_err(|e| DatasetError::array_shape_error(DIGITS_DATASET_NAME, "features", e))?;
        let labels_array = Array1::from_vec(labels);

        Ok((features_array, labels_array))
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(1797, 64)`
    ///   containing the 64 pixel intensities (`pixel_0_0` … `pixel_7_7`, each in
    ///   `0..=16`) of each flattened 8×8 image.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (1797 samples, 64 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the labels vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<u8>` - Reference to labels vector with shape `(1797,)` containing the digit classes (`0`–`9`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (1797 samples)
    pub fn labels(&self) -> Result<&Array1<u8>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&DigitsData` - reference to the cached `(features, labels)` tuple: the
    ///   feature matrix has shape `(1797, 64)` and the label vector has shape
    ///   `(1797,)` containing the digit classes (`0`–`9`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (1797 samples, 64 features)
    pub fn data(&self) -> Result<&DigitsData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and labels as references **without** triggering loading.
    ///
    /// Unlike [`Digits::data`], which loads the dataset on first call, this never
    /// runs the loader: if the data has not been loaded yet, it returns `None`
    /// instead of downloading and parsing. Use it when you only want the data if
    /// it is already cached and want to avoid paying the download/parse cost
    /// otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&DigitsData)` - reference to the cached `(features, labels)` tuple
    ///   (feature matrix `(1797, 64)`, label vector `(1797,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&DigitsData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize features,
    /// replace label values) with no `to_owned()` clone and without removing them
    /// from the cache: the changes persist, so later [`Digits::features`],
    /// [`Digits::data`], or [`Digits::get_data`] calls observe them.
    ///
    /// Like [`Digits::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call a loading accessor (e.g.
    /// [`Digits::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut DigitsData)` - mutable reference to the cached
    ///   `(features, labels)` tuple (feature matrix `(1797, 64)`, label vector
    ///   `(1797,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut DigitsData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`Digits::data`], which borrows the cached data, this moves it out and
    /// returns owned arrays directly — no `to_owned()` clone needed. The dataset is
    /// loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use [`Digits::take_data`]
    /// instead — it takes `&mut self` and leaves the instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<u8>)` - owned feature matrix with shape
    ///   `(1797, 64)` and owned label vector with shape `(1797,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn into_data(self) -> Result<DigitsData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`Digits::into_data`], this returns owned arrays with no `to_owned()`
    /// clone. But instead of consuming the instance, it takes `&mut self` and moves
    /// the cached data out, resetting the instance to its unloaded state: the next
    /// accessor call (e.g. [`Digits::features`] or [`Digits::data`]) loads the
    /// dataset again.
    ///
    /// Use [`Digits::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<u8>)` - owned feature matrix with shape
    ///   `(1797, 64)` and owned label vector with shape `(1797,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<DigitsData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(Digits, DigitsData, "digits");
