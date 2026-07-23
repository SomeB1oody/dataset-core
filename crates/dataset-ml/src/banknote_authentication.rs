//! Banknote Authentication dataset.
//!
//! Features extracted from images of genuine and forged banknote-like specimens.
//! The images were digitized with an industrial camera usually deployed for print
//! inspection, and a Wavelet Transform tool was used to derive four continuous
//! statistics from each image. The task is to predict the class of a specimen
//! from those four features.
//!
//! **Features (4, all numeric):** `variance`, `skewness`, `curtosis`, and
//! `entropy` of the Wavelet-Transformed image, all continuous `f64` values.
//!
//! **Target:** `class` — the raw integer code from the source, `0` or `1`
//!
//! **Samples:** 1372 total (762 of class `0`, 610 of class `1`)
//! **Application:** Binary classification / banknote authentication
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C55P57>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::{Array1, Array2};
use std::fs::File;

use csv::ReaderBuilder;

/// The URL for the Banknote Authentication dataset.
///
/// This is the UCI static package; it is a ZIP archive containing a single file,
/// `data_banknote_authentication.txt`.
///
/// # Citation
///
/// V. Lohweg. "Banknote Authentication," UCI Machine Learning Repository,
/// \[Online\]. Available: <https://doi.org/10.24432/C55P57>
const BANKNOTE_AUTHENTICATION_DATA_URL: &str =
    "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip";

/// The name the downloaded ZIP archive is saved under inside the temp directory.
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

/// The names of the four feature columns, in source order. `curtosis` keeps the
/// (misspelled) UCI attribute name so the schema matches the source exactly.
const FEATURE_NAMES: [&str; N_FEATURES] = ["variance", "skewness", "curtosis", "entropy"];

/// Type alias for the Banknote Authentication dataset: (features, labels).
type BanknoteAuthenticationData = (Array2<f64>, Array1<u8>);

/// A struct representing the Banknote Authentication dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The data were extracted from images taken of genuine and forged banknote-like
/// specimens. For digitization, an industrial camera usually deployed for print
/// inspection was used, yielding 400×400 pixel grayscale images with a resolution
/// of about 660 dpi. A Wavelet Transform tool was then used to extract four
/// continuous statistics from each image — the variance, skewness, curtosis, and
/// entropy of the transformed image — giving a compact pure-numeric feature
/// matrix over 1372 specimens.
///
/// # Feature columns
///
/// All 4 features are quantitative, stored in one `(1372, 4)` `Array2<f64>`
/// matrix. By 0-based column index:
///
/// | Column | Attribute  | Description                                          |
/// |--------|------------|------------------------------------------------------|
/// | `0`    | `variance` | variance of the Wavelet-Transformed image            |
/// | `1`    | `skewness` | skewness of the Wavelet-Transformed image            |
/// | `2`    | `curtosis` | curtosis of the Wavelet-Transformed image            |
/// | `3`    | `entropy`  | entropy of the image                                 |
///
/// `curtosis` keeps the source's spelling (UCI names the attribute that way)
/// so the schema matches the source exactly.
///
/// # Labels
///
/// - `class` (shape `(1372,)`): the `Array1<u8>` holds the raw integer code from
///   the source (`0` or `1`); UCI does not document which code corresponds to
///   genuine vs forged notes, so it is exposed verbatim.
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
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::banknote_authentication::BanknoteAuthentication;
///
/// let download_dir = "./banknote_authentication"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = BanknoteAuthentication::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
/// assert_eq!(features.shape(), &[1372, 4]);
/// assert_eq!(labels.len(), 1372);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 0.5;
///     labels[0] = 1;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable — the next access reloads from the cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[1372, 4]);
/// assert_eq!(owned_labels.len(), 1372);
///
/// // `into_data()` also returns owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[1372, 4]);
/// assert_eq!(owned_labels.len(), 1372);
/// ```
#[derive(Debug)]
pub struct BanknoteAuthentication {
    dataset: Dataset<BanknoteAuthenticationData, DatasetError>,
}

impl BanknoteAuthentication {
    /// Create a new BanknoteAuthentication instance without loading data.
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
    /// - `Self` - `BanknoteAuthentication` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        BanknoteAuthentication {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Banknote Authentication dataset.
    fn load_data(dir: &str) -> Result<BanknoteAuthenticationData, DatasetError> {
        // Prepare the dataset file: download the UCI ZIP package, extract it, and
        // surface the single `data_banknote_authentication.txt` file it contains.
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

        let mut features: Vec<f64> = Vec::with_capacity(N_SAMPLES * N_FEATURES);
        let mut labels: Vec<u8> = Vec::with_capacity(N_SAMPLES);

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
            for (col, name) in FEATURE_NAMES.iter().enumerate() {
                let value: f64 = record[col].trim().parse().map_err(|e| {
                    DatasetError::parse_failed(
                        BANKNOTE_AUTHENTICATION_DATASET_NAME,
                        name,
                        line_num,
                        e,
                    )
                })?;
                features.push(value);
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
            labels.push(label);
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(
                BANKNOTE_AUTHENTICATION_DATASET_NAME,
            ));
        }

        // Banknote Authentication has a fixed schema of 4 numeric features per sample.
        let features_array =
            Array2::from_shape_vec((n_samples, N_FEATURES), features).map_err(|e| {
                DatasetError::array_shape_error(BANKNOTE_AUTHENTICATION_DATASET_NAME, "features", e)
            })?;

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
    /// - `&Array2<f64>` - Reference to the numeric feature matrix with shape
    ///   `(1372, 4)`: the `variance`, `skewness`, `curtosis`, and `entropy` of
    ///   each Wavelet-Transformed image.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (1372 samples, 4 features)
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
    /// - `&Array1<u8>` - Reference to labels vector with shape `(1372,)` containing the raw class codes (`0` or `1`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (1372 samples)
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
    /// - `&BanknoteAuthenticationData` - reference to the cached
    ///   `(features, labels)` tuple: the feature matrix has shape `(1372, 4)` and
    ///   the label vector has shape `(1372,)` containing the raw class codes
    ///   (`0` or `1`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (1372 samples, 4 features)
    pub fn data(&self) -> Result<&BanknoteAuthenticationData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and labels as references **without** triggering loading.
    ///
    /// Unlike [`BanknoteAuthentication::data`], which loads the dataset on first
    /// call, this never runs the loader: if the data has not been loaded yet, it
    /// returns `None` instead of downloading and parsing. Use it when you only
    /// want the data if it is already cached and want to avoid paying the
    /// download/parse cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&BanknoteAuthenticationData)` - reference to the cached
    ///   `(features, labels)` tuple (feature matrix `(1372, 4)`, label vector
    ///   `(1372,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&BanknoteAuthenticationData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize features,
    /// replace label values) with no `to_owned()` clone and without removing them
    /// from the cache: the changes persist, so later
    /// [`BanknoteAuthentication::features`], [`BanknoteAuthentication::data`], or
    /// [`BanknoteAuthentication::get_data`] calls observe them.
    ///
    /// Like [`BanknoteAuthentication::get_data`], this does **not** trigger
    /// loading: it returns `None` if the dataset has not been loaded. Call a
    /// loading accessor (e.g. [`BanknoteAuthentication::data`]) first if you need
    /// to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut BanknoteAuthenticationData)` - mutable reference to the cached
    ///   `(features, labels)` tuple (feature matrix `(1372, 4)`, label vector
    ///   `(1372,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut BanknoteAuthenticationData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`BanknoteAuthentication::data`], which borrows the cached data,
    /// this moves it out and returns owned arrays directly — no `to_owned()` clone
    /// needed. The dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`BanknoteAuthentication::take_data`] instead — it takes `&mut self` and
    /// leaves the instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<u8>)` - owned feature matrix with shape
    ///   `(1372, 4)` and owned label vector with shape `(1372,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn into_data(self) -> Result<BanknoteAuthenticationData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`BanknoteAuthentication::into_data`], this returns owned arrays with
    /// no `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out, resetting the instance to its
    /// unloaded state: the next accessor call (e.g.
    /// [`BanknoteAuthentication::features`] or [`BanknoteAuthentication::data`])
    /// loads the dataset again.
    ///
    /// Use [`BanknoteAuthentication::into_data`] instead if you are done with the
    /// instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<u8>)` - owned feature matrix with shape
    ///   `(1372, 4)` and owned label vector with shape `(1372,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<BanknoteAuthenticationData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(
    BanknoteAuthentication,
    BanknoteAuthenticationData,
    "banknote_authentication"
);
