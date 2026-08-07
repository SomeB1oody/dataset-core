//! Spambase dataset.
//!
//! Researchers gathered this collection of 4,601 e-mails at Hewlett-Packard
//! Labs in 1999. The dataset summarizes each e-mail with 57 hand-crafted
//! frequency statistics, not its raw text. The spam came from a postmaster
//! and from individuals who had filed spam. The non-spam came from filed
//! work and personal e-mail. The task is to predict whether an e-mail is
//! spam from those statistics.
//!
//! **Features (57, all numeric):** 48 `word_freq_WORD` percentages, 6
//! `char_freq_CHAR` percentages, and 3 capital-run-length statistics (average,
//! longest, total). All are non-negative. The frequency columns lie in `0..=100`.
//!
//! **Target:** `class` - one of `ham` or `spam`
//!
//! **Samples:** 4,601 total (2,788 ham, 1,813 spam)
//! **Application:** Binary classification / spam detection
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C53G6X>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::{Array1, Array2};
use std::fs::File;

use csv::ReaderBuilder;

/// The URL for the Spambase dataset.
///
/// This is the UCI static package. It is a ZIP archive that contains
/// `spambase.DOCUMENTATION`, `spambase.data`, and `spambase.names`. The loader
/// uses only the `spambase.data` file.
///
/// # Citation
///
/// M. Hopkins, E. Reeber, G. Forman, and J. Suermondt. "Spambase," UCI Machine
/// Learning Repository, \[Online\]. Available: <https://doi.org/10.24432/C53G6X>
const SPAMBASE_DATA_URL: &str = "https://archive.ics.uci.edu/static/public/94/spambase.zip";

/// The filename for the downloaded ZIP archive inside the temp directory.
const SPAMBASE_ZIP_FILENAME: &str = "spambase.zip";

/// The name of the file inside the archive that holds the records.
const SPAMBASE_SOURCE_FILENAME: &str = "spambase.data";

/// The name of the final cached Spambase dataset file.
const SPAMBASE_FILENAME: &str = "spambase.csv";

/// The SHA256 hash of the Spambase dataset file (`spambase.data`).
const SPAMBASE_SHA256: &str = "b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12";

/// The name of the dataset.
const SPAMBASE_DATASET_NAME: &str = "spambase";

/// Number of samples.
const N_SAMPLES: usize = 4601;

/// The number of numeric features per sample (48 word frequencies, 6 char
/// frequencies, and 3 capital-run-length statistics).
const N_FEATURES: usize = 57;

/// The number of columns per CSV record (57 features and 1 label).
const N_COLUMNS: usize = N_FEATURES + 1;

/// Source column index of the label (`class`). The label is the **last** column.
const LABEL_COLUMN: usize = N_FEATURES;

/// Type alias for the Spambase dataset: (features, labels).
type SpambaseData = (Array2<f64>, Array1<&'static str>);

/// A struct that represents the Spambase dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// Mark Hopkins, Erik Reeber, George Forman, and Jaap Suermondt generated the
/// Spambase collection at Hewlett-Packard Labs in June–July 1999. The "spam"
/// concept is diverse: advertisements for products or web sites, make-money-fast
/// schemes, chain letters, and pornography. The spam e-mails came from the lab's
/// postmaster and from individuals who had filed spam. The non-spam e-mails came
/// from filed work and personal e-mail. This is why the word `george` and the
/// area code `650` are strong non-spam indicators here. They are useful for a
/// personalized filter, but a general purpose filter would need to blind them.
///
/// The dataset reduces each e-mail to 57 continuous statistics: how often
/// selected words and characters occur, and how long its runs of capital letters
/// are.
///
/// # Feature columns
///
/// All 57 features are numeric. They form one `(4601, 57)` `Array2<f64>`
/// matrix. By 0-based column index:
///
/// | Columns   | Attributes                            | Unit                              |
/// |-----------|---------------------------------------|-----------------------------------|
/// | `0..=47`  | `word_freq_WORD` (48 words, below)    | percentage of words (`0..=100`)   |
/// | `48..=53` | `char_freq_CHAR` (6 chars, below)     | percentage of chars (`0..=100`)   |
/// | `54`      | `capital_run_length_average`          | average capital-run length        |
/// | `55`      | `capital_run_length_longest`          | longest capital-run length        |
/// | `56`      | `capital_run_length_total`            | total number of capital letters   |
///
/// A `word_freq_WORD` column is `100 * (times WORD appears) / (total words)`. A
/// `char_freq_CHAR` column is `100 * (occurrences of CHAR) / (total characters)`.
/// A "word" is any string of alphanumeric characters bounded by non-alphanumeric
/// characters or the end of the string. The capital-run-length columns measure
/// uninterrupted sequences of capital letters. They report the average, the
/// maximum, and the sum, that is, the total number of capital letters in the
/// e-mail.
///
/// The 48 words of columns `0..=47`, in order:
///
/// `make`, `address`, `all`, `3d`, `our`, `over`, `remove`, `internet`, `order`,
/// `mail`, `receive`, `will`, `people`, `report`, `addresses`, `free`,
/// `business`, `email`, `you`, `credit`, `your`, `font`, `000`, `money`, `hp`,
/// `hpl`, `george`, `650`, `lab`, `labs`, `telnet`, `857`, `data`, `415`, `85`,
/// `technology`, `1999`, `parts`, `pm`, `direct`, `cs`, `meeting`, `original`,
/// `project`, `re`, `edu`, `table`, `conference`.
///
/// The 6 characters of columns `48..=53`, in order: `;`, `(`, `[`, `!`, `$`, `#`.
///
/// # Labels
///
/// - `class` (shape `(4601,)`): the `Array1<&'static str>` maps the source's
///   nominal codes to readable names: `0` → `"ham"` (not spam), `1` → `"spam"`.
///
/// See more information at <https://archive.ics.uci.edu/dataset/94/spambase>.
///
/// # Citation
///
/// M. Hopkins, E. Reeber, G. Forman, and J. Suermondt. "Spambase," UCI Machine
/// Learning Repository, \[Online\]. Available: <https://doi.org/10.24432/C53G6X>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Spambase;
///
/// let download_dir = "./spambase"; // the loader creates the directory if it does not exist
///
/// let mut dataset = Spambase::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap();
/// assert_eq!(features.shape(), &[4601, 57]);
/// assert_eq!(labels.len(), 4601);
///
/// // `get_data()` borrows the cached arrays without a reload. `get_data_mut()`
/// // edits the arrays in place. This needs no clone and no reload. The change
/// // stays cached. If you only need to change values, prefer this method over
/// // `.to_owned()`.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 0.5;
///     labels[0] = "ham";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out with no `to_owned()` clone. This
/// // leaves the instance reusable. The next access reloads the data from the
/// // cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[4601, 57]);
/// assert_eq!(owned_labels.len(), 4601);
///
/// // `into_data()` also returns the owned arrays with no clone, but it
/// // consumes the instance. If you are done with the dataset, use it.
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[4601, 57]);
/// assert_eq!(owned_labels.len(), 4601);
/// ```
#[derive(Debug)]
pub struct Spambase {
    dataset: Dataset<SpambaseData, DatasetError>,
}

impl Spambase {
    /// Create a new Spambase instance without loading data.
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
    /// - `Self` - a `Spambase` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Spambase {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Spambase dataset.
    fn load_data(dir: &str) -> Result<SpambaseData, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            SPAMBASE_FILENAME,
            SPAMBASE_DATASET_NAME,
            Some(SPAMBASE_SHA256),
            |temp_path| {
                download_to_with_retries(
                    SPAMBASE_DATA_URL,
                    temp_path,
                    Some(SPAMBASE_ZIP_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                unzip(&temp_path.join(SPAMBASE_ZIP_FILENAME), temp_path)?;
                Ok(temp_path.join(SPAMBASE_SOURCE_FILENAME))
            },
        )?;

        // `spambase.data` is a headerless comma-separated file: every line is a
        // record of 57 numeric features followed by the class code.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features: Vec<f64> = Vec::with_capacity(N_SAMPLES * N_FEATURES);
        let mut labels: Vec<&'static str> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(SPAMBASE_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            // Skip blank lines, for example a trailing newline.
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    SPAMBASE_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // 57 numeric features.
            for (col, field) in record.iter().take(N_FEATURES).enumerate() {
                let value: f64 = field.trim().parse().map_err(|e| {
                    DatasetError::parse_failed(
                        SPAMBASE_DATASET_NAME,
                        &format!("feature_{}", col),
                        line_num,
                        e,
                    )
                })?;
                features.push(value);
            }

            // Map the source's nominal code to a readable label.
            let label = match record[LABEL_COLUMN].trim() {
                "0" => "ham",
                "1" => "spam",
                other => {
                    return Err(DatasetError::invalid_value(
                        SPAMBASE_DATASET_NAME,
                        "class",
                        other,
                        line_num,
                    ));
                }
            };
            labels.push(label);
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(SPAMBASE_DATASET_NAME));
        }

        // Spambase has a fixed schema of 57 numeric features per sample.
        let features_array = Array2::from_shape_vec((n_samples, N_FEATURES), features)
            .map_err(|e| DatasetError::array_shape_error(SPAMBASE_DATASET_NAME, "features", e))?;

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
    /// - `&Array2<f64>` - Reference to the numeric feature matrix with shape
    ///   `(4601, 57)`: 48 word frequencies, 6 character frequencies, and 3
    ///   capital-run-length statistics.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match the expected dimensions (4601 samples, 57 features)
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
    /// - `&Array1<&'static str>` - Reference to label vector with shape `(4601,)` containing class names (`"ham"`, `"spam"`)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match the expected dimensions (4601 samples)
    pub fn labels(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and labels as references.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&SpambaseData` - reference to the cached `(features, labels)` tuple: the
    ///   feature matrix has shape `(4601, 57)` and the label vector has shape
    ///   `(4601,)` containing class names (`"ham"`, `"spam"`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match the expected dimensions (4601 samples, 57 features)
    pub fn data(&self) -> Result<&SpambaseData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and labels as references **without** triggering loading.
    ///
    /// Unlike [`Spambase::data`], which loads the dataset on the first call, this
    /// method never runs the loader. If the data has not loaded yet, this method
    /// returns `None` instead of downloading and parsing it. Use this method only
    /// when you want data that is already cached. This avoids the download and
    /// parse cost if the dataset is not cached yet.
    ///
    /// # Returns
    ///
    /// - `Some(&SpambaseData)` - reference to the cached `(features, labels)` tuple
    ///   (feature matrix `(4601, 57)`, label vector `(4601,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&SpambaseData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This method lets you change the cached arrays in place (for example, to
    /// normalize features or replace label values). It needs no `to_owned()`
    /// clone, and it does not remove the data from the cache. The changes persist,
    /// so later calls to [`Spambase::features`], [`Spambase::data`], or
    /// [`Spambase::get_data`] see them.
    ///
    /// Like [`Spambase::get_data`], this method does **not** trigger loading. It
    /// returns `None` if the dataset has not loaded yet. If you need the data to
    /// be present, call a loading accessor first, for example [`Spambase::data`].
    ///
    /// # Returns
    ///
    /// - `Some(&mut SpambaseData)` - mutable reference to the cached
    ///   `(features, labels)` tuple (feature matrix `(4601, 57)`, label vector
    ///   `(4601,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut SpambaseData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`Spambase::data`], which borrows the cached data, this method moves
    /// the data out and returns owned arrays directly, with no `to_owned()` clone
    /// needed. The dataset loads on the first access if it has not loaded yet.
    ///
    /// This method **consumes** `self`, so you cannot use the instance afterward.
    /// If you want owned data but need to keep using the instance, use
    /// [`Spambase::take_data`] instead. That method takes `&mut self` and leaves
    /// the instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<&'static str>)` - owned feature matrix with shape
    ///   `(4601, 57)` and owned label vector with shape `(4601,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn into_data(self) -> Result<SpambaseData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and labels out of the dataset. This leaves the
    /// instance reusable.
    ///
    /// Like [`Spambase::into_data`], this method returns owned arrays with no
    /// `to_owned()` clone. Instead of consuming the instance, it takes `&mut self`
    /// and moves the cached data out. This resets the instance to its unloaded
    /// state. The next accessor call, for example [`Spambase::features`] or
    /// [`Spambase::data`], loads the dataset again.
    ///
    /// If you are done with the instance, use [`Spambase::into_data`] instead.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<&'static str>)` - owned feature matrix with shape
    ///   `(4601, 57)` and owned label vector with shape `(4601,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<SpambaseData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(Spambase, SpambaseData, "spambase");
