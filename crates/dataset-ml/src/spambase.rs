//! Spambase dataset.
//!
//! A collection of 4,601 e-mails gathered at Hewlett-Packard Labs in 1999, each
//! summarized by 57 hand-crafted frequency statistics rather than by its raw text.
//! The spam came from a postmaster and from individuals who had filed spam; the
//! non-spam came from filed work and personal e-mail. The task is to predict
//! whether a message is spam from those statistics.
//!
//! **Features (57, all numeric):** 48 `word_freq_WORD` percentages, 6
//! `char_freq_CHAR` percentages, and 3 capital-run-length statistics (average,
//! longest, total). All are non-negative; the frequency columns lie in `0..=100`.
//!
//! **Target:** `class` — one of `ham` or `spam`
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
/// This is the UCI static package; it is a ZIP archive containing
/// `spambase.DOCUMENTATION`, `spambase.data`, and `spambase.names`, of which only
/// the `spambase.data` file is used.
///
/// # Citation
///
/// M. Hopkins, E. Reeber, G. Forman, and J. Suermondt. "Spambase," UCI Machine
/// Learning Repository, \[Online\]. Available: <https://doi.org/10.24432/C53G6X>
const SPAMBASE_DATA_URL: &str = "https://archive.ics.uci.edu/static/public/94/spambase.zip";

/// The name the downloaded ZIP archive is saved under inside the temp directory.
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

/// The number of numeric features per sample (48 word + 6 char frequencies + 3
/// capital-run-length statistics).
const N_FEATURES: usize = 57;

/// The number of columns per CSV record (57 features + 1 label).
const N_COLUMNS: usize = N_FEATURES + 1;

/// Source column index of the label (`class`). The label is the **last** column.
const LABEL_COLUMN: usize = N_FEATURES;

/// Type alias for the Spambase dataset: (features, labels).
type SpambaseData = (Array2<f64>, Array1<&'static str>);

/// A struct representing the Spambase dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The Spambase collection was generated at Hewlett-Packard Labs in June–July
/// 1999 by Mark Hopkins, Erik Reeber, George Forman, and Jaap Suermondt. The
/// "spam" concept is diverse: advertisements for products or web sites,
/// make-money-fast schemes, chain letters, pornography. The spam e-mails came from
/// the lab's postmaster and from individuals who had filed spam; the non-spam
/// e-mails came from filed work and personal e-mail, which is why the word
/// `george` and the area code `650` are strong non-spam indicators here — useful
/// for a personalized filter, but they would have to be blinded to build a general
/// purpose one.
///
/// Each e-mail is reduced to 57 continuous statistics: how often selected words
/// and characters occur, plus how long its runs of capital letters are.
///
/// # Feature columns
///
/// All 57 features are quantitative, stored in one `(4601, 57)` `Array2<f64>`
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
/// A `word_freq_WORD` column is `100 * (times WORD appears) / (total words)`; a
/// `char_freq_CHAR` column is `100 * (occurrences of CHAR) / (total characters)`.
/// A "word" is any string of alphanumeric characters bounded by non-alphanumeric
/// characters or end-of-string. The capital-run-length columns measure
/// uninterrupted sequences of capital letters: their average, their maximum, and
/// their sum (i.e. the total number of capital letters in the e-mail).
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
///   nominal codes to readable names — `0` → `"ham"` (not spam), `1` → `"spam"`.
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
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::spambase::Spambase;
///
/// let download_dir = "./spambase"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = Spambase::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
/// assert_eq!(features.shape(), &[4601, 57]);
/// assert_eq!(labels.len(), 4601);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 0.5;
///     labels[0] = "ham";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable — the next access reloads from the cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[4601, 57]);
/// assert_eq!(owned_labels.len(), 4601);
///
/// // `into_data()` also returns owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
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
    /// The dataset will be loaded lazily when you first call any data accessor method.
    /// This is a lightweight operation that only stores the storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory where the dataset will be stored.
    ///
    /// # Returns
    ///
    /// - `Self` - `Spambase` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Spambase {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Spambase dataset.
    fn load_data(dir: &str) -> Result<SpambaseData, DatasetError> {
        // Prepare the dataset file: download the UCI ZIP package, extract it, and
        // surface the `spambase.data` records (cached under `spambase.csv`).
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

            // Skip blank lines defensively (e.g. a trailing newline).
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

            // Label, mapping the source's nominal code to a readable name.
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
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
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
    /// - Dataset size doesn't match expected dimensions (4601 samples, 57 features)
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
    /// - `&Array1<&'static str>` - Reference to labels vector with shape `(4601,)` containing class names (`"ham"`, `"spam"`)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (4601 samples)
    pub fn labels(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
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
    /// - Dataset size doesn't match expected dimensions (4601 samples, 57 features)
    pub fn data(&self) -> Result<&SpambaseData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and labels as references **without** triggering loading.
    ///
    /// Unlike [`Spambase::data`], which loads the dataset on first call, this never
    /// runs the loader: if the data has not been loaded yet, it returns `None`
    /// instead of downloading and parsing. Use it when you only want the data if it
    /// is already cached and want to avoid paying the download/parse cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&SpambaseData)` - reference to the cached `(features, labels)` tuple
    ///   (feature matrix `(4601, 57)`, label vector `(4601,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&SpambaseData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize features,
    /// replace label values) with no `to_owned()` clone and without removing them
    /// from the cache: the changes persist, so later [`Spambase::features`],
    /// [`Spambase::data`], or [`Spambase::get_data`] calls observe them.
    ///
    /// Like [`Spambase::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call a loading accessor (e.g.
    /// [`Spambase::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut SpambaseData)` - mutable reference to the cached
    ///   `(features, labels)` tuple (feature matrix `(4601, 57)`, label vector
    ///   `(4601,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut SpambaseData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`Spambase::data`], which borrows the cached data, this moves it out
    /// and returns owned arrays directly — no `to_owned()` clone needed. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`Spambase::take_data`] instead — it takes `&mut self` and leaves the
    /// instance reusable.
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

    /// Take **owned** features and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`Spambase::into_data`], this returns owned arrays with no `to_owned()`
    /// clone. But instead of consuming the instance, it takes `&mut self` and moves
    /// the cached data out, resetting the instance to its unloaded state: the next
    /// accessor call (e.g. [`Spambase::features`] or [`Spambase::data`]) loads the
    /// dataset again.
    ///
    /// Use [`Spambase::into_data`] instead if you are done with the instance.
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
