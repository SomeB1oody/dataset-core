//! SMS Spam Collection dataset.
//!
//! A set of SMS messages tagged as legitimate (`ham`) or spam, collected for SMS
//! spam research. This is the crate's first **text** dataset: the "features" are
//! the raw message strings themselves, so there is no numeric or categorical
//! feature matrix — you vectorize the text yourself (bag-of-words, TF-IDF,
//! embeddings, …). Accordingly the document accessor is [`SmsSpam::texts`]
//! (returning an `Array1<String>` of raw messages), not `features()`.
//!
//! **Documents:** `Array1<String>` of 5,574 raw SMS message bodies
//!
//! **Target:** `label` — one of `ham` or `spam`
//!
//! **Samples:** 5,574 (4,827 ham, 747 spam)
//! **Application:** Binary text classification / spam detection
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5CC84>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::Array1;
use std::fs::File;

/// Type alias for the SMS Spam dataset: (message texts, labels).
type SmsSpamData = (Array1<String>, Array1<&'static str>);

/// The URL for the SMS Spam Collection dataset (a ZIP archive).
const SMS_SPAM_DATA_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip";

/// The name of the downloaded ZIP archive (inside the temp dir).
const SMS_SPAM_ZIP_FILENAME: &str = "smsspamcollection.zip";

/// The name of the data file inside the ZIP archive.
const SMS_SPAM_SOURCE_FILENAME: &str = "SMSSpamCollection";

/// The name of the cached SMS Spam dataset file.
const SMS_SPAM_FILENAME: &str = "sms_spam.csv";

/// The SHA256 hash of the cached SMS Spam dataset file (the extracted
/// `SMSSpamCollection` file's bytes).
const SMS_SPAM_SHA256: &str = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d";

/// The name of the dataset.
const SMS_SPAM_DATASET_NAME: &str = "sms_spam";

/// Number of samples.
const N_SAMPLES: usize = 5_574;

/// Number of columns per record (1 label + 1 message text).
const N_COLUMNS: usize = 2;

/// Source column index of the label.
const LABEL_COLUMN: usize = 0;

/// Source column index of the message text.
const TEXT_COLUMN: usize = 1;

/// A struct representing the SMS Spam Collection dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The SMS Spam Collection is a set of SMS messages that have been collected for
/// SMS spam research. It contains 5,574 English messages, each tagged as either
/// `ham` (legitimate) or `spam`. The messages come from several sources,
/// including the Grumbletext website, the NUS SMS Corpus, and a PhD thesis
/// collection. It is a standard benchmark for text classification.
///
/// # Documents
///
/// Unlike the tabular loaders, there is no feature matrix: each sample is a raw
/// message string. [`SmsSpam::texts`] returns a `(5574,)` `Array1<String>` of the
/// message bodies — vectorize them (bag-of-words, TF-IDF, embeddings, …) yourself
/// before feeding a model.
///
/// # Labels
///
/// - `label` (shape `(5574,)`): the `Array1<&'static str>` is one of `"ham"`
///   (legitimate) or `"spam"`.
///
/// See more information at <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>.
///
/// # Citation
///
/// Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection \[Dataset\]. UCI Machine
/// Learning Repository. <https://doi.org/10.24432/C5CC84>
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::sms_spam::SmsSpam;
///
/// let download_dir = "./sms_spam"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = SmsSpam::new(download_dir);
/// let texts = dataset.texts().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (texts, labels) = dataset.data().unwrap(); // this is also a way to get texts and labels
/// assert_eq!(texts.len(), 5574);
/// assert_eq!(labels.len(), 5574);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((texts, labels)) = dataset.get_data_mut() {
///     texts[0] = "hello world".to_string();
///     labels[0] = "spam";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out (no `to_owned()` clone) and leaves
/// // the instance reusable — the next access reloads from the cached file.
/// let (owned_texts, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_texts.len(), 5574);
/// assert_eq!(owned_labels.len(), 5574);
///
/// // `into_data()` also returns the owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_texts, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_texts.len(), 5574);
/// assert_eq!(owned_labels.len(), 5574);
/// ```
#[derive(Debug)]
pub struct SmsSpam {
    dataset: Dataset<SmsSpamData, DatasetError>,
}

impl SmsSpam {
    /// Create a new SmsSpam instance without loading data.
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
    /// - `Self` - `SmsSpam` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        SmsSpam {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the SMS Spam dataset.
    fn load_data(dir: &str) -> Result<SmsSpamData, DatasetError> {
        // Prepare the dataset file: download the ZIP, extract it, and use the
        // `SMSSpamCollection` file (cached under `sms_spam.csv`).
        let file_path = acquire_dataset(
            dir,
            SMS_SPAM_FILENAME,
            SMS_SPAM_DATASET_NAME,
            Some(SMS_SPAM_SHA256),
            |temp_path| {
                download_to_with_retries(
                    SMS_SPAM_DATA_URL,
                    temp_path,
                    Some(SMS_SPAM_ZIP_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                unzip(&temp_path.join(SMS_SPAM_ZIP_FILENAME), temp_path)?;
                Ok(temp_path.join(SMS_SPAM_SOURCE_FILENAME))
            },
        )?;

        // The source is tab-separated with no header: `label<TAB>message`. The
        // messages are free text that can contain `"`, `,`, and other punctuation,
        // so quote processing is disabled — every record is split purely on tabs.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(false)
            .quoting(false)
            .from_reader(file);

        let mut texts: Vec<String> = Vec::with_capacity(N_SAMPLES);
        let mut labels: Vec<&'static str> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(SMS_SPAM_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            // Skip blank lines defensively (e.g. a trailing newline).
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    SMS_SPAM_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // Label, mapping the source token to a readable `&'static str`.
            let label = match &record[LABEL_COLUMN] {
                "ham" => "ham",
                "spam" => "spam",
                other => {
                    return Err(DatasetError::invalid_value(
                        SMS_SPAM_DATASET_NAME,
                        "label",
                        other,
                        line_num,
                    ));
                }
            };
            labels.push(label);

            // Message text, kept verbatim.
            texts.push(record[TEXT_COLUMN].to_string());
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(SMS_SPAM_DATASET_NAME));
        }

        let texts_array = Array1::from_vec(texts);
        let labels_array = Array1::from_vec(labels);

        Ok((texts_array, labels_array))
    }

    /// Get a reference to the message-text vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// This is the SMS Spam analogue of the tabular loaders' `features()`: because
    /// the data is text, the "features" are the raw message strings, so this
    /// returns a 1-D `Array1<String>` rather than a 2-D feature matrix.
    ///
    /// # Returns
    ///
    /// - `&Array1<String>` - Reference to the message-text vector with shape
    ///   `(5574,)`, each entry a raw SMS message body.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (5,574 samples)
    pub fn texts(&self) -> Result<&Array1<String>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the labels vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<&'static str>` - Reference to labels vector with shape `(5574,)` containing `"ham"` or `"spam"`
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (5,574 samples)
    pub fn labels(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both message texts and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&SmsSpamData` - reference to the cached `(texts, labels)` tuple: the
    ///   message-text vector `(5574,)` and the label vector `(5574,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (5,574 samples)
    pub fn data(&self) -> Result<&SmsSpamData, DatasetError> {
        self.dataset.load()
    }

    /// Get both message texts and labels as references **without** triggering loading.
    ///
    /// Unlike [`SmsSpam::data`], which loads the dataset on first call, this never
    /// runs the loader: if the data has not been loaded yet, it returns `None`
    /// instead of downloading and parsing. Use it when you only want the data if
    /// it is already cached and want to avoid paying the download/parse cost
    /// otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&SmsSpamData)` - reference to the cached `(texts, labels)` tuple
    ///   (`(5574,)`, `(5574,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&SmsSpamData> {
        self.dataset.get()
    }

    /// Get mutable references to message texts and labels for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize or clean the
    /// message text) with no `to_owned()` clone and without removing them from the
    /// cache: the changes persist, so later [`SmsSpam::texts`], [`SmsSpam::data`],
    /// or [`SmsSpam::get_data`] calls observe them.
    ///
    /// Like [`SmsSpam::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call a loading accessor (e.g.
    /// [`SmsSpam::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut SmsSpamData)` - mutable reference to the cached `(texts,
    ///   labels)` tuple (`(5574,)`, `(5574,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut SmsSpamData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** message texts and labels.
    ///
    /// Unlike [`SmsSpam::data`], which borrows the cached data, this moves it out
    /// and returns owned arrays directly — no `to_owned()` clone needed. The dataset
    /// is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`SmsSpam::take_data`] instead — it takes `&mut self` and leaves the instance
    /// reusable.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array1<&'static str>)` - owned message-text vector
    ///   `(5574,)` and owned label vector `(5574,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file extraction, I/O,
    /// parsing, invalid labels, or a dimension mismatch).
    pub fn into_data(self) -> Result<SmsSpamData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** message texts and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`SmsSpam::into_data`], this returns owned arrays with no `to_owned()`
    /// clone. But instead of consuming the instance, it takes `&mut self` and moves
    /// the cached data out, resetting the instance to its unloaded state: the next
    /// accessor call (e.g. [`SmsSpam::texts`] or [`SmsSpam::data`]) loads the
    /// dataset again.
    ///
    /// Use [`SmsSpam::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array1<&'static str>)` - owned message-text vector
    ///   `(5574,)` and owned label vector `(5574,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file extraction, I/O,
    /// parsing, invalid labels, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<SmsSpamData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(SmsSpam, SmsSpamData, "sms_spam");
