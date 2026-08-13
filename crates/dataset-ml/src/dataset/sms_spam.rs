//! SMS Spam Collection dataset.
//!
//! This dataset is a set of SMS messages tagged as legitimate (`ham`) or spam.
//! Researchers collected it for SMS spam research. Each sample is one raw
//! message body. Vectorize the text yourself (bag-of-words, TF-IDF, embeddings,
//! and so on) before you use it as model input.
//!
//! **Columns (2):**
//!
//! | Name    | Type     | Description              |
//! |---------|----------|--------------------------|
//! | `text`  | `String` | the raw SMS message body |
//! | `label` | `String` | `ham` or `spam`          |
//!
//! The source designates the message text as the input
//! ([`SmsSpam::FEATURE_NAMES`](crate::SmsSpam::FEATURE_NAMES)) and the tag as the label ([`SmsSpam::TARGET`](crate::SmsSpam::TARGET)).
//!
//! **Samples:** 5,574 (4,827 ham, 747 spam)
//! **Application:** Binary text classification / spam detection
//!
//! **Missing values:** none.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5CC84>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::Array1;
use std::fs::File;

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

/// A struct that represents the SMS Spam Collection dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The SMS Spam Collection is a set of SMS messages. Researchers collected the
/// messages for SMS spam research. It contains 5,574 English messages, each
/// tagged as `ham` (legitimate) or `spam`. The messages come from several
/// sources, including the Grumbletext website, the NUS SMS Corpus, and a PhD
/// thesis collection. It is a standard benchmark for text classification.
///
/// # Columns
///
/// | Name    | Type     | Description              |
/// |---------|----------|--------------------------|
/// | `text`  | `String` | the raw SMS message body |
/// | `label` | `String` | `ham` or `spam`          |
///
/// The source designates the message text as the input
/// ([`SmsSpam::FEATURE_NAMES`]) and the tag as the label ([`SmsSpam::TARGET`]).
///
/// Missing values: none.
///
/// The `text` column holds whole documents, not numbers. Vectorize the messages
/// yourself (bag-of-words, TF-IDF, embeddings, and so on) before you use them as
/// model input.
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
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::SmsSpam;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./sms_spam";
///
/// let mut dataset = SmsSpam::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 5574);
/// assert_eq!(table.n_columns(), 2);
///
/// // Reach one column by name.
/// let texts = table.column(SmsSpam::FEATURE_NAMES[0]).unwrap().as_string().unwrap();
/// assert_eq!(texts.len(), 5574);
/// let labels = table.column(SmsSpam::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(labels[0], "ham");
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("text") {
///         if let dataset_ml::ColumnData::String(values) = column.data_mut() {
///             values[0] = "hello world".to_string();
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 5574);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 5574);
/// ```
#[derive(Debug)]
pub struct SmsSpam {
    dataset: Dataset<Table, DatasetError>,
}

impl SmsSpam {
    /// The column the source designates as the model input.
    pub const FEATURE_NAMES: [&'static str; 1] = ["text"];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "label";

    /// Create a new SmsSpam instance without loading data.
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
    /// - `Self` - a `SmsSpam` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        SmsSpam {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the SMS Spam dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
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
        // messages are free text and can contain `"`, `,`, and other punctuation.
        // The loader disables quote processing, so it splits each record only on tabs.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(false)
            .quoting(false)
            .from_reader(file);

        let mut texts: Vec<String> = Vec::with_capacity(N_SAMPLES);
        let mut labels: Vec<String> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(SMS_SPAM_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            // Skip blank lines, such as a trailing newline.
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

            // Label. The loader maps the source token to a readable name.
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
            labels.push(label.to_string());

            // Message text. The loader stores it unchanged.
            texts.push(record[TEXT_COLUMN].to_string());
        }

        Table::new(
            SMS_SPAM_DATASET_NAME,
            vec![
                Column::new(
                    Self::FEATURE_NAMES[0],
                    ColumnData::String(Array1::from_vec(texts)),
                ),
                Column::new(Self::TARGET, ColumnData::String(Array1::from_vec(labels))),
            ],
        )
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 5,574 samples and 2 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, or an invalid label)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`SmsSpam::data`], this method never runs the loader. If the data
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
    /// changes stay in the cache. Later calls to [`SmsSpam::data`] or
    /// [`SmsSpam::get_data`] see them.
    ///
    /// Like [`SmsSpam::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`SmsSpam::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 5,574 samples and 2 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file extraction, I/O,
    /// or parsing).
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
    /// - `Table` - the owned table of 5,574 samples and 2 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file extraction, I/O,
    /// or parsing).
    pub fn take_data(&mut self) -> Result<Table, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(SmsSpam, "sms_spam");
