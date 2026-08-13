//! YouTube Spam Collection dataset.
//!
//! This is a set of YouTube comments, tagged as legitimate (`ham`) or spam. The
//! comments come from the comment sections of five popular music videos,
//! collected for spam research. Each sample is one raw comment body. Vectorize
//! the text yourself (bag-of-words, TF-IDF, embeddings, and so on) before you
//! use it as model input.
//!
//! **Columns (2):**
//!
//! | Name    | Type     | Description                                |
//! |---------|----------|--------------------------------------------|
//! | `text`  | `String` | the raw comment body (source `CONTENT`)    |
//! | `label` | `String` | `ham` (source `CLASS` `0`) or `spam` (`1`) |
//!
//! The source also holds `COMMENT_ID`, `AUTHOR`, and `DATE`. The table does not
//! carry them.
//!
//! The source designates the comment text as the input
//! ([`YoutubeSpam::FEATURE_NAMES`](crate::YoutubeSpam::FEATURE_NAMES)) and the tag as the label
//! ([`YoutubeSpam::TARGET`](crate::YoutubeSpam::TARGET)).
//!
//! **Samples:** 1,956 (951 ham, 1,005 spam)
//! **Application:** Binary text classification / spam detection
//!
//! **Missing values:** none.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5F591>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::Array1;
use std::fs::File;
use std::io::Write as _;

/// The URL for the YouTube Spam Collection dataset (a ZIP archive).
const YOUTUBE_SPAM_DATA_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip";

/// The name of the downloaded ZIP archive (inside the temp dir).
const YOUTUBE_SPAM_ZIP_FILENAME: &str = "YouTube-Spam-Collection-v1.zip";

/// The five per-video CSV files inside the ZIP archive, in the fixed order
/// the loader concatenates them into the cached corpus.
const YOUTUBE_SPAM_SOURCE_FILENAMES: [&str; 5] = [
    "Youtube01-Psy.csv",
    "Youtube02-KatyPerry.csv",
    "Youtube03-LMFAO.csv",
    "Youtube04-Eminem.csv",
    "Youtube05-Shakira.csv",
];

/// The name of the cached YouTube Spam dataset file (the five per-video CSVs
/// concatenated in order).
const YOUTUBE_SPAM_FILENAME: &str = "youtube_spam.csv";

/// The SHA256 hash of the cached YouTube Spam dataset file (the five source CSVs
/// concatenated in order).
const YOUTUBE_SPAM_SHA256: &str =
    "f172e32ca7b4ecadb926df0c836dbe6c6485c519a47a5e7d7f719f2b3553906b";

/// The name of the dataset.
const YOUTUBE_SPAM_DATASET_NAME: &str = "youtube_spam";

/// Number of samples.
const N_SAMPLES: usize = 1_956;

/// Number of columns per record (`COMMENT_ID`, `AUTHOR`, `DATE`, `CONTENT`, `CLASS`).
const N_COLUMNS: usize = 5;

/// Source column index of the comment text (`CONTENT`).
const CONTENT_COLUMN: usize = 3;

/// Source column index of the class label (`CLASS`).
const CLASS_COLUMN: usize = 4;

/// A struct that represents the YouTube Spam Collection dataset with lazy
/// loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The YouTube Spam Collection contains 1,956 real comments from five popular
/// YouTube videos. The videos are music clips by Psy, Katy Perry, LMFAO, Eminem,
/// and Shakira, five of the ten most-viewed videos during the second half of
/// 2015. Researchers manually tagged each comment as either `ham` (legitimate)
/// or `spam`. This dataset is a standard benchmark for text classification, and
/// a sibling of the SMS Spam Collection by the same authors.
///
/// # Columns
///
/// | Name    | Type     | Description                                |
/// |---------|----------|--------------------------------------------|
/// | `text`  | `String` | the raw comment body (source `CONTENT`)    |
/// | `label` | `String` | `ham` (source `CLASS` `0`) or `spam` (`1`) |
///
/// The source designates the comment text as the input
/// ([`YoutubeSpam::FEATURE_NAMES`]) and the tag as the label
/// ([`YoutubeSpam::TARGET`]).
///
/// Missing values: none.
///
/// The source also holds `COMMENT_ID`, `AUTHOR`, and `DATE`. The table does not
/// carry them. The `text` column holds whole documents, not numbers. Vectorize
/// the comments yourself (bag-of-words, TF-IDF, embeddings, and so on) before
/// you use them as model input.
///
/// See more information at <https://archive.ics.uci.edu/dataset/380/youtube+spam+collection>.
///
/// # Citation
///
/// Alberto, T., Lochter, J. & Almeida, T. (2017). YouTube Spam Collection
/// \[Dataset\]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5F591>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::YoutubeSpam;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./youtube_spam";
///
/// let mut dataset = YoutubeSpam::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 1956);
/// assert_eq!(table.n_columns(), 2);
///
/// // Reach one column by name.
/// let texts = table.column(YoutubeSpam::FEATURE_NAMES[0]).unwrap().as_string().unwrap();
/// assert_eq!(texts.len(), 1956);
/// let labels = table.column(YoutubeSpam::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(labels[0], "spam");
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
/// assert_eq!(owned.n_samples(), 1956);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 1956);
/// ```
#[derive(Debug)]
pub struct YoutubeSpam {
    dataset: Dataset<Table, DatasetError>,
}

impl YoutubeSpam {
    /// The column the source designates as the model input.
    pub const FEATURE_NAMES: [&'static str; 1] = ["text"];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "label";

    /// Create a new YoutubeSpam instance without loading data.
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
    /// - `Self` - a `YoutubeSpam` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        YoutubeSpam {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the YouTube Spam dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        // Download the ZIP, extract it, and concatenate the five per-video CSVs,
        // in a fixed order, into one corpus file. This lets one pinned SHA-256
        // cover the whole dataset. The loader caches the result as
        // `youtube_spam.csv`.
        let file_path = acquire_dataset(
            dir,
            YOUTUBE_SPAM_FILENAME,
            YOUTUBE_SPAM_DATASET_NAME,
            Some(YOUTUBE_SPAM_SHA256),
            |temp_path| {
                download_to_with_retries(
                    YOUTUBE_SPAM_DATA_URL,
                    temp_path,
                    Some(YOUTUBE_SPAM_ZIP_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                unzip(&temp_path.join(YOUTUBE_SPAM_ZIP_FILENAME), temp_path)?;

                // Concatenate the raw bytes of the five source CSVs in order. Each
                // file ends with a newline, so the byte concatenation is a valid
                // CSV whose SHA-256 is stable across platforms.
                let combined_path = temp_path.join(YOUTUBE_SPAM_FILENAME);
                let mut combined = File::create(&combined_path)?;
                for name in YOUTUBE_SPAM_SOURCE_FILENAMES {
                    let bytes = std::fs::read(temp_path.join(name))?;
                    combined.write_all(&bytes)?;
                }
                combined.flush()?;

                Ok(combined_path)
            },
        )?;

        // The corpus is a standard comma-separated CSV with quoted fields (one
        // comment even contains an embedded newline), so quote handling stays
        // enabled. Because the five concatenated files each keep their own header
        // row, the code skips headers by hand instead of using `has_headers(true)`.
        // That option would skip only the first header row.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut texts: Vec<String> = Vec::with_capacity(N_SAMPLES);
        let mut labels: Vec<String> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(YOUTUBE_SPAM_DATASET_NAME, e))?;
            let line_num = idx + 1;

            // Skip blank lines, for example a trailing newline.
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    YOUTUBE_SPAM_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // Each of the five concatenated files starts with the same header row.
            // Skip every occurrence of it.
            if &record[0] == "COMMENT_ID" {
                continue;
            }

            // Map the source `CLASS` code to a readable name (`0` = legitimate,
            // `1` = spam). This matches `SmsSpam`'s `ham`/`spam` labels.
            let label = match &record[CLASS_COLUMN] {
                "0" => "ham",
                "1" => "spam",
                other => {
                    return Err(DatasetError::invalid_value(
                        YOUTUBE_SPAM_DATASET_NAME,
                        "CLASS",
                        other,
                        line_num,
                    ));
                }
            };
            labels.push(label.to_string());

            // Comment text, kept verbatim.
            texts.push(record[CONTENT_COLUMN].to_string());
        }

        Table::new(
            YOUTUBE_SPAM_DATASET_NAME,
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
    /// - `&Table` - reference to the cached table of 1,956 samples and 2 columns.
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
    /// Unlike [`YoutubeSpam::data`], this method never runs the loader. If the
    /// data has not loaded yet, it returns `None` instead of downloading and
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
    /// changes stay in the cache. Later calls to [`YoutubeSpam::data`] or
    /// [`YoutubeSpam::get_data`] see them.
    ///
    /// Like [`YoutubeSpam::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`YoutubeSpam::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 1,956 samples and 2 columns.
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
    /// - `Table` - the owned table of 1,956 samples and 2 columns.
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

impl_ml_dataset!(YoutubeSpam, "youtube_spam");
