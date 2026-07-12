//! YouTube Spam Collection dataset.
//!
//! A set of YouTube comments tagged as legitimate (`ham`) or spam, collected from
//! the comment sections of five popular music videos for spam research. Like
//! [`SmsSpam`](crate::sms_spam::SmsSpam), this is a **text** dataset: the
//! "features" are the raw comment strings themselves, so there is no numeric or
//! categorical feature matrix — you vectorize the text yourself (bag-of-words,
//! TF-IDF, embeddings, …). Accordingly the document accessor is
//! [`YoutubeSpam::texts`] (returning an `Array1<String>` of raw comments), not
//! `features()`.
//!
//! **Documents:** `Array1<String>` of 1,956 raw YouTube comment bodies
//!
//! **Target:** `label` — one of `ham` or `spam`
//!
//! **Samples:** 1,956 (951 ham, 1,005 spam)
//! **Application:** Binary text classification / spam detection
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5F591>

use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to, unzip};
use ndarray::Array1;
use std::fs::File;
use std::io::Write as _;

/// Type alias for the YouTube Spam dataset: (comment texts, labels).
type YoutubeSpamData = (Array1<String>, Array1<&'static str>);

/// The URL for the YouTube Spam Collection dataset (a ZIP archive).
const YOUTUBE_SPAM_DATA_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip";

/// The name of the downloaded ZIP archive (inside the temp dir).
const YOUTUBE_SPAM_ZIP_FILENAME: &str = "YouTube-Spam-Collection-v1.zip";

/// The five per-video CSV files inside the ZIP archive, in the fixed order they
/// are concatenated into the cached corpus.
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

/// A struct representing the YouTube Spam Collection dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The YouTube Spam Collection is a set of comments extracted from five of the
/// ten most-viewed YouTube videos (music clips by Psy, Katy Perry, LMFAO,
/// Eminem, and Shakira) during the second half of 2015. It contains 1,956 real
/// comments, each manually tagged as either `ham` (legitimate) or `spam`. It is a
/// standard benchmark for text classification and a sibling of the SMS Spam
/// Collection by the same authors.
///
/// # Documents
///
/// Unlike the tabular loaders, there is no feature matrix: each sample is a raw
/// comment string. [`YoutubeSpam::texts`] returns a `(1956,)` `Array1<String>` of
/// the comment bodies (the source `CONTENT` column) — vectorize them
/// (bag-of-words, TF-IDF, embeddings, …) yourself before feeding a model. The
/// per-comment metadata columns (`COMMENT_ID`, `AUTHOR`, `DATE`) are not exposed.
///
/// # Labels
///
/// - `label` (shape `(1956,)`): the `Array1<&'static str>` is one of `"ham"`
///   (legitimate, the source `CLASS` value `0`) or `"spam"` (the source `CLASS`
///   value `1`).
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
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::youtube_spam::YoutubeSpam;
///
/// let download_dir = "./youtube_spam"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = YoutubeSpam::new(download_dir);
/// let texts = dataset.texts().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (texts, labels) = dataset.data().unwrap(); // this is also a way to get texts and labels
/// assert_eq!(texts.len(), 1956);
/// assert_eq!(labels.len(), 1956);
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
/// assert_eq!(owned_texts.len(), 1956);
/// assert_eq!(owned_labels.len(), 1956);
///
/// // `into_data()` also returns the owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_texts, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_texts.len(), 1956);
/// assert_eq!(owned_labels.len(), 1956);
/// ```
#[derive(Debug)]
pub struct YoutubeSpam {
    dataset: Dataset<YoutubeSpamData, DatasetError>,
}

impl YoutubeSpam {
    /// Create a new YoutubeSpam instance without loading data.
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
    /// - `Self` - `YoutubeSpam` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        YoutubeSpam {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the YouTube Spam dataset.
    fn load_data(dir: &str) -> Result<YoutubeSpamData, DatasetError> {
        // Prepare the dataset file: download the ZIP, extract it, and concatenate
        // the five per-video CSVs (in a fixed order) into a single corpus file so
        // one pinned SHA-256 covers the whole dataset (cached as
        // `youtube_spam.csv`).
        let file_path = acquire_dataset(
            dir,
            YOUTUBE_SPAM_FILENAME,
            YOUTUBE_SPAM_DATASET_NAME,
            Some(YOUTUBE_SPAM_SHA256),
            |temp_path| {
                download_to(
                    YOUTUBE_SPAM_DATA_URL,
                    temp_path,
                    Some(YOUTUBE_SPAM_ZIP_FILENAME),
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
        // row, headers are skipped by hand rather than with `has_headers(true)`
        // (which would only skip the very first one).
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut texts: Vec<String> = Vec::with_capacity(N_SAMPLES);
        let mut labels: Vec<&'static str> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(YOUTUBE_SPAM_DATASET_NAME, e))?;
            let line_num = idx + 1;

            // Skip blank lines defensively (e.g. a trailing newline).
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

            // Each of the five concatenated files starts with the same header
            // row; skip every occurrence.
            if &record[0] == "COMMENT_ID" {
                continue;
            }

            // Label, mapping the source `CLASS` code to a readable `&'static str`
            // (`0` = legitimate, `1` = spam) — matching `SmsSpam`'s `ham`/`spam`.
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
            labels.push(label);

            // Comment text, kept verbatim.
            texts.push(record[CONTENT_COLUMN].to_string());
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(YOUTUBE_SPAM_DATASET_NAME));
        }

        let texts_array = Array1::from_vec(texts);
        let labels_array = Array1::from_vec(labels);

        Ok((texts_array, labels_array))
    }

    /// Get a reference to the comment-text vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// This is the YouTube Spam analogue of the tabular loaders' `features()`:
    /// because the data is text, the "features" are the raw comment strings, so
    /// this returns a 1-D `Array1<String>` rather than a 2-D feature matrix.
    ///
    /// # Returns
    ///
    /// - `&Array1<String>` - Reference to the comment-text vector with shape
    ///   `(1956,)`, each entry a raw YouTube comment body.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (1,956 samples)
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
    /// - `&Array1<&'static str>` - Reference to labels vector with shape `(1956,)` containing `"ham"` or `"spam"`
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (1,956 samples)
    pub fn labels(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both comment texts and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&YoutubeSpamData` - reference to the cached `(texts, labels)` tuple: the
    ///   comment-text vector `(1956,)` and the label vector `(1956,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (1,956 samples)
    pub fn data(&self) -> Result<&YoutubeSpamData, DatasetError> {
        self.dataset.load()
    }

    /// Get both comment texts and labels as references **without** triggering loading.
    ///
    /// Unlike [`YoutubeSpam::data`], which loads the dataset on first call, this
    /// never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&YoutubeSpamData)` - reference to the cached `(texts, labels)` tuple
    ///   (`(1956,)`, `(1956,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&YoutubeSpamData> {
        self.dataset.get()
    }

    /// Get mutable references to comment texts and labels for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize or clean the
    /// comment text) with no `to_owned()` clone and without removing them from the
    /// cache: the changes persist, so later [`YoutubeSpam::texts`],
    /// [`YoutubeSpam::data`], or [`YoutubeSpam::get_data`] calls observe them.
    ///
    /// Like [`YoutubeSpam::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call a loading accessor (e.g.
    /// [`YoutubeSpam::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut YoutubeSpamData)` - mutable reference to the cached `(texts,
    ///   labels)` tuple (`(1956,)`, `(1956,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut YoutubeSpamData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** comment texts and labels.
    ///
    /// Unlike [`YoutubeSpam::data`], which borrows the cached data, this moves it
    /// out and returns owned arrays directly — no `to_owned()` clone needed. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`YoutubeSpam::take_data`] instead — it takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array1<&'static str>)` - owned comment-text vector
    ///   `(1956,)` and owned label vector `(1956,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file extraction, I/O,
    /// parsing, invalid labels, or a dimension mismatch).
    pub fn into_data(self) -> Result<YoutubeSpamData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** comment texts and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`YoutubeSpam::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out, resetting the instance to its
    /// unloaded state: the next accessor call (e.g. [`YoutubeSpam::texts`] or
    /// [`YoutubeSpam::data`]) loads the dataset again.
    ///
    /// Use [`YoutubeSpam::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array1<&'static str>)` - owned comment-text vector
    ///   `(1956,)` and owned label vector `(1956,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file extraction, I/O,
    /// parsing, invalid labels, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<YoutubeSpamData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}
