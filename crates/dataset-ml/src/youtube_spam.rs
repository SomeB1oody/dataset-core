//! YouTube Spam Collection dataset.
//!
//! This is a set of YouTube comments, tagged as legitimate (`ham`) or spam. The
//! comments come from the comment sections of five popular music videos,
//! collected for spam research. Like [`SmsSpam`](crate::sms_spam::SmsSpam), this
//! is a **text** dataset: the "features" are the raw comment strings. There is
//! no numeric or categorical feature matrix, so you vectorize the text yourself,
//! for example with bag-of-words, TF-IDF, or embeddings. The document accessor
//! is [`YoutubeSpam::texts`], which returns an `Array1<String>` of raw comments,
//! not `features()`.
//!
//! **Documents:** `Array1<String>` of 1,956 raw YouTube comment bodies
//!
//! **Target:** `label` - one of `ham` or `spam`
//!
//! **Samples:** 1,956 (951 ham, 1,005 spam)
//! **Application:** Binary text classification / spam detection
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5F591>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
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

/// This struct represents the YouTube Spam Collection dataset. It loads data
/// lazily: the dataset does not load until you call a data accessor method. Once
/// loaded, the data stays cached for later accesses.
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
/// # Documents
///
/// Unlike the tabular loaders, there is no feature matrix: each sample is a raw
/// comment string. [`YoutubeSpam::texts`] returns a `(1956,)` `Array1<String>` of
/// the comment bodies, from the source `CONTENT` column. Vectorize them
/// yourself, for example with bag-of-words, TF-IDF, or embeddings, before you
/// feed a model. The loader does not expose the per-comment metadata columns
/// (`COMMENT_ID`, `AUTHOR`, `DATE`).
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
/// This struct implements `Send` and `Sync` because all its fields implement them.
/// This makes it safe to share the struct across threads. The internal
/// [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::youtube_spam::YoutubeSpam;
///
/// let download_dir = "./youtube_spam"; // the code creates the directory if it does not exist
///
/// let mut dataset = YoutubeSpam::new(download_dir);
/// let texts = dataset.texts().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (texts, labels) = dataset.data().unwrap(); // this is also a way to get texts and labels
/// assert_eq!(texts.len(), 1956);
/// assert_eq!(labels.len(), 1956);
///
/// // `get_data()` borrows the cached arrays without reloading. `get_data_mut()`
/// // edits them in place, with no clone and no reload. The change stays cached.
/// // Prefer this method over `.to_owned()` when you only need to change values.
/// if let Some((texts, labels)) = dataset.get_data_mut() {
///     texts[0] = "hello world".to_string();
///     labels[0] = "spam";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out with no `to_owned()` clone. It
/// // leaves the instance reusable. The next access reloads data from the cached
/// // file.
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
    /// The dataset does not load immediately. It loads the first time you call a
    /// data accessor method. This call is lightweight: it only stores the storage
    /// directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory that holds the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - `YoutubeSpam` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        YoutubeSpam {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the YouTube Spam dataset.
    fn load_data(dir: &str) -> Result<YoutubeSpamData, DatasetError> {
        // Download the ZIP, extract it, and concatenate the five per-video CSVs,
        // in a fixed order, into one corpus file. This lets one pinned SHA-256
        // cover the whole dataset. The result is cached as `youtube_spam.csv`.
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
        let mut labels: Vec<&'static str> = Vec::with_capacity(N_SAMPLES);

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

            // Map the source `CLASS` code to a readable `&'static str` (`0` = legitimate,
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
    /// This method loads the dataset lazily on the first call. Later calls return
    /// the cached data instantly.
    ///
    /// This method is the YouTube Spam analogue of the tabular loaders' `features()`.
    /// Because the data is text, the "features" are the raw comment strings. This
    /// method returns a 1-D `Array1<String>` instead of a 2-D feature matrix.
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
    /// - Dataset size does not match the expected dimensions (1,956 samples)
    pub fn texts(&self) -> Result<&Array1<String>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the labels vector.
    ///
    /// This method loads the dataset lazily on the first call. Later calls return
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
    /// - Dataset size does not match the expected dimensions (1,956 samples)
    pub fn labels(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both comment texts and labels as references.
    ///
    /// This method loads the dataset lazily on the first call. Later calls return
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
    /// - Dataset size does not match the expected dimensions (1,956 samples)
    pub fn data(&self) -> Result<&YoutubeSpamData, DatasetError> {
        self.dataset.load()
    }

    /// Get both comment texts and labels as references **without** triggering loading.
    ///
    /// Unlike [`YoutubeSpam::data`], which loads the dataset on the first call,
    /// this method never runs the loader. If the data has not loaded yet, this
    /// method returns `None` instead of downloading and parsing it. Use this
    /// method only when you want data that is already cached. This avoids the
    /// download and parse cost if the dataset is not cached yet.
    ///
    /// # Returns
    ///
    /// - `Some(&YoutubeSpamData)` - reference to the cached `(texts, labels)` tuple
    ///   (`(1956,)`, `(1956,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&YoutubeSpamData> {
        self.dataset.get()
    }

    /// Get mutable references to comment texts and labels for **in-place** editing.
    ///
    /// This method lets you change the cached arrays in place (for example, to
    /// normalize or clean the comment text). It needs no `to_owned()` clone, and
    /// it does not remove the data from the cache. The changes persist, so later
    /// calls to [`YoutubeSpam::texts`], [`YoutubeSpam::data`], or
    /// [`YoutubeSpam::get_data`] see them.
    ///
    /// Like [`YoutubeSpam::get_data`], this method does **not** trigger loading.
    /// It returns `None` if the dataset has not loaded yet. If you need the data
    /// to be present, call a loading accessor first, for example
    /// [`YoutubeSpam::data`].
    ///
    /// # Returns
    ///
    /// - `Some(&mut YoutubeSpamData)` - mutable reference to the cached `(texts,
    ///   labels)` tuple (`(1956,)`, `(1956,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut YoutubeSpamData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** comment texts and labels.
    ///
    /// Unlike [`YoutubeSpam::data`], which borrows the cached data, this method
    /// moves the data out and returns owned arrays directly, with no
    /// `to_owned()` clone needed. The dataset loads on the first access if it has
    /// not loaded yet.
    ///
    /// This method **consumes** `self`, so you cannot use the instance afterward.
    /// If you want owned data but need to keep using the instance, use
    /// [`YoutubeSpam::take_data`] instead. That method takes `&mut self` and
    /// leaves the instance reusable.
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

    /// Take **owned** comment texts and labels out of the dataset. This leaves it
    /// reusable.
    ///
    /// Like [`YoutubeSpam::into_data`], this method returns owned arrays with no
    /// `to_owned()` clone. Instead of consuming the instance, it takes `&mut self`
    /// and moves the cached data out. This resets the instance to its unloaded
    /// state. The next accessor call, for example [`YoutubeSpam::texts`] or
    /// [`YoutubeSpam::data`], loads the dataset again.
    ///
    /// If you are done with the instance, use [`YoutubeSpam::into_data`] instead.
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

impl_ml_dataset!(YoutubeSpam, YoutubeSpamData, "youtube_spam");
