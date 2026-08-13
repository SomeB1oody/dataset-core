//! Sentiment Labelled Sentences dataset.
//!
//! This dataset is a set of 3,000 short sentences from three product and
//! service review sites (Amazon, IMDb, Yelp). Human annotators hand-labeled
//! each sentence with a binary sentiment (`positive` or `negative`). Each
//! sample is one raw sentence. Vectorize the text yourself (bag-of-words,
//! TF-IDF, embeddings, and so on) before you use it as model input.
//!
//! **Columns (3):**
//!
//! | Name     | Type     | Description                          |
//! |----------|----------|--------------------------------------|
//! | `text`   | `String` | one raw review sentence              |
//! | `source` | `String` | `amazon`, `imdb`, or `yelp`          |
//! | `label`  | `String` | `positive` (`1`) or `negative` (`0`) |
//!
//! The source designates the review text as the input
//! ([`SentimentSentences::FEATURE_NAMES`](crate::SentimentSentences::FEATURE_NAMES)) and the sentiment as the label
//! ([`SentimentSentences::TARGET`](crate::SentimentSentences::TARGET)).
//!
//! **Samples:** 3,000 total (1,500 positive, 1,500 negative, 1,000 per site,
//! balanced)
//! **Application:** Binary text classification / sentiment analysis
//!
//! **Missing values:** none.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C57604>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::Array1;
use std::fs::File;
use std::io::Write as _;

/// The URL for the Sentiment Labelled Sentences dataset (a ZIP archive).
const SENTIMENT_SENTENCES_DATA_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip";

/// The name of the downloaded ZIP archive (inside the temp dir).
const SENTIMENT_SENTENCES_ZIP_FILENAME: &str = "sentiment_labelled_sentences.zip";

/// The name of the folder inside the ZIP archive that holds the data files
/// (note the spaces in the upstream directory name).
const SENTIMENT_SENTENCES_SUBDIR: &str = "sentiment labelled sentences";

/// The three per-site source files inside the archive, paired with the
/// `source` value each one carries, in the fixed order the loader combines
/// them.
const SENTIMENT_SENTENCES_SOURCE_FILES: [(&str, &str); 3] = [
    ("amazon", "amazon_cells_labelled.txt"),
    ("imdb", "imdb_labelled.txt"),
    ("yelp", "yelp_labelled.txt"),
];

/// The name of the cached Sentiment Labelled Sentences dataset file (the three
/// per-site files combined, each line prefixed with its `source`).
const SENTIMENT_SENTENCES_FILENAME: &str = "sentiment_sentences.csv";

/// The SHA256 hash of the cached Sentiment Labelled Sentences dataset file (the
/// combined `source<TAB>sentence<TAB>label` corpus).
const SENTIMENT_SENTENCES_SHA256: &str =
    "3a6aac64fa37c8075d49678cd73140eaa70a95c984d540ddf93ec7b021e05725";

/// The name of the dataset.
const SENTIMENT_SENTENCES_DATASET_NAME: &str = "sentiment_sentences";

/// Number of samples.
const N_SAMPLES: usize = 3_000;

/// Number of columns per record in the combined file (`source`, `sentence`, `label`).
const N_COLUMNS: usize = 3;

/// Column index of the source site (`amazon` / `imdb` / `yelp`).
const SOURCE_COLUMN: usize = 0;

/// Column index of the sentence text.
const SENTENCE_COLUMN: usize = 1;

/// Column index of the sentiment label (`0` / `1`).
const LABEL_COLUMN: usize = 2;

/// A struct that represents the Sentiment Labelled Sentences dataset with lazy
/// loading.
///
/// The dataset loads only when you call a data accessor method. After the
/// first load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The Sentiment Labelled Sentences dataset (Kotzias, Denil, de Freitas & Smyth,
/// 2015) collects 3,000 sentences from three review sites. It takes 1,000
/// sentences from each site: Amazon product reviews, IMDb movie reviews, and
/// Yelp restaurant reviews. Human annotators hand-labeled each sentence
/// `positive` or `negative` (500 of each per site, so the corpus is perfectly
/// balanced). It is a compact benchmark for sentence-level sentiment
/// classification and for studying cross-domain transfer between the three
/// sources.
///
/// # Columns
///
/// | Name     | Type     | Description                          |
/// |----------|----------|--------------------------------------|
/// | `text`   | `String` | one raw review sentence              |
/// | `source` | `String` | `amazon`, `imdb`, or `yelp`          |
/// | `label`  | `String` | `positive` (`1`) or `negative` (`0`) |
///
/// The source designates the review text as the input
/// ([`SentimentSentences::FEATURE_NAMES`]) and the sentiment as the label
/// ([`SentimentSentences::TARGET`]).
///
/// Missing values: none.
///
/// The `text` column holds whole documents, not numbers. Vectorize the
/// sentences yourself (bag-of-words, TF-IDF, embeddings, and so on) before you
/// use them as model input. The `source` column names the review site the
/// sentence came from. Use it to slice the corpus by domain, or to design
/// cross-domain experiments.
///
/// See more information at <https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences>.
///
/// # Citation
///
/// Kotzias, D., Denil, M., de Freitas, N. & Smyth, P. (2015). "From Group to
/// Individual Labels using Deep Features," KDD. Sentiment Labelled Sentences
/// \[Dataset\]. UCI Machine Learning Repository. <https://doi.org/10.24432/C57604>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::SentimentSentences;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./sentiment_sentences";
///
/// let mut dataset = SentimentSentences::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 3000);
/// assert_eq!(table.n_columns(), 3);
///
/// // Reach one column by name.
/// let texts = table.column(SentimentSentences::FEATURE_NAMES[0]).unwrap().as_string().unwrap();
/// assert_eq!(texts.len(), 3000);
/// let sources = table.column("source").unwrap().as_string().unwrap();
/// assert_eq!(sources[0], "amazon");
/// let labels = table.column(SentimentSentences::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(labels[0], "negative");
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
/// assert_eq!(owned.n_samples(), 3000);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 3000);
/// ```
#[derive(Debug)]
pub struct SentimentSentences {
    dataset: Dataset<Table, DatasetError>,
}

impl SentimentSentences {
    /// The column the source designates as the model input.
    pub const FEATURE_NAMES: [&'static str; 1] = ["text"];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "label";

    /// Create a new SentimentSentences instance without loading data.
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
    /// - `Self` - a `SentimentSentences` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        SentimentSentences {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Sentiment Labelled Sentences dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        // Each source file holds only `sentence<TAB>label`. The filename is
        // the only place that names the site, so the loader prepends a
        // `source` column, producing `source<TAB>sentence<TAB>label` lines.
        // One pinned SHA-256 covers the combined file, cached as
        // `sentiment_sentences.csv`.
        let file_path = acquire_dataset(
            dir,
            SENTIMENT_SENTENCES_FILENAME,
            SENTIMENT_SENTENCES_DATASET_NAME,
            Some(SENTIMENT_SENTENCES_SHA256),
            |temp_path| {
                download_to_with_retries(
                    SENTIMENT_SENTENCES_DATA_URL,
                    temp_path,
                    Some(SENTIMENT_SENTENCES_ZIP_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                unzip(&temp_path.join(SENTIMENT_SENTENCES_ZIP_FILENAME), temp_path)?;

                // The three data files live inside a folder whose name contains
                // spaces. Writing explicit `\t`/`\n` bytes keeps the combined
                // file's SHA-256 stable across platforms.
                let src_dir = temp_path.join(SENTIMENT_SENTENCES_SUBDIR);
                let combined_path = temp_path.join(SENTIMENT_SENTENCES_FILENAME);
                let mut combined = File::create(&combined_path)?;
                for (source, filename) in SENTIMENT_SENTENCES_SOURCE_FILES {
                    let content = std::fs::read_to_string(src_dir.join(filename))?;
                    for line in content.lines() {
                        if line.trim().is_empty() {
                            continue;
                        }
                        combined.write_all(source.as_bytes())?;
                        combined.write_all(b"\t")?;
                        combined.write_all(line.as_bytes())?;
                        combined.write_all(b"\n")?;
                    }
                }
                combined.flush()?;

                Ok(combined_path)
            },
        )?;

        // The combined corpus is tab-separated `source<TAB>sentence<TAB>label`. The
        // sentences are free text that can contain `"` and `,` (but never a tab). As
        // in `SmsSpam`, the reader disables quote handling and splits every record
        // purely on tabs.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(false)
            .quoting(false)
            .from_reader(file);

        let mut texts: Vec<String> = Vec::with_capacity(N_SAMPLES);
        let mut sources: Vec<String> = Vec::with_capacity(N_SAMPLES);
        let mut labels: Vec<String> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record = result
                .map_err(|e| DatasetError::csv_read_error(SENTIMENT_SENTENCES_DATASET_NAME, e))?;
            let line_num = idx + 1;

            // Skip blank lines defensively (e.g. a trailing newline).
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    SENTIMENT_SENTENCES_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // Source site.
            let source = match &record[SOURCE_COLUMN] {
                "amazon" => "amazon",
                "imdb" => "imdb",
                "yelp" => "yelp",
                other => {
                    return Err(DatasetError::invalid_value(
                        SENTIMENT_SENTENCES_DATASET_NAME,
                        "source",
                        other,
                        line_num,
                    ));
                }
            };

            // Sentiment label, mapping the source code to a readable name
            // (`0` = negative, `1` = positive).
            let label = match &record[LABEL_COLUMN] {
                "0" => "negative",
                "1" => "positive",
                other => {
                    return Err(DatasetError::invalid_value(
                        SENTIMENT_SENTENCES_DATASET_NAME,
                        "label",
                        other,
                        line_num,
                    ));
                }
            };

            sources.push(source.to_string());
            labels.push(label.to_string());
            texts.push(record[SENTENCE_COLUMN].to_string());
        }

        Table::new(
            SENTIMENT_SENTENCES_DATASET_NAME,
            vec![
                Column::new(
                    Self::FEATURE_NAMES[0],
                    ColumnData::String(Array1::from_vec(texts)),
                ),
                Column::new("source", ColumnData::String(Array1::from_vec(sources))),
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
    /// - `&Table` - reference to the cached table of 3,000 samples and 3 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, or an invalid source
    ///   or label)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`SentimentSentences::data`], this method never runs the loader.
    /// If the data has not loaded yet, it returns `None` instead of downloading
    /// and parsing it.
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
    /// changes stay in the cache. Later calls to [`SentimentSentences::data`] or
    /// [`SentimentSentences::get_data`] see them.
    ///
    /// Like [`SentimentSentences::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`SentimentSentences::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 3,000 samples and 3 columns.
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
    /// - `Table` - the owned table of 3,000 samples and 3 columns.
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

impl_ml_dataset!(SentimentSentences, "sentiment_sentences");
