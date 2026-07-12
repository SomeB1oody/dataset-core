//! Sentiment Labelled Sentences dataset.
//!
//! A set of 3,000 short sentences drawn from three product/service review sites
//! (Amazon, IMDb, Yelp), each hand-labelled with a binary sentiment (`positive`
//! or `negative`). Like [`SmsSpam`](crate::sms_spam::SmsSpam) and
//! [`YoutubeSpam`](crate::youtube_spam::YoutubeSpam) this is a **text** dataset,
//! so the document accessor is [`SentimentSentences::texts`] (an
//! `Array1<String>` of raw sentences), not `features()`. Unlike those two, it
//! also carries a piece of per-sample **metadata** — which site the sentence came
//! from — exposed via [`SentimentSentences::sources`].
//!
//! **Documents:** `Array1<String>` of 3,000 raw review sentences
//!
//! **Metadata:** `source` — one of `amazon`, `imdb`, or `yelp`
//!
//! **Target:** `label` — one of `positive` or `negative`
//!
//! **Samples:** 3,000 (1,500 positive, 1,500 negative; 1,000 per site, balanced)
//! **Application:** Binary text classification / sentiment analysis
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C57604>

use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to, unzip};
use ndarray::Array1;
use std::fs::File;
use std::io::Write as _;

/// Type alias for the Sentiment Labelled Sentences dataset:
/// (sentence texts, source sites, sentiment labels).
type SentimentSentencesData = (Array1<String>, Array1<&'static str>, Array1<&'static str>);

/// The URL for the Sentiment Labelled Sentences dataset (a ZIP archive).
const SENTIMENT_SENTENCES_DATA_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip";

/// The name of the downloaded ZIP archive (inside the temp dir).
const SENTIMENT_SENTENCES_ZIP_FILENAME: &str = "sentiment_labelled_sentences.zip";

/// The name of the folder inside the ZIP archive that holds the data files
/// (note the spaces in the upstream directory name).
const SENTIMENT_SENTENCES_SUBDIR: &str = "sentiment labelled sentences";

/// The three per-site source files inside the archive, paired with the `source`
/// label they are tagged with, in the fixed order they are combined.
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

/// A struct representing the Sentiment Labelled Sentences dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The Sentiment Labelled Sentences dataset (Kotzias, Denil, de Freitas & Smyth,
/// 2015) collects 3,000 sentences taken from reviews on three sites — Amazon
/// product reviews, IMDb movie reviews, and Yelp restaurant reviews — 1,000 from
/// each. Every sentence was hand-labelled with a clearly `positive` or `negative`
/// sentiment (500 of each per site, so the corpus is perfectly balanced). It is a
/// compact benchmark for sentence-level sentiment classification and for studying
/// cross-domain transfer between the three sources.
///
/// # Documents
///
/// Unlike the tabular loaders, there is no feature matrix: each sample is a raw
/// sentence string. [`SentimentSentences::texts`] returns a `(3000,)`
/// `Array1<String>` of the sentences — vectorize them (bag-of-words, TF-IDF,
/// embeddings, …) yourself before feeding a model.
///
/// # Metadata
///
/// - `source` (shape `(3000,)`): [`SentimentSentences::sources`] returns an
///   `Array1<&'static str>`, each one of `"amazon"`, `"imdb"`, or `"yelp"` — the
///   review site the sentence came from. Use it to slice the corpus by domain or
///   set up cross-domain experiments.
///
/// # Labels
///
/// - `label` (shape `(3000,)`): the `Array1<&'static str>` is one of `"positive"`
///   (the source label `1`) or `"negative"` (the source label `0`).
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
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::sentiment_sentences::SentimentSentences;
///
/// let download_dir = "./sentiment_sentences"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = SentimentSentences::new(download_dir);
/// let texts = dataset.texts().unwrap();
/// let sources = dataset.sources().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (texts, sources, labels) = dataset.data().unwrap(); // also gets all three at once
/// assert_eq!(texts.len(), 3000);
/// assert_eq!(sources.len(), 3000);
/// assert_eq!(labels.len(), 3000);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((texts, _sources, labels)) = dataset.get_data_mut() {
///     texts[0] = "hello world".to_string();
///     labels[0] = "positive";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out (no `to_owned()` clone) and leaves
/// // the instance reusable — the next access reloads from the cached file.
/// let (owned_texts, owned_sources, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_texts.len(), 3000);
/// assert_eq!(owned_sources.len(), 3000);
/// assert_eq!(owned_labels.len(), 3000);
///
/// // `into_data()` also returns the owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_texts, owned_sources, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_texts.len(), 3000);
/// assert_eq!(owned_sources.len(), 3000);
/// assert_eq!(owned_labels.len(), 3000);
/// ```
#[derive(Debug)]
pub struct SentimentSentences {
    dataset: Dataset<SentimentSentencesData, DatasetError>,
}

impl SentimentSentences {
    /// Create a new SentimentSentences instance without loading data.
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
    /// - `Self` - `SentimentSentences` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        SentimentSentences {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Sentiment Labelled Sentences dataset.
    fn load_data(dir: &str) -> Result<SentimentSentencesData, DatasetError> {
        // Prepare the dataset file: download the ZIP, extract it, and combine the
        // three per-site files into one corpus. Each source file is only
        // `sentence<TAB>label`, so the site name (which is otherwise implied only
        // by the filename) is prepended as a `source` column, giving
        // `source<TAB>sentence<TAB>label` lines. Combining them under one pinned
        // SHA-256 covers the whole dataset (cached as `sentiment_sentences.csv`).
        let file_path = acquire_dataset(
            dir,
            SENTIMENT_SENTENCES_FILENAME,
            SENTIMENT_SENTENCES_DATASET_NAME,
            Some(SENTIMENT_SENTENCES_SHA256),
            |temp_path| {
                download_to(
                    SENTIMENT_SENTENCES_DATA_URL,
                    temp_path,
                    Some(SENTIMENT_SENTENCES_ZIP_FILENAME),
                )?;
                unzip(&temp_path.join(SENTIMENT_SENTENCES_ZIP_FILENAME), temp_path)?;

                // The three data files live inside a folder whose name contains
                // spaces. Read each in a fixed order and prefix every line with
                // its source; writing explicit `\t`/`\n` bytes keeps the combined
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

        // The combined corpus is tab-separated `source<TAB>sentence<TAB>label`.
        // The sentences are free text that can contain `"` and `,` (but never a
        // tab), so — as in `SmsSpam` — quote handling is disabled and every record
        // is split purely on tabs.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(false)
            .quoting(false)
            .from_reader(file);

        let mut texts: Vec<String> = Vec::with_capacity(N_SAMPLES);
        let mut sources: Vec<&'static str> = Vec::with_capacity(N_SAMPLES);
        let mut labels: Vec<&'static str> = Vec::with_capacity(N_SAMPLES);

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

            // Source site, mapped to a `&'static str`.
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

            // Sentiment label, mapping the source code to a readable `&'static str`
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

            sources.push(source);
            labels.push(label);
            texts.push(record[SENTENCE_COLUMN].to_string());
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(
                SENTIMENT_SENTENCES_DATASET_NAME,
            ));
        }

        let texts_array = Array1::from_vec(texts);
        let sources_array = Array1::from_vec(sources);
        let labels_array = Array1::from_vec(labels);

        Ok((texts_array, sources_array, labels_array))
    }

    /// Get a reference to the sentence-text vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// This is the Sentiment Labelled Sentences analogue of the tabular loaders'
    /// `features()`: because the data is text, the "features" are the raw sentence
    /// strings, so this returns a 1-D `Array1<String>` rather than a 2-D feature
    /// matrix.
    ///
    /// # Returns
    ///
    /// - `&Array1<String>` - Reference to the sentence-text vector with shape
    ///   `(3000,)`, each entry a raw review sentence.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, or invalid source/label)
    /// - Dataset size doesn't match expected dimensions (3,000 samples)
    pub fn texts(&self) -> Result<&Array1<String>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the source-site vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// This is metadata unique to this loader (the other text loaders have no such
    /// column): each entry records which review site the sentence came from.
    ///
    /// # Returns
    ///
    /// - `&Array1<&'static str>` - Reference to the source-site vector with shape
    ///   `(3000,)`, each entry one of `"amazon"`, `"imdb"`, or `"yelp"`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, or invalid source/label)
    /// - Dataset size doesn't match expected dimensions (3,000 samples)
    pub fn sources(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get a reference to the labels vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<&'static str>` - Reference to labels vector with shape `(3000,)` containing `"positive"` or `"negative"`
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, or invalid source/label)
    /// - Dataset size doesn't match expected dimensions (3,000 samples)
    pub fn labels(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.2)
    }

    /// Get sentence texts, source sites, and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&SentimentSentencesData` - reference to the cached `(texts, sources,
    ///   labels)` triple: the sentence-text vector `(3000,)`, the source-site
    ///   vector `(3000,)`, and the label vector `(3000,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, or invalid source/label)
    /// - Dataset size doesn't match expected dimensions (3,000 samples)
    pub fn data(&self) -> Result<&SentimentSentencesData, DatasetError> {
        self.dataset.load()
    }

    /// Get texts, sources, and labels as references **without** triggering loading.
    ///
    /// Unlike [`SentimentSentences::data`], which loads the dataset on first call,
    /// this never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&SentimentSentencesData)` - reference to the cached `(texts,
    ///   sources, labels)` triple (`(3000,)` each), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&SentimentSentencesData> {
        self.dataset.get()
    }

    /// Get mutable references to texts, sources, and labels for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize or clean the
    /// sentence text) with no `to_owned()` clone and without removing them from the
    /// cache: the changes persist, so later [`SentimentSentences::texts`],
    /// [`SentimentSentences::data`], or [`SentimentSentences::get_data`] calls
    /// observe them.
    ///
    /// Like [`SentimentSentences::get_data`], this does **not** trigger loading: it
    /// returns `None` if the dataset has not been loaded. Call a loading accessor
    /// (e.g. [`SentimentSentences::data`]) first if you need to ensure the data is
    /// present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut SentimentSentencesData)` - mutable reference to the cached
    ///   `(texts, sources, labels)` triple (`(3000,)` each), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut SentimentSentencesData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** texts, sources, and labels.
    ///
    /// Unlike [`SentimentSentences::data`], which borrows the cached data, this
    /// moves it out and returns owned arrays directly — no `to_owned()` clone
    /// needed. The dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`SentimentSentences::take_data`] instead — it takes `&mut self` and leaves
    /// the instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array1<&'static str>, Array1<&'static str>)` - owned
    ///   sentence-text vector `(3000,)`, source-site vector `(3000,)`, and label
    ///   vector `(3000,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file extraction, I/O,
    /// parsing, invalid source/label, or a dimension mismatch).
    pub fn into_data(self) -> Result<SentimentSentencesData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** texts, sources, and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`SentimentSentences::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out, resetting the instance to its
    /// unloaded state: the next accessor call (e.g. [`SentimentSentences::texts`]
    /// or [`SentimentSentences::data`]) loads the dataset again.
    ///
    /// Use [`SentimentSentences::into_data`] instead if you are done with the
    /// instance.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array1<&'static str>, Array1<&'static str>)` - owned
    ///   sentence-text vector `(3000,)`, source-site vector `(3000,)`, and label
    ///   vector `(3000,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file extraction, I/O,
    /// parsing, invalid source/label, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<SentimentSentencesData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}
