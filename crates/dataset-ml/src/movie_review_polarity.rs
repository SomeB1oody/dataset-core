//! Cornell Movie Review Polarity dataset (polarity dataset v2.0).
//!
//! Pang & Lee's classic sentiment-polarity benchmark: 2,000 full movie reviews
//! from IMDb, split evenly into 1,000 `positive` and 1,000 `negative` reviews.
//! Like the other text loaders ([`SmsSpam`](crate::sms_spam::SmsSpam),
//! [`Newsgroups20`](crate::newsgroups20::Newsgroups20)) it is a **text** dataset,
//! so the document accessor is [`MovieReviewPolarity::texts`] (an
//! `Array1<String>` of raw reviews), not `features()`. It complements the
//! sentence-level [`SentimentSentences`](crate::sentiment_sentences::SentimentSentences)
//! with full-document reviews.
//!
//! **Documents:** `Array1<String>` of 2,000 movie reviews (already tokenized and
//! lowercased, one review per document)
//!
//! **Target:** `label` — one of `positive` or `negative`
//!
//! **Samples:** 2,000 (1,000 positive, 1,000 negative; balanced)
//! **Application:** Binary text classification / sentiment analysis
//!
//! **Source:** Cornell movie-review data (polarity dataset v2.0)
//! <http://www.cs.cornell.edu/people/pabo/movie-review-data/>

use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to, untar_gz};
use ndarray::Array1;
use std::fs;
use std::path::Path;

/// Type alias for the Movie Review Polarity dataset: (review texts, labels).
type MovieReviewPolarityData = (Array1<String>, Array1<&'static str>);

/// The URL for the polarity dataset v2.0 (a gzip-compressed tarball).
const MOVIE_REVIEW_POLARITY_DATA_URL: &str =
    "http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz";

/// The name of the cached archive (the `.tar.gz` is cached as-is; its SHA-256 is
/// the integrity check, and it is re-extracted in memory on load).
const MOVIE_REVIEW_POLARITY_ARCHIVE_FILENAME: &str = "review_polarity.tar.gz";

/// The SHA256 hash of the cached `review_polarity.tar.gz` archive.
const MOVIE_REVIEW_POLARITY_SHA256: &str =
    "fc0dccc2671af5db3c5d8f81f77a1ebfec953ecdd422334062df61ede36b2179";

/// The name of the dataset.
const MOVIE_REVIEW_POLARITY_DATASET_NAME: &str = "movie_review_polarity";

/// The folder inside the archive holding the tokenized reviews (`pos`/`neg` subdirs).
const DATA_SUBDIR: &str = "txt_sentoken";

/// Number of samples.
const N_SAMPLES: usize = 2_000;

/// The class subdirectories paired with their `&'static str` labels, in the fixed
/// (lexicographic) order they are walked.
const CLASS_DIRS: [(&str, &str); 2] = [("neg", "negative"), ("pos", "positive")];

/// A struct representing the Movie Review Polarity dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The polarity dataset v2.0 (Pang & Lee, 2004) collects 2,000 movie reviews
/// pulled from the IMDb archive — 1,000 with an overall positive rating and 1,000
/// with an overall negative rating — for document-level sentiment classification.
/// The reviews are distributed pre-tokenized and lowercased (the `txt_sentoken`
/// form, one sentence per line). It is one of the most widely cited sentiment
/// benchmarks.
///
/// # Documents
///
/// Unlike the tabular loaders, there is no feature matrix: each sample is a raw
/// review string. [`MovieReviewPolarity::texts`] returns a `(2000,)`
/// `Array1<String>` of the reviews (the whole tokenized document, newlines
/// included) — vectorize them (bag-of-words, TF-IDF, embeddings, …) yourself
/// before feeding a model.
///
/// # Labels
///
/// - `label` (shape `(2000,)`): the `Array1<&'static str>` is one of `"positive"`
///   (from the `pos` folder) or `"negative"` (from the `neg` folder).
///
/// See more information at <http://www.cs.cornell.edu/people/pabo/movie-review-data/>.
///
/// # Citation
///
/// Pang, B. & Lee, L. (2004). "A Sentimental Education: Sentiment Analysis Using
/// Subjectivity Summarization Based on Minimum Cuts," ACL. Polarity dataset v2.0,
/// <http://www.cs.cornell.edu/people/pabo/movie-review-data/>.
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::movie_review_polarity::MovieReviewPolarity;
///
/// let download_dir = "./movie_review_polarity"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = MovieReviewPolarity::new(download_dir);
/// let texts = dataset.texts().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (texts, labels) = dataset.data().unwrap(); // this is also a way to get texts and labels
/// assert_eq!(texts.len(), 2000);
/// assert_eq!(labels.len(), 2000);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((texts, labels)) = dataset.get_data_mut() {
///     texts[0] = "hello world".to_string();
///     labels[0] = "positive";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out (no `to_owned()` clone) and leaves
/// // the instance reusable — the next access reloads from the cached archive.
/// let (owned_texts, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_texts.len(), 2000);
/// assert_eq!(owned_labels.len(), 2000);
///
/// // `into_data()` also returns the owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_texts, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_texts.len(), 2000);
/// assert_eq!(owned_labels.len(), 2000);
/// ```
#[derive(Debug)]
pub struct MovieReviewPolarity {
    dataset: Dataset<MovieReviewPolarityData, DatasetError>,
}

impl MovieReviewPolarity {
    /// Create a new MovieReviewPolarity instance without loading data.
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
    /// - `Self` - `MovieReviewPolarity` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        MovieReviewPolarity {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Movie Review Polarity dataset.
    fn load_data(dir: &str) -> Result<MovieReviewPolarityData, DatasetError> {
        // Cache the compressed tarball as-is (its SHA-256 is the integrity check).
        // Like `Newsgroups20`, the reviews are multi-line raw documents, so rather
        // than re-serialize them we keep the canonical archive and re-extract it in
        // memory on load.
        let archive_path = acquire_dataset(
            dir,
            MOVIE_REVIEW_POLARITY_ARCHIVE_FILENAME,
            MOVIE_REVIEW_POLARITY_DATASET_NAME,
            Some(MOVIE_REVIEW_POLARITY_SHA256),
            |temp_path| {
                download_to(
                    MOVIE_REVIEW_POLARITY_DATA_URL,
                    temp_path,
                    Some(MOVIE_REVIEW_POLARITY_ARCHIVE_FILENAME),
                )?;
                Ok(temp_path.join(MOVIE_REVIEW_POLARITY_ARCHIVE_FILENAME))
            },
        )?;

        // Extract into a temp dir under `dir` that is cleaned up when it drops.
        let extract_dir = tempfile::Builder::new()
            .prefix("polarity-")
            .tempdir_in(dir)?;
        untar_gz(&archive_path, extract_dir.path())?;

        let data_root = extract_dir.path().join(DATA_SUBDIR);
        let mut texts: Vec<String> = Vec::with_capacity(N_SAMPLES);
        let mut labels: Vec<&'static str> = Vec::with_capacity(N_SAMPLES);

        // Walk `neg` then `pos`, files in lexicographic order, so the sample
        // ordering is deterministic.
        for (folder, label) in CLASS_DIRS {
            let class_path = data_root.join(folder);
            for file_name in sorted_file_names(&class_path)? {
                let bytes = fs::read(class_path.join(&file_name))?;
                // Decode as Latin-1 (byte -> Unicode scalar), like scikit-learn's
                // text loaders, so any non-UTF-8 byte is preserved losslessly.
                let text: String = bytes.iter().map(|&b| b as char).collect();
                texts.push(text);
                labels.push(label);
            }
        }

        if texts.is_empty() {
            return Err(DatasetError::empty_dataset(
                MOVIE_REVIEW_POLARITY_DATASET_NAME,
            ));
        }

        Ok((Array1::from_vec(texts), Array1::from_vec(labels)))
    }

    /// Get a reference to the review-text vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// This is the Movie Review Polarity analogue of the tabular loaders'
    /// `features()`: because the data is text, the "features" are the raw review
    /// strings, so this returns a 1-D `Array1<String>` rather than a 2-D feature
    /// matrix.
    ///
    /// # Returns
    ///
    /// - `&Array1<String>` - Reference to the review-text vector with shape
    ///   `(2000,)`, each entry a tokenized movie review.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - Archive extraction or I/O operations fail
    /// - Dataset size doesn't match expected dimensions (2,000 samples)
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
    /// - `&Array1<&'static str>` - Reference to labels vector with shape `(2000,)` containing `"positive"` or `"negative"`
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - Archive extraction or I/O operations fail
    /// - Dataset size doesn't match expected dimensions (2,000 samples)
    pub fn labels(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both review texts and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&MovieReviewPolarityData` - reference to the cached `(texts, labels)`
    ///   tuple: the review-text vector `(2000,)` and the label vector `(2000,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - Archive extraction or I/O operations fail
    /// - Dataset size doesn't match expected dimensions (2,000 samples)
    pub fn data(&self) -> Result<&MovieReviewPolarityData, DatasetError> {
        self.dataset.load()
    }

    /// Get both review texts and labels as references **without** triggering loading.
    ///
    /// Unlike [`MovieReviewPolarity::data`], which loads the dataset on first call,
    /// this never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&MovieReviewPolarityData)` - reference to the cached `(texts,
    ///   labels)` tuple (`(2000,)`, `(2000,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&MovieReviewPolarityData> {
        self.dataset.get()
    }

    /// Get mutable references to review texts and labels for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize or clean the
    /// review text) with no `to_owned()` clone and without removing them from the
    /// cache: the changes persist, so later [`MovieReviewPolarity::texts`],
    /// [`MovieReviewPolarity::data`], or [`MovieReviewPolarity::get_data`] calls
    /// observe them.
    ///
    /// Like [`MovieReviewPolarity::get_data`], this does **not** trigger loading: it
    /// returns `None` if the dataset has not been loaded. Call a loading accessor
    /// (e.g. [`MovieReviewPolarity::data`]) first if you need to ensure the data is
    /// present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut MovieReviewPolarityData)` - mutable reference to the cached
    ///   `(texts, labels)` tuple (`(2000,)`, `(2000,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut MovieReviewPolarityData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** review texts and labels.
    ///
    /// Unlike [`MovieReviewPolarity::data`], which borrows the cached data, this
    /// moves it out and returns owned arrays directly — no `to_owned()` clone
    /// needed. The dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`MovieReviewPolarity::take_data`] instead — it takes `&mut self` and leaves
    /// the instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array1<&'static str>)` - owned review-text vector
    ///   `(2000,)` and owned label vector `(2000,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, archive extraction, I/O,
    /// or a dimension mismatch).
    pub fn into_data(self) -> Result<MovieReviewPolarityData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** review texts and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`MovieReviewPolarity::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out, resetting the instance to its
    /// unloaded state: the next accessor call (e.g. [`MovieReviewPolarity::texts`]
    /// or [`MovieReviewPolarity::data`]) loads the dataset again.
    ///
    /// Use [`MovieReviewPolarity::into_data`] instead if you are done with the
    /// instance.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array1<&'static str>)` - owned review-text vector
    ///   `(2000,)` and owned label vector `(2000,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, archive extraction, I/O,
    /// or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<MovieReviewPolarityData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

/// List a directory's regular-file children, sorted lexicographically.
fn sorted_file_names(path: &Path) -> Result<Vec<String>, DatasetError> {
    let mut names: Vec<String> = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_file()
            && let Some(name) = entry.file_name().to_str()
        {
            names.push(name.to_string());
        }
    }
    names.sort();
    Ok(names)
}
