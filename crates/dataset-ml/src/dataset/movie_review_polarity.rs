//! Cornell Movie Review Polarity dataset (polarity dataset v2.0).
//!
//! Pang and Lee's classic sentiment-polarity benchmark holds 2,000 full movie
//! reviews from IMDb, split evenly into 1,000 `positive` and 1,000 `negative`
//! reviews. Like the other text loaders ([`SmsSpam`](crate::SmsSpam),
//! [`Newsgroups20`](crate::Newsgroups20)) it is a **text** dataset. So the
//! document accessor is
//! [`MovieReviewPolarity::texts`](crate::MovieReviewPolarity::texts) (an
//! `Array1<String>` of raw reviews), not `features()`. It complements the
//! sentence-level [`SentimentSentences`](crate::SentimentSentences) with
//! full-document reviews.
//!
//! **Documents:** `Array1<String>` of 2,000 movie reviews (already tokenized and
//! lowercased, one review per document)
//!
//! **Target:** `label`, one of `positive` or `negative`
//!
//! **Samples:** 2,000 (1,000 positive, 1,000 negative: a balanced split)
//! **Application:** Binary text classification / sentiment analysis
//!
//! **Source:** Cornell movie-review data (polarity dataset v2.0)
//! <http://www.cs.cornell.edu/people/pabo/movie-review-data/>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, untar_gz};
use ndarray::Array1;
use std::fs;
use std::path::Path;

/// Type alias for the Movie Review Polarity dataset: (review texts, labels).
type MovieReviewPolarityData = (Array1<String>, Array1<&'static str>);

/// The URL for the polarity dataset v2.0 (a gzip-compressed tarball).
const MOVIE_REVIEW_POLARITY_DATA_URL: &str =
    "http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz";

/// The name of the cached archive. The code caches the `.tar.gz` file as-is, uses
/// its SHA-256 as the integrity check, and re-extracts it in memory on load.
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

/// The class subdirectories paired with their `&'static str` labels, in the
/// fixed (lexicographic) order the loader walks them.
const CLASS_DIRS: [(&str, &str); 2] = [("neg", "negative"), ("pos", "positive")];

/// A struct that represents the Movie Review Polarity dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The polarity dataset v2.0 (Pang and Lee, 2004) collects 2,000 movie reviews
/// pulled from the IMDb archive, for document-level sentiment classification.
/// Of these, 1,000 have an overall positive rating and 1,000 have an overall
/// negative rating. The dataset provides the reviews pre-tokenized and
/// lowercased (the `txt_sentoken` form, one sentence per line). It is one of
/// the most widely cited sentiment benchmarks.
///
/// # Documents
///
/// Unlike the tabular loaders, there is no feature matrix. Each sample is a raw
/// review string. [`MovieReviewPolarity::texts`] returns a `(2000,)`
/// `Array1<String>` of the reviews (the whole tokenized document, newlines
/// included). Vectorize the reviews yourself (bag-of-words, TF-IDF, embeddings,
/// and so on) before you use them as model input.
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
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::MovieReviewPolarity;
///
/// let download_dir = "./movie_review_polarity"; // the loader creates the directory if it does not exist
///
/// let mut dataset = MovieReviewPolarity::new(download_dir);
/// let texts = dataset.texts().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (texts, labels) = dataset.data().unwrap();
/// assert_eq!(texts.len(), 2000);
/// assert_eq!(labels.len(), 2000);
///
/// // `get_data()` borrows the cached arrays without a reload. `get_data_mut()`
/// // edits the arrays in place. This needs no clone and no reload. The change
/// // stays cached. If you only need to change values, prefer this method over
/// // `.to_owned()`.
/// if let Some((texts, labels)) = dataset.get_data_mut() {
///     texts[0] = "hello world".to_string();
///     labels[0] = "positive";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out with no `to_owned()` clone. This
/// // leaves the instance reusable. The next access reloads the data from the
/// // cached file.
/// let (owned_texts, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_texts.len(), 2000);
/// assert_eq!(owned_labels.len(), 2000);
///
/// // `into_data()` also returns the owned arrays with no clone, but it
/// // consumes the instance. If you are done with the dataset, use it.
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
    /// The dataset loads lazily, on your first call to a data accessor method.
    /// This is a lightweight operation that only stores the storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - The directory that stores the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - a `MovieReviewPolarity` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        MovieReviewPolarity {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Movie Review Polarity dataset.
    fn load_data(dir: &str) -> Result<MovieReviewPolarityData, DatasetError> {
        // The loader caches the compressed tarball as-is, and uses its SHA-256
        // hash as the integrity check. Like `Newsgroups20`, the reviews are
        // multi-line raw documents. Instead of re-serializing them, the code
        // keeps the canonical archive and re-extracts it in memory on load.
        let archive_path = acquire_dataset(
            dir,
            MOVIE_REVIEW_POLARITY_ARCHIVE_FILENAME,
            MOVIE_REVIEW_POLARITY_DATASET_NAME,
            Some(MOVIE_REVIEW_POLARITY_SHA256),
            |temp_path| {
                download_to_with_retries(
                    MOVIE_REVIEW_POLARITY_DATA_URL,
                    temp_path,
                    Some(MOVIE_REVIEW_POLARITY_ARCHIVE_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(MOVIE_REVIEW_POLARITY_ARCHIVE_FILENAME))
            },
        )?;

        // The code extracts the archive into a temp dir under `dir`. The temp
        // dir cleans up when it drops.
        let extract_dir = tempfile::Builder::new()
            .prefix("polarity-")
            .tempdir_in(dir)?;
        untar_gz(&archive_path, extract_dir.path())?;

        let data_root = extract_dir.path().join(DATA_SUBDIR);
        let mut texts: Vec<String> = Vec::with_capacity(N_SAMPLES);
        let mut labels: Vec<&'static str> = Vec::with_capacity(N_SAMPLES);

        // The code walks `neg`, then `pos`, and walks files within each folder in
        // lexicographic order. This makes the sample order deterministic.
        for (folder, label) in CLASS_DIRS {
            let class_path = data_root.join(folder);
            for file_name in sorted_file_names(&class_path)? {
                let bytes = fs::read(class_path.join(&file_name))?;
                // The code decodes each byte as Latin-1 (byte -> Unicode scalar), like
                // scikit-learn's text loaders. This preserves non-UTF-8 bytes losslessly.
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
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// This method is the Movie Review Polarity equivalent of the tabular loaders'
    /// `features()`. The data is text, so the "features" are the raw review
    /// strings. This method returns a 1-D `Array1<String>`, not a 2-D feature
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
    /// - File extraction or I/O operations fail
    /// - Dataset size does not match the expected dimensions (2,000 samples)
    pub fn texts(&self) -> Result<&Array1<String>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the label vector.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array1<&'static str>` - Reference to label vector with shape `(2000,)` containing `"positive"` or `"negative"`
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Dataset size does not match the expected dimensions (2,000 samples)
    pub fn labels(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both review texts and labels as references.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
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
    /// - File extraction or I/O operations fail
    /// - Dataset size does not match the expected dimensions (2,000 samples)
    pub fn data(&self) -> Result<&MovieReviewPolarityData, DatasetError> {
        self.dataset.load()
    }

    /// Get both review texts and labels as references **without** triggering loading.
    ///
    /// Unlike [`MovieReviewPolarity::data`], which loads the dataset on first call,
    /// this method never runs the loader. If the data has not loaded yet, this
    /// method returns `None` instead of downloading and parsing it. Use this
    /// method when you want the data only if it is already cached. This avoids
    /// the download and parse cost when the data is not yet cached.
    ///
    /// # Returns
    ///
    /// - `Some(&MovieReviewPolarityData)` - reference to the cached `(texts,
    ///   labels)` tuple (`(2000,)`, `(2000,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&MovieReviewPolarityData> {
        self.dataset.get()
    }

    /// Get mutable references to review texts and labels for **in-place** editing.
    ///
    /// This lets you change the cached arrays directly, for example to normalize or
    /// clean the review text. It needs no `to_owned()` clone, and it does not
    /// remove the arrays from the cache. The changes persist, so later calls to
    /// [`MovieReviewPolarity::texts`], [`MovieReviewPolarity::data`], or
    /// [`MovieReviewPolarity::get_data`] observe them.
    ///
    /// Like [`MovieReviewPolarity::get_data`], this method does not trigger
    /// loading. It returns `None` if the dataset has not loaded yet. If you need
    /// the data to be present, call a loading accessor first, for example
    /// [`MovieReviewPolarity::data`].
    ///
    /// # Returns
    ///
    /// - `Some(&mut MovieReviewPolarityData)` - mutable reference to the cached
    ///   `(texts, labels)` tuple (`(2000,)`, `(2000,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut MovieReviewPolarityData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** review texts and labels.
    ///
    /// Unlike [`MovieReviewPolarity::data`], which borrows the cached data, this
    /// method moves the data out and returns owned arrays directly. It needs no
    /// `to_owned()` clone. If the dataset has not loaded yet, it loads on first
    /// access.
    ///
    /// This method consumes `self`, so you cannot use the instance afterward. If
    /// you want owned data but need to keep using the instance, use
    /// [`MovieReviewPolarity::take_data`] instead. It takes `&mut self` and leaves
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

    /// Take **owned** review texts and labels out of the dataset. This leaves the
    /// instance reusable.
    ///
    /// Like [`MovieReviewPolarity::into_data`], this method returns owned arrays
    /// with no `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out. This resets the instance to its
    /// unloaded state. The next accessor call, for example
    /// [`MovieReviewPolarity::texts`] or [`MovieReviewPolarity::data`], loads the
    /// dataset again.
    ///
    /// If you are done with the instance, use [`MovieReviewPolarity::into_data`]
    /// instead.
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

/// List a directory's regular-file children in lexicographic order.
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

impl_ml_dataset!(
    MovieReviewPolarity,
    MovieReviewPolarityData,
    "movie_review_polarity"
);
