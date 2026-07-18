//! 20 Newsgroups dataset.
//!
//! The classic **20 Newsgroups** text-classification benchmark: roughly 18,846
//! Usenet posts partitioned (nearly evenly) across 20 different newsgroups. It is
//! the multi-class counterpart to the binary text loaders
//! ([`SmsSpam`](crate::sms_spam::SmsSpam),
//! [`YoutubeSpam`](crate::youtube_spam::YoutubeSpam),
//! [`SentimentSentences`](crate::sentiment_sentences::SentimentSentences),
//! [`MovieReviewPolarity`](crate::movie_review_polarity::MovieReviewPolarity)) and
//! the framework-agnostic
//! analogue of scikit-learn's `fetch_20newsgroups`. Like those loaders it is a
//! **text** dataset, so the document accessor is [`Newsgroups20::texts`] (an
//! `Array1<String>` of raw posts), not `features()`.
//!
//! **Documents:** `Array1<String>` of raw newsgroup posts (full text, including
//! the email-style headers — nothing is stripped)
//!
//! **Target:** `label` — one of the 20 newsgroup names (e.g. `sci.space`)
//!
//! **Samples:** 18,846 total — 11,314 train / 7,532 test (the standard "bydate"
//! chronological split)
//! **Application:** Multi-class text classification (20 classes)
//!
//! **Source:** Jason Rennie's 20 Newsgroups page (the `bydate` tarball, the same
//! one scikit-learn downloads) <http://qwone.com/~jason/20Newsgroups/>

use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to, untar_gz};
use ndarray::Array1;
use std::fs;
use std::path::Path;

/// Type alias for the 20 Newsgroups dataset: (document texts, newsgroup labels).
type Newsgroups20Data = (Array1<String>, Array1<&'static str>);

/// The URL for the 20 Newsgroups dataset (the `bydate` gzip-compressed tarball).
const NEWSGROUPS20_DATA_URL: &str = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz";

/// The name of the cached archive (the `.tar.gz` is cached as-is; its SHA-256 is
/// the integrity check, and it is re-extracted in memory on load).
const NEWSGROUPS20_ARCHIVE_FILENAME: &str = "20news-bydate.tar.gz";

/// The SHA256 hash of the cached `20news-bydate.tar.gz` archive.
const NEWSGROUPS20_SHA256: &str =
    "8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610";

/// The name of the dataset.
const NEWSGROUPS20_DATASET_NAME: &str = "newsgroups20";

/// The top-level folder holding the training partition inside the archive.
const TRAIN_DIR: &str = "20news-bydate-train";

/// The top-level folder holding the test partition inside the archive.
const TEST_DIR: &str = "20news-bydate-test";

/// Subset selector: the training partition (11,314 posts), scikit-learn's default.
const SUBSET_TRAIN: &[&str] = &[TRAIN_DIR];

/// Subset selector: the test partition (7,532 posts).
const SUBSET_TEST: &[&str] = &[TEST_DIR];

/// Subset selector: the full dataset (18,846 posts, train followed by test).
const SUBSET_ALL: &[&str] = &[TRAIN_DIR, TEST_DIR];

/// The 20 newsgroup category names (the per-class subdirectory names). Every
/// post's label is one of these `&'static str`s.
const CATEGORIES: [&str; 20] = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
];

/// Map a category subdirectory name to its `&'static str` label, or `None` if it
/// is not one of the 20 known newsgroups.
fn category_label(name: &str) -> Option<&'static str> {
    CATEGORIES.iter().copied().find(|&c| c == name)
}

/// A struct representing the 20 Newsgroups dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The 20 Newsgroups dataset is a collection of ~18,846 Usenet posts, partitioned
/// (nearly evenly) across 20 newsgroups on distinct topics — from `sci.space` and
/// `comp.graphics` to `talk.politics.mideast` and `rec.sport.hockey`. It is one of
/// the most widely used benchmarks for text classification and clustering. This
/// loader uses the canonical **"bydate"** version, which sorts the posts by date
/// into a fixed train/test split (11,314 / 7,532) and removes duplicates and some
/// newsgroup-identifying headers — the exact tarball scikit-learn's
/// `fetch_20newsgroups` downloads.
///
/// # Subsets
///
/// Mirroring scikit-learn's `subset` argument, there are three constructors, all
/// sharing the same cached archive:
///
/// - [`Newsgroups20::new`] — the **train** partition (11,314 posts), the default.
/// - [`Newsgroups20::new_test`] — the **test** partition (7,532 posts).
/// - [`Newsgroups20::new_all`] — **all** 18,846 posts (train followed by test).
///
/// # Documents
///
/// Unlike the tabular loaders, there is no feature matrix: each sample is a raw
/// post string. [`Newsgroups20::texts`] returns an `Array1<String>` of the full
/// post text **including** the email-style headers (`From:`, `Subject:`, …) —
/// nothing is stripped, matching scikit-learn's default. The files are decoded as
/// Latin-1 (each byte maps to one Unicode scalar, as scikit-learn does), so any
/// non-UTF-8 bytes are preserved losslessly. Vectorize the text (bag-of-words,
/// TF-IDF, embeddings, …) yourself before feeding a model.
///
/// # Labels
///
/// - `label` (shape `(n_samples,)`): the `Array1<&'static str>` is one of the 20
///   newsgroup names (e.g. `"sci.space"`, `"alt.atheism"`).
///
/// See more information at <http://qwone.com/~jason/20Newsgroups/>.
///
/// # Citation
///
/// Lang, K. (1995). "NewsWeeder: Learning to Filter Netnews," ICML. Dataset
/// curated by Jason Rennie, <http://qwone.com/~jason/20Newsgroups/>.
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::newsgroups20::Newsgroups20;
///
/// let download_dir = "./newsgroups20"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = Newsgroups20::new(download_dir); // the train partition
/// let texts = dataset.texts().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (texts, labels) = dataset.data().unwrap(); // this is also a way to get texts and labels
/// assert_eq!(texts.len(), 11314);
/// assert_eq!(labels.len(), 11314);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((texts, labels)) = dataset.get_data_mut() {
///     texts[0] = "hello world".to_string();
///     labels[0] = "sci.space";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out (no `to_owned()` clone) and leaves
/// // the instance reusable — the next access reloads from the cached archive.
/// let (owned_texts, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_texts.len(), 11314);
/// assert_eq!(owned_labels.len(), 11314);
///
/// // `into_data()` also returns the owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_texts, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_texts.len(), 11314);
/// assert_eq!(owned_labels.len(), 11314);
/// ```
#[derive(Debug)]
pub struct Newsgroups20 {
    dataset: Dataset<Newsgroups20Data, DatasetError>,
}

impl Newsgroups20 {
    /// Create a new Newsgroups20 instance for the **train** partition (11,314
    /// posts) without loading data.
    ///
    /// This mirrors scikit-learn's default `subset="train"`. The dataset will be
    /// loaded lazily when you first call any data accessor method. This is a
    /// lightweight operation that only stores the storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory where the dataset will be stored.
    ///
    /// # Returns
    ///
    /// - `Self` - `Newsgroups20` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, SUBSET_TRAIN)
    }

    /// Create a new Newsgroups20 instance for the **test** partition (7,532
    /// posts) without loading data.
    ///
    /// This mirrors scikit-learn's `subset="test"`. See [`Newsgroups20::new`] for
    /// the loading semantics.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory where the dataset will be stored.
    ///
    /// # Returns
    ///
    /// - `Self` - `Newsgroups20` instance ready for lazy loading.
    pub fn new_test(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, SUBSET_TEST)
    }

    /// Create a new Newsgroups20 instance for **all** 18,846 posts (train
    /// followed by test) without loading data.
    ///
    /// This mirrors scikit-learn's `subset="all"`. See [`Newsgroups20::new`] for
    /// the loading semantics.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory where the dataset will be stored.
    ///
    /// # Returns
    ///
    /// - `Self` - `Newsgroups20` instance ready for lazy loading.
    pub fn new_all(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, SUBSET_ALL)
    }

    /// Construct an instance whose loader walks the given subset directories.
    fn with_subset(storage_dir: &str, subset_dirs: &'static [&'static str]) -> Self {
        Newsgroups20 {
            dataset: Dataset::new(storage_dir, move |dir| Self::load_data(dir, subset_dirs)),
        }
    }

    /// Acquire and parse the 20 Newsgroups dataset for the requested subset.
    fn load_data(
        dir: &str,
        subset_dirs: &'static [&'static str],
    ) -> Result<Newsgroups20Data, DatasetError> {
        // Cache the compressed tarball as-is (its SHA-256 is the integrity check).
        // Unlike the combined-file text loaders, the posts are multi-line raw
        // documents, so rather than re-serialize them we keep the canonical
        // archive and re-extract it in memory on load.
        let archive_path = acquire_dataset(
            dir,
            NEWSGROUPS20_ARCHIVE_FILENAME,
            NEWSGROUPS20_DATASET_NAME,
            Some(NEWSGROUPS20_SHA256),
            |temp_path| {
                download_to(
                    NEWSGROUPS20_DATA_URL,
                    temp_path,
                    Some(NEWSGROUPS20_ARCHIVE_FILENAME),
                )?;
                Ok(temp_path.join(NEWSGROUPS20_ARCHIVE_FILENAME))
            },
        )?;

        // Extract into a temp dir under `dir` that is cleaned up when it drops.
        let extract_dir = tempfile::Builder::new().prefix("20news-").tempdir_in(dir)?;
        untar_gz(&archive_path, extract_dir.path())?;

        let mut texts: Vec<String> = Vec::new();
        let mut labels: Vec<&'static str> = Vec::new();

        // Walk each requested partition, categories then files in a deterministic
        // (lexicographic) order so the sample ordering is stable.
        for subset in subset_dirs {
            let subset_path = extract_dir.path().join(subset);
            for category in sorted_child_names(&subset_path, /* dirs = */ true)? {
                let label = category_label(&category).ok_or_else(|| {
                    DatasetError::invalid_value(NEWSGROUPS20_DATASET_NAME, "category", &category, 0)
                })?;
                let category_path = subset_path.join(&category);
                for file_name in sorted_child_names(&category_path, /* dirs = */ false)? {
                    let bytes = fs::read(category_path.join(&file_name))?;
                    // Decode as Latin-1 (byte -> Unicode scalar), like scikit-learn,
                    // so non-UTF-8 bytes are preserved losslessly.
                    let text: String = bytes.iter().map(|&b| b as char).collect();
                    texts.push(text);
                    labels.push(label);
                }
            }
        }

        if texts.is_empty() {
            return Err(DatasetError::empty_dataset(NEWSGROUPS20_DATASET_NAME));
        }

        Ok((Array1::from_vec(texts), Array1::from_vec(labels)))
    }

    /// Get a reference to the document-text vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// This is the 20 Newsgroups analogue of the tabular loaders' `features()`:
    /// because the data is text, the "features" are the raw post strings, so this
    /// returns a 1-D `Array1<String>` rather than a 2-D feature matrix.
    ///
    /// # Returns
    ///
    /// - `&Array1<String>` - Reference to the document-text vector, each entry a
    ///   raw newsgroup post (length depends on the subset: 11,314 / 7,532 / 18,846).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - Archive extraction or I/O operations fail
    /// - Data format is invalid (an unexpected category directory)
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
    /// - `&Array1<&'static str>` - Reference to labels vector, each entry one of
    ///   the 20 newsgroup names (e.g. `"sci.space"`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - Archive extraction or I/O operations fail
    /// - Data format is invalid (an unexpected category directory)
    pub fn labels(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both document texts and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Newsgroups20Data` - reference to the cached `(texts, labels)` tuple.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - Archive extraction or I/O operations fail
    /// - Data format is invalid (an unexpected category directory)
    pub fn data(&self) -> Result<&Newsgroups20Data, DatasetError> {
        self.dataset.load()
    }

    /// Get both document texts and labels as references **without** triggering loading.
    ///
    /// Unlike [`Newsgroups20::data`], which loads the dataset on first call, this
    /// never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&Newsgroups20Data)` - reference to the cached `(texts, labels)`
    ///   tuple, if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&Newsgroups20Data> {
        self.dataset.get()
    }

    /// Get mutable references to document texts and labels for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. strip headers or clean
    /// the post text) with no `to_owned()` clone and without removing them from the
    /// cache: the changes persist, so later [`Newsgroups20::texts`],
    /// [`Newsgroups20::data`], or [`Newsgroups20::get_data`] calls observe them.
    ///
    /// Like [`Newsgroups20::get_data`], this does **not** trigger loading: it
    /// returns `None` if the dataset has not been loaded. Call a loading accessor
    /// (e.g. [`Newsgroups20::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut Newsgroups20Data)` - mutable reference to the cached `(texts,
    ///   labels)` tuple, if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut Newsgroups20Data> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** document texts and labels.
    ///
    /// Unlike [`Newsgroups20::data`], which borrows the cached data, this moves it
    /// out and returns owned arrays directly — no `to_owned()` clone needed. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`Newsgroups20::take_data`] instead — it takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array1<&'static str>)` - owned document-text vector and
    ///   owned label vector.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, archive extraction, I/O,
    /// or an unexpected category directory).
    pub fn into_data(self) -> Result<Newsgroups20Data, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** document texts and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`Newsgroups20::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out, resetting the instance to its
    /// unloaded state: the next accessor call (e.g. [`Newsgroups20::texts`] or
    /// [`Newsgroups20::data`]) loads the dataset again.
    ///
    /// Use [`Newsgroups20::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array1<String>, Array1<&'static str>)` - owned document-text vector and
    ///   owned label vector.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, archive extraction, I/O,
    /// or an unexpected category directory).
    pub fn take_data(&mut self) -> Result<Newsgroups20Data, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

/// List the names of a directory's children, keeping only directories (when
/// `dirs` is `true`) or only files (when `false`), sorted lexicographically.
fn sorted_child_names(path: &Path, dirs: bool) -> Result<Vec<String>, DatasetError> {
    let mut names: Vec<String> = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() == dirs
            && let Some(name) = entry.file_name().to_str()
        {
            names.push(name.to_string());
        }
    }
    names.sort();
    Ok(names)
}
