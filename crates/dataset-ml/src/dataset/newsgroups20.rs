//! 20 Newsgroups dataset.
//!
//! The **20 Newsgroups** text-classification benchmark holds about 18,846 Usenet
//! posts, split almost evenly across 20 newsgroups. Each sample is one whole
//! post, including the email-style headers. The loader strips nothing.
//! Vectorize the text yourself (bag-of-words, TF-IDF, embeddings, and so on)
//! before you use it as model input.
//!
//! **Columns (2):**
//!
//! | Name    | Type     | Description                                     |
//! |---------|----------|-------------------------------------------------|
//! | `text`  | `String` | one raw post, headers included                  |
//! | `label` | `String` | one of the 20 newsgroup names, e.g. `sci.space` |
//!
//! The source designates the post text as the input
//! ([`Newsgroups20::FEATURE_NAMES`](crate::Newsgroups20::FEATURE_NAMES)) and the newsgroup name as the label
//! ([`Newsgroups20::TARGET`](crate::Newsgroups20::TARGET)).
//!
//! **Samples:** 18,846 total: 11,314 train, 7,532 test (the standard "bydate"
//! chronological split)
//! **Application:** Multi-class text classification (20 classes)
//!
//! **Missing values:** none.
//!
//! **Source:** Jason Rennie's 20 Newsgroups page (the `bydate` tarball, the same
//! one scikit-learn downloads) <http://qwone.com/~jason/20Newsgroups/>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, untar_gz};
use ndarray::Array1;
use std::fs;
use std::path::Path;

/// The URL for the 20 Newsgroups dataset (the `bydate` gzip-compressed tarball).
const NEWSGROUPS20_DATA_URL: &str = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz";

/// The name of the cached archive. The loader caches the `.tar.gz` file as-is,
/// checks its SHA-256 hash for integrity, and re-extracts it in memory on load.
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
/// value of the `label` column is one of these names.
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

/// A struct that represents the 20 Newsgroups dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The 20 Newsgroups dataset has about 18,846 Usenet posts, split almost evenly
/// across 20 newsgroups on distinct topics. Topics range from `sci.space` and
/// `comp.graphics` to `talk.politics.mideast` and `rec.sport.hockey`. It is one of
/// the most widely used benchmarks for text classification and clustering. This
/// loader uses the canonical **"bydate"** version. This version sorts the posts by
/// date into a fixed train/test split (11,314 / 7,532). It also removes
/// duplicates and some newsgroup-identifying headers. This is the exact tarball
/// that scikit-learn's `fetch_20newsgroups` downloads.
///
/// # Subsets
///
/// This loader mirrors scikit-learn's `subset` argument with three constructors.
/// All three share the same cached archive:
///
/// - [`Newsgroups20::new`]: the **train** partition (11,314 posts), the default.
/// - [`Newsgroups20::new_test`]: the **test** partition (7,532 posts).
/// - [`Newsgroups20::new_all`]: **all** 18,846 posts (train followed by test).
///
/// # Columns
///
/// | Name    | Type     | Description                                     |
/// |---------|----------|-------------------------------------------------|
/// | `text`  | `String` | one raw post, headers included                  |
/// | `label` | `String` | one of the 20 newsgroup names, e.g. `sci.space` |
///
/// The source designates the post text as the input
/// ([`Newsgroups20::FEATURE_NAMES`]) and the newsgroup name as the label
/// ([`Newsgroups20::TARGET`]).
///
/// Missing values: none.
///
/// The `text` column holds the full post text, **including** the email-style
/// headers (`From:`, `Subject:`, …). This matches scikit-learn's default. The
/// loader decodes the files as Latin-1 (each byte maps to one Unicode scalar).
/// This is the same method scikit-learn uses, so non-UTF-8 bytes stay intact.
/// Vectorize the text (bag-of-words, TF-IDF, embeddings, …) yourself before you
/// feed it to a model.
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
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Newsgroups20;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./newsgroups20";
///
/// let mut dataset = Newsgroups20::new(download_dir); // the train partition
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 11314);
/// assert_eq!(table.n_columns(), 2);
///
/// // Reach one column by name.
/// let texts = table.column(Newsgroups20::FEATURE_NAMES[0]).unwrap().as_string().unwrap();
/// assert_eq!(texts.len(), 11314);
/// let labels = table.column(Newsgroups20::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(labels[0], "alt.atheism");
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
/// assert_eq!(owned.n_samples(), 11314);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 11314);
/// ```
#[derive(Debug)]
pub struct Newsgroups20 {
    dataset: Dataset<Table, DatasetError>,
}

impl Newsgroups20 {
    /// The column the source designates as the model input.
    pub const FEATURE_NAMES: [&'static str; 1] = ["text"];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "label";

    /// Create a new Newsgroups20 instance for the **train** partition (11,314
    /// posts) without loading data.
    ///
    /// This mirrors scikit-learn's default `subset="train"`. The dataset loads
    /// lazily, on your first call to a data accessor method. This is a
    /// lightweight operation that only stores the storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - The directory that stores the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - a `Newsgroups20` instance ready for lazy loading.
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
    /// - `storage_dir` - The directory that stores the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - a `Newsgroups20` instance ready for lazy loading.
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
    /// - `storage_dir` - The directory that stores the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - a `Newsgroups20` instance ready for lazy loading.
    pub fn new_all(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, SUBSET_ALL)
    }

    /// Construct an instance whose loader walks the given subset directories.
    fn with_subset(storage_dir: &str, subset_dirs: &'static [&'static str]) -> Self {
        Newsgroups20 {
            dataset: Dataset::new(storage_dir, move |dir| Self::load_data(dir, subset_dirs)),
        }
    }

    /// Get and parse the 20 Newsgroups dataset for the requested subset.
    fn load_data(dir: &str, subset_dirs: &'static [&'static str]) -> Result<Table, DatasetError> {
        // The loader caches the compressed tarball as-is. Its SHA-256 hash is
        // the integrity check.
        let archive_path = acquire_dataset(
            dir,
            NEWSGROUPS20_ARCHIVE_FILENAME,
            NEWSGROUPS20_DATASET_NAME,
            Some(NEWSGROUPS20_SHA256),
            |temp_path| {
                download_to_with_retries(
                    NEWSGROUPS20_DATA_URL,
                    temp_path,
                    Some(NEWSGROUPS20_ARCHIVE_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(NEWSGROUPS20_ARCHIVE_FILENAME))
            },
        )?;

        // The code extracts the archive into a temp dir under `dir`. The temp
        // dir deletes itself when it drops.
        let extract_dir = tempfile::Builder::new().prefix("20news-").tempdir_in(dir)?;
        untar_gz(&archive_path, extract_dir.path())?;

        let mut texts: Vec<String> = Vec::new();
        let mut labels: Vec<String> = Vec::new();

        // The code walks each requested partition, then categories, then files,
        // in a deterministic (lexicographic) order. This keeps the sample
        // ordering stable.
        for subset in subset_dirs {
            let subset_path = extract_dir.path().join(subset);
            for category in sorted_child_names(&subset_path, /* dirs = */ true)? {
                let label = category_label(&category).ok_or_else(|| {
                    DatasetError::invalid_value(NEWSGROUPS20_DATASET_NAME, "category", &category, 0)
                })?;
                let category_path = subset_path.join(&category);
                for file_name in sorted_child_names(&category_path, /* dirs = */ false)? {
                    let bytes = fs::read(category_path.join(&file_name))?;
                    // The loader decodes bytes as Latin-1 (byte -> Unicode scalar),
                    // like scikit-learn. This keeps non-UTF-8 bytes intact.
                    let text: String = bytes.iter().map(|&b| b as char).collect();
                    texts.push(text);
                    labels.push(label.to_string());
                }
            }
        }

        Table::new(
            NEWSGROUPS20_DATASET_NAME,
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
    /// - `&Table` - reference to the cached table of 2 columns. The sample count
    ///   depends on the subset: 11,314 / 7,532 / 18,846.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (an unexpected category directory)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`Newsgroups20::data`], this method never runs the loader. If the
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
    /// changes stay in the cache. Later calls to [`Newsgroups20::data`] or
    /// [`Newsgroups20::get_data`] see them.
    ///
    /// Like [`Newsgroups20::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`Newsgroups20::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 2 columns. The sample count depends on the
    ///   subset: 11,314 / 7,532 / 18,846.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, archive extraction,
    /// I/O, or an unexpected category directory).
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
    /// - `Table` - the owned table of 2 columns. The sample count depends on the
    ///   subset: 11,314 / 7,532 / 18,846.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, archive extraction,
    /// I/O, or an unexpected category directory).
    pub fn take_data(&mut self) -> Result<Table, DatasetError> {
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

impl_ml_dataset!(Newsgroups20, "newsgroups20");
