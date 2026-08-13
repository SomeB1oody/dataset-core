//! Cornell Movie Review Polarity dataset (polarity dataset v2.0).
//!
//! Pang and Lee's sentiment-polarity benchmark holds 2,000 full movie reviews
//! from IMDb, split evenly into 1,000 `positive` and 1,000 `negative` reviews.
//! Each sample is one whole review document, already tokenized and lowercased.
//! Vectorize the text yourself (bag-of-words, TF-IDF, embeddings, and so on)
//! before you use it as model input.
//!
//! **Columns (2):**
//!
//! | Name    | Type     | Description                                     |
//! |---------|----------|-------------------------------------------------|
//! | `text`  | `String` | one tokenized review, newlines included         |
//! | `label` | `String` | `positive` (`pos` folder) or `negative` (`neg`) |
//!
//! The source designates the review text as the input
//! ([`MovieReviewPolarity::FEATURE_NAMES`](crate::MovieReviewPolarity::FEATURE_NAMES)) and the sentiment as the label
//! ([`MovieReviewPolarity::TARGET`](crate::MovieReviewPolarity::TARGET)).
//!
//! **Samples:** 2,000 (1,000 positive, 1,000 negative: a balanced split)
//! **Application:** Binary text classification / sentiment analysis
//!
//! **Missing values:** none.
//!
//! **Source:** Cornell movie-review data (polarity dataset v2.0)
//! <http://www.cs.cornell.edu/people/pabo/movie-review-data/>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, untar_gz};
use ndarray::Array1;
use std::fs;
use std::path::Path;

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

/// The class subdirectories paired with their labels, in the fixed
/// (lexicographic) order the loader walks them.
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
/// # Columns
///
/// | Name    | Type     | Description                                     |
/// |---------|----------|-------------------------------------------------|
/// | `text`  | `String` | one tokenized review, newlines included         |
/// | `label` | `String` | `positive` (`pos` folder) or `negative` (`neg`) |
///
/// The source designates the review text as the input
/// ([`MovieReviewPolarity::FEATURE_NAMES`]) and the sentiment as the label
/// ([`MovieReviewPolarity::TARGET`]).
///
/// Missing values: none.
///
/// The `text` column holds whole documents, not numbers. Vectorize the reviews
/// yourself (bag-of-words, TF-IDF, embeddings, and so on) before you use them as
/// model input. The loader walks the `neg` folder first, then the `pos` folder,
/// and takes the files of each folder in lexicographic order. The sample order
/// is therefore stable.
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
/// // the loader creates the directory if it does not exist
/// let download_dir = "./movie_review_polarity";
///
/// let mut dataset = MovieReviewPolarity::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 2000);
/// assert_eq!(table.n_columns(), 2);
///
/// // Reach one column by name.
/// let texts = table.column(MovieReviewPolarity::FEATURE_NAMES[0]).unwrap().as_string().unwrap();
/// assert_eq!(texts.len(), 2000);
/// let labels = table.column(MovieReviewPolarity::TARGET).unwrap().as_string().unwrap();
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
/// assert_eq!(owned.n_samples(), 2000);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 2000);
/// ```
#[derive(Debug)]
pub struct MovieReviewPolarity {
    dataset: Dataset<Table, DatasetError>,
}

impl MovieReviewPolarity {
    /// The column the source designates as the model input.
    pub const FEATURE_NAMES: [&'static str; 1] = ["text"];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "label";

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
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        // The loader caches the compressed tarball as-is, and uses its SHA-256
        // hash as the integrity check.
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
        let mut labels: Vec<String> = Vec::with_capacity(N_SAMPLES);

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
                labels.push(label.to_string());
            }
        }

        Table::new(
            MOVIE_REVIEW_POLARITY_DATASET_NAME,
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
    /// - `&Table` - reference to the cached table of 2,000 samples and 2 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - The archive holds no review
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`MovieReviewPolarity::data`], this method never runs the loader.
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
    /// changes stay in the cache. Later calls to [`MovieReviewPolarity::data`] or
    /// [`MovieReviewPolarity::get_data`] see them.
    ///
    /// Like [`MovieReviewPolarity::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`MovieReviewPolarity::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 2,000 samples and 2 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, archive extraction,
    /// I/O, or parsing).
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
    /// - `Table` - the owned table of 2,000 samples and 2 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, archive extraction,
    /// I/O, or parsing).
    pub fn take_data(&mut self) -> Result<Table, DatasetError> {
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

impl_ml_dataset!(MovieReviewPolarity, "movie_review_polarity");
