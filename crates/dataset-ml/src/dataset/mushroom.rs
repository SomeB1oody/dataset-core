//! Mushroom dataset.
//!
//! The dataset holds records from *The Audubon Society Field Guide to North
//! American Mushrooms* (1981). The records describe 23 species of gilled
//! mushrooms in the Agaricus and Lepiota family. The task is to predict
//! whether a mushroom is edible or poisonous. This is the first
//! **all-categorical** loader. Every feature is a string code, so there is no
//! numeric feature matrix.
//!
//! **Features (22, all categorical):** `cap-shape`, `cap-surface`, `cap-color`,
//! `bruises`, `odor`, `gill-attachment`, `gill-spacing`, `gill-size`,
//! `gill-color`, `stalk-shape`, `stalk-root`, `stalk-surface-above-ring`,
//! `stalk-surface-below-ring`, `stalk-color-above-ring`, `stalk-color-below-ring`,
//! `veil-type`, `veil-color`, `ring-number`, `ring-type`, `spore-print-color`,
//! `population`, `habitat`. Each value is a single-letter code.
//!
//! **Target:** `class`, binary label kept verbatim (`e` = edible, `p` = poisonous)
//!
//! **Samples:** 8,124
//! **Application:** Binary classification / edibility prediction
//!
//! **Source:** UCI Machine Learning Repository
//! <https://archive.ics.uci.edu/dataset/73/mushroom>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::{Array1, Array2};
use std::fs::File;

/// Type alias for Mushroom dataset: (categorical features, labels).
type MushroomData = (Array2<String>, Array1<String>);

/// The URL for the Mushroom dataset (the `agaricus-lepiota.data` file).
const MUSHROOM_DATA_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data";

/// The name of the cached Mushroom dataset file.
const MUSHROOM_FILENAME: &str = "mushroom.csv";

/// The SHA256 hash of the cached Mushroom dataset file (`agaricus-lepiota.data`'s bytes).
const MUSHROOM_SHA256: &str = "e65d082030501a3ebcbcd7c9f7c71aa9d28fdfff463bf4cf4716a3fe13ac360e";

/// The name of the dataset.
const MUSHROOM_DATASET_NAME: &str = "mushroom";

/// Number of samples.
const N_SAMPLES: usize = 8_124;

/// Number of categorical features.
const N_FEATURES: usize = 22;

/// Number of columns per record (1 label + 22 features).
const N_COLUMNS: usize = 23;

/// Source column index of the label (`class`). The label is the **first** column.
const LABEL_COLUMN: usize = 0;

/// Categorical feature columns, as `(source column index, name)`, in output order.
/// All 22 features follow the leading `class` label column.
const FEATURE_COLUMNS: [(usize, &str); N_FEATURES] = [
    (1, "cap-shape"),
    (2, "cap-surface"),
    (3, "cap-color"),
    (4, "bruises"),
    (5, "odor"),
    (6, "gill-attachment"),
    (7, "gill-spacing"),
    (8, "gill-size"),
    (9, "gill-color"),
    (10, "stalk-shape"),
    (11, "stalk-root"),
    (12, "stalk-surface-above-ring"),
    (13, "stalk-surface-below-ring"),
    (14, "stalk-color-above-ring"),
    (15, "stalk-color-below-ring"),
    (16, "veil-type"),
    (17, "veil-color"),
    (18, "ring-number"),
    (19, "ring-type"),
    (20, "spore-print-color"),
    (21, "population"),
    (22, "habitat"),
];

/// The token marking a missing categorical value in the source (only in `stalk-root`).
const MISSING_TOKEN: &str = "?";

/// This struct represents the Mushroom dataset and loads it lazily.
///
/// Nothing loads until you call a data accessor method. After loading, the
/// data stays cached for later accesses.
///
/// # About Dataset
///
/// The Mushroom dataset describes hypothetical samples that correspond to 23
/// species of gilled mushrooms in the Agaricus and Lepiota family. The records
/// come from *The Audubon Society Field Guide to North American Mushrooms*
/// (1981). Each species is labeled edible or poisonous. The poisonous label
/// also covers species of unknown edibility and species not recommended for
/// eating. The classification task is to predict edibility from 22 categorical
/// attributes. No simple rule determines the edibility of a mushroom, and this
/// makes the dataset a difficult classification problem.
///
/// # Feature columns
///
/// All 22 features are categorical, stored as single-letter string codes in one
/// `(8124, 22)` `Array2<String>` matrix (there is no numeric matrix). By 0-based
/// column:
///
/// | Column | Attribute                  |
/// |--------|----------------------------|
/// | `0`    | `cap-shape`                |
/// | `1`    | `cap-surface`              |
/// | `2`    | `cap-color`                |
/// | `3`    | `bruises`                  |
/// | `4`    | `odor`                     |
/// | `5`    | `gill-attachment`          |
/// | `6`    | `gill-spacing`             |
/// | `7`    | `gill-size`                |
/// | `8`    | `gill-color`               |
/// | `9`    | `stalk-shape`              |
/// | `10`   | `stalk-root`               |
/// | `11`   | `stalk-surface-above-ring` |
/// | `12`   | `stalk-surface-below-ring` |
/// | `13`   | `stalk-color-above-ring`   |
/// | `14`   | `stalk-color-below-ring`   |
/// | `15`   | `veil-type`                |
/// | `16`   | `veil-color`               |
/// | `17`   | `ring-number`              |
/// | `18`   | `ring-type`                |
/// | `19`   | `spore-print-color`        |
/// | `20`   | `population`               |
/// | `21`   | `habitat`                  |
///
/// # Labels
///
/// - `class` (shape `(8124,)`): the `Array1<String>` is kept verbatim. Each
///   entry is either `e` (edible) or `p` (poisonous).
///
/// Missing values:
/// - The source marks missing values with `?` (only in `stalk-root`, 2,480
///   samples). The loader maps these to empty strings `""`.
///
/// See more information at <https://archive.ics.uci.edu/dataset/73/mushroom>.
///
/// # Citation
///
/// Mushroom (1987). UCI Machine Learning Repository.
/// <https://doi.org/10.24432/C5959T>
///
/// # Thread Safety
///
/// Every field implements `Send` and `Sync`, so this struct implements them too. It is safe
/// to share across threads.
/// The internal [`Dataset`] makes initialization thread-safe and lazy.
///
/// # Example
/// ```no_run
/// use dataset_ml::Mushroom;
///
/// let download_dir = "./mushroom"; // creates the directory if it is missing
///
/// let mut dataset = Mushroom::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap(); // this is also a way to get all data
/// assert_eq!(features.shape(), &[8124, 22]);
/// assert_eq!(labels.len(), 8124);
///
/// // `get_data()` borrows the cached arrays without reloading. `get_data_mut()`
/// // edits them in place. It needs no clone and no reload, and the change
/// // stays cached. Prefer this method over cloning with `.to_owned()` when
/// // you only need to change values.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = "x".to_string();
///     labels[0] = "e".to_string();
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out (no `to_owned()` clone) and leaves
/// // the instance reusable. The next access reloads from the cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[8124, 22]);
/// assert_eq!(owned_labels.len(), 8124);
///
/// // `into_data()` also returns the owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[8124, 22]);
/// assert_eq!(owned_labels.len(), 8124);
/// ```
#[derive(Debug)]
pub struct Mushroom {
    dataset: Dataset<MushroomData, DatasetError>,
}

impl Mushroom {
    /// Create a new Mushroom instance without loading data.
    ///
    /// This does not load the dataset. The dataset loads on the first call to a
    /// data accessor method. This is a lightweight operation: it only stores the
    /// storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory used to store the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - `Mushroom` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Mushroom {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Mushroom dataset.
    fn load_data(dir: &str) -> Result<MushroomData, DatasetError> {
        // Prepare the dataset file. The source file is `agaricus-lepiota.data`.
        // The code caches it as `mushroom.csv`.
        let file_path = acquire_dataset(
            dir,
            MUSHROOM_FILENAME,
            MUSHROOM_DATASET_NAME,
            Some(MUSHROOM_SHA256),
            |temp_path| {
                download_to_with_retries(
                    MUSHROOM_DATA_URL,
                    temp_path,
                    Some(MUSHROOM_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(MUSHROOM_FILENAME))
            },
        )?;

        // The source is plain comma-separated with no header and single-letter codes.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features: Vec<String> = Vec::with_capacity(N_SAMPLES * N_FEATURES);
        let mut labels: Vec<String> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(MUSHROOM_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            // Skip blank lines defensively (e.g. a trailing newline).
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    MUSHROOM_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // Categorical features, mapping the `?` missing token to an empty string.
            for &(col, _name) in FEATURE_COLUMNS.iter() {
                let value = &record[col];
                if value == MISSING_TOKEN {
                    features.push(String::new());
                } else {
                    features.push(value.to_string());
                }
            }

            // Label, kept verbatim (`e` or `p`).
            let label = &record[LABEL_COLUMN];
            if label.is_empty() {
                return Err(DatasetError::invalid_value(
                    MUSHROOM_DATASET_NAME,
                    "class",
                    label,
                    line_num,
                ));
            }
            labels.push(label.to_string());
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(MUSHROOM_DATASET_NAME));
        }

        let features_array = Array2::from_shape_vec((n_samples, N_FEATURES), features)
            .map_err(|e| DatasetError::array_shape_error(MUSHROOM_DATASET_NAME, "features", e))?;

        let labels_array = Array1::from_vec(labels);

        Ok((features_array, labels_array))
    }

    /// Get a reference to the categorical feature matrix.
    ///
    /// This method triggers lazy loading on first call. Later calls return the
    /// cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<String>` - Reference to the categorical feature matrix with shape
    ///   `(8124, 22)`. Each value is a single-letter code, except missing `stalk-root`
    ///   entries, which are empty strings.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (8,124 samples)
    pub fn features(&self) -> Result<&Array2<String>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the label vector.
    ///
    /// This method triggers lazy loading on first call. Later calls return the
    /// cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<String>` - Reference to label vector with shape `(8124,)`
    ///   containing `class` values (`e` = edible or `p` = poisonous)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (8,124 samples)
    pub fn labels(&self) -> Result<&Array1<String>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get features and labels as references.
    ///
    /// This method triggers lazy loading on first call. Later calls return the
    /// cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&MushroomData` - reference to the cached `(features, labels)` tuple: the
    ///   categorical feature matrix `(8124, 22)` and the label vector `(8124,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (8,124 samples)
    pub fn data(&self) -> Result<&MushroomData, DatasetError> {
        self.dataset.load()
    }

    /// Get features and labels as references, without triggering loading.
    ///
    /// Unlike [`Mushroom::data`], which loads the dataset on first call, this never
    /// runs the loader. If the data has not been loaded yet, it returns `None`
    /// instead of downloading and parsing.
    ///
    /// Use this method when you want the data only if it is already cached. This
    /// avoids the download and parse cost when the data is not cached.
    ///
    /// # Returns
    ///
    /// - `Some(&MushroomData)` - reference to the cached `(features, labels)` tuple
    ///   (`(8124, 22)`, `(8124,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&MushroomData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This lets you change the cached arrays directly (e.g. encode categorical
    /// features). It needs no `to_owned()` clone, and the arrays stay in the
    /// cache. The changes persist, so later calls to [`Mushroom::features`],
    /// [`Mushroom::data`], or [`Mushroom::get_data`] see them.
    ///
    /// Like [`Mushroom::get_data`], this does **not** trigger loading. It returns
    /// `None` if the dataset has not been loaded. If you need to make sure the
    /// data is present, call a loading accessor first (e.g. [`Mushroom::data`]).
    ///
    /// # Returns
    ///
    /// - `Some(&mut MushroomData)` - mutable reference to the cached `(features,
    ///   labels)` tuple (`(8124, 22)`, `(8124,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut MushroomData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`Mushroom::data`], which borrows the cached data, this moves it out
    /// and returns owned arrays directly. It needs no `to_owned()` clone. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`Mushroom::take_data`] instead. It takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array1<String>)` - owned categorical feature matrix
    ///   `(8124, 22)` and owned label vector `(8124,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<MushroomData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and labels out of the dataset. The instance stays
    /// reusable.
    ///
    /// Like [`Mushroom::into_data`], this returns owned arrays with no `to_owned()`
    /// clone. But instead of consuming the instance, it takes `&mut self` and moves
    /// the cached data out. This resets the instance to its unloaded state. The
    /// next accessor call (e.g. [`Mushroom::features`] or [`Mushroom::data`])
    /// loads the dataset again.
    ///
    /// If you are done with the instance, use [`Mushroom::into_data`] instead.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array1<String>)` - owned categorical feature matrix
    ///   `(8124, 22)` and owned label vector `(8124,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<MushroomData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(Mushroom, MushroomData, "mushroom");
