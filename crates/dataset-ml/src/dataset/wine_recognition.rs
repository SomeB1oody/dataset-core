//! Wine Recognition dataset.
//!
//! The dataset holds results from a chemical analysis of wines grown in the
//! same region in Italy but derived from three different cultivars. The
//! analysis determined the quantities of 13 constituents found in each of the
//! three types of wine. The task is to predict the cultivar (one of three
//! classes) from the constituents.
//!
//! This is the **Wine recognition** dataset (the same one bundled with
//! scikit-learn as `load_wine`). It is distinct from the **Wine Quality**
//! datasets in [`crate::dataset::wine_quality`], which are a regression task on red/white
//! wine quality scores.
//!
//! **Features (13):**
//! - `alcohol`
//! - `malic_acid`
//! - `ash`
//! - `alcalinity_of_ash`
//! - `magnesium`
//! - `total_phenols`
//! - `flavanoids`
//! - `nonflavanoid_phenols`
//! - `proanthocyanins`
//! - `color_intensity`
//! - `hue`
//! - `od280_od315_of_diluted_wines`
//! - `proline`
//!
//! **Target:** `class` - one of `class_1`, `class_2`, or `class_3` (the cultivar)
//!
//! **Samples:** 178 total (59 of class 1, 71 of class 2, 48 of class 3)
//! **Application:** Multi-class classification / cultivar recognition
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5PC7J>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::fs::File;

/// The URL for the Wine Recognition dataset.
///
/// # Citation
///
/// S. Aeberhard and M. Forina. "Wine," UCI Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C5PC7J>
const WINE_RECOGNITION_DATA_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data";

/// The name of the Wine Recognition dataset file.
const WINE_RECOGNITION_FILENAME: &str = "wine_recognition.csv";

/// The SHA256 hash of the Wine Recognition dataset file.
const WINE_RECOGNITION_SHA256: &str =
    "6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659";

/// The name of the dataset
const WINE_RECOGNITION_DATASET_NAME: &str = "wine_recognition";

/// The number of features per sample (13 chemical constituents).
const N_FEATURES: usize = 13;

/// Type alias for the Wine Recognition dataset: (features, labels).
type WineRecognitionData = (Array2<f64>, Array1<&'static str>);

/// One CSV record of the Wine Recognition dataset: the `1`/`2`/`3` class label
/// followed by the 13 `f64` constituent measurements.
///
/// This struct declares fields in CSV column order. It deserializes them
/// **positionally** (the loader disables csv's header handling). This matches
/// the headerless `wine.data` layout, where the class is the first column.
#[derive(Deserialize)]
struct WineRecognitionRecord {
    class: String,
    alcohol: f64,
    malic_acid: f64,
    ash: f64,
    alcalinity_of_ash: f64,
    magnesium: f64,
    total_phenols: f64,
    flavanoids: f64,
    nonflavanoid_phenols: f64,
    proanthocyanins: f64,
    color_intensity: f64,
    hue: f64,
    od280_od315_of_diluted_wines: f64,
    proline: f64,
}

/// A struct that represents the Wine Recognition dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// This dataset is the result of a chemical analysis of wines grown in the same
/// region in Italy but derived from three different cultivars. The analysis
/// determined the quantities of 13 constituents found in each of the three types
/// of wine.
///
/// This is the **Wine recognition** dataset (scikit-learn's `load_wine`), a
/// multi-class classification task. It is **not** the same as the
/// [`crate::dataset::wine_quality`] datasets, which predict a quality score (regression).
///
/// # Feature columns
///
/// The 13 numeric feature columns are the chemical constituents measured for
/// each wine sample. By 0-based column index in the feature matrix:
///
/// | Columns | Attributes                      | Unit |
/// |---------|---------------------------------|------|
/// | `0`     | `alcohol`                       |      |
/// | `1`     | `malic_acid`                    |      |
/// | `2`     | `ash`                           |      |
/// | `3`     | `alcalinity_of_ash`             |      |
/// | `4`     | `magnesium`                     |      |
/// | `5`     | `total_phenols`                 |      |
/// | `6`     | `flavanoids`                    |      |
/// | `7`     | `nonflavanoid_phenols`          |      |
/// | `8`     | `proanthocyanins`               |      |
/// | `9`     | `color_intensity`               |      |
/// | `10`    | `hue`                           |      |
/// | `11`    | `od280_od315_of_diluted_wines`  |      |
/// | `12`    | `proline`                       |      |
///
/// # Labels
///
/// - class (in `&str`): `"class_1"`, `"class_2"`, `"class_3"`
///
/// See more information at <https://archive.ics.uci.edu/dataset/109/wine>
///
/// # Citation
///
/// S. Aeberhard and M. Forina. "Wine," UCI Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C5PC7J>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::WineRecognition;
///
/// let download_dir = "./wine_recognition"; // the loader creates the directory if it does not exist
///
/// let mut dataset = WineRecognition::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap();
/// assert_eq!(features.shape(), &[178, 13]);
/// assert_eq!(labels.len(), 178);
///
/// // `get_data()` borrows the cached arrays without a reload. `get_data_mut()`
/// // edits the arrays in place. This needs no clone and no reload. The change
/// // stays cached. If you only need to change values, prefer this method over
/// // `.to_owned()`.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 13.5;
///     labels[0] = "class_2";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out with no `to_owned()` clone. This
/// // leaves the instance reusable. The next access reloads the data from the
/// // cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[178, 13]);
/// assert_eq!(owned_labels.len(), 178);
///
/// // `into_data()` also returns the owned arrays with no clone, but it
/// // consumes the instance. If you are done with the dataset, use it.
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[178, 13]);
/// assert_eq!(owned_labels.len(), 178);
/// ```
#[derive(Debug)]
pub struct WineRecognition {
    dataset: Dataset<WineRecognitionData, DatasetError>,
}

impl WineRecognition {
    /// Create a new WineRecognition instance without loading data.
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
    /// - `Self` - a `WineRecognition` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        WineRecognition {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Wine Recognition dataset.
    fn load_data(dir: &str) -> Result<WineRecognitionData, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            WINE_RECOGNITION_FILENAME,
            WINE_RECOGNITION_DATASET_NAME,
            Some(WINE_RECOGNITION_SHA256),
            |temp_path| {
                download_to_with_retries(
                    WINE_RECOGNITION_DATA_URL,
                    temp_path,
                    Some(WINE_RECOGNITION_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(WINE_RECOGNITION_FILENAME))
            },
        )?;

        // `wine.data` has no header row, so every line is a record. Do not skip
        // the first one.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features = Vec::new();
        let mut labels = Vec::new();

        for (idx, result) in rdr.deserialize::<WineRecognitionRecord>().enumerate() {
            let WineRecognitionRecord {
                class,
                alcohol,
                malic_acid,
                ash,
                alcalinity_of_ash,
                magnesium,
                total_phenols,
                flavanoids,
                nonflavanoid_phenols,
                proanthocyanins,
                color_intensity,
                hue,
                od280_od315_of_diluted_wines,
                proline,
            } = result
                .map_err(|e| DatasetError::csv_read_error(WINE_RECOGNITION_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            features.extend_from_slice(&[
                alcohol,
                malic_acid,
                ash,
                alcalinity_of_ash,
                magnesium,
                total_phenols,
                flavanoids,
                nonflavanoid_phenols,
                proanthocyanins,
                color_intensity,
                hue,
                od280_od315_of_diluted_wines,
                proline,
            ]);

            labels.push(match class.as_str() {
                "1" => "class_1",
                "2" => "class_2",
                "3" => "class_3",
                other => {
                    return Err(DatasetError::invalid_value(
                        WINE_RECOGNITION_DATASET_NAME,
                        "class",
                        other,
                        line_num,
                    ));
                }
            });
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(WINE_RECOGNITION_DATASET_NAME));
        }

        // Wine Recognition has a fixed schema of 13 numeric features per sample.
        let features_array =
            Array2::from_shape_vec((n_samples, N_FEATURES), features).map_err(|e| {
                DatasetError::array_shape_error(WINE_RECOGNITION_DATASET_NAME, "features", e)
            })?;
        let labels_array = Array1::from_vec(labels);

        Ok((features_array, labels_array))
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(178, 13)`
    ///   containing the 13 chemical constituents (alcohol, malic acid, ash, …,
    ///   proline).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match the expected dimensions (178 samples, 13 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the label vector.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array1<&'static str>` - Reference to label vector with shape `(178,)`
    ///   containing cultivar classes (`"class_1"`, `"class_2"`, `"class_3"`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match the expected dimensions (178 samples)
    pub fn labels(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and labels as references.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&WineRecognitionData` - reference to the cached `(features, labels)`
    ///   tuple. The feature matrix has shape `(178, 13)`. The label vector has
    ///   shape `(178,)` and contains cultivar classes (`"class_1"`, `"class_2"`,
    ///   `"class_3"`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match the expected dimensions (178 samples, 13 features)
    pub fn data(&self) -> Result<&WineRecognitionData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and labels as references **without** triggering loading.
    ///
    /// Unlike [`WineRecognition::data`], which loads the dataset on first call,
    /// this method never runs the loader. If the data is not in the cache yet, it
    /// returns `None` instead of downloading and parsing it. Use this method to
    /// get data only when it is already cached. This avoids the download and
    /// parse cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&WineRecognitionData)` - reference to the cached `(features, labels)`
    ///   tuple (feature matrix `(178, 13)`, label vector `(178,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&WineRecognitionData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This lets you change the cached arrays directly (e.g. normalize features,
    /// replace label values), with no `to_owned()` clone and without removing them
    /// from the cache. The changes persist, so later [`WineRecognition::features`],
    /// [`WineRecognition::data`], or [`WineRecognition::get_data`] calls see them.
    ///
    /// Like [`WineRecognition::get_data`], this method does **not** trigger
    /// loading. It returns `None` if the dataset is not loaded. If you need the
    /// data to be present, call a loading accessor first (e.g.
    /// [`WineRecognition::data`]).
    ///
    /// # Returns
    ///
    /// - `Some(&mut WineRecognitionData)` - mutable reference to the cached
    ///   `(features, labels)` tuple (feature matrix `(178, 13)`, label vector
    ///   `(178,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut WineRecognitionData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`WineRecognition::data`], which borrows the cached data, this moves
    /// it out and returns owned arrays directly. There is no `to_owned()` clone.
    /// This method loads the dataset on first access if it has not loaded yet.
    ///
    /// This **consumes** `self`. You cannot use the instance afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`WineRecognition::take_data`] instead. It takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<&'static str>)` - owned feature matrix with shape
    ///   `(178, 13)` and owned label vector with shape `(178,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn into_data(self) -> Result<WineRecognitionData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and labels out of the dataset. This leaves the
    /// instance reusable.
    ///
    /// Like [`WineRecognition::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. Instead of consuming the instance, it takes `&mut self`
    /// and moves the cached data out. This resets the instance to its unloaded
    /// state, so the next accessor call (e.g. [`WineRecognition::features`] or
    /// [`WineRecognition::data`]) loads the dataset again.
    ///
    /// If you are done with the instance, use [`WineRecognition::into_data`] instead.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<&'static str>)` - owned feature matrix with shape
    ///   `(178, 13)` and owned label vector with shape `(178,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<WineRecognitionData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(WineRecognition, WineRecognitionData, "wine_recognition");
