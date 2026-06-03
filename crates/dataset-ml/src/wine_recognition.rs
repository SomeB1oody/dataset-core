//! Wine Recognition dataset.
//!
//! Results of a chemical analysis of wines grown in the same region in Italy but
//! derived from three different cultivars. The analysis determined the
//! quantities of 13 constituents found in each of the three types of wine. The
//! task is to predict the cultivar (one of three classes) from the constituents.
//!
//! This is the **Wine recognition** dataset (the same one bundled with
//! scikit-learn as `load_wine`); it is distinct from the **Wine Quality**
//! datasets in [`crate::wine_quality`], which are a regression task on red/white
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

use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to};
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
/// Fields are declared in CSV column order and deserialized **positionally**
/// (the loader disables csv's header handling), matching the headerless
/// `wine.data` layout where the class is the first column.
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

/// A struct representing the Wine Recognition dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// These data are the results of a chemical analysis of wines grown in the same
/// region in Italy but derived from three different cultivars. The analysis
/// determined the quantities of 13 constituents found in each of the three types
/// of wine.
///
/// This is the **Wine recognition** dataset (scikit-learn's `load_wine`), a
/// multi-class classification task. It is **not** the same as the
/// [`crate::wine_quality`] datasets, which predict a quality score (regression).
///
/// Features:
/// - alcohol
/// - malic acid
/// - ash
/// - alcalinity of ash
/// - magnesium
/// - total phenols
/// - flavanoids
/// - nonflavanoid phenols
/// - proanthocyanins
/// - color intensity
/// - hue
/// - OD280/OD315 of diluted wines
/// - proline
///
/// Labels:
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
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::wine_recognition::WineRecognition;
///
/// let download_dir = "./wine_recognition"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = WineRecognition::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
/// assert_eq!(features.shape(), &[178, 13]);
/// assert_eq!(labels.len(), 178);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 13.5;
///     labels[0] = "class_2";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable — the next access reloads from the cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[178, 13]);
/// assert_eq!(owned_labels.len(), 178);
///
/// // `into_data()` also returns owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
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
    /// The dataset will be loaded lazily when you first call any data accessor method.
    /// This is a lightweight operation that only stores the storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory where the dataset will be stored.
    ///
    /// # Returns
    ///
    /// - `Self` - `WineRecognition` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        WineRecognition {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Wine Recognition dataset.
    fn load_data(dir: &str) -> Result<WineRecognitionData, DatasetError> {
        // Prepare the dataset file
        let file_path = acquire_dataset(
            dir,
            WINE_RECOGNITION_FILENAME,
            WINE_RECOGNITION_DATASET_NAME,
            Some(WINE_RECOGNITION_SHA256),
            |temp_path| {
                download_to(
                    WINE_RECOGNITION_DATA_URL,
                    temp_path,
                    Some(WINE_RECOGNITION_FILENAME),
                )?;
                Ok(temp_path.join(WINE_RECOGNITION_FILENAME))
            },
        )?;

        // csv deserializes into the struct. `wine.data` has no header row, so
        // every line is a record — do not skip the first one.
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
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
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
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (178 samples, 13 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the labels vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<&'static str>` - Reference to labels vector with shape `(178,)` containing cultivar classes (`"class_1"`, `"class_2"`, `"class_3"`)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (178 samples)
    pub fn labels(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&WineRecognitionData` - reference to the cached `(features, labels)`
    ///   tuple: the feature matrix has shape `(178, 13)` and the label vector has
    ///   shape `(178,)` containing cultivar classes (`"class_1"`, `"class_2"`,
    ///   `"class_3"`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (178 samples, 13 features)
    pub fn data(&self) -> Result<&WineRecognitionData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and labels as references **without** triggering loading.
    ///
    /// Unlike [`WineRecognition::data`], which loads the dataset on first call,
    /// this never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&WineRecognitionData)` - reference to the cached `(features, labels)`
    ///   tuple (feature matrix `(178, 13)`, label vector `(178,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&WineRecognitionData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize features,
    /// replace label values) with no `to_owned()` clone and without removing them
    /// from the cache: the changes persist, so later [`WineRecognition::features`],
    /// [`WineRecognition::data`], or [`WineRecognition::get_data`] calls observe them.
    ///
    /// Like [`WineRecognition::get_data`], this does **not** trigger loading: it
    /// returns `None` if the dataset has not been loaded. Call a loading accessor
    /// (e.g. [`WineRecognition::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut WineRecognitionData)` - mutable reference to the cached
    ///   `(features, labels)` tuple (feature matrix `(178, 13)`, label vector
    ///   `(178,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut WineRecognitionData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`WineRecognition::data`], which borrows the cached data, this moves
    /// it out and returns owned arrays directly — no `to_owned()` clone needed. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`WineRecognition::take_data`] instead — it takes `&mut self` and leaves the
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

    /// Take **owned** features and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`WineRecognition::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out, resetting the instance to its
    /// unloaded state: the next accessor call (e.g. [`WineRecognition::features`] or
    /// [`WineRecognition::data`]) loads the dataset again.
    ///
    /// Use [`WineRecognition::into_data`] instead if you are done with the instance.
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
