//! Iris flower dataset.
//!
//! The classic Fisher Iris dataset for multi-class classification.
//! It contains measurements for three Iris species: `setosa`, `versicolor`,
//! and `virginica`.
//!
//! **Features (4):**
//! - `sepal_length` - sepal length in cm
//! - `sepal_width` - sepal width in cm
//! - `petal_length` - petal length in cm
//! - `petal_width` - petal width in cm
//!
//! **Target:** `species` - one of `setosa`, `versicolor`, or `virginica`
//!
//! **Samples:** 150 total, with 50 samples per species
//! **Application:** Multi-class classification / species recognition
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C56C76>

use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::fs::File;

/// The URL for the Iris dataset.
///
/// # Citation
///
/// R. A. Fisher. "Iris," UCI Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C56C76>
const IRIS_DATA_URL: &str = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv";

/// The name of the Iris dataset file.
const IRIS_FILENAME: &str = "iris.csv";

/// The SHA256 hash of the Iris dataset file.
const IRIS_SHA256: &str = "c52742e50315a99f956a383faedf7575552675f6409ef0f9a47076dd08479930";

/// The name of the dataset
const IRIS_DATASET_NAME: &str = "iris";

/// Type alias for the Iris dataset: (features, labels).
type IrisData = (Array2<f64>, Array1<&'static str>);

/// One CSV record of the Iris dataset: four `f64` measurements followed by the
/// species label.
///
/// Fields are declared in CSV column order and deserialized **positionally**
/// (the loader disables csv's header handling), so this struct is independent
/// of the exact header spelling and of any byte-order mark on the header row.
#[derive(Deserialize)]
struct IrisRecord {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    species: String,
}

/// A struct representing the Iris dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The Iris dataset is a classic dataset for classification tasks. It includes three iris species
/// with 50 samples each as well as some properties about each flower. One flower species is
/// linearly separable from the other two, but the other two are not linearly separable from each other.
///
/// Features:
/// - sepal length in cm
/// - sepal width in cm
/// - petal length in cm
/// - petal width in cm
///
/// Labels:
/// - species name (in `&str`): `"setosa"`, `"versicolor"`, `"virginica"`
///
/// See more information at <https://archive.ics.uci.edu/dataset/53/iris>
///
/// # Citation
///
/// R. A. Fisher. "Iris," UCI Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C56C76>
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::iris::Iris;
///
/// let download_dir = "./iris"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = Iris::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
/// assert_eq!(features.shape(), &[150, 4]);
/// assert_eq!(labels.len(), 150);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 5.5;
///     labels[0] = "setosa-modified";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable — the next access reloads from the cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[150, 4]);
/// assert_eq!(owned_labels.len(), 150);
///
/// // `into_data()` also returns owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[150, 4]);
/// assert_eq!(owned_labels.len(), 150);
/// ```
#[derive(Debug)]
pub struct Iris {
    dataset: Dataset<IrisData, DatasetError>,
}

impl Iris {
    /// Create a new Iris instance without loading data.
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
    /// - `Self` - `Iris` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Iris {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Iris dataset.
    fn load_data(dir: &str) -> Result<IrisData, DatasetError> {
        // Prepare the dataset file
        let file_path = acquire_dataset(
            dir,
            IRIS_FILENAME,
            IRIS_DATASET_NAME,
            Some(IRIS_SHA256),
            |temp_path| {
                download_to(IRIS_DATA_URL, temp_path, None)?;
                Ok(temp_path.join(IRIS_FILENAME))
            },
        )?;

        // csv deserializes into the struct
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features = Vec::new();
        let mut labels = Vec::new();

        for (idx, result) in rdr.deserialize::<IrisRecord>().skip(1).enumerate() {
            let IrisRecord {
                sepal_length,
                sepal_width,
                petal_length,
                petal_width,
                species,
            } = result.map_err(|e| DatasetError::csv_read_error(IRIS_DATASET_NAME, e))?;
            let line_num = idx + 2; // +1 for 0-indexed, +1 for header

            features.push(sepal_length);
            features.push(sepal_width);
            features.push(petal_length);
            features.push(petal_width);

            labels.push(match species.as_str() {
                "setosa" => "setosa",
                "versicolor" => "versicolor",
                "virginica" => "virginica",
                other => {
                    return Err(DatasetError::invalid_value(
                        IRIS_DATASET_NAME,
                        "label",
                        other,
                        line_num,
                    ));
                }
            });
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(IRIS_DATASET_NAME));
        }

        // Iris has a fixed schema of 4 numeric features per sample.
        let features_array = Array2::from_shape_vec((n_samples, 4), features)
            .map_err(|e| DatasetError::array_shape_error(IRIS_DATASET_NAME, "features", e))?;
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
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(150, 4)` containing:
    ///     - sepal length in cm
    ///     - sepal width in cm
    ///     - petal length in cm
    ///     - petal width in cm
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (150 samples, 4 features)
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
    /// - `&Array1<&'static str>` - Reference to labels vector with shape `(150,)` containing species names (`"setosa"`, `"versicolor"`, `"virginica"`)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (150 samples)
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
    /// - `&IrisData` - reference to the cached `(features, labels)` tuple: the
    ///   feature matrix has shape `(150, 4)` (sepal length/width, petal
    ///   length/width, all in cm) and the label vector has shape `(150,)`
    ///   containing species names (`"setosa"`, `"versicolor"`, `"virginica"`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (150 samples, 4 features)
    pub fn data(&self) -> Result<&IrisData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and labels as references **without** triggering loading.
    ///
    /// Unlike [`Iris::data`], which loads the dataset on first call, this never
    /// runs the loader: if the data has not been loaded yet, it returns `None`
    /// instead of downloading and parsing. Use it when you only want the data if
    /// it is already cached and want to avoid paying the download/parse cost
    /// otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&IrisData)` - reference to the cached `(features, labels)` tuple
    ///   (feature matrix `(150, 4)`, label vector `(150,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&IrisData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize features,
    /// replace label values) with no `to_owned()` clone and without removing them
    /// from the cache: the changes persist, so later [`Iris::features`],
    /// [`Iris::data`], or [`Iris::get_data`] calls observe them.
    ///
    /// Like [`Iris::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call a loading accessor (e.g.
    /// [`Iris::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut IrisData)` - mutable reference to the cached
    ///   `(features, labels)` tuple (feature matrix `(150, 4)`, label vector
    ///   `(150,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut IrisData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`Iris::data`], which borrows the cached data, this moves it out and
    /// returns owned arrays directly — no `to_owned()` clone needed. The dataset is
    /// loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use [`Iris::take_data`]
    /// instead — it takes `&mut self` and leaves the instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<&'static str>)` - owned feature matrix with shape
    ///   `(150, 4)` and owned label vector with shape `(150,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn into_data(self) -> Result<IrisData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`Iris::into_data`], this returns owned arrays with no `to_owned()`
    /// clone. But instead of consuming the instance, it takes `&mut self` and moves
    /// the cached data out, resetting the instance to its unloaded state: the next
    /// accessor call (e.g. [`Iris::features`] or [`Iris::data`]) loads the dataset
    /// again.
    ///
    /// Use [`Iris::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<&'static str>)` - owned feature matrix with shape
    ///   `(150, 4)` and owned label vector with shape `(150,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<IrisData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}
