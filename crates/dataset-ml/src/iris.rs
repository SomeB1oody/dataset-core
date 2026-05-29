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

/// One CSV record of the Iris dataset: four `f64` measurements followed by the
/// species label.
///
/// Records are deserialized **positionally** (by column order), so this struct
/// is independent of the exact header spelling and of any byte-order mark on the
/// header row.
#[derive(Deserialize)]
struct IrisRecord(f64, f64, f64, f64, String);

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
/// let dataset = Iris::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
/// // you can use `.to_owned()` to get owned copies of the data
/// let mut features_owned = features.to_owned();
/// let mut labels_owned = labels.to_owned();
///
/// // Example: Modify feature values
/// features_owned[[0, 0]] = 5.5;
/// labels_owned[0] = "setosa-modified";
///
/// assert_eq!(features.shape(), &[150, 4]);
/// assert_eq!(labels.len(), 150);
/// ```
#[derive(Debug)]
pub struct Iris {
    dataset: Dataset<(Array2<f64>, Array1<&'static str>)>,
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
            dataset: Dataset::new(storage_dir),
        }
    }

    /// Acquire and parse the Iris dataset.
    fn load_data(dir: &str) -> Result<(Array2<f64>, Array1<&'static str>), DatasetError> {
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

        // Stream the cached file through csv, deserializing one record at a time.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

        let mut features = Vec::new();
        let mut labels = Vec::new();

        for (idx, result) in rdr.deserialize::<IrisRecord>().enumerate() {
            let IrisRecord(sepal_length, sepal_width, petal_length, petal_width, species) =
                result.map_err(|e| DatasetError::csv_read_error(IRIS_DATASET_NAME, e))?;
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
        Ok(&self.dataset.load(Self::load_data)?.0)
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
        Ok(&self.dataset.load(Self::load_data)?.1)
    }

    /// Get both features and labels as references.
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
    /// - `&Array1<&'static str>` - Reference to labels vector with shape `(150,)` containing species names (`"setosa"`, `"versicolor"`, `"virginica"`)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (150 samples, 4 features)
    pub fn data(&self) -> Result<(&Array2<f64>, &Array1<&'static str>), DatasetError> {
        let data = self.dataset.load(Self::load_data)?;
        Ok((&data.0, &data.1))
    }
}
