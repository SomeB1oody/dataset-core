use crate::{
    DatasetError, create_temp_dir, download_to, file_sha256_matches, prepare_download_dir, unzip,
};
use ndarray::{Array1, Array2};
use std::fs::{File, remove_file, rename};
use csv::ReaderBuilder;
use std::path::Path;
use std::sync::OnceLock;

/// The URL for the Iris dataset.
///
/// # Citation
///
/// R. A. Fisher. "Iris," UCI Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C56C76>
const IRIS_DATA_URL: &str = "https://archive.ics.uci.edu/static/public/53/iris.zip";

/// The prefix for temporary files created during dataset download and extraction.
const IRIS_TEMP_FILE_PREFIX: &str = ".tmp-iris-";

/// The name of the zip file downloaded.
const IRIS_ZIP_FILENAME: &str = "iris.zip";

/// The name of the file in the zip after extraction.
const IRIS_FILENAME: &str = "iris.data";

/// The expected number of features in the Iris dataset.
const IRIS_NUM_FEATURES: usize = 4;

/// The SHA256 hash of the Iris dataset file.
const IRIS_SHA256: &str = "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0";

/// The name of the dataset
const IRIS_DATASET_NAME: &str = "iris";

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
/// The `OnceLock` ensures thread-safe lazy initialization.
///
/// # Fields
///
/// - `storage_dir` - Directory where the dataset will be stored.
/// - `data` - Cached data as a tuple of references to `Array2<f64>` and `Array1<&'static str>`. (`OnceLock` is used to ensure thread-safety)
///
/// # Example
/// ```rust
/// use rustyml_dataset::iris::Iris;
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
///
/// // clean up: remove the downloaded files (dispensable)
/// std::fs::remove_dir_all(download_dir).unwrap();
/// ```
#[derive(Clone)]
pub struct Iris {
    storage_dir: String,
    data: OnceLock<(Array2<f64>, Array1<&'static str>)>,
}

impl std::fmt::Debug for Iris {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Iris")
            .field("storage_dir", &self.storage_dir)
            .field("data_loaded", &self.data.get().is_some())
            .finish()
    }
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
            storage_dir: storage_dir.to_string(),
            data: OnceLock::new(),
        }
    }

    /// Internal function to load the dataset from disk or download it.
    ///
    /// This function is called automatically by the accessor methods.
    fn load_data_internal(dir: &str) -> Result<(Array2<f64>, Array1<&'static str>), DatasetError> {
        let dir = Path::new(dir);
        let dst = dir.join(IRIS_FILENAME);
        let (need_download, need_overwrite) = prepare_download_dir(dir, &dst, IRIS_SHA256)?;

        // download and extract iris dataset if needed
        if need_download {
            // temporary directory to store the downloaded zip file
            let temp_dir = create_temp_dir(dir, IRIS_TEMP_FILE_PREFIX)?;
            let dir_temp = temp_dir.path();
            // download and extract iris dataset
            download_to(IRIS_DATA_URL, dir_temp)?;
            unzip(&dir_temp.join(IRIS_ZIP_FILENAME), dir_temp)?;
            let src = dir_temp.join(IRIS_FILENAME);
            // check if the file exists and matches the expected SHA256 hash
            if !file_sha256_matches(src.as_path(), IRIS_SHA256)? {
                // clean up temporary directory
                drop(temp_dir);
                return Err(DatasetError::sha256_validation_failed(
                    IRIS_DATASET_NAME,
                    IRIS_FILENAME,
                ));
            }
            if need_overwrite {
                remove_file(&dst)?;
            }
            // move iris.data out of the temporary directory
            rename(src, &dst)?;
        }

        let file = File::open(&dst)?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(false)
            .from_reader(file);

        let mut features = Vec::new();
        let mut labels = Vec::new();

        for (idx, result) in rdr.records().enumerate() {
            let record = result.map_err(|e| {
                DatasetError::csv_read_error(IRIS_DATASET_NAME, e)
            })?;
            let line_num = idx + 1; // +1 for 0-indexed, no header

            if record.len() != IRIS_NUM_FEATURES + 1 {
                return Err(DatasetError::invalid_column_count(
                    IRIS_DATASET_NAME,
                    IRIS_NUM_FEATURES + 1,
                    record.len(),
                    line_num,
                    &format!("{:?}", record),
                ));
            }

            // Features are columns 0-3 (4 features)
            for i in 0..IRIS_NUM_FEATURES {
                let field = format!("feature[{i}]");
                features.push(record[i].parse::<f64>().map_err(|e| {
                    DatasetError::parse_failed(
                        IRIS_DATASET_NAME,
                        &field,
                        line_num,
                        &format!("{:?}", record),
                        e,
                    )
                })?);
            }

            // Label is column 4
            labels.push(match &record[4] {
                "Iris-setosa" => "setosa",
                "Iris-versicolor" => "versicolor",
                "Iris-virginica" => "virginica",
                other => {
                    return Err(DatasetError::invalid_value(
                        IRIS_DATASET_NAME,
                        "label",
                        other,
                        line_num,
                        &format!("{:?}", record),
                    ));
                }
            });
        }
        
        // Verify the dataset is not empty
        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::length_mismatch(
                IRIS_DATASET_NAME,
                "samples",
                1, // At least 1 expected
                0,
            ));
        }

        let features_array =
            Array2::from_shape_vec((n_samples, IRIS_NUM_FEATURES), features)
                .map_err(|e| DatasetError::array_shape_error(IRIS_DATASET_NAME, "features", e))?;
        let labels_array = Array1::from_vec(labels);

        Ok((features_array, labels_array))
    }

    /// Internal helper to ensure data is loaded and return a reference.
    fn load_data(&self) -> Result<&(Array2<f64>, Array1<&'static str>), DatasetError> {
        // if already initialized
        if let Some(cache) = self.data.get() {
            return Ok(cache);
        }
        // if not, initialize then store
        let (features, labels) = Self::load_data_internal(&self.storage_dir)?;

        // Try to set the value. If another thread already set it, that's fine - just use the existing value
        let _ = self.data.set((features, labels));

        let cache = self
            .data
            .get()
            .expect("IRIS_DATA should be initialized after set");
        Ok(cache)
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
        Ok(&self.load_data()?.0)
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
        Ok(&self.load_data()?.1)
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
        let data = self.load_data()?;
        Ok((&data.0, &data.1))
    }
}