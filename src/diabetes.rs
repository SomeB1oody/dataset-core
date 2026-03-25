use crate::{
    DatasetError, create_temp_dir, download_to, file_sha256_matches, prepare_download_dir,
};
use ndarray::{Array1, Array2};
use std::fs::{File, remove_file, rename};
use csv::ReaderBuilder;
use std::path::Path;
use std::sync::OnceLock;

/// The URL for the Diabetes dataset.
const DIABETES_DATA_URL: &str = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv";

/// The prefix for temporary files created during dataset download and parsing.
const DIABETES_TEMP_FILE_PREFIX: &str = ".tmp-diabetes-";

/// A static string slice containing the name of the Diabetes dataset file.
const DIABETES_FILENAME: &str = "diabetes.csv";

/// The expected number of features in the Diabetes dataset.
const DIABETES_NUM_FEATURES: usize = 8;

/// The SHA256 hash of the Diabetes dataset file.
const DIABETES_SHA256: &str = "698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72";

/// The name of the dataset
const DIABETES_DATASET_NAME: &str = "diabetes";

/// A struct representing the Diabetes dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
/// The objective is to predict based on diagnostic measurements whether a patient has diabetes.
///
/// Features:
/// - Pregnancies: Number of times pregnant
/// - Glucose: Plasma glucose concentration at 2 hours in an oral glucose tolerance test
/// - BloodPressure: Diastolic blood pressure (mm Hg)
/// - SkinThickness: Triceps skin fold thickness (mm)
/// - Insulin: 2-Hour serum insulin (mu U/ml)
/// - BMI: Body mass index (weight in kg/(height in m)^2)
/// - DiabetesPedigreeFunction: Diabetes pedigree function
/// - Age: Age (years)
///
/// Labels:
/// - Outcome: Class variable (0 or 1)
///
/// See more information at <https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data>
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The `OnceLock` ensures thread-safe lazy initialization.
///
/// # Fields
///
/// - `storage_dir` - Directory where the dataset will be stored.
/// - `data` - Cached data as a tuple of references to `Array2<f64>` and `Array1<f64>`. (`OnceLock` is used to ensure thread-safety)
///
/// # Example
/// ```rust
/// use rustyml_dataset::diabetes::Diabetes;
///
/// let download_dir = "./diabetes"; // the code will create the directory if it doesn't exist
///
/// let dataset = Diabetes::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
/// // you can use `.to_owned()` to get owned copies of the data
/// let mut features_owned = features.to_owned();
/// let mut labels_owned = labels.to_owned();
///
/// // Example: Modify feature values
/// features_owned[[0, 0]] = 10.0;
/// labels_owned[0] = 1.0;
///
/// assert_eq!(features.shape(), &[768, 8]);
/// assert_eq!(labels.len(), 768);
///
/// // clean up: remove the downloaded files (dispensable)
/// std::fs::remove_dir_all(download_dir).unwrap();
/// ```
#[derive(Clone)]
pub struct Diabetes {
    storage_dir: String,
    data: OnceLock<(Array2<f64>, Array1<f64>)>,
}

impl std::fmt::Debug for Diabetes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Diabetes")
            .field("storage_dir", &self.storage_dir)
            .field("data_loaded", &self.data.get().is_some())
            .finish()
    }
}

impl Diabetes {
    /// Create a new Diabetes instance without loading data.
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
    /// - `Self` - `Diabetes` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Diabetes {
            storage_dir: storage_dir.to_string(),
            data: OnceLock::new(),
        }
    }

    /// Internal function to load the dataset from disk or download it.
    ///
    /// This function is called automatically by the accessor methods.
    fn load_data_internal(dir: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
        let dir = Path::new(dir);
        let dst = dir.join(DIABETES_FILENAME);
        let (need_download, need_overwrite) = prepare_download_dir(dir, &dst, DIABETES_SHA256)?;

        if need_download {
            // temporary directory
            let temp_dir = create_temp_dir(dir, DIABETES_TEMP_FILE_PREFIX)?;
            let dir_temp = temp_dir.path();
            // download file to temporary directory
            download_to(DIABETES_DATA_URL, dir_temp)?;
            // move downloaded file to final location
            let src = dir_temp.join(DIABETES_FILENAME);
            // check if the file matches the expected SHA256 hash
            if !file_sha256_matches(src.as_path(), DIABETES_SHA256)? {
                // clean up temporary directory
                drop(temp_dir);
                return Err(DatasetError::sha256_validation_failed(
                    DIABETES_DATASET_NAME,
                    DIABETES_FILENAME,
                ));
            }
            if need_overwrite {
                remove_file(&dst)?;
            }
            rename(src, &dst)?;
        }

        let file = File::open(&dst)?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        let mut features = Vec::new();
        let mut labels = Vec::new();

        for (idx, result) in rdr.records().enumerate() {
            let record = result.map_err(|e| {
                DatasetError::csv_read_error(DIABETES_DATASET_NAME, e)
            })?;
            let line_num = idx + 2; // +1 for 0-indexed, +1 for header

            if record.len() != DIABETES_NUM_FEATURES + 1 {
                return Err(DatasetError::invalid_column_count(
                    DIABETES_DATASET_NAME,
                    DIABETES_NUM_FEATURES + 1,
                    record.len(),
                    line_num,
                    &format!("{:?}", record),
                ));
            }

            for i in 0..DIABETES_NUM_FEATURES {
                let field = format!("feature[{i}]");
                features.push(record[i].parse::<f64>().map_err(|e| {
                    DatasetError::parse_failed(
                        DIABETES_DATASET_NAME,
                        &field,
                        line_num,
                        &format!("{:?}", record),
                        e,
                    )
                })?);
            }

            labels.push(record[8].parse::<f64>().map_err(|e| {
                DatasetError::parse_failed(
                    DIABETES_DATASET_NAME,
                    "label",
                    line_num,
                    &format!("{:?}", record),
                    e,
                )
            })?);
        }

        // Verify the dataset is not empty
        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::length_mismatch(
                DIABETES_DATASET_NAME,
                "samples",
                1, // At least 1 expected
                0,
            ));
        }

        let features_array =
            Array2::from_shape_vec((n_samples, DIABETES_NUM_FEATURES), features)
                .map_err(|e| {
                    DatasetError::array_shape_error(DIABETES_DATASET_NAME, "features", e)
                })?;
        let labels_array = Array1::from_vec(labels);

        Ok((features_array, labels_array))
    }

    /// Internal helper to ensure data is loaded and return a reference.
    fn load_data(&self) -> Result<&(Array2<f64>, Array1<f64>), DatasetError> {
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
            .expect("DIABETES_DATA should be initialized after set");
        Ok(cache)
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(768, 8)` containing:
    ///     - Pregnancies: Number of times pregnant
    ///     - Glucose: Plasma glucose concentration at 2 hours in an oral glucose tolerance test
    ///     - BloodPressure: Diastolic blood pressure (mm Hg)
    ///     - SkinThickness: Triceps skin fold thickness (mm)
    ///     - Insulin: 2-Hour serum insulin (mu U/ml)
    ///     - BMI: Body mass index (weight in kg/(height in m)^2)
    ///     - DiabetesPedigreeFunction: Diabetes pedigree function
    ///     - Age: Age (years)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (768 samples, 8 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.load_data()?.0)
    }

    /// Get a reference to the label vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<f64>` - Reference to label vector with shape `(768,)` containing class variable (0 or 1)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (768 samples)
    pub fn labels(&self) -> Result<&Array1<f64>, DatasetError> {
        Ok(&self.load_data()?.1)
    }

    /// Get both features and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(768, 8)` containing:
    ///     - Pregnancies: Number of times pregnant
    ///     - Glucose: Plasma glucose concentration at 2 hours in an oral glucose tolerance test
    ///     - BloodPressure: Diastolic blood pressure (mm Hg)
    ///     - SkinThickness: Triceps skin fold thickness (mm)
    ///     - Insulin: 2-Hour serum insulin (mu U/ml)
    ///     - BMI: Body mass index (weight in kg/(height in m)^2)
    ///     - DiabetesPedigreeFunction: Diabetes pedigree function
    ///     - Age: Age (years)
    /// - `&Array1<f64>` - Reference to label vector with shape `(768,)` containing class variable (0 or 1)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (768 samples, 8 features)
    pub fn data(&self) -> Result<(&Array2<f64>, &Array1<f64>), DatasetError> {
        let data = self.load_data()?;
        Ok((&data.0, &data.1))
    }
}