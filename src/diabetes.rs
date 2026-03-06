use std::fs::{remove_file, rename, File};
use std::io::Read;
use std::path::Path;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;
use crate::{download_to, DatasetError, create_temp_dir, file_sha256_matches, prepare_download_dir};

/// The URL for the Diabetes dataset.
const DIABETES_DATA_URL: &str = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv";

/// The prefix for temporary files created during dataset download and parsing.
const DIABETES_TEMP_FILE_PREFIX: &str = ".tmp-diabetes-";

/// A static string slice containing the name of the Diabetes dataset file.
const DIABETES_FILENAME: &str = "diabetes.csv";

/// The number of samples in the Diabetes dataset.
const DIABETES_SAMPLE_SIZE: usize = 768;

/// The number of features in the Diabetes dataset.
const DIABETES_NUM_FEATURES: usize = 8;

/// The SHA256 hash of the Diabetes dataset file.
const DIABETES_SHA256: &str = "698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72";

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
/// # Fields
///
/// - `storage_path` - Directory path where the dataset will be stored.
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
/// // clean up: remove the downloaded files
/// std::fs::remove_dir_all(download_dir).unwrap();
/// ```
pub struct Diabetes {
    storage_path: String,
    data: OnceLock<(Array2<f64>, Array1<f64>)>,
}

impl Diabetes {
    /// Create a new Diabetes instance without loading data.
    ///
    /// The dataset will be loaded lazily when you first call any data accessor method.
    /// This is a lightweight operation that only stores the storage path.
    ///
    /// # Parameters
    ///
    /// - `storage_path` - Directory path where the dataset will be stored.
    ///
    /// # Returns
    ///
    /// - `Self` - `Diabetes` instance ready for lazy loading.
    pub fn new(storage_path: &str) -> Self {
        Diabetes {
            storage_path: storage_path.to_string(),
            data: OnceLock::new(),
        }
    }

    /// Internal function to load the dataset from disk or download it.
    ///
    /// This function is called automatically by the accessor methods.
    fn load_data_internal(path: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
        let path = Path::new(path);
        let dst = path.join(DIABETES_FILENAME);
        let (need_download, need_overwrite) =
            prepare_download_dir(path, &dst, DIABETES_SHA256)?;

        if need_download {
            // temporary directory
            let temp_dir = create_temp_dir(path, DIABETES_TEMP_FILE_PREFIX)?;
            let path_temp = temp_dir.path();
            // download file to temporary directory
            download_to(DIABETES_DATA_URL, path_temp)?;
            // move downloaded file to final location
            let src = path_temp.join(DIABETES_FILENAME);
            // check if the file matches the expected SHA256 hash
            if !file_sha256_matches(src.as_path(), DIABETES_SHA256)? {
                return Err(DatasetError::ValidationError(format!("{} SHA256 validation failed", DIABETES_FILENAME)));
            }
            if need_overwrite {
                remove_file(&dst).map_err(|e| DatasetError::StdIoError(e))?;
            }
            rename(src, &dst).map_err(|e| DatasetError::StdIoError(e))?;
        }

        let mut file = File::open(dst).map_err(|e| DatasetError::StdIoError(e))?;
        let mut data = String::new();
        file.read_to_string(&mut data).map_err(|e| DatasetError::StdIoError(e))?;

        let mut features = Vec::with_capacity(DIABETES_SAMPLE_SIZE * DIABETES_NUM_FEATURES);
        let mut labels = Vec::with_capacity(DIABETES_SAMPLE_SIZE);

        let lines: Vec<&str> = data.trim().lines().collect();

        // Process lines as data (skip header)
        for line in &lines[1..] {
            if line.is_empty() { continue; }
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() != DIABETES_NUM_FEATURES + 1 {
                return Err(DatasetError::DataFormatError(
                    format!("Expected {} columns, got {} at line {}",
                            DIABETES_NUM_FEATURES + 1,
                            cols.len(),
                            line
                    )
                ));
            }

            for i in 0..DIABETES_NUM_FEATURES {
                features.push(cols[i].parse::<f64>().map_err(
                    |e| DatasetError::DataFormatError(
                        format!("Failed to parse Diabetes dataset features {} at line {}: {}", i, line, e)
                    ))?);
            }

            labels.push(cols[8].parse::<f64>().map_err(
                |e| DatasetError::DataFormatError(
                    format!("Failed to parse Diabetes label at line {}: {}", line, e)
                )
            )?);
        }

        if features.len() != DIABETES_SAMPLE_SIZE * DIABETES_NUM_FEATURES {
            return Err(DatasetError::DataFormatError(
                format!("Expected {} * {} elements in features, got {}",
                        DIABETES_SAMPLE_SIZE,
                        DIABETES_NUM_FEATURES,
                        features.len()
                )
            ));
        }
        if labels.len() != DIABETES_SAMPLE_SIZE {
            return Err(DatasetError::DataFormatError(
                format!("Expected {} elements in labels, got {}", DIABETES_SAMPLE_SIZE, labels.len())
            ));
        }

        let features_array = Array2::from_shape_vec((DIABETES_SAMPLE_SIZE, DIABETES_NUM_FEATURES), features)
            .map_err(|e| DatasetError::DataFormatError(
                format!("Failed to create features array: {}", e)
            ))?;
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
        let (features, labels) = Self::load_data_internal(&self.storage_path)?;

        // Try to set the value. If another thread already set it, that's fine - just use the existing value
        let _ = self.data.set((features, labels));

        let cache = self.data
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