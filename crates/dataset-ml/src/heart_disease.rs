//! Heart Disease (Cleveland) dataset.
//!
//! The dataset holds clinical records from the Cleveland Clinic Foundation,
//! collected by Robert Detrano. The task is to predict the presence of heart
//! disease in a patient. This loader uses the canonical
//! `processed.cleveland.data` partition. This is the 14-column subset that
//! virtually all published experiments on this database use (303 patients,
//! 13 features plus the diagnosis).
//!
//! **Features (13, all numeric):**
//! - `age`: age in years
//! - `sex`: `1` = male, `0` = female
//! - `cp`: chest pain type (`1`–`4`)
//! - `trestbps`: resting blood pressure (mm Hg)
//! - `chol`: serum cholesterol (mg/dl)
//! - `fbs`: fasting blood sugar > 120 mg/dl (`1` = true, `0` = false)
//! - `restecg`: resting electrocardiographic results (`0`, `1`, `2`)
//! - `thalach`: maximum heart rate achieved
//! - `exang`: exercise-induced angina (`1` = yes, `0` = no)
//! - `oldpeak`: ST depression induced by exercise relative to rest
//! - `slope`: slope of the peak exercise ST segment (`1`–`3`)
//! - `ca`: number of major vessels (`0`–`3`) colored by fluoroscopy (has missing values)
//! - `thal`: `3` = normal, `6` = fixed defect, `7` = reversible defect (has missing values)
//!
//! **Target:** `num`, diagnosis of heart disease from `0` (absence) through
//! `4` (increasing presence). Commonly binarized to absence (`0`) vs presence (`> 0`).
//!
//! **Samples:** 303
//! **Application:** (Multi-class) classification / heart-disease diagnosis
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C52P4X>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::{Array1, Array2};
use std::fs::File;

use csv::ReaderBuilder;

/// Type alias for the Heart Disease dataset: (features, labels).
type HeartDiseaseData = (Array2<f64>, Array1<u8>);

/// The URL for the Heart Disease dataset (the `processed.cleveland.data` file).
const HEART_DISEASE_DATA_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data";

/// The name of the cached Heart Disease dataset file.
const HEART_DISEASE_FILENAME: &str = "heart_disease.csv";

/// The SHA256 hash of the cached Heart Disease dataset file (`processed.cleveland.data`'s bytes).
const HEART_DISEASE_SHA256: &str =
    "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8";

/// The name of the dataset.
const HEART_DISEASE_DATASET_NAME: &str = "heart_disease";

/// Number of samples.
const N_SAMPLES: usize = 303;

/// Number of numeric features.
const N_FEATURES: usize = 13;

/// Number of columns per record (13 features + 1 target).
const N_COLUMNS: usize = 14;

/// Source column index of the target (`num`). The target is the **last** column.
const TARGET_COLUMN: usize = 13;

/// Numeric feature columns, as `(source column index, name)`, in output order.
const FEATURE_COLUMNS: [(usize, &str); N_FEATURES] = [
    (0, "age"),
    (1, "sex"),
    (2, "cp"),
    (3, "trestbps"),
    (4, "chol"),
    (5, "fbs"),
    (6, "restecg"),
    (7, "thalach"),
    (8, "exang"),
    (9, "oldpeak"),
    (10, "slope"),
    (11, "ca"),
    (12, "thal"),
];

/// The token marking a missing value in the source (only in `ca` and `thal`).
const MISSING_TOKEN: &str = "?";

/// This struct represents the Heart Disease (Cleveland) dataset and loads it lazily.
///
/// Nothing loads until you call a data accessor method. After loading, the
/// data stays cached for later accesses.
///
/// # About Dataset
///
/// This database contains 76 attributes, but all published experiments use a
/// subset of 14 of them: the `processed.cleveland.data` file used here. The
/// "goal" field (`num`) refers to the presence of heart disease in the
/// patient. It is an integer value from `0` (no presence) to `4`. Most
/// experiments simply try to distinguish presence (values `1`, `2`, `3`, `4`)
/// from absence (value `0`). The data comes from the Cleveland Clinic
/// Foundation. Robert Detrano supplied it.
///
/// # Feature columns
///
/// All 13 features are numeric, stored in one `(303, 13)` `Array2<f64>` matrix.
/// Several are integer-coded categoricals kept as `f64`:
///
/// | Column | Attribute  | Meaning                                                  |
/// |--------|------------|----------------------------------------------------------|
/// | `0`    | `age`      | age in years                                             |
/// | `1`    | `sex`      | `1` = male, `0` = female                                 |
/// | `2`    | `cp`       | chest pain type (`1`–`4`)                                |
/// | `3`    | `trestbps` | resting blood pressure (mm Hg)                           |
/// | `4`    | `chol`     | serum cholesterol (mg/dl)                                |
/// | `5`    | `fbs`      | fasting blood sugar > 120 mg/dl (`1`/`0`)                |
/// | `6`    | `restecg`  | resting ECG results (`0`, `1`, `2`)                      |
/// | `7`    | `thalach`  | maximum heart rate achieved                              |
/// | `8`    | `exang`    | exercise-induced angina (`1`/`0`)                        |
/// | `9`    | `oldpeak`  | ST depression induced by exercise relative to rest       |
/// | `10`   | `slope`    | slope of the peak exercise ST segment (`1`–`3`)          |
/// | `11`   | `ca`       | number of major vessels (`0`–`3`) colored by fluoroscopy |
/// | `12`   | `thal`     | `3` = normal, `6` = fixed defect, `7` = reversible defect |
///
/// # Labels
///
/// - `num` (shape `(303,)`): the `Array1<u8>` diagnosis, `0` (absence) through
///   `4` (increasing presence). It is commonly binarized to absence (`0`) vs
///   presence (`> 0`).
///
/// Missing values:
/// - The source marks missing values with `?`: 4 in `ca` (column `11`) and 2 in
///   `thal` (column `12`), for 6 affected patients. The loader maps these to
///   `NaN` (like the missing numeric values in [`crate::titanic`] and
///   [`crate::palmer_penguins`]).
///
/// See more information at <https://archive.ics.uci.edu/dataset/45/heart+disease>.
///
/// # Citation
///
/// Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease
/// \[Dataset\]. UCI Machine Learning Repository. <https://doi.org/10.24432/C52P4X>
///
/// # Thread Safety
///
/// Every field implements `Send` and `Sync`, so this struct implements them too. It is safe
/// to share across threads.
/// The internal [`Dataset`] makes initialization thread-safe and lazy.
///
/// # Example
/// ```no_run
/// use dataset_ml::heart_disease::HeartDisease;
///
/// let download_dir = "./heart_disease"; // creates the directory if it is missing
///
/// let mut dataset = HeartDisease::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap(); // also a way to get features and labels
/// assert_eq!(features.shape(), &[303, 13]);
/// assert_eq!(labels.len(), 303);
///
/// // `get_data()` borrows the cached arrays without reloading. `get_data_mut()`
/// // edits them in place. It needs no clone and no reload, and the change
/// // stays cached. Prefer this method over cloning with `.to_owned()` when
/// // you only need to change values.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 60.0;
///     labels[0] = 1;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable. The next access reloads from the cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[303, 13]);
/// assert_eq!(owned_labels.len(), 303);
///
/// // `into_data()` also returns owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[303, 13]);
/// assert_eq!(owned_labels.len(), 303);
/// ```
#[derive(Debug)]
pub struct HeartDisease {
    dataset: Dataset<HeartDiseaseData, DatasetError>,
}

impl HeartDisease {
    /// Create a new HeartDisease instance without loading data.
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
    /// - `Self` - `HeartDisease` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        HeartDisease {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Heart Disease dataset.
    fn load_data(dir: &str) -> Result<HeartDiseaseData, DatasetError> {
        // Prepare the dataset file. The source file is `processed.cleveland.data`.
        // The code caches it as `heart_disease.csv`.
        let file_path = acquire_dataset(
            dir,
            HEART_DISEASE_FILENAME,
            HEART_DISEASE_DATASET_NAME,
            Some(HEART_DISEASE_SHA256),
            |temp_path| {
                download_to_with_retries(
                    HEART_DISEASE_DATA_URL,
                    temp_path,
                    Some(HEART_DISEASE_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(HEART_DISEASE_FILENAME))
            },
        )?;

        // The source is plain comma-separated with no header.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features: Vec<f64> = Vec::with_capacity(N_SAMPLES * N_FEATURES);
        let mut labels: Vec<u8> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(HEART_DISEASE_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            // Skip blank lines defensively (e.g. a trailing newline).
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    HEART_DISEASE_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // Numeric features, mapping the `?` missing token to NaN.
            for &(col, name) in FEATURE_COLUMNS.iter() {
                let raw = &record[col];
                if raw == MISSING_TOKEN {
                    features.push(f64::NAN);
                } else {
                    let value: f64 = raw.parse().map_err(|e| {
                        DatasetError::parse_failed(HEART_DISEASE_DATASET_NAME, name, line_num, e)
                    })?;
                    features.push(value);
                }
            }

            // Target, an integer diagnosis in 0..=4. The source stores it as a
            // float-formatted integer in some related files, but the Cleveland
            // partition stores a plain integer.
            let target: u8 = record[TARGET_COLUMN].parse().map_err(|e| {
                DatasetError::parse_failed(HEART_DISEASE_DATASET_NAME, "num", line_num, e)
            })?;
            labels.push(target);
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(HEART_DISEASE_DATASET_NAME));
        }

        let features_array =
            Array2::from_shape_vec((n_samples, N_FEATURES), features).map_err(|e| {
                DatasetError::array_shape_error(HEART_DISEASE_DATASET_NAME, "features", e)
            })?;

        let labels_array = Array1::from_vec(labels);

        Ok((features_array, labels_array))
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on first call. Later calls return the
    /// cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to the numeric feature matrix with shape
    ///   `(303, 13)`. Missing `ca`/`thal` values are `NaN`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (303 samples, 13 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the labels vector.
    ///
    /// This method triggers lazy loading on first call. Later calls return the
    /// cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<u8>` - Reference to labels vector with shape `(303,)` containing
    ///   the `num` diagnosis (`0` absence through `4` increasing presence).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (303 samples)
    pub fn labels(&self) -> Result<&Array1<u8>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and labels as references.
    ///
    /// This method triggers lazy loading on first call. Later calls return the
    /// cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&HeartDiseaseData` - reference to the cached `(features, labels)` tuple:
    ///   the feature matrix has shape `(303, 13)` and the label vector has shape
    ///   `(303,)` containing the `num` diagnosis (`0`–`4`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (303 samples, 13 features)
    pub fn data(&self) -> Result<&HeartDiseaseData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and labels as references, without triggering loading.
    ///
    /// Unlike [`HeartDisease::data`], which loads the dataset on first call, this
    /// never runs the loader. If the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing.
    ///
    /// Use this method when you want the data only if it is already cached. This
    /// avoids the download and parse cost when the data is not cached.
    ///
    /// # Returns
    ///
    /// - `Some(&HeartDiseaseData)` - reference to the cached `(features, labels)`
    ///   tuple (feature matrix `(303, 13)`, label vector `(303,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&HeartDiseaseData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This lets you change the cached arrays directly (e.g. impute the missing
    /// `NaN` values, binarize the target). It needs no `to_owned()` clone, and
    /// the arrays stay in the cache. The changes persist, so later calls to
    /// [`HeartDisease::features`], [`HeartDisease::data`], or
    /// [`HeartDisease::get_data`] see them.
    ///
    /// Like [`HeartDisease::get_data`], this does **not** trigger loading. It
    /// returns `None` if the dataset has not been loaded. If you need to make
    /// sure the data is present, call a loading accessor first (e.g.
    /// [`HeartDisease::data`]).
    ///
    /// # Returns
    ///
    /// - `Some(&mut HeartDiseaseData)` - mutable reference to the cached
    ///   `(features, labels)` tuple (feature matrix `(303, 13)`, label vector
    ///   `(303,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut HeartDiseaseData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`HeartDisease::data`], which borrows the cached data, this moves it
    /// out and returns owned arrays directly. It needs no `to_owned()` clone. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`HeartDisease::take_data`] instead. It takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<u8>)` - owned feature matrix with shape `(303, 13)`
    ///   and owned label vector with shape `(303,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<HeartDiseaseData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and labels out of the dataset. The instance stays
    /// reusable.
    ///
    /// Like [`HeartDisease::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out. This resets the instance to
    /// its unloaded state. The next accessor call (e.g. [`HeartDisease::features`]
    /// or [`HeartDisease::data`]) loads the dataset again.
    ///
    /// If you are done with the instance, use [`HeartDisease::into_data`] instead.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<u8>)` - owned feature matrix with shape `(303, 13)`
    ///   and owned label vector with shape `(303,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<HeartDiseaseData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(HeartDisease, HeartDiseaseData, "heart_disease");
