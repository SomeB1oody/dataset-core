//! Breast Cancer Wisconsin (Diagnostic) dataset.
//!
//! Features computed from a digitized image of a fine needle aspirate (FNA) of a
//! breast mass, describing characteristics of the cell nuclei present in the
//! image. The task is to predict whether a tumor is malignant or benign.
//!
//! For each of 10 base measurements (`radius`, `texture`, `perimeter`, `area`,
//! `smoothness`, `compactness`, `concavity`, `concave_points`, `symmetry`,
//! `fractal_dimension`) the dataset reports three statistics — the `mean`, the
//! standard error (`se`), and the `worst` (mean of the three largest values) —
//! giving **30 features** in total.
//!
//! **Features (30):** `<measurement>_mean`, `<measurement>_se`, and
//! `<measurement>_worst` for each of the 10 measurements listed above.
//!
//! **Target:** `diagnosis` - one of `malignant` or `benign`
//!
//! **Samples:** 569 total (212 malignant, 357 benign)
//! **Application:** Binary classification / tumor diagnosis
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5DW2B>

use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::fs::File;

/// The URL for the Breast Cancer Wisconsin (Diagnostic) dataset.
///
/// # Citation
///
/// W. Wolberg, O. Mangasarian, N. Street, and W. Street. "Breast Cancer
/// Wisconsin (Diagnostic)," UCI Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C5DW2B>
const BREAST_CANCER_DATA_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data";

/// The name of the Breast Cancer dataset file.
const BREAST_CANCER_FILENAME: &str = "breast_cancer.csv";

/// The SHA256 hash of the Breast Cancer dataset file.
const BREAST_CANCER_SHA256: &str =
    "d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a";

/// The name of the dataset
const BREAST_CANCER_DATASET_NAME: &str = "breast_cancer";

/// The number of features per sample (10 measurements × {mean, se, worst}).
const N_FEATURES: usize = 30;

/// Type alias for the Breast Cancer dataset: (features, labels).
type BreastCancerData = (Array2<f64>, Array1<&'static str>);

/// One CSV record of the Breast Cancer dataset: an ID column, the `M`/`B`
/// diagnosis, then the 30 `f64` features (`mean`, `se`, and `worst` for each of
/// the 10 base measurements, in that block order).
///
/// Fields are declared in CSV column order and deserialized **positionally**
/// (the loader disables csv's header handling), matching the headerless
/// `wdbc.data` layout.
#[derive(Deserialize)]
struct BreastCancerRecord {
    /// The sample ID. Present only to consume the first CSV column positionally;
    /// it is not exposed as a feature, so it is intentionally never read.
    #[allow(dead_code)]
    id: u64,
    diagnosis: String,
    radius_mean: f64,
    texture_mean: f64,
    perimeter_mean: f64,
    area_mean: f64,
    smoothness_mean: f64,
    compactness_mean: f64,
    concavity_mean: f64,
    concave_points_mean: f64,
    symmetry_mean: f64,
    fractal_dimension_mean: f64,
    radius_se: f64,
    texture_se: f64,
    perimeter_se: f64,
    area_se: f64,
    smoothness_se: f64,
    compactness_se: f64,
    concavity_se: f64,
    concave_points_se: f64,
    symmetry_se: f64,
    fractal_dimension_se: f64,
    radius_worst: f64,
    texture_worst: f64,
    perimeter_worst: f64,
    area_worst: f64,
    smoothness_worst: f64,
    compactness_worst: f64,
    concavity_worst: f64,
    concave_points_worst: f64,
    symmetry_worst: f64,
    fractal_dimension_worst: f64,
}

/// A struct representing the Breast Cancer Wisconsin (Diagnostic) dataset with
/// lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// Features are computed from a digitized image of a fine needle aspirate (FNA)
/// of a breast mass and describe characteristics of the cell nuclei present in
/// the image. For each of 10 base measurements (radius, texture, perimeter,
/// area, smoothness, compactness, concavity, concave points, symmetry, fractal
/// dimension) the mean, standard error, and "worst" (mean of the three largest
/// values) are recorded, for 30 features in total.
///
/// # Feature columns
///
/// All 30 features are quantitative. They are laid out in three blocks of 10:
/// the `mean` of each base measurement (columns `0`–`9`), its standard error
/// (`se`, columns `10`–`19`), and its `worst` value (columns `20`–`29`).
///
/// | Columns | Attributes                | Unit |
/// |---------|---------------------------|------|
/// | `0`     | `radius_mean`             |      |
/// | `1`     | `texture_mean`            |      |
/// | `2`     | `perimeter_mean`          |      |
/// | `3`     | `area_mean`               |      |
/// | `4`     | `smoothness_mean`         |      |
/// | `5`     | `compactness_mean`        |      |
/// | `6`     | `concavity_mean`          |      |
/// | `7`     | `concave_points_mean`     |      |
/// | `8`     | `symmetry_mean`           |      |
/// | `9`     | `fractal_dimension_mean`  |      |
/// | `10`    | `radius_se`               |      |
/// | `11`    | `texture_se`              |      |
/// | `12`    | `perimeter_se`            |      |
/// | `13`    | `area_se`                 |      |
/// | `14`    | `smoothness_se`           |      |
/// | `15`    | `compactness_se`          |      |
/// | `16`    | `concavity_se`            |      |
/// | `17`    | `concave_points_se`       |      |
/// | `18`    | `symmetry_se`             |      |
/// | `19`    | `fractal_dimension_se`    |      |
/// | `20`    | `radius_worst`            |      |
/// | `21`    | `texture_worst`           |      |
/// | `22`    | `perimeter_worst`         |      |
/// | `23`    | `area_worst`              |      |
/// | `24`    | `smoothness_worst`        |      |
/// | `25`    | `compactness_worst`       |      |
/// | `26`    | `concavity_worst`         |      |
/// | `27`    | `concave_points_worst`    |      |
/// | `28`    | `symmetry_worst`          |      |
/// | `29`    | `fractal_dimension_worst` |      |
///
/// # Labels
///
/// - diagnosis (in `&str`): `"malignant"`, `"benign"`
///
/// See more information at <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>
///
/// # Citation
///
/// W. Wolberg, O. Mangasarian, N. Street, and W. Street. "Breast Cancer
/// Wisconsin (Diagnostic)," UCI Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C5DW2B>
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::breast_cancer::BreastCancer;
///
/// let download_dir = "./breast_cancer"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = BreastCancer::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
/// assert_eq!(features.shape(), &[569, 30]);
/// assert_eq!(labels.len(), 569);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 15.0;
///     labels[0] = "benign";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable — the next access reloads from the cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[569, 30]);
/// assert_eq!(owned_labels.len(), 569);
///
/// // `into_data()` also returns owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[569, 30]);
/// assert_eq!(owned_labels.len(), 569);
/// ```
#[derive(Debug)]
pub struct BreastCancer {
    dataset: Dataset<BreastCancerData, DatasetError>,
}

impl BreastCancer {
    /// Create a new BreastCancer instance without loading data.
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
    /// - `Self` - `BreastCancer` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        BreastCancer {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Breast Cancer dataset.
    fn load_data(dir: &str) -> Result<BreastCancerData, DatasetError> {
        // Prepare the dataset file
        let file_path = acquire_dataset(
            dir,
            BREAST_CANCER_FILENAME,
            BREAST_CANCER_DATASET_NAME,
            Some(BREAST_CANCER_SHA256),
            |temp_path| {
                download_to(
                    BREAST_CANCER_DATA_URL,
                    temp_path,
                    Some(BREAST_CANCER_FILENAME),
                )?;
                Ok(temp_path.join(BREAST_CANCER_FILENAME))
            },
        )?;

        // csv deserializes into the struct. `wdbc.data` has no header row, so
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features = Vec::new();
        let mut labels = Vec::new();

        for (idx, result) in rdr.deserialize::<BreastCancerRecord>().enumerate() {
            let BreastCancerRecord {
                id: _,
                diagnosis,
                radius_mean,
                texture_mean,
                perimeter_mean,
                area_mean,
                smoothness_mean,
                compactness_mean,
                concavity_mean,
                concave_points_mean,
                symmetry_mean,
                fractal_dimension_mean,
                radius_se,
                texture_se,
                perimeter_se,
                area_se,
                smoothness_se,
                compactness_se,
                concavity_se,
                concave_points_se,
                symmetry_se,
                fractal_dimension_se,
                radius_worst,
                texture_worst,
                perimeter_worst,
                area_worst,
                smoothness_worst,
                compactness_worst,
                concavity_worst,
                concave_points_worst,
                symmetry_worst,
                fractal_dimension_worst,
            } = result.map_err(|e| DatasetError::csv_read_error(BREAST_CANCER_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            features.extend_from_slice(&[
                radius_mean,
                texture_mean,
                perimeter_mean,
                area_mean,
                smoothness_mean,
                compactness_mean,
                concavity_mean,
                concave_points_mean,
                symmetry_mean,
                fractal_dimension_mean,
                radius_se,
                texture_se,
                perimeter_se,
                area_se,
                smoothness_se,
                compactness_se,
                concavity_se,
                concave_points_se,
                symmetry_se,
                fractal_dimension_se,
                radius_worst,
                texture_worst,
                perimeter_worst,
                area_worst,
                smoothness_worst,
                compactness_worst,
                concavity_worst,
                concave_points_worst,
                symmetry_worst,
                fractal_dimension_worst,
            ]);

            labels.push(match diagnosis.as_str() {
                "M" => "malignant",
                "B" => "benign",
                other => {
                    return Err(DatasetError::invalid_value(
                        BREAST_CANCER_DATASET_NAME,
                        "diagnosis",
                        other,
                        line_num,
                    ));
                }
            });
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(BREAST_CANCER_DATASET_NAME));
        }

        // Breast Cancer has a fixed schema of 30 numeric features per sample.
        let features_array =
            Array2::from_shape_vec((n_samples, N_FEATURES), features).map_err(|e| {
                DatasetError::array_shape_error(BREAST_CANCER_DATASET_NAME, "features", e)
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
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(569, 30)`
    ///   containing the `mean`, `se`, and `worst` statistics for each of the 10
    ///   base nucleus measurements.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (569 samples, 30 features)
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
    /// - `&Array1<&'static str>` - Reference to labels vector with shape `(569,)` containing diagnoses (`"malignant"`, `"benign"`)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (569 samples)
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
    /// - `&BreastCancerData` - reference to the cached `(features, labels)` tuple:
    ///   the feature matrix has shape `(569, 30)` and the label vector has shape
    ///   `(569,)` containing diagnoses (`"malignant"`, `"benign"`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (569 samples, 30 features)
    pub fn data(&self) -> Result<&BreastCancerData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and labels as references **without** triggering loading.
    ///
    /// Unlike [`BreastCancer::data`], which loads the dataset on first call, this
    /// never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&BreastCancerData)` - reference to the cached `(features, labels)`
    ///   tuple (feature matrix `(569, 30)`, label vector `(569,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&BreastCancerData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize features,
    /// replace label values) with no `to_owned()` clone and without removing them
    /// from the cache: the changes persist, so later [`BreastCancer::features`],
    /// [`BreastCancer::data`], or [`BreastCancer::get_data`] calls observe them.
    ///
    /// Like [`BreastCancer::get_data`], this does **not** trigger loading: it
    /// returns `None` if the dataset has not been loaded. Call a loading accessor
    /// (e.g. [`BreastCancer::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut BreastCancerData)` - mutable reference to the cached
    ///   `(features, labels)` tuple (feature matrix `(569, 30)`, label vector
    ///   `(569,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut BreastCancerData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`BreastCancer::data`], which borrows the cached data, this moves it
    /// out and returns owned arrays directly — no `to_owned()` clone needed. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`BreastCancer::take_data`] instead — it takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<&'static str>)` - owned feature matrix with shape
    ///   `(569, 30)` and owned label vector with shape `(569,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn into_data(self) -> Result<BreastCancerData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`BreastCancer::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out, resetting the instance to its
    /// unloaded state: the next accessor call (e.g. [`BreastCancer::features`] or
    /// [`BreastCancer::data`]) loads the dataset again.
    ///
    /// Use [`BreastCancer::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<&'static str>)` - owned feature matrix with shape
    ///   `(569, 30)` and owned label vector with shape `(569,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<BreastCancerData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}
