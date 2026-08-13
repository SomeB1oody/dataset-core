//! Breast Cancer Wisconsin (Diagnostic) dataset.
//!
//! This dataset's features come from a digitized image of a fine needle
//! aspirate (FNA) of a breast mass. They describe characteristics of the cell
//! nuclei in the image. The task is to predict whether a tumor is malignant or
//! benign.
//!
//! The source measures 10 properties of each cell nucleus: `radius`, `texture`,
//! `perimeter`, `area`, `smoothness`, `compactness`, `concavity`,
//! `concave_points`, `symmetry`, and `fractal_dimension`. For each property the
//! source reports three statistics: the mean (`_mean`), the standard error
//! (`_se`), and the worst value (`_worst`, the mean of the three largest
//! values). This gives 30 feature columns.
//!
//! **Columns (31):**
//!
//! | Name | Type | Description |
//! |------|------|-------------|
//! | `diagnosis` | `String` | `malignant` or `benign` |
//! | `radius_mean` | `Numeric` | mean of `radius` |
//! | `texture_mean` | `Numeric` | mean of `texture` |
//! | `perimeter_mean` | `Numeric` | mean of `perimeter` |
//! | `area_mean` | `Numeric` | mean of `area` |
//! | `smoothness_mean` | `Numeric` | mean of `smoothness` |
//! | `compactness_mean` | `Numeric` | mean of `compactness` |
//! | `concavity_mean` | `Numeric` | mean of `concavity` |
//! | `concave_points_mean` | `Numeric` | mean of `concave_points` |
//! | `symmetry_mean` | `Numeric` | mean of `symmetry` |
//! | `fractal_dimension_mean` | `Numeric` | mean of `fractal_dimension` |
//! | `radius_se` | `Numeric` | standard error of `radius` |
//! | `texture_se` | `Numeric` | standard error of `texture` |
//! | `perimeter_se` | `Numeric` | standard error of `perimeter` |
//! | `area_se` | `Numeric` | standard error of `area` |
//! | `smoothness_se` | `Numeric` | standard error of `smoothness` |
//! | `compactness_se` | `Numeric` | standard error of `compactness` |
//! | `concavity_se` | `Numeric` | standard error of `concavity` |
//! | `concave_points_se` | `Numeric` | standard error of `concave_points` |
//! | `symmetry_se` | `Numeric` | standard error of `symmetry` |
//! | `fractal_dimension_se` | `Numeric` | standard error of `fractal_dimension` |
//! | `radius_worst` | `Numeric` | worst value of `radius` |
//! | `texture_worst` | `Numeric` | worst value of `texture` |
//! | `perimeter_worst` | `Numeric` | worst value of `perimeter` |
//! | `area_worst` | `Numeric` | worst value of `area` |
//! | `smoothness_worst` | `Numeric` | worst value of `smoothness` |
//! | `compactness_worst` | `Numeric` | worst value of `compactness` |
//! | `concavity_worst` | `Numeric` | worst value of `concavity` |
//! | `concave_points_worst` | `Numeric` | worst value of `concave_points` |
//! | `symmetry_worst` | `Numeric` | worst value of `symmetry` |
//! | `fractal_dimension_worst` | `Numeric` | worst value of `fractal_dimension` |
//!
//! The source designates the 30 cell-nucleus measurements as the inputs
//! ([`BreastCancer::FEATURE_NAMES`](crate::BreastCancer::FEATURE_NAMES)) and `diagnosis` as the label
//! ([`BreastCancer::TARGET`](crate::BreastCancer::TARGET)).
//!
//! **Samples:** 569 total (212 malignant, 357 benign)
//! **Application:** Binary classification / tumor diagnosis
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5DW2B>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
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

/// One CSV record of the Breast Cancer dataset holds an ID column, the
/// `M`/`B` diagnosis, and 30 `f64` features. The features are `mean`, `se`,
/// and `worst` for each of the 10 base measurements, in that block order.
///
/// The struct declares its fields in CSV column order. The loader disables
/// csv's header handling, so csv deserializes the fields **positionally**.
/// This matches the headerless `wdbc.data` layout.
#[derive(Deserialize)]
struct BreastCancerRecord {
    /// The sample ID. This field consumes the first CSV column positionally.
    /// It is not a feature. The loader never reads it.
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

/// A struct that represents the Breast Cancer Wisconsin (Diagnostic) dataset
/// with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// This dataset's features come from a digitized image of a fine needle
/// aspirate (FNA) of a breast mass. They describe characteristics of the cell
/// nuclei in the image. The source measures 10 properties of each cell nucleus:
/// radius, texture, perimeter, area, smoothness, compactness, concavity,
/// concave points, symmetry, and fractal dimension. For each property the source
/// reports the mean, the standard error, and the "worst" value (the mean of the
/// three largest values). This gives 30 feature columns.
///
/// # Columns
///
/// | Name | Type | Description |
/// |------|------|-------------|
/// | `diagnosis` | `String` | `malignant` or `benign` |
/// | `radius_mean` | `Numeric` | mean of `radius` |
/// | `texture_mean` | `Numeric` | mean of `texture` |
/// | `perimeter_mean` | `Numeric` | mean of `perimeter` |
/// | `area_mean` | `Numeric` | mean of `area` |
/// | `smoothness_mean` | `Numeric` | mean of `smoothness` |
/// | `compactness_mean` | `Numeric` | mean of `compactness` |
/// | `concavity_mean` | `Numeric` | mean of `concavity` |
/// | `concave_points_mean` | `Numeric` | mean of `concave_points` |
/// | `symmetry_mean` | `Numeric` | mean of `symmetry` |
/// | `fractal_dimension_mean` | `Numeric` | mean of `fractal_dimension` |
/// | `radius_se` | `Numeric` | standard error of `radius` |
/// | `texture_se` | `Numeric` | standard error of `texture` |
/// | `perimeter_se` | `Numeric` | standard error of `perimeter` |
/// | `area_se` | `Numeric` | standard error of `area` |
/// | `smoothness_se` | `Numeric` | standard error of `smoothness` |
/// | `compactness_se` | `Numeric` | standard error of `compactness` |
/// | `concavity_se` | `Numeric` | standard error of `concavity` |
/// | `concave_points_se` | `Numeric` | standard error of `concave_points` |
/// | `symmetry_se` | `Numeric` | standard error of `symmetry` |
/// | `fractal_dimension_se` | `Numeric` | standard error of `fractal_dimension` |
/// | `radius_worst` | `Numeric` | worst value of `radius` |
/// | `texture_worst` | `Numeric` | worst value of `texture` |
/// | `perimeter_worst` | `Numeric` | worst value of `perimeter` |
/// | `area_worst` | `Numeric` | worst value of `area` |
/// | `smoothness_worst` | `Numeric` | worst value of `smoothness` |
/// | `compactness_worst` | `Numeric` | worst value of `compactness` |
/// | `concavity_worst` | `Numeric` | worst value of `concavity` |
/// | `concave_points_worst` | `Numeric` | worst value of `concave_points` |
/// | `symmetry_worst` | `Numeric` | worst value of `symmetry` |
/// | `fractal_dimension_worst` | `Numeric` | worst value of `fractal_dimension` |
///
/// The source designates the 30 cell-nucleus measurements as the inputs
/// ([`BreastCancer::FEATURE_NAMES`]) and `diagnosis` as the label
/// ([`BreastCancer::TARGET`]).
///
/// The source file also starts every record with a sample ID. The ID names no
/// measurement, so the table drops it.
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
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::BreastCancer;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./breast_cancer";
///
/// let mut dataset = BreastCancer::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 569);
/// assert_eq!(table.n_columns(), 31);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&BreastCancer::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[569, 30]);
///
/// // Reach one column by name.
/// let diagnosis = table.column(BreastCancer::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(diagnosis.len(), 569);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("radius_mean") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 15.0;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 569);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 569);
/// ```
#[derive(Debug)]
pub struct BreastCancer {
    dataset: Dataset<Table, DatasetError>,
}

impl BreastCancer {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave_points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave_points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave_points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "diagnosis";

    /// Create a new BreastCancer instance without loading data.
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
    /// - `Self` - a `BreastCancer` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        BreastCancer {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Breast Cancer dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            BREAST_CANCER_FILENAME,
            BREAST_CANCER_DATASET_NAME,
            Some(BREAST_CANCER_SHA256),
            |temp_path| {
                download_to_with_retries(
                    BREAST_CANCER_DATA_URL,
                    temp_path,
                    Some(BREAST_CANCER_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(BREAST_CANCER_FILENAME))
            },
        )?;

        // The csv crate deserializes each row into the struct. `wdbc.data` has no
        // header row, so the loader disables header handling.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features = Vec::new();
        let mut diagnoses = Vec::new();

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

            diagnoses.push(
                match diagnosis.as_str() {
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
                }
                .to_string(),
            );
        }

        // The source lists the diagnosis before the 30 measurements.
        let mut columns = Vec::with_capacity(N_FEATURES + 1);
        columns.push(Column::new(
            Self::TARGET,
            ColumnData::String(Array1::from_vec(diagnoses)),
        ));
        for (index, &name) in Self::FEATURE_NAMES.iter().enumerate() {
            let values: Vec<f64> = features[index..]
                .iter()
                .step_by(N_FEATURES)
                .copied()
                .collect();
            columns.push(Column::new(
                name,
                ColumnData::Numeric(Array1::from_vec(values)),
            ));
        }

        Table::new(BREAST_CANCER_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 569 samples and 31 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (unparseable values, an unknown diagnosis)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`BreastCancer::data`], this method never runs the loader. If the
    /// data has not loaded yet, it returns `None` instead of downloading and
    /// parsing it.
    ///
    /// # Returns
    ///
    /// - `Some(&Table)` - reference to the cached table, if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&Table> {
        self.dataset.get()
    }

    /// Get a mutable reference to the parsed table for **in-place** editing.
    ///
    /// This needs no clone, and it does not remove the data from the cache. The
    /// changes stay in the cache. Later calls to [`BreastCancer::data`] or
    /// [`BreastCancer::get_data`] see them.
    ///
    /// Like [`BreastCancer::get_data`], this does **not** trigger loading.
    ///
    /// # Returns
    ///
    /// - `Some(&mut Table)` - mutable reference to the cached table, if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut Table> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return the **owned** table.
    ///
    /// This **consumes** `self`. If you want owned data but need to keep using
    /// the instance, use [`BreastCancer::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 569 samples and 31 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, or parsing).
    pub fn into_data(self) -> Result<Table, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take the **owned** table out of the dataset. This leaves the instance
    /// reusable.
    ///
    /// This resets the instance to its unloaded state. The next accessor call
    /// loads the dataset again.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 569 samples and 31 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, or parsing).
    pub fn take_data(&mut self) -> Result<Table, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(BreastCancer, "breast_cancer");
