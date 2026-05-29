//! Pima Indians Diabetes dataset.
//!
//! Diagnostic measurements from the National Institute of Diabetes and
//! Digestive and Kidney Diseases, used to predict whether a patient has diabetes.
//!
//! **Features (8):**
//! - `Pregnancies` - number of times pregnant
//! - `Glucose` - plasma glucose concentration at 2 hours in an oral glucose tolerance test
//! - `BloodPressure` - diastolic blood pressure (mm Hg)
//! - `SkinThickness` - triceps skin fold thickness (mm)
//! - `Insulin` - 2-hour serum insulin (mu U/ml)
//! - `BMI` - body mass index (weight in kg / (height in m)^2)
//! - `DiabetesPedigreeFunction` - diabetes pedigree function
//! - `Age` - age in years
//!
//! **Target:** `Outcome` - binary class label (`0` or `1`)
//!
//! **Samples:** 768
//! **Application:** Binary classification / diabetes prediction
//!
//! **Source:** UCI Machine Learning Repository
//! <https://archive.ics.uci.edu/dataset/34/diabetes>

use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::fs::File;

/// The URL for the Diabetes dataset.
const DIABETES_DATA_URL: &str =
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv";

/// A static string slice containing the name of the Diabetes dataset file.
const DIABETES_FILENAME: &str = "diabetes.csv";

/// The SHA256 hash of the Diabetes dataset file.
const DIABETES_SHA256: &str = "698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72";

/// The name of the dataset
const DIABETES_DATASET_NAME: &str = "diabetes";

/// One CSV record of the Diabetes dataset: 8 `f64` feature columns followed by
/// the binary `Outcome` label.
///
/// Fields are declared in CSV column order and deserialized **positionally**
/// (the loader disables csv's header handling), so this struct is independent
/// of the exact header spelling.
#[derive(Deserialize)]
struct DiabetesRecord {
    pregnancies: f64,
    glucose: f64,
    blood_pressure: f64,
    skin_thickness: f64,
    insulin: f64,
    bmi: f64,
    diabetes_pedigree_function: f64,
    age: f64,
    outcome: f64,
}

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
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::diabetes::Diabetes;
///
/// let download_dir = "./diabetes"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = Diabetes::new(download_dir);
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
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place (no clone, no reload — the change stays cached).
/// if let Some((features, _labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 10.0;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable — the next access reloads from the cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[768, 8]);
/// assert_eq!(owned_labels.len(), 768);
///
/// // `into_data()` also returns owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[768, 8]);
/// assert_eq!(owned_labels.len(), 768);
/// ```
#[derive(Debug)]
pub struct Diabetes {
    dataset: Dataset<(Array2<f64>, Array1<f64>)>,
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
            dataset: Dataset::new(storage_dir),
        }
    }

    /// Acquire and parse the Diabetes dataset.
    fn load_data(dir: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
        // Prepare the dataset file
        let file_path = acquire_dataset(
            dir,
            DIABETES_FILENAME,
            DIABETES_DATASET_NAME,
            Some(DIABETES_SHA256),
            |temp_path| {
                download_to(DIABETES_DATA_URL, temp_path, None)?;
                Ok(temp_path.join(DIABETES_FILENAME))
            },
        )?;

        // Stream the cached file through csv, deserializing one record at a time.
        // `has_headers(false)` makes csv deserialize into the named struct
        // *positionally* (by column order) rather than by header name, keeping
        // parsing independent of the exact header spelling. We skip the header
        // row ourselves with `.skip(1)`.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features = Vec::new();
        let mut labels = Vec::new();

        for result in rdr.deserialize::<DiabetesRecord>().skip(1) {
            let DiabetesRecord {
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree_function,
                age,
                outcome,
            } = result.map_err(|e| DatasetError::csv_read_error(DIABETES_DATASET_NAME, e))?;

            features.extend_from_slice(&[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree_function,
                age,
            ]);
            labels.push(outcome);
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(DIABETES_DATASET_NAME));
        }

        // Diabetes has a fixed schema of 8 numeric features per sample.
        let features_array = Array2::from_shape_vec((n_samples, 8), features)
            .map_err(|e| DatasetError::array_shape_error(DIABETES_DATASET_NAME, "features", e))?;
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
        Ok(&self.dataset.load(Self::load_data)?.0)
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
        Ok(&self.dataset.load(Self::load_data)?.1)
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
        let data = self.dataset.load(Self::load_data)?;
        Ok((&data.0, &data.1))
    }

    /// Get both features and labels as references **without** triggering loading.
    ///
    /// Unlike [`Diabetes::data`], which loads the dataset on first call, this
    /// never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some((&Array2<f64>, &Array1<f64>))` - references to the cached feature
    ///   matrix `(768, 8)` and label vector `(768,)`, if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<(&Array2<f64>, &Array1<f64>)> {
        self.dataset.get().map(|(f, l)| (f, l))
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize features,
    /// fill in missing entries) with no `to_owned()` clone and without removing
    /// them from the cache: the changes persist, so later [`Diabetes::features`],
    /// [`Diabetes::data`], or [`Diabetes::get_data`] calls observe them.
    ///
    /// Like [`Diabetes::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call a loading accessor (e.g.
    /// [`Diabetes::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some((&mut Array2<f64>, &mut Array1<f64>))` - mutable references to the
    ///   cached feature matrix `(768, 8)` and label vector `(768,)`, if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<(&mut Array2<f64>, &mut Array1<f64>)> {
        self.dataset.get_mut().map(|(f, l)| (f, l))
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`Diabetes::data`], which borrows the cached data, this moves it out
    /// and returns owned arrays directly — no `to_owned()` clone needed. The dataset
    /// is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`Diabetes::take_data`] instead — it takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<f64>)` - owned feature matrix with shape `(768, 8)`
    ///   and owned label vector with shape `(768,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
        self.dataset.load(Self::load_data)?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`Diabetes::into_data`], this returns owned arrays with no `to_owned()`
    /// clone. But instead of consuming the instance, it takes `&mut self` and moves
    /// the cached data out, resetting the instance to its unloaded state: the next
    /// accessor call (e.g. [`Diabetes::features`] or [`Diabetes::data`]) loads the
    /// dataset again.
    ///
    /// Use [`Diabetes::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<f64>)` - owned feature matrix with shape `(768, 8)`
    ///   and owned label vector with shape `(768,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
        self.dataset.load(Self::load_data)?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}
