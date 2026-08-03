//! Diabetes dataset (scikit-learn `load_diabetes`).
//!
//! The dataset has ten baseline physiological variables for 442 diabetes patients.
//! Researchers use the variables to predict a quantitative measure of disease
//! progression one year after baseline. The dataset is a common small regression
//! benchmark.
//!
//! This loader reproduces the default output of scikit-learn's `load_diabetes()`
//! function. It **standardizes** the ten feature columns. For each column, it
//! subtracts the mean, then divides the result by the column's L2 norm (equal to
//! `std * sqrt(n_samples)`). After this step, each column has a mean of 0 and a
//! sum of squares of 1. The target stays **unscaled**. The source file is the
//! original tab-separated data from the "Least Angle Regression" paper (Efron et
//! al., 2004).
//!
//! **Features (10):** in scikit-learn column order
//! - `age` - age in years
//! - `sex` - sex
//! - `bmi` - body mass index
//! - `bp` - average blood pressure
//! - `s1` - tc, total serum cholesterol
//! - `s2` - ldl, low-density lipoproteins
//! - `s3` - hdl, high-density lipoproteins
//! - `s4` - tch, total cholesterol / HDL
//! - `s5` - ltg, possibly the log of the serum triglycerides level
//! - `s6` - glu, blood sugar level
//!
//! **Target:** quantitative measure of disease progression one year after
//!   baseline (unscaled, integer-valued in the range 25–346).
//!
//! **Samples:** 442
//! **Application:** Regression / disease progression prediction
//!
//! **Source:** Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani
//! (2004), "Least Angle Regression," *Annals of Statistics* (with discussion),
//! 407–499. <https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::fs::File;

/// The URL for the Diabetes dataset (the original tab-separated file that
/// scikit-learn cites as the source for `load_diabetes`).
const DIABETES_DATA_URL: &str = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt";

/// The name of the Diabetes dataset file.
const DIABETES_FILENAME: &str = "diabetes.tab";

/// The SHA256 hash of the Diabetes dataset file.
const DIABETES_SHA256: &str = "4733febee697862c22139cdac87478a300ce0d101593deb07ed6c0f3328a99cd";

/// The name of the dataset.
const DIABETES_DATASET_NAME: &str = "diabetes";

/// The number of feature columns per sample.
const N_FEATURES: usize = 10;

/// Type alias for the Diabetes dataset: (features, targets).
type DiabetesData = (Array2<f64>, Array1<f64>);

/// This struct represents one tab-separated record of the Diabetes dataset. It has
/// 10 `f64` feature columns (`age`, `sex`, `bmi`, `bp`, `s1`–`s6`), followed by the
/// `f64` regression target `Y` (disease progression).
///
/// The struct declares its fields in source column order. The loader disables
/// csv's header handling, so csv deserializes the fields **positionally**. This
/// design makes the struct independent of the exact header spelling.
#[derive(Deserialize)]
struct DiabetesRecord {
    age: f64,
    sex: f64,
    bmi: f64,
    bp: f64,
    s1: f64,
    s2: f64,
    s3: f64,
    s4: f64,
    s5: f64,
    s6: f64,
    y: f64,
}

/// This struct represents the Diabetes dataset and loads it lazily.
///
/// The dataset loads only when you call a data accessor method. Later calls
/// return the cached data without loading again.
///
/// # About Dataset
///
/// Researchers measured ten baseline variables for each of 442 diabetes patients:
/// age, sex, body mass index, average blood pressure, and six blood serum
/// measurements. Researchers also recorded a quantitative measure of disease
/// progression one year after baseline as the response of interest. This loader
/// reproduces the default output of scikit-learn's `load_diabetes()` function. It
/// **standardizes** each of the ten feature columns: it mean-centers each column,
/// then divides the result by its L2 norm. After this step, every column has a
/// mean of 0 and a sum of squares of 1. The target stays unscaled.
///
/// # Feature columns
///
/// The loader standardizes all ten feature columns: it mean-centers each column,
/// then divides it by its L2 norm, so the stored values are dimensionless (mean 0,
/// sum of squares 1). The `Unit` column below records the unit of the *original*
/// (pre-standardization) measurement where known. The parenthetical text in
/// `Attributes` expands each abbreviated name. By 0-based column index in the
/// feature matrix, in scikit-learn column order:
///
/// | Columns | Attributes                                            | Unit  |
/// |---------|-------------------------------------------------------|-------|
/// | `0`     | `age`                                                 | years |
/// | `1`     | `sex`                                                 |       |
/// | `2`     | `bmi` (body mass index)                               |       |
/// | `3`     | `bp` (average blood pressure)                         |       |
/// | `4`     | `s1` (tc, total serum cholesterol)                    |       |
/// | `5`     | `s2` (ldl, low-density lipoproteins)                  |       |
/// | `6`     | `s3` (hdl, high-density lipoproteins)                 |       |
/// | `7`     | `s4` (tch, total cholesterol / HDL)                   |       |
/// | `8`     | `s5` (ltg, possibly log of serum triglycerides level) |       |
/// | `9`     | `s6` (glu, blood sugar level)                         |       |
///
/// # Targets
///
/// - quantitative measure of disease progression one year after baseline
///   (unscaled, integer-valued in the range 25–346)
///
/// See more information at <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>
///
/// # Citation
///
/// Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004),
/// "Least Angle Regression," Annals of Statistics (with discussion), 407–499.
///
/// # Thread Safety
///
/// All fields of this struct implement `Send` and `Sync`, so the struct implements
/// them too. This makes the struct safe to share across threads. The internal
/// [`Dataset`] makes sure initialization is lazy and thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Diabetes;
///
/// let download_dir = "./diabetes"; // the code creates the directory if it does not exist
///
/// let mut dataset = Diabetes::new(download_dir);
/// let features = dataset.features().unwrap();
/// let targets = dataset.targets().unwrap();
///
/// let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
/// assert_eq!(features.shape(), &[442, 10]);
/// assert_eq!(targets.len(), 442);
///
/// // `get_data()` borrows the cached arrays and does not reload them.
/// // `get_data_mut()` edits the arrays in place. It makes no clone, and the
/// // change stays in the cache. Prefer `get_data_mut()` over `.to_owned()` when
/// // you only need to change values.
/// if let Some((features, targets)) = dataset.get_data_mut() {
///     features[[0, 0]] = 0.05;
///     targets[0] = 200.0;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out with no `.to_owned()` clone. It leaves
/// // the instance reusable. The next access reloads the data from the cached
/// // file.
/// let (owned_features, owned_targets) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[442, 10]);
/// assert_eq!(owned_targets.len(), 442);
///
/// // `into_data()` also returns owned arrays with no clone. But it consumes the
/// // instance. Use it when you are done with the dataset.
/// let (owned_features, owned_targets) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[442, 10]);
/// assert_eq!(owned_targets.len(), 442);
/// ```
#[derive(Debug)]
pub struct Diabetes {
    dataset: Dataset<DiabetesData, DatasetError>,
}

impl Diabetes {
    /// Create a new Diabetes instance without loading data.
    ///
    /// The dataset loads on the first call to a data accessor method. This function
    /// only stores the storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - the directory that stores the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - a `Diabetes` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Diabetes {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Diabetes dataset.
    fn load_data(dir: &str) -> Result<DiabetesData, DatasetError> {
        // Prepare the dataset file
        let file_path = acquire_dataset(
            dir,
            DIABETES_FILENAME,
            DIABETES_DATASET_NAME,
            Some(DIABETES_SHA256),
            |temp_path| {
                download_to_with_retries(
                    DIABETES_DATA_URL,
                    temp_path,
                    Some(DIABETES_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(DIABETES_FILENAME))
            },
        )?;

        // The source is tab-separated with a header row, so skip the header.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(false)
            .from_reader(file);

        // Collect the raw feature rows and targets first. Standardization needs a
        // full pass over each column to compute its mean and norm.
        let mut raw: Vec<[f64; N_FEATURES]> = Vec::new();
        let mut targets: Vec<f64> = Vec::new();

        for result in rdr.deserialize::<DiabetesRecord>().skip(1) {
            let DiabetesRecord {
                age,
                sex,
                bmi,
                bp,
                s1,
                s2,
                s3,
                s4,
                s5,
                s6,
                y,
            } = result.map_err(|e| DatasetError::csv_read_error(DIABETES_DATASET_NAME, e))?;

            raw.push([age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]);
            targets.push(y);
        }

        let n_samples = targets.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(DIABETES_DATASET_NAME));
        }

        // This step reproduces scikit-learn's standardization. For each column, it
        // subtracts the mean, then divides the result by the L2 norm of the
        // centered column (equal to `std * sqrt(n_samples)`). After this step, each
        // column has a mean of 0 and a sum of squares of 1. Every column varies, so
        // its norm is never zero.
        let n_f = n_samples as f64;
        let mut means = [0.0f64; N_FEATURES];
        for row in &raw {
            for (m, &v) in means.iter_mut().zip(row.iter()) {
                *m += v;
            }
        }
        for m in &mut means {
            *m /= n_f;
        }

        let mut norms = [0.0f64; N_FEATURES];
        for row in &raw {
            for (j, norm) in norms.iter_mut().enumerate() {
                let centered = row[j] - means[j];
                *norm += centered * centered;
            }
        }
        for norm in &mut norms {
            *norm = norm.sqrt();
        }

        let mut features = Vec::with_capacity(n_samples * N_FEATURES);
        for row in &raw {
            for (j, &v) in row.iter().enumerate() {
                features.push((v - means[j]) / norms[j]);
            }
        }

        // Diabetes has a fixed schema of 10 standardized features per sample.
        let features_array = Array2::from_shape_vec((n_samples, N_FEATURES), features)
            .map_err(|e| DatasetError::array_shape_error(DIABETES_DATASET_NAME, "features", e))?;
        let targets_array = Array1::from_vec(targets);

        Ok((features_array, targets_array))
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(442, 10)`
    ///   containing the standardized scikit-learn features (`age`, `sex`, `bmi`,
    ///   `bp`, `s1`–`s6`). Each column has mean 0 and a sum of squares of 1.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match expected dimensions (442 samples, 10 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the target vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<f64>` - Reference to target vector with shape `(442,)`
    ///   containing the unscaled measure of disease progression one year after
    ///   baseline (integer-valued in the range 25–346).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match expected dimensions (442 samples)
    pub fn targets(&self) -> Result<&Array1<f64>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and targets as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&DiabetesData` - reference to the cached `(features, targets)` tuple:
    ///   feature matrix with shape `(442, 10)` (standardized `age`, `sex`, `bmi`,
    ///   `bp`, `s1`–`s6`) and target vector with shape `(442,)` (unscaled disease
    ///   progression).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match expected dimensions (442 samples, 10 features)
    pub fn data(&self) -> Result<&DiabetesData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and targets as references **without** triggering loading.
    ///
    /// Unlike [`Diabetes::data`], this method never runs the loader. If the
    /// dataset has not loaded yet, it returns `None` instead of downloading and
    /// parsing the data. Use this method when you want the data only if it is
    /// already cached. This choice avoids the download and parse cost.
    ///
    /// # Returns
    ///
    /// - `Some(&DiabetesData)` - reference to the cached `(features, targets)` tuple
    ///   (feature matrix `(442, 10)`, target vector `(442,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&DiabetesData> {
        self.dataset.get()
    }

    /// Get mutable references to features and targets for **in-place** editing.
    ///
    /// This lets you change the cached arrays directly (for example, re-scale
    /// features or clip outliers) with no `to_owned()` clone. The changes stay in
    /// the cache: later calls to [`Diabetes::features`], [`Diabetes::data`], or
    /// [`Diabetes::get_data`] see them.
    ///
    /// Like [`Diabetes::get_data`], this method does **not** trigger loading. It
    /// returns `None` if the dataset has not loaded. If you need to make sure the
    /// data is present, call a loading accessor first (for example,
    /// [`Diabetes::data`]).
    ///
    /// # Returns
    ///
    /// - `Some(&mut DiabetesData)` - a mutable reference to the cached
    ///   `(features, targets)` tuple (feature matrix `(442, 10)`, target vector
    ///   `(442,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut DiabetesData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and targets.
    ///
    /// Unlike [`Diabetes::data`], which borrows the cached data, this method moves
    /// the data out and returns owned arrays. It needs no `to_owned()` clone. The
    /// dataset loads on the first access if it has not loaded yet.
    ///
    /// This **consumes** `self`, so you cannot use the instance afterward. If you
    /// want owned data but need to keep using the instance, use
    /// [`Diabetes::take_data`] instead. It takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<f64>)` - an owned feature matrix with shape
    ///   `(442, 10)` and an owned target vector with shape `(442,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<DiabetesData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and targets out of the dataset. Leave the dataset
    /// reusable.
    ///
    /// Like [`Diabetes::into_data`], this returns owned arrays with no `to_owned()`
    /// clone. But instead of consuming the instance, it takes `&mut self` and moves
    /// the cached data out. This resets the instance to its unloaded state. The
    /// next accessor call (for example, [`Diabetes::features`] or
    /// [`Diabetes::data`]) loads the dataset again.
    ///
    /// If you are done with the instance, use [`Diabetes::into_data`] instead.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<f64>)` - an owned feature matrix with shape
    ///   `(442, 10)` and an owned target vector with shape `(442,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<DiabetesData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(Diabetes, DiabetesData, "diabetes");
