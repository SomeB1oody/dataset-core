//! Diabetes dataset (scikit-learn `load_diabetes`).
//!
//! Ten baseline physiological variables measured on 442 diabetes patients, used
//! to predict a quantitative measure of disease progression one year after
//! baseline. A classic small regression benchmark.
//!
//! This loader reproduces scikit-learn's **default** `load_diabetes()` output:
//! the ten feature columns are **standardized** — each is mean-centered and
//! divided by its L2 norm (equivalently `std * sqrt(n_samples)`), so every column
//! has mean 0 and a sum of squares of 1. The target is left **unscaled**. The
//! underlying file is the original tab-separated data distributed with the
//! "Least Angle Regression" paper (Efron et al., 2004).
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
//! - `s5` - ltg, possibly log of serum triglycerides level
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

/// A static string slice containing the name of the Diabetes dataset file.
const DIABETES_FILENAME: &str = "diabetes.tab";

/// The SHA256 hash of the Diabetes dataset file.
const DIABETES_SHA256: &str = "4733febee697862c22139cdac87478a300ce0d101593deb07ed6c0f3328a99cd";

/// The name of the dataset
const DIABETES_DATASET_NAME: &str = "diabetes";

/// The number of feature columns per sample.
const N_FEATURES: usize = 10;

/// Type alias for the Diabetes dataset: (features, targets).
type DiabetesData = (Array2<f64>, Array1<f64>);

/// One tab-separated record of the Diabetes dataset: 10 `f64` feature columns
/// (`age`, `sex`, `bmi`, `bp`, `s1`–`s6`) followed by the `f64` regression
/// target `Y` (disease progression).
///
/// Fields are declared in source column order and deserialized **positionally**
/// (the loader disables csv's header handling), so this struct is independent of
/// the exact header spelling.
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

/// A struct representing the Diabetes dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// Ten baseline variables — age, sex, body mass index, average blood pressure,
/// and six blood serum measurements — were obtained for each of 442 diabetes
/// patients, along with the response of interest: a quantitative measure of
/// disease progression one year after baseline. This loader reproduces
/// scikit-learn's default `load_diabetes()` output by **standardizing** each of
/// the ten feature columns (mean-centered and divided by its L2 norm, so every
/// column has mean 0 and a sum of squares of 1). The target is left unscaled.
///
/// # Feature columns
///
/// All ten feature columns are **standardized** — each is mean-centered and
/// divided by its L2 norm, so the stored values are dimensionless (mean 0, sum
/// of squares 1). The `Unit` column below records the unit of the *original*
/// (pre-standardization) measurement where known; the parenthetical text in
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
/// let targets = dataset.targets().unwrap();
///
/// let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
/// assert_eq!(features.shape(), &[442, 10]);
/// assert_eq!(targets.len(), 442);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((features, targets)) = dataset.get_data_mut() {
///     features[[0, 0]] = 0.05;
///     targets[0] = 200.0;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable — the next access reloads from the cached file.
/// let (owned_features, owned_targets) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[442, 10]);
/// assert_eq!(owned_targets.len(), 442);
///
/// // `into_data()` also returns owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
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
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Diabetes dataset.
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

        // Collect the raw feature rows and targets first; standardization needs a
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

        // Reproduce scikit-learn's standardization: for each column, subtract the
        // mean and divide by the L2 norm of the centered column (equivalently
        // `std * sqrt(n_samples)`), so each column ends up with mean 0 and a sum
        // of squares of 1. Every column varies, so the norm is never zero.
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
    /// - Dataset size doesn't match expected dimensions (442 samples, 10 features)
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
    /// - Dataset size doesn't match expected dimensions (442 samples)
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
    /// - Dataset size doesn't match expected dimensions (442 samples, 10 features)
    pub fn data(&self) -> Result<&DiabetesData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and targets as references **without** triggering loading.
    ///
    /// Unlike [`Diabetes::data`], which loads the dataset on first call, this
    /// never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&DiabetesData)` - reference to the cached `(features, targets)` tuple
    ///   (feature matrix `(442, 10)`, target vector `(442,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&DiabetesData> {
        self.dataset.get()
    }

    /// Get mutable references to features and targets for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. re-scale features,
    /// clip outliers) with no `to_owned()` clone and without removing them from
    /// the cache: the changes persist, so later [`Diabetes::features`],
    /// [`Diabetes::data`], or [`Diabetes::get_data`] calls observe them.
    ///
    /// Like [`Diabetes::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call a loading accessor (e.g.
    /// [`Diabetes::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut DiabetesData)` - mutable reference to the cached
    ///   `(features, targets)` tuple (feature matrix `(442, 10)`, target vector
    ///   `(442,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut DiabetesData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and targets.
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
    /// - `(Array2<f64>, Array1<f64>)` - owned feature matrix with shape `(442, 10)`
    ///   and owned target vector with shape `(442,)`.
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

    /// Take **owned** features and targets out of the dataset, leaving it reusable.
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
    /// - `(Array2<f64>, Array1<f64>)` - owned feature matrix with shape `(442, 10)`
    ///   and owned target vector with shape `(442,)`.
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
