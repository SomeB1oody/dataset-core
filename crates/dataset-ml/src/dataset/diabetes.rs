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
//! **Columns (11):** in scikit-learn column order
//!
//! | Name     | Type      | Description                                                      |
//! |----------|-----------|----------------------------------------------------------------------|
//! | `age`    | `Numeric` | standardized age in years                                        |
//! | `sex`    | `Numeric` | standardized sex                                                 |
//! | `bmi`    | `Numeric` | standardized body mass index                                     |
//! | `bp`     | `Numeric` | standardized average blood pressure                              |
//! | `s1`     | `Numeric` | standardized tc, total serum cholesterol                         |
//! | `s2`     | `Numeric` | standardized ldl, low-density lipoproteins                       |
//! | `s3`     | `Numeric` | standardized hdl, high-density lipoproteins                      |
//! | `s4`     | `Numeric` | standardized tch, total cholesterol / HDL                        |
//! | `s5`     | `Numeric` | standardized ltg, possibly log of serum triglycerides level      |
//! | `s6`     | `Numeric` | standardized glu, blood sugar level                              |
//! | `target` | `Numeric` | disease progression one year after baseline, unscaled, 25 to 346 |
//!
//! The source designates the ten baseline variables as the inputs
//! ([`Diabetes::FEATURE_NAMES`](crate::Diabetes::FEATURE_NAMES)) and `target` as the label
//! ([`Diabetes::TARGET`](crate::Diabetes::TARGET)).
//!
//! **Samples:** 442
//! **Application:** Regression / disease progression prediction
//!
//! **Source:** Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani
//! (2004), "Least Angle Regression," *Annals of Statistics* (with discussion),
//! 407–499. <https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
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

/// A struct that represents the Diabetes dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
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
/// # Columns
///
/// The loader standardizes all ten feature columns, so their stored values are
/// dimensionless. The description below names the unit of the *original*
/// (pre-standardization) measurement where the source records one. The columns
/// keep scikit-learn column order:
///
/// | Name     | Type      | Description                                                      |
/// |----------|-----------|----------------------------------------------------------------------|
/// | `age`    | `Numeric` | standardized age in years                                        |
/// | `sex`    | `Numeric` | standardized sex                                                 |
/// | `bmi`    | `Numeric` | standardized body mass index                                     |
/// | `bp`     | `Numeric` | standardized average blood pressure                              |
/// | `s1`     | `Numeric` | standardized tc, total serum cholesterol                         |
/// | `s2`     | `Numeric` | standardized ldl, low-density lipoproteins                       |
/// | `s3`     | `Numeric` | standardized hdl, high-density lipoproteins                      |
/// | `s4`     | `Numeric` | standardized tch, total cholesterol / HDL                        |
/// | `s5`     | `Numeric` | standardized ltg, possibly log of serum triglycerides level      |
/// | `s6`     | `Numeric` | standardized glu, blood sugar level                              |
/// | `target` | `Numeric` | disease progression one year after baseline, unscaled, 25 to 346 |
///
/// The source designates the ten baseline variables as the inputs
/// ([`Diabetes::FEATURE_NAMES`]) and `target` as the label
/// ([`Diabetes::TARGET`]).
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
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Diabetes;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./diabetes";
///
/// let mut dataset = Diabetes::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 442);
/// assert_eq!(table.n_columns(), 11);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&Diabetes::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[442, 10]);
///
/// // Reach one column by name.
/// let target = table.column(Diabetes::TARGET).unwrap().as_numeric().unwrap();
/// assert_eq!(target.len(), 442);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("age") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 0.05;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 442);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 442);
/// ```
#[derive(Debug)]
pub struct Diabetes {
    dataset: Dataset<Table, DatasetError>,
}

impl Diabetes {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "target";

    /// Create a new Diabetes instance without loading data.
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
    /// - `Self` - a `Diabetes` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Diabetes {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Diabetes dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
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

        let mut features: [Vec<f64>; N_FEATURES] =
            std::array::from_fn(|_| Vec::with_capacity(n_samples));
        for row in &raw {
            for (j, &v) in row.iter().enumerate() {
                features[j].push((v - means[j]) / norms[j]);
            }
        }

        let mut columns: Vec<Column> = Self::FEATURE_NAMES
            .iter()
            .zip(features)
            .map(|(&name, values)| Column::new(name, ColumnData::Numeric(Array1::from_vec(values))))
            .collect();
        columns.push(Column::new(
            Self::TARGET,
            ColumnData::Numeric(Array1::from_vec(targets)),
        ));

        Table::new(DIABETES_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 442 samples and 11 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`Diabetes::data`], this method never runs the loader. If the data
    /// has not loaded yet, it returns `None` instead of downloading and parsing
    /// it.
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
    /// changes stay in the cache. Later calls to [`Diabetes::data`] or
    /// [`Diabetes::get_data`] see them.
    ///
    /// Like [`Diabetes::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`Diabetes::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 442 samples and 11 columns.
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
    /// - `Table` - the owned table of 442 samples and 11 columns.
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

impl_ml_dataset!(Diabetes, "diabetes");
