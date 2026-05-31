//! Red wine subset of the Wine Quality dataset.
//!
//! See [`crate::wine_quality`] for the full dataset description,
//! including features, target, application scenarios, and source.
//!
//! **Samples:** 1599
//! **Feature shape:** `(1599, 11)`
//! **Target shape:** `(1599,)`

use crate::wine_quality::{WineData, parse_wine_data_to_array};
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to};
use ndarray::{Array1, Array2};
use std::fs::File;

/// The URL for the Red Wine Quality dataset.
const RED_WINE_DATA_URL: &str = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-red.csv";

/// The red wine file of the CSV files inside the zip archive.
const RED_WINE_QUALITY_FILENAME: &str = "winequality-red.csv";

/// The SHA256 hash of the red wine quality dataset.
const RED_WINE_QUALITY_SHA256: &str =
    "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e";

/// A struct representing the Red Wine Quality dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The dataset contains physicochemical properties of Portuguese "Vinho Verde"
/// red wine samples and a quality score for each sample.
///
/// Features (11 total, all `f64`):
///   - fixed acidity
///   - volatile acidity
///   - citric acid
///   - residual sugar
///   - chlorides
///   - free sulfur dioxide
///   - total sulfur dioxide
///   - density
///   - pH
///   - sulphates
///   - alcohol
///
/// Targets:
/// - quality (score between 0 and 10, stored as `f64`)
///
/// See more information at <https://archive.ics.uci.edu/dataset/186/wine+quality>
///
/// # Citation
///
/// P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. "Wine Quality," UCI Machine Learning Repository, 2009. \[Online\]. Available: <https://doi.org/10.24432/C56S3T>.
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::RedWineQuality;
///
/// let download_dir = "./red_wine"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = RedWineQuality::new(download_dir);
/// let features = dataset.features().unwrap();
/// let targets = dataset.targets().unwrap();
///
/// let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
/// // you can use `.to_owned()` to get owned copies of the data
/// let mut features_owned = features.to_owned();
/// let mut targets_owned = targets.to_owned();
///
/// // Example: Modify feature values
/// features_owned[[0, 0]] = 10.0;
/// targets_owned[0] = 7.0;
///
/// assert_eq!(features.shape(), &[1599, 11]);
/// assert_eq!(targets.len(), 1599);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place (no clone, no reload — the change stays cached).
/// if let Some((features, _targets)) = dataset.get_data_mut() {
///     features[[0, 0]] = 10.0;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable — the next access reloads from the cached file.
/// let (owned_features, owned_targets) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[1599, 11]);
/// assert_eq!(owned_targets.len(), 1599);
///
/// // `into_data()` also returns owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_features, owned_targets) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[1599, 11]);
/// assert_eq!(owned_targets.len(), 1599);
/// ```
#[derive(Debug)]
pub struct RedWineQuality {
    dataset: Dataset<WineData, DatasetError>,
}

impl RedWineQuality {
    /// Create a new RedWineQuality instance without loading data.
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
    /// - `Self` - `RedWineQuality` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        RedWineQuality {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Red Wine Quality dataset.
    fn load_data(dir: &str) -> Result<WineData, DatasetError> {
        // Prepare the dataset file
        let file_path = acquire_dataset(
            dir,
            RED_WINE_QUALITY_FILENAME,
            "red_wine_quality",
            Some(RED_WINE_QUALITY_SHA256),
            |temp_path| {
                download_to(RED_WINE_DATA_URL, temp_path, None)?;
                Ok(temp_path.join(RED_WINE_QUALITY_FILENAME))
            },
        )?;

        // Parse the file
        let file = File::open(&file_path)?;
        parse_wine_data_to_array("red_wine_quality", file)
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(1599, 11)` containing:
    ///     - fixed acidity
    ///     - volatile acidity
    ///     - citric acid
    ///     - residual sugar
    ///     - chlorides
    ///     - free sulfur dioxide
    ///     - total sulfur dioxide
    ///     - density
    ///     - pH
    ///     - sulphates
    ///     - alcohol
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (1599 samples, 11 features)
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
    /// - `&Array1<f64>` - Reference to target vector with shape `(1599,)` containing quality scores (0-10)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (1599 samples)
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
    /// - `&WineData` - reference to the cached `(features, targets)` tuple: feature
    ///   matrix with shape `(1599, 11)` (the 11 physicochemical properties) and
    ///   target vector with shape `(1599,)` (quality scores, 0-10).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (1599 samples, 11 features)
    pub fn data(&self) -> Result<&WineData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and targets as references **without** triggering loading.
    ///
    /// Unlike [`RedWineQuality::data`], which loads the dataset on first call,
    /// this never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&WineData)` - reference to the cached `(features, targets)` tuple
    ///   (feature matrix `(1599, 11)`, target vector `(1599,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&WineData> {
        self.dataset.get()
    }

    /// Get mutable references to features and targets for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize features,
    /// rescale targets) with no `to_owned()` clone and without removing them from
    /// the cache: the changes persist, so later [`RedWineQuality::features`],
    /// [`RedWineQuality::data`], or [`RedWineQuality::get_data`] calls observe
    /// them.
    ///
    /// Like [`RedWineQuality::get_data`], this does **not** trigger loading: it
    /// returns `None` if the dataset has not been loaded. Call a loading accessor
    /// (e.g. [`RedWineQuality::data`]) first if you need to ensure the data is
    /// present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut WineData)` - mutable reference to the cached `(features,
    ///   targets)` tuple (feature matrix `(1599, 11)`, target vector `(1599,)`),
    ///   if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut WineData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and targets.
    ///
    /// Unlike [`RedWineQuality::data`], which borrows the cached data, this moves it
    /// out and returns owned arrays directly — no `to_owned()` clone needed. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`RedWineQuality::take_data`] instead — it takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<f64>)` - owned feature matrix with shape
    ///   `(1599, 11)` and owned target vector with shape `(1599,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<WineData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and targets out of the dataset, leaving it reusable.
    ///
    /// Like [`RedWineQuality::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out, resetting the instance to its
    /// unloaded state: the next accessor call (e.g. [`RedWineQuality::features`] or
    /// [`RedWineQuality::data`]) loads the dataset again.
    ///
    /// Use [`RedWineQuality::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<f64>)` - owned feature matrix with shape
    ///   `(1599, 11)` and owned target vector with shape `(1599,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<WineData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}
