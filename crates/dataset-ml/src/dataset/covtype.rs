//! Forest Cover Type dataset.
//!
//! The Forest CoverType dataset supports multi-class classification. It matches
//! the dataset that scikit-learn exposes through `fetch_covtype`. Each of the
//! 581,012 samples describes a 30×30 meter cell of wilderness in the Roosevelt
//! National Forest of northern Colorado. The task is to predict the forest cover
//! type of the cell, one of seven classes, from 54 cartographic features.
//!
//! **Features (54):** these encode 12 logical attributes (10 numeric and 2
//! categorical). The two categorical attributes are already **one-hot expanded**.
//! By 0-based column index:
//! - cols `0..=9`, 10 distinct quantitative variables: `Elevation`, `Aspect`,
//!   `Slope`, `Horizontal_Distance_To_Hydrology`, `Vertical_Distance_To_Hydrology`,
//!   `Horizontal_Distance_To_Roadways`, `Hillshade_9am`, `Hillshade_Noon`,
//!   `Hillshade_3pm`, `Horizontal_Distance_To_Fire_Points`
//! - cols `10..=13`, `Wilderness_Area`: **one** categorical attribute (4 areas),
//!   one-hot encoded. Exactly one of these columns is `1` and the rest are `0`.
//! - cols `14..=53`, `Soil_Type`: **one** categorical attribute (40 soil types),
//!   one-hot encoded. Exactly one of these columns is `1` and the rest are `0`.
//!
//! All 54 columns use `f64` values (the one-hot columns hold `0.0`/`1.0`), so each
//! one-hot block sums to `1` per row. See the struct docs for a per-column table.
//!
//! **Target:** `cover_type` - the forest cover type, one of `1`–`7` (stored as
//! `u8`):
//!
//! - `1` = Spruce/Fir
//! - `2` = Lodgepole Pine
//! - `3` = Ponderosa Pine
//! - `4` = Cottonwood/Willow
//! - `5` = Aspen
//! - `6` = Douglas-fir
//! - `7` = Krummholz
//!
//! **Samples:** 581,012 total
//! **Application:** Multi-class classification / forest cover type prediction
//!
//! **Source:** UCI Machine Learning Repository, via the gzip-compressed
//! `covtype.data.gz` mirror that scikit-learn's `fetch_covtype` downloads.
//! <https://archive.ics.uci.edu/dataset/31/covertype>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, gunzip};
use ndarray::{Array1, Array2};
use std::fs::File;

/// The URL for the Forest Cover Type dataset.
///
/// This is the gzip-compressed `covtype.data.gz` mirror that scikit-learn's
/// `fetch_covtype` uses. The URL has no filename segment, so the code saves the
/// download under an explicit name ([`COVTYPE_GZ_FILENAME`]).
///
/// # Citation
///
/// J. A. Blackard and D. J. Dean. "Covertype," UCI Machine Learning Repository,
/// \[Online\]. Available: <https://doi.org/10.24432/C50K5N>
const COVTYPE_DATA_URL: &str = "https://ndownloader.figshare.com/files/5976039";

/// The filename for the downloaded gzip archive inside the temp directory.
const COVTYPE_GZ_FILENAME: &str = "covtype.data.gz";

/// The name of the final cached (decompressed) Cover Type dataset file.
const COVTYPE_FILENAME: &str = "covtype.csv";

/// The SHA256 hash of the **decompressed** Cover Type dataset file (`covtype.csv`).
///
/// This hash covers the uncompressed comma-separated data, not the downloaded
/// `covtype.data.gz`, because the cached file is the decompressed one.
const COVTYPE_SHA256: &str = "0a9371cef7c964b5475d6053cc3e0894a5aa6f65ad1ed3ecb01c45aa96217945";

/// The name of the dataset
const COVTYPE_DATASET_NAME: &str = "covtype";

/// The number of cartographic features per sample.
const N_FEATURES: usize = 54;

/// The number of columns per CSV record (54 features and 1 cover-type label).
const N_COLUMNS: usize = N_FEATURES + 1;

/// The expected number of samples, used only to pre-allocate the parse buffers.
const N_SAMPLES: usize = 581_012;

/// Type alias for the Cover Type dataset: (features, labels).
type CovtypeData = (Array2<f64>, Array1<u8>);

/// A struct that represents the Forest Cover Type dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The Forest CoverType dataset contains 581,012 cartographic samples, each
/// describing a 30×30 meter cell of the Roosevelt National Forest in northern
/// Colorado. The 54 features combine 10 quantitative measurements (elevation,
/// slope, distances to hydrology, roadways, and fire points, and hillshade
/// indices) with two one-hot blocks. The one-hot blocks are 4 `Wilderness_Area`
/// columns and 40 `Soil_Type` columns. The target is the forest cover type
/// (`1`–`7`).
///
/// This is the same data that scikit-learn exposes through `fetch_covtype`.
///
/// # Feature columns
///
/// The 54 feature columns are **not** 54 independent variables. They encode 12
/// logical attributes: 10 numeric and 2 categorical. The two categorical
/// attributes are already **one-hot expanded** into many binary indicator columns.
/// The table below lists them by 0-based column index in the feature matrix:
///
/// | Columns   | Attribute(s)                                  | Encoding                                   |
/// |-----------|-----------------------------------------------|--------------------------------------------|
/// | `0`       | `Elevation`                                   | quantitative (meters)                      |
/// | `1`       | `Aspect`                                      | quantitative (azimuth degrees)             |
/// | `2`       | `Slope`                                       | quantitative (degrees)                     |
/// | `3`       | `Horizontal_Distance_To_Hydrology`            | quantitative                               |
/// | `4`       | `Vertical_Distance_To_Hydrology`              | quantitative (may be negative)             |
/// | `5`       | `Horizontal_Distance_To_Roadways`             | quantitative                               |
/// | `6`       | `Hillshade_9am`                               | quantitative (`0..=255`)                   |
/// | `7`       | `Hillshade_Noon`                              | quantitative (`0..=255`)                   |
/// | `8`       | `Hillshade_3pm`                               | quantitative (`0..=255`)                   |
/// | `9`       | `Horizontal_Distance_To_Fire_Points`          | quantitative                               |
/// | `10..=13` | `Wilderness_Area` (one attribute, 4 areas)    | one-hot: exactly one column is `1`, rest `0` |
/// | `14..=53` | `Soil_Type` (one attribute, 40 soil types)    | one-hot: exactly one column is `1`, rest `0` |
///
/// Columns `0..=9` hold ten distinct numeric features. Columns `10..=13` jointly
/// answer "which of 4 wilderness areas", and columns `14..=53` jointly answer
/// "which of 40 soil types". Each of these two blocks is a single categorical
/// variable: `1` marks the active category, and `0` marks every other column in
/// the block. Each block sums to `1`. All 54 columns use `f64` values (the one-hot
/// columns hold `0.0` or `1.0`), which matches scikit-learn's dense `fetch_covtype`
/// matrix.
///
/// # Labels
///
/// - cover type (in `u8`): `1` = Spruce/Fir, `2` = Lodgepole Pine,
///   `3` = Ponderosa Pine, `4` = Cottonwood/Willow, `5` = Aspen,
///   `6` = Douglas-fir, `7` = Krummholz
///
/// See more information at
/// <https://archive.ics.uci.edu/dataset/31/covertype>
///
/// # Citation
///
/// J. A. Blackard and D. J. Dean. "Covertype," UCI Machine Learning Repository,
/// \[Online\]. Available: <https://doi.org/10.24432/C50K5N>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Covtype;
///
/// let download_dir = "./covtype"; // the loader creates the directory if it does not exist
///
/// let mut dataset = Covtype::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap();
/// assert_eq!(features.shape(), &[581012, 54]);
/// assert_eq!(labels.len(), 581012);
///
/// // `get_data()` borrows the cached arrays without a reload. `get_data_mut()`
/// // edits the arrays in place. This needs no clone and no reload. The change
/// // stays cached. If you only need to change values, prefer this method over
/// // `.to_owned()`.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 2596.0;
///     labels[0] = 5;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out with no `to_owned()` clone. This
/// // leaves the instance reusable. The next access reloads the data from the
/// // cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[581012, 54]);
/// assert_eq!(owned_labels.len(), 581012);
///
/// // `into_data()` also returns the owned arrays with no clone, but it
/// // consumes the instance. If you are done with the dataset, use it.
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[581012, 54]);
/// assert_eq!(owned_labels.len(), 581012);
/// ```
#[derive(Debug)]
pub struct Covtype {
    dataset: Dataset<CovtypeData, DatasetError>,
}

impl Covtype {
    /// Create a new Covtype instance without loading data.
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
    /// - `Self` - a `Covtype` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Covtype {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Forest Cover Type dataset.
    fn load_data(dir: &str) -> Result<CovtypeData, DatasetError> {
        // Download the gzip-compressed `covtype.data.gz` file, then decompress it
        // into the plain comma-separated `covtype.csv` file.
        let file_path = acquire_dataset(
            dir,
            COVTYPE_FILENAME,
            COVTYPE_DATASET_NAME,
            Some(COVTYPE_SHA256),
            |temp_path| {
                download_to_with_retries(
                    COVTYPE_DATA_URL,
                    temp_path,
                    Some(COVTYPE_GZ_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                let gz_path = temp_path.join(COVTYPE_GZ_FILENAME);
                let csv_path = temp_path.join(COVTYPE_FILENAME);
                gunzip(&gz_path, &csv_path)?;
                Ok(csv_path)
            },
        )?;

        // `covtype.data` is a headerless comma-separated file: every line is a
        // record of 54 cartographic features followed by the cover-type label.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        // Pre-allocate for the known sample count. This avoids repeatedly growing
        // a ~250 MB feature buffer. Parsing still works for any actual row count.
        let mut features = Vec::with_capacity(N_SAMPLES * N_FEATURES);
        let mut labels = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(COVTYPE_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    COVTYPE_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            for (col, field) in record.iter().take(N_FEATURES).enumerate() {
                let value: f64 = field.trim().parse().map_err(|e| {
                    DatasetError::parse_failed(
                        COVTYPE_DATASET_NAME,
                        &format!("feature_{}", col),
                        line_num,
                        e,
                    )
                })?;
                features.push(value);
            }

            let raw_label = record[N_FEATURES].trim();
            let label: u8 = raw_label.parse().map_err(|e| {
                DatasetError::parse_failed(COVTYPE_DATASET_NAME, "cover_type", line_num, e)
            })?;
            if !(1..=7).contains(&label) {
                return Err(DatasetError::invalid_value(
                    COVTYPE_DATASET_NAME,
                    "cover_type",
                    raw_label,
                    line_num,
                ));
            }
            labels.push(label);
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(COVTYPE_DATASET_NAME));
        }

        // Cover Type has a fixed schema of 54 numeric features per sample.
        let features_array = Array2::from_shape_vec((n_samples, N_FEATURES), features)
            .map_err(|e| DatasetError::array_shape_error(COVTYPE_DATASET_NAME, "features", e))?;
        let labels_array = Array1::from_vec(labels);

        Ok((features_array, labels_array))
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(581012, 54)`
    ///   containing the 10 quantitative variables followed by the 4 one-hot
    ///   `Wilderness_Area` and 40 one-hot `Soil_Type` columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File decompression or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match the expected dimensions (581012 samples, 54 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get a reference to the label vector.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array1<u8>` - Reference to label vector with shape `(581012,)` containing the cover-type classes (`1`–`7`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File decompression or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match the expected dimensions (581012 samples)
    pub fn labels(&self) -> Result<&Array1<u8>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and labels as references.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&CovtypeData` - reference to the cached `(features, labels)` tuple.
    ///   The feature matrix has shape `(581012, 54)`. The label vector has shape
    ///   `(581012,)` and contains the cover-type classes (`1`–`7`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File decompression or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match the expected dimensions (581012 samples, 54 features)
    pub fn data(&self) -> Result<&CovtypeData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and labels as references **without** triggering loading.
    ///
    /// Unlike [`Covtype::data`], which loads the dataset on the first call, this
    /// method never runs the loader. If the data has not loaded yet, this method
    /// returns `None` instead of downloading and parsing it. Use this method only
    /// when you want data that is already cached. This avoids the download and
    /// parse cost if the dataset is not cached yet.
    ///
    /// # Returns
    ///
    /// - `Some(&CovtypeData)` - reference to the cached `(features, labels)` tuple
    ///   (feature matrix `(581012, 54)`, label vector `(581012,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&CovtypeData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This method lets you change the cached arrays in place (for example, to
    /// normalize features or replace label values). It needs no `to_owned()`
    /// clone, and it does not remove the data from the cache. The changes persist,
    /// so later calls to [`Covtype::features`], [`Covtype::data`], or
    /// [`Covtype::get_data`] see them.
    ///
    /// Like [`Covtype::get_data`], this method does **not** trigger loading. It
    /// returns `None` if the dataset has not loaded yet. If you need the data to
    /// be present, call a loading accessor first, for example [`Covtype::data`].
    ///
    /// # Returns
    ///
    /// - `Some(&mut CovtypeData)` - mutable reference to the cached
    ///   `(features, labels)` tuple (feature matrix `(581012, 54)`, label vector
    ///   `(581012,)`), if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut CovtypeData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`Covtype::data`], which borrows the cached data, this method moves
    /// the data out and returns owned arrays directly, with no `to_owned()` clone
    /// needed. The dataset loads on the first access if it has not loaded yet.
    ///
    /// This method **consumes** `self`, so you cannot use the instance afterward.
    /// If you want owned data but need to keep using the instance, use
    /// [`Covtype::take_data`] instead. That method takes `&mut self` and leaves
    /// the instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<u8>)` - owned feature matrix with shape
    ///   `(581012, 54)` and owned label vector with shape `(581012,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn into_data(self) -> Result<CovtypeData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and labels out of the dataset. This leaves the
    /// instance reusable.
    ///
    /// Like [`Covtype::into_data`], this method returns owned arrays with no
    /// `to_owned()` clone. Instead of consuming the instance, it takes `&mut self`
    /// and moves the cached data out. This resets the instance to its unloaded
    /// state. The next accessor call, for example [`Covtype::features`] or
    /// [`Covtype::data`], loads the dataset again.
    ///
    /// If you are done with the instance, use [`Covtype::into_data`] instead.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<u8>)` - owned feature matrix with shape
    ///   `(581012, 54)` and owned label vector with shape `(581012,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<CovtypeData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(Covtype, CovtypeData, "covtype");
