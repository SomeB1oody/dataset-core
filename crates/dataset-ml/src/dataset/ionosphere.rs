//! Ionosphere dataset.
//!
//! A radar system in Goose Bay, Labrador, collected these returns. The system
//! aimed at free electrons in the ionosphere. "Good" (`g`) radar returns show
//! evidence of some type of structure in the ionosphere. "Bad" (`b`) returns
//! pass through the ionosphere instead. The task is to predict the quality of a
//! return from 34 continuous features.
//!
//! **Features (34, all numeric):** 17 pulses, each described by two attributes:
//! the real and imaginary components of the complex electromagnetic signal,
//! processed by an autocorrelation function. The first attribute is `0` or `1`
//! (whether the return was usable), and the second attribute is constant `0` in
//! this collection. The loader still exposes both attributes verbatim as `f64`
//! columns.
//!
//! **Target:** `class` - one of `good` or `bad`
//!
//! **Samples:** 351 total (225 good, 126 bad)
//! **Application:** Binary classification / radar return quality
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5W01B>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::{Array1, Array2};
use std::fs::File;

use csv::ReaderBuilder;

/// The URL for the Ionosphere dataset (the `ionosphere.data` file).
///
/// # Citation
///
/// V. Sigillito, S. Wing, L. Hutton, and K. Baker. "Ionosphere," UCI Machine
/// Learning Repository, \[Online\]. Available: <https://doi.org/10.24432/C5W01B>
const IONOSPHERE_DATA_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data";

/// The name of the cached Ionosphere dataset file.
const IONOSPHERE_FILENAME: &str = "ionosphere.csv";

/// The SHA256 hash of the cached Ionosphere dataset file (`ionosphere.data`'s bytes).
const IONOSPHERE_SHA256: &str = "46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd";

/// The name of the dataset.
const IONOSPHERE_DATASET_NAME: &str = "ionosphere";

/// Number of samples.
const N_SAMPLES: usize = 351;

/// Number of numeric features.
const N_FEATURES: usize = 34;

/// Number of columns per record (34 features + 1 label).
const N_COLUMNS: usize = 35;

/// Source column index of the label (`class`). The label is the **last** column.
const LABEL_COLUMN: usize = 34;

/// Type alias for the Ionosphere dataset: (features, labels).
type IonosphereData = (Array2<f64>, Array1<&'static str>);

/// This struct represents the Ionosphere dataset and loads data lazily.
///
/// You do not load the dataset until you call one of the data accessor
/// methods. After that, the dataset caches the data for later calls.
///
/// # About Dataset
///
/// A system in Goose Bay, Labrador, collected this radar data. The system used
/// a phased array of 16 high-frequency antennas with a total transmitted power
/// of about 6.4 kilowatts. Its targets were free electrons in the ionosphere.
/// "Good" radar returns show evidence of some type of structure in the
/// ionosphere. "Bad" returns do not: their signals pass through the ionosphere
/// instead. The system processed received signals with an autocorrelation
/// function. The function took the pulse time and the pulse number as
/// arguments. The Goose Bay system used 17 pulse numbers. Each pulse has two
/// attributes, the real and imaginary parts of the complex electromagnetic
/// signal, for a total of 34 continuous features.
///
/// # Feature columns
///
/// The `(351, 34)` `Array2<f64>` matrix holds all 34 quantitative features.
/// Columns come in 17 real/imaginary pairs (pulses `1`–`17`):
///
/// | Columns   | Attribute                                   |
/// |-----------|---------------------------------------------|
/// | `0`       | pulse 1 - real part (`0` or `1`)            |
/// | `1`       | pulse 1 - imaginary part (constant `0` here)|
/// | `2`, `3`  | pulse 2 - real, imaginary                   |
/// | …         | …                                           |
/// | `32`, `33`| pulse 17 - real, imaginary                  |
///
/// The values are normalized to roughly `-1..=1`. The first two columns are
/// degenerate in this collection: column `0` is `0` or `1`, and column `1` is
/// always `0`. The loader keeps them verbatim, so the schema matches the source
/// exactly.
///
/// # Labels
///
/// - `class` (shape `(351,)`): the `Array1<&'static str>` maps the source's
///   single-letter codes to readable names, `g` → `"good"` and `b` → `"bad"`.
///
/// See more information at <https://archive.ics.uci.edu/dataset/52/ionosphere>.
///
/// # Citation
///
/// V. Sigillito, S. Wing, L. Hutton, and K. Baker. "Ionosphere," UCI Machine
/// Learning Repository, \[Online\]. Available: <https://doi.org/10.24432/C5W01B>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because every field
/// does. This makes it safe to share across threads. The internal [`Dataset`]
/// keeps lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Ionosphere;
///
/// let download_dir = "./ionosphere"; // the code creates the directory if it does not exist
///
/// let mut dataset = Ionosphere::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
/// assert_eq!(features.shape(), &[351, 34]);
/// assert_eq!(labels.len(), 351);
///
/// // `get_data()` borrows the cached arrays without reloading. `get_data_mut()`
/// // edits them in place: no clone, no reload, and the change stays cached.
/// // Prefer this over cloning with `.to_owned()` when you only need to tweak
/// // values.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 0.5;
///     labels[0] = "bad";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable. The next access reloads from the cached file.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[351, 34]);
/// assert_eq!(owned_labels.len(), 351);
///
/// // `into_data()` also returns owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[351, 34]);
/// assert_eq!(owned_labels.len(), 351);
/// ```
#[derive(Debug)]
pub struct Ionosphere {
    dataset: Dataset<IonosphereData, DatasetError>,
}

impl Ionosphere {
    /// Create a new Ionosphere instance without loading data.
    ///
    /// The dataset loads lazily, on your first call to a data accessor method.
    /// This function only stores the storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory where the dataset will be stored.
    ///
    /// # Returns
    ///
    /// - `Self` - `Ionosphere` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Ionosphere {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Ionosphere dataset.
    fn load_data(dir: &str) -> Result<IonosphereData, DatasetError> {
        // Prepare the dataset file. The source file is `ionosphere.data`. Cache it
        // under `ionosphere.csv`.
        let file_path = acquire_dataset(
            dir,
            IONOSPHERE_FILENAME,
            IONOSPHERE_DATASET_NAME,
            Some(IONOSPHERE_SHA256),
            |temp_path| {
                download_to_with_retries(
                    IONOSPHERE_DATA_URL,
                    temp_path,
                    Some(IONOSPHERE_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(IONOSPHERE_FILENAME))
            },
        )?;

        // The source is plain comma-separated with no header.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features: Vec<f64> = Vec::with_capacity(N_SAMPLES * N_FEATURES);
        let mut labels: Vec<&'static str> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(IONOSPHERE_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            // Skip blank lines defensively (e.g. a trailing newline).
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    IONOSPHERE_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // 34 numeric features.
            for col in 0..N_FEATURES {
                let value: f64 = record[col].parse().map_err(|e| {
                    DatasetError::parse_failed(
                        IONOSPHERE_DATASET_NAME,
                        &format!("attribute_{}", col + 1),
                        line_num,
                        e,
                    )
                })?;
                features.push(value);
            }

            // Label, mapped from the source's single-letter code to a readable name.
            let label = match &record[LABEL_COLUMN] {
                "g" => "good",
                "b" => "bad",
                other => {
                    return Err(DatasetError::invalid_value(
                        IONOSPHERE_DATASET_NAME,
                        "class",
                        other,
                        line_num,
                    ));
                }
            };
            labels.push(label);
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(IONOSPHERE_DATASET_NAME));
        }

        let features_array = Array2::from_shape_vec((n_samples, N_FEATURES), features)
            .map_err(|e| DatasetError::array_shape_error(IONOSPHERE_DATASET_NAME, "features", e))?;

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
    ///   `(351, 34)`: 17 pulses × (real, imaginary) autocorrelation components.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match expected dimensions (351 samples, 34 features)
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
    /// - `&Array1<&'static str>` - Reference to labels vector with shape `(351,)` containing class names (`"good"`, `"bad"`)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match expected dimensions (351 samples)
    pub fn labels(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get both features and labels as references.
    ///
    /// This method triggers lazy loading on first call. Later calls return the
    /// cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&IonosphereData` - reference to the cached `(features, labels)` tuple: the
    ///   feature matrix has shape `(351, 34)` and the label vector has shape
    ///   `(351,)` containing class names (`"good"`, `"bad"`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size does not match expected dimensions (351 samples, 34 features)
    pub fn data(&self) -> Result<&IonosphereData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and labels as references **without** triggering loading.
    ///
    /// Unlike [`Ionosphere::data`], which loads the dataset on first call, this
    /// never runs the loader. If the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. If the data is already cached
    /// and you want to avoid the download and parse cost, use this method.
    ///
    /// # Returns
    ///
    /// - `Some(&IonosphereData)` - reference to the cached `(features, labels)`
    ///   tuple (feature matrix `(351, 34)`, label vector `(351,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&IonosphereData> {
        self.dataset.get()
    }

    /// Get mutable references to features and labels for **in-place** editing.
    ///
    /// This lets you change the cached arrays directly (e.g. normalize features,
    /// replace label values), with no `to_owned()` clone. The cache keeps the
    /// change: it does not remove the arrays. Later calls to
    /// [`Ionosphere::features`], [`Ionosphere::data`], or [`Ionosphere::get_data`]
    /// observe the change.
    ///
    /// Like [`Ionosphere::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. If you need to make sure the data
    /// is present, call a loading accessor first (e.g. [`Ionosphere::data`]).
    ///
    /// # Returns
    ///
    /// - `Some(&mut IonosphereData)` - mutable reference to the cached
    ///   `(features, labels)` tuple (feature matrix `(351, 34)`, label vector
    ///   `(351,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut IonosphereData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** features and labels.
    ///
    /// Unlike [`Ionosphere::data`], which borrows the cached data, this moves it
    /// out and returns owned arrays directly. It needs no `to_owned()` clone. If
    /// the dataset is not loaded yet, this call loads it.
    ///
    /// This consumes `self`, so you cannot use the instance afterward. If you want
    /// owned data but need to keep using the instance, use
    /// [`Ionosphere::take_data`] instead. It takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<&'static str>)` - owned feature matrix with shape
    ///   `(351, 34)` and owned label vector with shape `(351,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn into_data(self) -> Result<IonosphereData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** features and labels out of the dataset, leaving it reusable.
    ///
    /// Like [`Ionosphere::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. Unlike that method, it takes `&mut self` instead of
    /// consuming the instance. It moves the cached data out and resets the
    /// instance to its unloaded state. The next accessor call (e.g.
    /// [`Ionosphere::features`] or [`Ionosphere::data`]) loads the dataset again.
    ///
    /// If you are done with the instance, use [`Ionosphere::into_data`] instead.
    ///
    /// # Returns
    ///
    /// - `(Array2<f64>, Array1<&'static str>)` - owned feature matrix with shape
    ///   `(351, 34)` and owned label vector with shape `(351,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<IonosphereData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(Ionosphere, IonosphereData, "ionosphere");
