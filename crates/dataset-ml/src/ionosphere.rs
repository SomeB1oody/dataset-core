//! Ionosphere dataset.
//!
//! Radar returns collected by a system in Goose Bay, Labrador, aimed at free
//! electrons in the ionosphere. "Good" (`g`) radar returns are those showing
//! evidence of some type of structure in the ionosphere; "bad" (`b`) returns are
//! those that pass through the ionosphere. The task is to predict the quality of
//! a return from 34 continuous features.
//!
//! **Features (34, all numeric):** 17 pulses, each described by two attributes —
//! the real and imaginary components of the complex electromagnetic signal
//! processed by an autocorrelation function. The first attribute is `0` or `1`
//! (whether the return was usable) and the second attribute is constant `0` in
//! this collection; both are still exposed verbatim as `f64` columns.
//!
//! **Target:** `class` — one of `good` or `bad`
//!
//! **Samples:** 351 total (225 good, 126 bad)
//! **Application:** Binary classification / radar return quality
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5W01B>

use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to};
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

/// A struct representing the Ionosphere dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// This radar data was collected by a system in Goose Bay, Labrador, consisting
/// of a phased array of 16 high-frequency antennas with a total transmitted power
/// on the order of 6.4 kilowatts. The targets were free electrons in the
/// ionosphere. "Good" radar returns are those showing evidence of some type of
/// structure in the ionosphere; "bad" returns are those that do not — their
/// signals pass through the ionosphere. Received signals were processed using an
/// autocorrelation function whose arguments are the time of a pulse and the pulse
/// number; there were 17 pulse numbers for the Goose Bay system, and each pulse is
/// described by two attributes (the real and imaginary parts of the complex
/// electromagnetic signal), giving 34 continuous features.
///
/// # Feature columns
///
/// All 34 features are quantitative, stored in one `(351, 34)` `Array2<f64>`
/// matrix. Columns come in 17 real/imaginary pairs (pulses `1`–`17`):
///
/// | Columns   | Attribute                                   |
/// |-----------|---------------------------------------------|
/// | `0`       | pulse 1 — real part (`0` or `1`)            |
/// | `1`       | pulse 1 — imaginary part (constant `0` here)|
/// | `2`, `3`  | pulse 2 — real, imaginary                   |
/// | …         | …                                           |
/// | `32`, `33`| pulse 17 — real, imaginary                  |
///
/// The values are normalized to roughly `-1..=1`. The first two columns are
/// degenerate in this collection (column `0` is `0`/`1`, column `1` is always
/// `0`), but they are kept verbatim so the schema matches the source exactly.
///
/// # Labels
///
/// - `class` (shape `(351,)`): the `Array1<&'static str>` maps the source's
///   single-letter codes to readable names — `g` → `"good"`, `b` → `"bad"`.
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
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::ionosphere::Ionosphere;
///
/// let download_dir = "./ionosphere"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = Ionosphere::new(download_dir);
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
/// assert_eq!(features.shape(), &[351, 34]);
/// assert_eq!(labels.len(), 351);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((features, labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 0.5;
///     labels[0] = "bad";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves owned arrays out (no `to_owned()` clone) and leaves the
/// // instance reusable — the next access reloads from the cached file.
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
    /// The dataset will be loaded lazily when you first call any data accessor method.
    /// This is a lightweight operation that only stores the storage directory.
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

    /// Acquire and parse the Ionosphere dataset.
    fn load_data(dir: &str) -> Result<IonosphereData, DatasetError> {
        // Prepare the dataset file. The source file is `ionosphere.data`; cache it
        // under `ionosphere.csv`.
        let file_path = acquire_dataset(
            dir,
            IONOSPHERE_FILENAME,
            IONOSPHERE_DATASET_NAME,
            Some(IONOSPHERE_SHA256),
            |temp_path| {
                download_to(IONOSPHERE_DATA_URL, temp_path, Some(IONOSPHERE_FILENAME))?;
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

            // Label, mapping the source's single-letter code to a readable name.
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
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
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
    /// - Dataset size doesn't match expected dimensions (351 samples, 34 features)
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
    /// - `&Array1<&'static str>` - Reference to labels vector with shape `(351,)` containing class names (`"good"`, `"bad"`)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (351 samples)
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
    /// - Dataset size doesn't match expected dimensions (351 samples, 34 features)
    pub fn data(&self) -> Result<&IonosphereData, DatasetError> {
        self.dataset.load()
    }

    /// Get both features and labels as references **without** triggering loading.
    ///
    /// Unlike [`Ionosphere::data`], which loads the dataset on first call, this
    /// never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
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
    /// This lets you modify the cached arrays directly (e.g. normalize features,
    /// replace label values) with no `to_owned()` clone and without removing them
    /// from the cache: the changes persist, so later [`Ionosphere::features`],
    /// [`Ionosphere::data`], or [`Ionosphere::get_data`] calls observe them.
    ///
    /// Like [`Ionosphere::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call a loading accessor (e.g.
    /// [`Ionosphere::data`]) first if you need to ensure the data is present.
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
    /// out and returns owned arrays directly — no `to_owned()` clone needed. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`Ionosphere::take_data`] instead — it takes `&mut self` and leaves the
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
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out, resetting the instance to its
    /// unloaded state: the next accessor call (e.g. [`Ionosphere::features`] or
    /// [`Ionosphere::data`]) loads the dataset again.
    ///
    /// Use [`Ionosphere::into_data`] instead if you are done with the instance.
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
