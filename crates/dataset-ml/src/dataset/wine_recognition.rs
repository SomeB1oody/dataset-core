//! Wine Recognition dataset.
//!
//! The dataset holds results from a chemical analysis of wines grown in the
//! same region in Italy but derived from three different cultivars. The
//! analysis determined the quantities of 13 constituents found in each of the
//! three types of wine. The task is to predict the cultivar (one of three
//! classes) from the constituents.
//!
//! This is the **Wine recognition** dataset (the same one bundled with
//! scikit-learn as `load_wine`). It is distinct from the **Wine Quality**
//! datasets in [`crate::dataset::wine_quality`], which are a regression task on red/white
//! wine quality scores.
//!
//! **Columns (14):**
//!
//! | Name                           | Type      | Description                                       |
//! |--------------------------------|-----------|---------------------------------------------------|
//! | `class`                        | `String`  | `class_1`, `class_2`, or `class_3` (the cultivar) |
//! | `alcohol`                      | `Numeric` | alcohol content                                   |
//! | `malic_acid`                   | `Numeric` | malic acid content                                |
//! | `ash`                          | `Numeric` | ash content                                       |
//! | `alcalinity_of_ash`            | `Numeric` | alcalinity of the ash                             |
//! | `magnesium`                    | `Numeric` | magnesium content                                 |
//! | `total_phenols`                | `Numeric` | total phenol content                              |
//! | `flavanoids`                   | `Numeric` | flavanoid content                                 |
//! | `nonflavanoid_phenols`         | `Numeric` | nonflavanoid phenol content                       |
//! | `proanthocyanins`              | `Numeric` | proanthocyanin content                            |
//! | `color_intensity`              | `Numeric` | color intensity                                   |
//! | `hue`                          | `Numeric` | hue                                               |
//! | `od280_od315_of_diluted_wines` | `Numeric` | OD280/OD315 ratio of diluted wines                |
//! | `proline`                      | `Numeric` | proline content                                   |
//!
//! The source designates the 13 constituents as the inputs
//! ([`WineRecognition::FEATURE_NAMES`](crate::WineRecognition::FEATURE_NAMES)) and `class` as the label
//! ([`WineRecognition::TARGET`](crate::WineRecognition::TARGET)).
//!
//! **Samples:** 178 total (59 of class 1, 71 of class 2, 48 of class 3)
//! **Application:** Multi-class classification / cultivar recognition
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5PC7J>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
use serde::Deserialize;
use std::fs::File;

/// The URL for the Wine Recognition dataset.
///
/// # Citation
///
/// S. Aeberhard and M. Forina. "Wine," UCI Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C5PC7J>
const WINE_RECOGNITION_DATA_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data";

/// The name of the Wine Recognition dataset file.
const WINE_RECOGNITION_FILENAME: &str = "wine_recognition.csv";

/// The SHA256 hash of the Wine Recognition dataset file.
const WINE_RECOGNITION_SHA256: &str =
    "6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659";

/// The name of the dataset
const WINE_RECOGNITION_DATASET_NAME: &str = "wine_recognition";

/// The number of features per sample (13 chemical constituents).
const N_FEATURES: usize = 13;

/// One CSV record of the Wine Recognition dataset: the `1`/`2`/`3` class label
/// followed by the 13 `f64` constituent measurements.
///
/// This struct declares fields in CSV column order. It deserializes them
/// **positionally** (the loader disables csv's header handling). This matches
/// the headerless `wine.data` layout, where the class is the first column.
#[derive(Deserialize)]
struct WineRecognitionRecord {
    class: String,
    alcohol: f64,
    malic_acid: f64,
    ash: f64,
    alcalinity_of_ash: f64,
    magnesium: f64,
    total_phenols: f64,
    flavanoids: f64,
    nonflavanoid_phenols: f64,
    proanthocyanins: f64,
    color_intensity: f64,
    hue: f64,
    od280_od315_of_diluted_wines: f64,
    proline: f64,
}

/// A struct that represents the Wine Recognition dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// This dataset is the result of a chemical analysis of wines grown in the same
/// region in Italy but derived from three different cultivars. The analysis
/// determined the quantities of 13 constituents found in each of the three types
/// of wine.
///
/// This is the **Wine recognition** dataset (scikit-learn's `load_wine`), a
/// multi-class classification task. It is **not** the same as the
/// [`crate::dataset::wine_quality`] datasets, which predict a quality score (regression).
///
/// # Columns
///
/// | Name                           | Type      | Description                                       |
/// |--------------------------------|-----------|---------------------------------------------------|
/// | `class`                        | `String`  | `class_1`, `class_2`, or `class_3` (the cultivar) |
/// | `alcohol`                      | `Numeric` | alcohol content                                   |
/// | `malic_acid`                   | `Numeric` | malic acid content                                |
/// | `ash`                          | `Numeric` | ash content                                       |
/// | `alcalinity_of_ash`            | `Numeric` | alcalinity of the ash                             |
/// | `magnesium`                    | `Numeric` | magnesium content                                 |
/// | `total_phenols`                | `Numeric` | total phenol content                              |
/// | `flavanoids`                   | `Numeric` | flavanoid content                                 |
/// | `nonflavanoid_phenols`         | `Numeric` | nonflavanoid phenol content                       |
/// | `proanthocyanins`              | `Numeric` | proanthocyanin content                            |
/// | `color_intensity`              | `Numeric` | color intensity                                   |
/// | `hue`                          | `Numeric` | hue                                               |
/// | `od280_od315_of_diluted_wines` | `Numeric` | OD280/OD315 ratio of diluted wines                |
/// | `proline`                      | `Numeric` | proline content                                   |
///
/// The source designates the 13 constituents as the inputs
/// ([`WineRecognition::FEATURE_NAMES`]) and `class` as the label
/// ([`WineRecognition::TARGET`]).
///
/// See more information at <https://archive.ics.uci.edu/dataset/109/wine>
///
/// # Citation
///
/// S. Aeberhard and M. Forina. "Wine," UCI Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C5PC7J>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::WineRecognition;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./wine_recognition";
///
/// let mut dataset = WineRecognition::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 178);
/// assert_eq!(table.n_columns(), 14);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&WineRecognition::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[178, 13]);
///
/// // Reach one column by name.
/// let class = table.column(WineRecognition::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(class.len(), 178);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("alcohol") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 13.5;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 178);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 178);
/// ```
#[derive(Debug)]
pub struct WineRecognition {
    dataset: Dataset<Table, DatasetError>,
}

impl WineRecognition {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280_od315_of_diluted_wines",
        "proline",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "class";

    /// Create a new WineRecognition instance without loading data.
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
    /// - `Self` - a `WineRecognition` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        WineRecognition {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Wine Recognition dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            WINE_RECOGNITION_FILENAME,
            WINE_RECOGNITION_DATASET_NAME,
            Some(WINE_RECOGNITION_SHA256),
            |temp_path| {
                download_to_with_retries(
                    WINE_RECOGNITION_DATA_URL,
                    temp_path,
                    Some(WINE_RECOGNITION_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(WINE_RECOGNITION_FILENAME))
            },
        )?;

        // `wine.data` has no header row, so every line is a record. Do not skip
        // the first one.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features = Vec::new();
        let mut classes = Vec::new();

        for (idx, result) in rdr.deserialize::<WineRecognitionRecord>().enumerate() {
            let WineRecognitionRecord {
                class,
                alcohol,
                malic_acid,
                ash,
                alcalinity_of_ash,
                magnesium,
                total_phenols,
                flavanoids,
                nonflavanoid_phenols,
                proanthocyanins,
                color_intensity,
                hue,
                od280_od315_of_diluted_wines,
                proline,
            } = result
                .map_err(|e| DatasetError::csv_read_error(WINE_RECOGNITION_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            features.extend_from_slice(&[
                alcohol,
                malic_acid,
                ash,
                alcalinity_of_ash,
                magnesium,
                total_phenols,
                flavanoids,
                nonflavanoid_phenols,
                proanthocyanins,
                color_intensity,
                hue,
                od280_od315_of_diluted_wines,
                proline,
            ]);

            classes.push(
                match class.as_str() {
                    "1" => "class_1",
                    "2" => "class_2",
                    "3" => "class_3",
                    other => {
                        return Err(DatasetError::invalid_value(
                            WINE_RECOGNITION_DATASET_NAME,
                            Self::TARGET,
                            other,
                            line_num,
                        ));
                    }
                }
                .to_string(),
            );
        }

        // The source lists the class before the 13 constituents.
        let mut columns = Vec::with_capacity(N_FEATURES + 1);
        columns.push(Column::new(
            Self::TARGET,
            ColumnData::String(Array1::from_vec(classes)),
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

        Table::new(WINE_RECOGNITION_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 178 samples and 14 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (unparseable values, an unknown class)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`WineRecognition::data`], this method never runs the loader. If
    /// the data has not loaded yet, it returns `None` instead of downloading and
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
    /// changes stay in the cache. Later calls to [`WineRecognition::data`] or
    /// [`WineRecognition::get_data`] see them.
    ///
    /// Like [`WineRecognition::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`WineRecognition::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 178 samples and 14 columns.
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
    /// - `Table` - the owned table of 178 samples and 14 columns.
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

impl_ml_dataset!(WineRecognition, "wine_recognition");
