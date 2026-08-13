//! Iris flower dataset.
//!
//! The classic Fisher Iris dataset for multi-class classification. It holds
//! measurements of three Iris species: `setosa`, `versicolor`, and `virginica`.
//!
//! **Columns (5):**
//!
//! | Name           | Type      | Description                            |
//! |----------------|-----------|----------------------------------------|
//! | `sepal_length` | `Numeric` | sepal length in cm                     |
//! | `sepal_width`  | `Numeric` | sepal width in cm                      |
//! | `petal_length` | `Numeric` | petal length in cm                     |
//! | `petal_width`  | `Numeric` | petal width in cm                      |
//! | `species`      | `String`  | `setosa`, `versicolor`, or `virginica` |
//!
//! The source designates the four measurements as the inputs
//! ([`Iris::FEATURE_NAMES`](crate::Iris::FEATURE_NAMES)) and `species` as the label ([`Iris::TARGET`](crate::Iris::TARGET)).
//!
//! **Samples:** 150 total, 50 per species
//! **Application:** Multi-class classification / species recognition
//!
//! **Missing values:** none.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C56C76>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
use serde::Deserialize;
use std::fs::File;

/// The URL for the Iris dataset.
///
/// # Citation
///
/// R. A. Fisher. "Iris," UCI Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C56C76>
const IRIS_DATA_URL: &str = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv";

/// The name of the Iris dataset file.
const IRIS_FILENAME: &str = "iris.csv";

/// The SHA256 hash of the Iris dataset file.
const IRIS_SHA256: &str = "c52742e50315a99f956a383faedf7575552675f6409ef0f9a47076dd08479930";

/// The name of the dataset.
const IRIS_DATASET_NAME: &str = "iris";

/// Number of samples.
const N_SAMPLES: usize = 150;

/// Number of measurement columns.
const N_FEATURES: usize = 4;

/// The three species the `species` column names.
const SPECIES: [&str; 3] = ["setosa", "versicolor", "virginica"];

/// One CSV record of the Iris dataset, in source column order.
#[derive(Deserialize)]
struct IrisRecord {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    species: String,
}

/// A struct that represents the Iris dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The Iris dataset holds four measurements of 150 iris flowers, 50 of each of
/// three species. The task is to recognize the species. `setosa` separates from
/// the other two by a straight line, and `versicolor` and `virginica` overlap.
///
/// # Columns
///
/// | Name           | Type      | Description                |
/// |----------------|-----------|----------------------------|
/// | `sepal_length` | `Numeric` | sepal length in cm         |
/// | `sepal_width`  | `Numeric` | sepal width in cm          |
/// | `petal_length` | `Numeric` | petal length in cm         |
/// | `petal_width`  | `Numeric` | petal width in cm          |
/// | `species`      | `String`  | one of three species names |
///
/// The source designates the four measurements as the inputs
/// ([`Iris::FEATURE_NAMES`]) and `species` as the label ([`Iris::TARGET`]).
///
/// Missing values: none.
///
/// See more information at <https://archive.ics.uci.edu/dataset/53/iris>.
///
/// # Citation
///
/// R. A. Fisher. "Iris," UCI Machine Learning Repository, \[Online\].
/// Available: <https://doi.org/10.24432/C56C76>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Iris;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./iris";
///
/// let mut dataset = Iris::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 150);
/// assert_eq!(table.n_columns(), 5);
///
/// // Name the columns you want in the matrix.
/// let features = table.numeric_matrix(&Iris::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[150, 4]);
///
/// // Two of them, in the order you ask for.
/// let petals = table.numeric_matrix(&["petal_width", "petal_length"]).unwrap();
/// assert_eq!(petals.shape(), &[150, 2]);
///
/// // Reach one column by name.
/// let species = table.column(Iris::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(species.len(), 150);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("sepal_length") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 9.9;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 150);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 150);
/// ```
#[derive(Debug)]
pub struct Iris {
    dataset: Dataset<Table, DatasetError>,
}

impl Iris {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] =
        ["sepal_length", "sepal_width", "petal_length", "petal_width"];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "species";

    /// Create a new Iris instance without loading data.
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
    /// - `Self` - an `Iris` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Iris {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Iris dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            IRIS_FILENAME,
            IRIS_DATASET_NAME,
            Some(IRIS_SHA256),
            |temp_path| {
                download_to_with_retries(IRIS_DATA_URL, temp_path, None, DOWNLOAD_RETRIES)?;
                Ok(temp_path.join(IRIS_FILENAME))
            },
        )?;

        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut sepal_length = Vec::with_capacity(N_SAMPLES);
        let mut sepal_width = Vec::with_capacity(N_SAMPLES);
        let mut petal_length = Vec::with_capacity(N_SAMPLES);
        let mut petal_width = Vec::with_capacity(N_SAMPLES);
        let mut species = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.deserialize::<IrisRecord>().skip(1).enumerate() {
            let record = result.map_err(|e| DatasetError::csv_read_error(IRIS_DATASET_NAME, e))?;
            let line_num = idx + 2; // +1 for 0-indexed, +1 for the header

            if !SPECIES.contains(&record.species.as_str()) {
                return Err(DatasetError::invalid_value(
                    IRIS_DATASET_NAME,
                    Self::TARGET,
                    &record.species,
                    line_num,
                ));
            }

            sepal_length.push(record.sepal_length);
            sepal_width.push(record.sepal_width);
            petal_length.push(record.petal_length);
            petal_width.push(record.petal_width);
            species.push(record.species);
        }

        Table::new(
            IRIS_DATASET_NAME,
            vec![
                Column::new(
                    Self::FEATURE_NAMES[0],
                    ColumnData::Numeric(Array1::from_vec(sepal_length)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[1],
                    ColumnData::Numeric(Array1::from_vec(sepal_width)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[2],
                    ColumnData::Numeric(Array1::from_vec(petal_length)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[3],
                    ColumnData::Numeric(Array1::from_vec(petal_width)),
                ),
                Column::new(Self::TARGET, ColumnData::String(Array1::from_vec(species))),
            ],
        )
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 150 samples and 5 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (unparseable values, an unknown species)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`Iris::data`], this method never runs the loader. If the data has
    /// not loaded yet, it returns `None` instead of downloading and parsing it.
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
    /// changes stay in the cache. Later calls to [`Iris::data`] or
    /// [`Iris::get_data`] see them.
    ///
    /// Like [`Iris::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`Iris::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 150 samples and 5 columns.
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
    /// - `Table` - the owned table of 150 samples and 5 columns.
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

impl_ml_dataset!(Iris, "iris");
