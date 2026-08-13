//! White wine subset of the Wine Quality dataset.
//!
//! See [`crate::dataset::wine_quality`] for the full dataset description,
//! including the columns, the target, application scenarios, and the source.
//!
//! **Samples:** 4898
//! **Columns:** 12

use crate::DOWNLOAD_RETRIES;
use crate::dataset::wine_quality::parse_wine_data_to_table;
use crate::table::Table;
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use std::fs::File;

/// Number of feature columns.
const N_FEATURES: usize = 11;

/// The URL for the White Wine Quality dataset.
const WHITE_WINE_DATA_URL: &str = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-white.csv";

/// The white wine file of the CSV files inside the zip archive.
const WHITE_WINE_QUALITY_FILENAME: &str = "winequality-white.csv";

/// The SHA256 hash of the white wine quality dataset.
const WHITE_WINE_QUALITY_SHA256: &str =
    "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836";

/// A struct that represents the White Wine Quality dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The dataset contains physicochemical properties of Portuguese "Vinho Verde"
/// white wine samples and a quality score for each sample.
///
/// # Columns
///
/// | Name                   | Type      | Description                        |
/// |------------------------|-----------|-------------------------------------|
/// | `fixed acidity`        | `Numeric` | fixed acidity                      |
/// | `volatile acidity`     | `Numeric` | volatile acidity                   |
/// | `citric acid`          | `Numeric` | citric acid                        |
/// | `residual sugar`       | `Numeric` | residual sugar                     |
/// | `chlorides`            | `Numeric` | chlorides                          |
/// | `free sulfur dioxide`  | `Numeric` | free sulfur dioxide                |
/// | `total sulfur dioxide` | `Numeric` | total sulfur dioxide               |
/// | `density`              | `Numeric` | density                            |
/// | `pH`                   | `Numeric` | pH                                 |
/// | `sulphates`            | `Numeric` | sulphates                          |
/// | `alcohol`              | `Numeric` | alcohol                            |
/// | `quality`              | `Numeric` | quality score between `0` and `10` |
///
/// The source designates the 11 physicochemical measurements as the inputs
/// ([`WhiteWineQuality::FEATURE_NAMES`]) and `quality` as the label
/// ([`WhiteWineQuality::TARGET`]).
///
/// See more information at <https://archive.ics.uci.edu/dataset/186/wine+quality>
///
/// # Citation
///
/// P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. "Wine Quality," UCI Machine Learning Repository, 2009. \[Online\]. Available: <https://doi.org/10.24432/C56S3T>.
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::WhiteWineQuality;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./white_wine";
///
/// let mut dataset = WhiteWineQuality::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 4898);
/// assert_eq!(table.n_columns(), 12);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&WhiteWineQuality::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[4898, 11]);
///
/// // Reach one column by name.
/// let quality = table.column("quality").unwrap().as_numeric().unwrap();
/// assert_eq!(quality.len(), 4898);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("fixed acidity") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 10.0;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 4898);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 4898);
/// ```
#[derive(Debug)]
pub struct WhiteWineQuality {
    dataset: Dataset<Table, DatasetError>,
}

impl WhiteWineQuality {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "quality";

    /// Create a new WhiteWineQuality instance without loading data.
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
    /// - `Self` - a `WhiteWineQuality` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        WhiteWineQuality {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the White Wine Quality dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            WHITE_WINE_QUALITY_FILENAME,
            "white_wine_quality",
            Some(WHITE_WINE_QUALITY_SHA256),
            |temp_path| {
                download_to_with_retries(WHITE_WINE_DATA_URL, temp_path, None, DOWNLOAD_RETRIES)?;
                Ok(temp_path.join(WHITE_WINE_QUALITY_FILENAME))
            },
        )?;

        let file = File::open(&file_path)?;
        parse_wine_data_to_table("white_wine_quality", file)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 4898 samples and 12
    ///   columns.
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
    /// Unlike [`WhiteWineQuality::data`], this method never runs the loader. If
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
    /// changes stay in the cache. Later calls to [`WhiteWineQuality::data`] or
    /// [`WhiteWineQuality::get_data`] see them.
    ///
    /// Like [`WhiteWineQuality::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`WhiteWineQuality::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 4898 samples and 12 columns.
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
    /// - `Table` - the owned table of 4898 samples and 12 columns.
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

impl_ml_dataset!(WhiteWineQuality, "white_wine_quality");
