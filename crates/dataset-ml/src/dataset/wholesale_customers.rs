//! Wholesale Customers dataset.
//!
//! Annual spending of 440 clients of a Portuguese wholesale distributor, across
//! six product categories, with the sales channel and the region of each client.
//! The dataset has **no target column**. The usual task is to cluster the
//! clients by their spending, then compare the clusters against the channel and
//! the region.
//!
//! **Columns (8):**
//!
//! | Name               | Type      | Description                   |
//! |--------------------|-----------|---------------------------------|
//! | `Channel`          | `Numeric` | `1` = Horeca (hotel, restaurant, or cafe), `2` = Retail |
//! | `Region`           | `Numeric` | `1` = Lisbon, `2` = Oporto, `3` = other |
//! | `Fresh`            | `Numeric` | annual spending on fresh products |
//! | `Milk`             | `Numeric` | annual spending on milk products |
//! | `Grocery`          | `Numeric` | annual spending on grocery products |
//! | `Frozen`           | `Numeric` | annual spending on frozen products |
//! | `Detergents_Paper` | `Numeric` | annual spending on detergents and paper |
//! | `Delicassen`       | `Numeric` | annual spending on delicatessen products |
//!
//! `Channel` and `Region` are categorical codes, stored as `Numeric`. The other
//! six columns are the annual spending on that product category, in monetary
//! units. The source has no label column. The eight columns above are the
//! model inputs ([`WholesaleCustomers::COLUMN_NAMES`](crate::WholesaleCustomers::COLUMN_NAMES)).
//!
//! **Samples:** 440
//! **Application:** Clustering / customer segmentation
//!
//! **Missing values:** none.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5030X>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
use std::fs::File;

/// The URL for the Wholesale Customers dataset.
///
/// # Citation
///
/// Cardoso, M. (2013). Wholesale customers \[Dataset\]. UCI Machine Learning
/// Repository. <https://doi.org/10.24432/C5030X>
const WHOLESALE_DATA_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv";

/// The name of the cached Wholesale Customers dataset file.
const WHOLESALE_FILENAME: &str = "wholesale_customers.csv";

/// The SHA256 hash of the cached Wholesale Customers dataset file.
const WHOLESALE_SHA256: &str = "c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef";

/// The name of the dataset.
const WHOLESALE_DATASET_NAME: &str = "wholesale_customers";

/// Number of samples.
const N_SAMPLES: usize = 440;

/// Number of feature columns.
const N_FEATURES: usize = 8;

/// A struct that represents the Wholesale Customers dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The Wholesale Customers dataset records the annual spending of 440 clients of
/// a wholesale distributor in Portugal, across six product categories. It also
/// records the sales channel and the region of each client.
///
/// The dataset has **no target column**, so every column is a model input. The
/// usual task is to cluster the clients by their spending, then compare the
/// clusters against `Channel` and `Region`.
///
/// # Columns
///
/// | Name               | Type      | Description                   |
/// |--------------------|-----------|---------------------------------|
/// | `Channel`          | `Numeric` | `1` = Horeca (hotel, restaurant, or cafe), `2` = Retail |
/// | `Region`           | `Numeric` | `1` = Lisbon, `2` = Oporto, `3` = other |
/// | `Fresh`            | `Numeric` | annual spending on fresh products |
/// | `Milk`             | `Numeric` | annual spending on milk products |
/// | `Grocery`          | `Numeric` | annual spending on grocery products |
/// | `Frozen`           | `Numeric` | annual spending on frozen products |
/// | `Detergents_Paper` | `Numeric` | annual spending on detergents and paper |
/// | `Delicassen`       | `Numeric` | annual spending on delicatessen products |
///
/// [`WholesaleCustomers::COLUMN_NAMES`] holds these names in the same order.
///
/// The six spending columns hold monetary units. Every value is a whole number
/// in the source, and the loader stores all 8 columns as `Numeric`.
///
/// `Channel` and `Region` are categorical codes, not amounts. A distance-based
/// method reads them as numbers and treats `Region` `3` as three times `Region`
/// `1`. Cluster on the six spending columns, and keep the two codes to check the
/// result.
///
/// The six spending columns have a long right tail. `Fresh` runs from `3` to
/// `112,151` around a mean of `12,000`. Consider a log transform or
/// [`min_max_scale`](crate::preprocessing::min_max_scale) before a
/// distance-based method.
///
/// # Class balance of the two codes
///
/// | Code      | Value          | Clients |
/// |-----------|----------------|---------|
/// | `Channel` | `1` Horeca     | 298     |
/// | `Channel` | `2` Retail     | 142     |
/// | `Region`  | `1` Lisbon     | 77      |
/// | `Region`  | `2` Oporto     | 47      |
/// | `Region`  | `3` other      | 316     |
///
/// The UCI web page marks `Region` as the dataset's target. The published work
/// on this dataset clusters the clients instead, so this loader keeps `Region`
/// as a feature column and exposes no target.
///
/// Missing values: none. No field is empty.
///
/// See more information at <https://archive.ics.uci.edu/dataset/292/wholesale+customers>.
///
/// # Citation
///
/// Cardoso, M. (2013). Wholesale customers \[Dataset\]. UCI Machine Learning
/// Repository. <https://doi.org/10.24432/C5030X>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::WholesaleCustomers;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./wholesale_customers";
///
/// let mut dataset = WholesaleCustomers::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 440);
/// assert_eq!(table.n_columns(), 8);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&WholesaleCustomers::COLUMN_NAMES).unwrap();
/// assert_eq!(features.shape(), &[440, 8]);
///
/// // Reach one spending column by name.
/// let fresh = table.column("Fresh").unwrap().as_numeric().unwrap();
/// assert_eq!(fresh.len(), 440);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("Fresh") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = values[0].ln();
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 440);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 440);
/// ```
#[derive(Debug)]
pub struct WholesaleCustomers {
    dataset: Dataset<Table, DatasetError>,
}

impl WholesaleCustomers {
    /// The column names, in the order the source lists them.
    ///
    /// A column index of `4` names `COLUMN_NAMES[4]`, which is `"Grocery"`.
    ///
    /// # Example
    /// ```
    /// use dataset_ml::WholesaleCustomers;
    ///
    /// assert_eq!(WholesaleCustomers::COLUMN_NAMES[0], "Channel");
    /// assert_eq!(WholesaleCustomers::COLUMN_NAMES[7], "Delicassen");
    /// ```
    pub const COLUMN_NAMES: [&'static str; N_FEATURES] = [
        "Channel",
        "Region",
        "Fresh",
        "Milk",
        "Grocery",
        "Frozen",
        "Detergents_Paper",
        "Delicassen",
    ];

    /// Create a new WholesaleCustomers instance without loading data.
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
    /// - `Self` - a `WholesaleCustomers` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        WholesaleCustomers {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Wholesale Customers dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            WHOLESALE_FILENAME,
            WHOLESALE_DATASET_NAME,
            Some(WHOLESALE_SHA256),
            |temp_path| {
                download_to_with_retries(
                    WHOLESALE_DATA_URL,
                    temp_path,
                    Some(WHOLESALE_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(WHOLESALE_FILENAME))
            },
        )?;

        // The source is comma-separated with a header row and CRLF line endings.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new()
            .delimiter(b',')
            .has_headers(true)
            .from_reader(file);

        let mut values: Vec<Vec<f64>> = (0..N_FEATURES)
            .map(|_| Vec::with_capacity(N_SAMPLES))
            .collect();

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(WHOLESALE_DATASET_NAME, e))?;
            let line_num = idx + 2; // +1 for the header, +1 for 1-based lines

            // Skip blank lines, such as a trailing newline at the end of the file.
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_FEATURES {
                return Err(DatasetError::invalid_column_count(
                    WHOLESALE_DATASET_NAME,
                    N_FEATURES,
                    record.len(),
                    line_num,
                ));
            }

            for (col, name) in Self::COLUMN_NAMES.iter().enumerate() {
                let value: f64 = record[col].trim().parse().map_err(|e| {
                    DatasetError::parse_failed(WHOLESALE_DATASET_NAME, name, line_num, e)
                })?;
                values[col].push(value);
            }
        }

        let columns = Self::COLUMN_NAMES
            .iter()
            .copied()
            .zip(values)
            .map(|(name, column)| Column::new(name, ColumnData::Numeric(Array1::from_vec(column))))
            .collect();

        Table::new(WHOLESALE_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 440 samples and 8 columns.
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
    /// Unlike [`WholesaleCustomers::data`], this method never runs the loader. If
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
    /// changes stay in the cache. Later calls to [`WholesaleCustomers::data`] or
    /// [`WholesaleCustomers::get_data`] see them.
    ///
    /// Like [`WholesaleCustomers::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`WholesaleCustomers::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 440 samples and 8 columns.
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
    /// - `Table` - the owned table of 440 samples and 8 columns.
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

impl_ml_dataset!(WholesaleCustomers, "wholesale_customers");
