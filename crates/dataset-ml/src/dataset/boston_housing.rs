//! Boston Housing dataset.
//!
//! This dataset holds housing data for Boston suburbs, drawn from U.S.
//! Census-derived information. Researchers commonly use it as a regression
//! benchmark.
//!
//! **Columns (14):**
//!
//! | Name      | Type      | Description                                                             |
//! |-----------|-----------|---------------------------------------------------------------------------|
//! | `CRIM`    | `Numeric` | per capita crime rate by town                                           |
//! | `ZN`      | `Numeric` | proportion of residential land zoned for lots over 25,000 sq.ft.        |
//! | `INDUS`   | `Numeric` | proportion of non-retail business acres per town                        |
//! | `CHAS`    | `Numeric` | Charles River dummy variable, 1 if the tract bounds the river, else 0   |
//! | `NOX`     | `Numeric` | nitric oxides concentration in parts per 10 million                     |
//! | `RM`      | `Numeric` | average number of rooms per dwelling                                    |
//! | `AGE`     | `Numeric` | proportion of owner-occupied units built before 1940                    |
//! | `DIS`     | `Numeric` | weighted distances to five Boston employment centers                    |
//! | `RAD`     | `Numeric` | index of accessibility to radial highways                               |
//! | `TAX`     | `Numeric` | full-value property-tax rate per $10,000                                |
//! | `PTRATIO` | `Numeric` | pupil-teacher ratio by town                                             |
//! | `B`       | `Numeric` | 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town |
//! | `LSTAT`   | `Numeric` | percentage of lower-status population                                   |
//! | `MEDV`    | `Numeric` | median value of owner-occupied homes in $1000s                          |
//!
//! The source designates the first 13 columns as the inputs
//! ([`BostonHousing::FEATURE_NAMES`](crate::BostonHousing::FEATURE_NAMES)) and `MEDV` as the label
//! ([`BostonHousing::TARGET`](crate::BostonHousing::TARGET)).
//!
//! **Samples:** 506
//! **Application:** Regression / housing value prediction
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5C88K>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
use serde::Deserialize;
use std::fs::File;

/// The URL for the Boston Housing dataset.
const BOSTON_HOUSING_DATA_URL: &str =
    "https://github.com/selva86/datasets/raw/master/BostonHousing.csv";

/// The name of the file inside the extracted folder
const BOSTON_HOUSING_FILENAME: &str = "BostonHousing.csv";

/// The SHA256 hash of the dataset file
const BOSTON_HOUSING_SHA256: &str =
    "ab16ba38fbbbbcc69fe930aab1293104f1442c8279c130d9eba03dd864bef675";

/// The name of the dataset
const BOSTON_HOUSING_DATASET_NAME: &str = "boston_housing";

/// Number of samples.
const N_SAMPLES: usize = 506;

/// One CSV record of the Boston Housing dataset: 13 `f64` feature columns
/// followed by the `medv` target.
///
/// The struct declares its fields in CSV column order. The loader disables
/// csv's header handling, so csv deserializes the fields **positionally**.
/// This design makes the struct independent of the exact header spelling.
#[derive(Deserialize)]
struct BostonHousingRecord {
    crim: f64,
    zn: f64,
    indus: f64,
    chas: f64,
    nox: f64,
    rm: f64,
    age: f64,
    dis: f64,
    rad: f64,
    tax: f64,
    ptratio: f64,
    b: f64,
    lstat: f64,
    medv: f64,
}

/// A struct that represents the Boston Housing dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The U.S. Census Service collected the information behind the Boston Housing
/// dataset. It describes housing in the Boston, MA area.
///
/// # Columns
///
/// | Name      | Type      | Description                                                             |
/// |-----------|-----------|---------------------------------------------------------------------------|
/// | `CRIM`    | `Numeric` | per capita crime rate by town                                           |
/// | `ZN`      | `Numeric` | proportion of residential land zoned for lots over 25,000 sq.ft.        |
/// | `INDUS`   | `Numeric` | proportion of non-retail business acres per town                        |
/// | `CHAS`    | `Numeric` | Charles River dummy variable, 1 if the tract bounds the river, else 0   |
/// | `NOX`     | `Numeric` | nitric oxides concentration in parts per 10 million                     |
/// | `RM`      | `Numeric` | average number of rooms per dwelling                                    |
/// | `AGE`     | `Numeric` | proportion of owner-occupied units built before 1940                    |
/// | `DIS`     | `Numeric` | weighted distances to five Boston employment centers                    |
/// | `RAD`     | `Numeric` | index of accessibility to radial highways                               |
/// | `TAX`     | `Numeric` | full-value property-tax rate per $10,000                                |
/// | `PTRATIO` | `Numeric` | pupil-teacher ratio by town                                             |
/// | `B`       | `Numeric` | 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town |
/// | `LSTAT`   | `Numeric` | percentage of lower-status population                                   |
/// | `MEDV`    | `Numeric` | median value of owner-occupied homes in $1000s                          |
///
/// The source designates the first 13 columns as the inputs
/// ([`BostonHousing::FEATURE_NAMES`]) and `MEDV` as the label
/// ([`BostonHousing::TARGET`]).
///
/// # Citation
///
/// D. Harrison and D. L. Rubinfeld, "Hedonic prices and the demand for clean
/// air," Journal of Environmental Economics and Management, vol. 5, no. 1,
/// pp. 81–102, 1978. Dataset via UCI Machine Learning Repository,
/// <https://doi.org/10.24432/C5C88K>.
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::BostonHousing;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./boston_housing";
///
/// let mut dataset = BostonHousing::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 506);
/// assert_eq!(table.n_columns(), 14);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&BostonHousing::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[506, 13]);
///
/// // Reach one column by name.
/// let medv = table.column(BostonHousing::TARGET).unwrap().as_numeric().unwrap();
/// assert_eq!(medv.len(), 506);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("CRIM") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 0.1;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 506);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 506);
/// ```
#[derive(Debug)]
pub struct BostonHousing {
    dataset: Dataset<Table, DatasetError>,
}

impl BostonHousing {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; 13] = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B",
        "LSTAT",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "MEDV";

    /// Create a new BostonHousing instance without loading data.
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
    /// - `Self` - a `BostonHousing` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        BostonHousing {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Boston Housing dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            BOSTON_HOUSING_FILENAME,
            BOSTON_HOUSING_DATASET_NAME,
            Some(BOSTON_HOUSING_SHA256),
            |temp_path| {
                download_to_with_retries(
                    BOSTON_HOUSING_DATA_URL,
                    temp_path,
                    None,
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(BOSTON_HOUSING_FILENAME))
            },
        )?;

        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut crim = Vec::with_capacity(N_SAMPLES);
        let mut zn = Vec::with_capacity(N_SAMPLES);
        let mut indus = Vec::with_capacity(N_SAMPLES);
        let mut chas = Vec::with_capacity(N_SAMPLES);
        let mut nox = Vec::with_capacity(N_SAMPLES);
        let mut rm = Vec::with_capacity(N_SAMPLES);
        let mut age = Vec::with_capacity(N_SAMPLES);
        let mut dis = Vec::with_capacity(N_SAMPLES);
        let mut rad = Vec::with_capacity(N_SAMPLES);
        let mut tax = Vec::with_capacity(N_SAMPLES);
        let mut ptratio = Vec::with_capacity(N_SAMPLES);
        let mut b = Vec::with_capacity(N_SAMPLES);
        let mut lstat = Vec::with_capacity(N_SAMPLES);
        let mut medv = Vec::with_capacity(N_SAMPLES);

        for result in rdr.deserialize::<BostonHousingRecord>().skip(1) {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(BOSTON_HOUSING_DATASET_NAME, e))?;

            crim.push(record.crim);
            zn.push(record.zn);
            indus.push(record.indus);
            chas.push(record.chas);
            nox.push(record.nox);
            rm.push(record.rm);
            age.push(record.age);
            dis.push(record.dis);
            rad.push(record.rad);
            tax.push(record.tax);
            ptratio.push(record.ptratio);
            b.push(record.b);
            lstat.push(record.lstat);
            medv.push(record.medv);
        }

        Table::new(
            BOSTON_HOUSING_DATASET_NAME,
            vec![
                Column::new(
                    Self::FEATURE_NAMES[0],
                    ColumnData::Numeric(Array1::from_vec(crim)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[1],
                    ColumnData::Numeric(Array1::from_vec(zn)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[2],
                    ColumnData::Numeric(Array1::from_vec(indus)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[3],
                    ColumnData::Numeric(Array1::from_vec(chas)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[4],
                    ColumnData::Numeric(Array1::from_vec(nox)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[5],
                    ColumnData::Numeric(Array1::from_vec(rm)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[6],
                    ColumnData::Numeric(Array1::from_vec(age)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[7],
                    ColumnData::Numeric(Array1::from_vec(dis)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[8],
                    ColumnData::Numeric(Array1::from_vec(rad)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[9],
                    ColumnData::Numeric(Array1::from_vec(tax)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[10],
                    ColumnData::Numeric(Array1::from_vec(ptratio)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[11],
                    ColumnData::Numeric(Array1::from_vec(b)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[12],
                    ColumnData::Numeric(Array1::from_vec(lstat)),
                ),
                Column::new(Self::TARGET, ColumnData::Numeric(Array1::from_vec(medv))),
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
    /// - `&Table` - reference to the cached table of 506 samples and 14 columns.
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
    /// Unlike [`BostonHousing::data`], this method never runs the loader. If the
    /// data has not loaded yet, it returns `None` instead of downloading and
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
    /// changes stay in the cache. Later calls to [`BostonHousing::data`] or
    /// [`BostonHousing::get_data`] see them.
    ///
    /// Like [`BostonHousing::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`BostonHousing::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 506 samples and 14 columns.
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
    /// - `Table` - the owned table of 506 samples and 14 columns.
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

impl_ml_dataset!(BostonHousing, "boston_housing");
