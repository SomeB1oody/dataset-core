//! Titanic survival dataset.
//!
//! The dataset holds passenger records from the Kaggle `Titanic: Machine
//! Learning from Disaster` competition. The task is to predict survival on
//! the RMS Titanic.
//!
//! **Columns (12):**
//!
//! | Name          | Type      | Description                           |
//! |---------------|-----------|----------------------------------------|
//! | `PassengerId` | `Numeric` | passenger record number               |
//! | `Survived`    | `Numeric` | `0.0` for died, `1.0` for survived    |
//! | `Pclass`      | `Numeric` | ticket class: `1`, `2`, or `3`        |
//! | `Name`        | `String`  | passenger name                        |
//! | `Sex`         | `String`  | `male` or `female`                    |
//! | `Age`         | `Numeric` | age in years                          |
//! | `SibSp`       | `Numeric` | count of siblings and spouses aboard  |
//! | `Parch`       | `Numeric` | count of parents and children aboard  |
//! | `Ticket`      | `String`  | ticket number                         |
//! | `Fare`        | `Numeric` | passenger fare                        |
//! | `Cabin`       | `String`  | cabin number                          |
//! | `Embarked`    | `String`  | port of embarkation: `C`, `Q`, or `S` |
//!
//! The source designates ten columns as the inputs
//! ([`Titanic::FEATURE_NAMES`](crate::Titanic::FEATURE_NAMES)) and `Survived` as the label
//! ([`Titanic::TARGET`](crate::Titanic::TARGET)).
//!
//! **Samples:** 891
//! **Application:** Binary classification / survival prediction
//!
//! **Missing values:** a missing numeric field becomes `NaN`. A missing text
//! field becomes an empty string. The source omits `Age`, `Cabin`, and
//! `Embarked` values.
//!
//! **Source:** Kaggle competition
//! <https://www.kaggle.com/c/titanic/data>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
use serde::Deserialize;
use std::fs::File;

/// The URL for the Titanic dataset.
const TITANIC_DATA_URL: &str =
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv";

/// The name of the Titanic dataset file.
const TITANIC_FILENAME: &str = "titanic.csv";

/// The SHA256 hash of the Titanic dataset file.
const TITANIC_SHA256: &str = "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7";

/// The name of the dataset
const TITANIC_DATASET_NAME: &str = "titanic";

/// One CSV record of the Titanic dataset, with fields in source column order.
///
/// Numeric columns are `Option<f64>` so that empty fields deserialize to `None`
/// (later mapped to `NaN`). Text columns are `String`, and empty fields become
/// `""`. This struct declares fields in CSV column order. It deserializes
/// them **positionally**. The loader disables csv's header handling, so this
/// struct does not depend on the exact header spelling.
#[derive(Deserialize)]
struct TitanicRecord {
    passenger_id: Option<f64>,
    survived: Option<f64>,
    pclass: Option<f64>,
    name: String,
    sex: String,
    age: Option<f64>,
    sib_sp: Option<f64>,
    parch: Option<f64>,
    ticket: String,
    fare: Option<f64>,
    cabin: String,
    embarked: String,
}

/// A struct that represents the Titanic dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// On April 15, 1912, during her maiden voyage, the widely considered
/// "unsinkable" RMS Titanic sank after it collided with an iceberg. The ship
/// did not have enough lifeboats for everyone on board. As a result, 1502 of
/// the 2224 passengers and crew died. Luck played some role in survival, but
/// some groups of people were more likely to survive than others.
///
/// # Columns
///
/// | Name          | Type      | Description                           |
/// |---------------|-----------|----------------------------------------|
/// | `PassengerId` | `Numeric` | passenger record number               |
/// | `Survived`    | `Numeric` | `0.0` for died, `1.0` for survived    |
/// | `Pclass`      | `Numeric` | ticket class: `1`, `2`, or `3`        |
/// | `Name`        | `String`  | passenger name                        |
/// | `Sex`         | `String`  | `male` or `female`                    |
/// | `Age`         | `Numeric` | age in years                          |
/// | `SibSp`       | `Numeric` | count of siblings and spouses aboard  |
/// | `Parch`       | `Numeric` | count of parents and children aboard  |
/// | `Ticket`      | `String`  | ticket number                         |
/// | `Fare`        | `Numeric` | passenger fare                        |
/// | `Cabin`       | `String`  | cabin number                          |
/// | `Embarked`    | `String`  | port of embarkation: `C`, `Q`, or `S` |
///
/// The source designates ten columns as the inputs
/// ([`Titanic::FEATURE_NAMES`]) and `Survived` as the label
/// ([`Titanic::TARGET`]).
///
/// Missing values: a missing numeric field becomes `NaN`. A missing text field
/// becomes an empty string. The source omits `Age`, `Cabin`, and `Embarked`
/// values.
///
/// See more information at <https://www.kaggle.com/c/titanic/data>.
///
/// # Citation
///
/// Kaggle, "Titanic: Machine Learning from Disaster." \[Online\].
/// Available: <https://www.kaggle.com/c/titanic>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Titanic;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./titanic";
///
/// let mut dataset = Titanic::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 891);
/// assert_eq!(table.n_columns(), 12);
///
/// // Name the numeric feature columns you want in the matrix. `FEATURE_NAMES`
/// // also holds text columns, so a numeric matrix call must name a subset.
/// let numeric_features = table
///     .numeric_matrix(&["Pclass", "Age", "SibSp", "Parch", "Fare"])
///     .unwrap();
/// assert_eq!(numeric_features.shape(), &[891, 5]);
///
/// // Reach a text column by name.
/// let sex = table.column("Sex").unwrap().as_string().unwrap();
/// assert_eq!(sex.len(), 891);
///
/// // Reach the label column by name.
/// let survived = table.column(Titanic::TARGET).unwrap().as_numeric().unwrap();
/// assert_eq!(survived.len(), 891);
///
/// // Reach one column by name.
/// let age = table.column("Age").unwrap().as_numeric().unwrap();
/// assert_eq!(age.len(), 891);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("Age") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 30.0;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 891);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 891);
/// ```
#[derive(Debug)]
pub struct Titanic {
    dataset: Dataset<Table, DatasetError>,
}

impl Titanic {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; 10] = [
        "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "Survived";

    /// Create a new Titanic instance without loading data.
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
    /// - `Self` - a `Titanic` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Titanic {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Titanic dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            TITANIC_FILENAME,
            TITANIC_DATASET_NAME,
            Some(TITANIC_SHA256),
            |temp_path| {
                download_to_with_retries(TITANIC_DATA_URL, temp_path, None, DOWNLOAD_RETRIES)?;
                Ok(temp_path.join(TITANIC_FILENAME))
            },
        )?;

        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut passenger_id = Vec::new();
        let mut survived = Vec::new();
        let mut pclass = Vec::new();
        let mut name = Vec::new();
        let mut sex = Vec::new();
        let mut age = Vec::new();
        let mut sib_sp = Vec::new();
        let mut parch = Vec::new();
        let mut ticket = Vec::new();
        let mut fare = Vec::new();
        let mut cabin = Vec::new();
        let mut embarked = Vec::new();

        for result in rdr.deserialize::<TitanicRecord>().skip(1) {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(TITANIC_DATASET_NAME, e))?;

            // Missing numeric fields (`None`) become `NaN`.
            passenger_id.push(record.passenger_id.unwrap_or(f64::NAN));
            survived.push(record.survived.unwrap_or(f64::NAN));
            pclass.push(record.pclass.unwrap_or(f64::NAN));
            name.push(record.name);
            sex.push(record.sex);
            age.push(record.age.unwrap_or(f64::NAN));
            sib_sp.push(record.sib_sp.unwrap_or(f64::NAN));
            parch.push(record.parch.unwrap_or(f64::NAN));
            ticket.push(record.ticket);
            fare.push(record.fare.unwrap_or(f64::NAN));
            cabin.push(record.cabin);
            embarked.push(record.embarked);
        }

        Table::new(
            TITANIC_DATASET_NAME,
            vec![
                Column::new(
                    "PassengerId",
                    ColumnData::Numeric(Array1::from_vec(passenger_id)),
                ),
                Column::new(
                    Self::TARGET,
                    ColumnData::Numeric(Array1::from_vec(survived)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[0],
                    ColumnData::Numeric(Array1::from_vec(pclass)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[1],
                    ColumnData::String(Array1::from_vec(name)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[2],
                    ColumnData::String(Array1::from_vec(sex)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[3],
                    ColumnData::Numeric(Array1::from_vec(age)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[4],
                    ColumnData::Numeric(Array1::from_vec(sib_sp)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[5],
                    ColumnData::Numeric(Array1::from_vec(parch)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[6],
                    ColumnData::String(Array1::from_vec(ticket)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[7],
                    ColumnData::Numeric(Array1::from_vec(fare)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[8],
                    ColumnData::String(Array1::from_vec(cabin)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[9],
                    ColumnData::String(Array1::from_vec(embarked)),
                ),
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
    /// - `&Table` - reference to the cached table of 891 samples and 12 columns.
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
    /// Unlike [`Titanic::data`], this method never runs the loader. If the data
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
    /// changes stay in the cache. Later calls to [`Titanic::data`] or
    /// [`Titanic::get_data`] see them.
    ///
    /// Like [`Titanic::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`Titanic::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 891 samples and 12 columns.
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
    /// - `Table` - the owned table of 891 samples and 12 columns.
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

impl_ml_dataset!(Titanic, "titanic");
