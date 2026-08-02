//! Titanic survival dataset.
//!
//! The dataset holds passenger records from the Kaggle `Titanic: Machine
//! Learning from Disaster` competition. The task is to predict survival on
//! the RMS Titanic.
//!
//! **Features (11, mixed):**
//! - String features: `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`
//! - Numeric features: `PassengerId`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`
//!
//! **Target:** `Survived` - binary label (`0` = died, `1` = survived)
//!
//! **Samples:** 891
//! **Application:** Binary classification / survival prediction
//!
//! **Source:** Kaggle competition
//! <https://www.kaggle.com/c/titanic/data>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::fs::File;

/// Type alias for Titanic dataset: (string features, numeric features, labels)
type TitanicData = (Array2<String>, Array2<f64>, Array1<f64>);

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
/// `""`. Fields are declared in CSV column order and deserialized
/// **positionally**. The loader disables csv's header handling, so this
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

/// This struct represents the Titanic dataset and loads it lazily.
///
/// Nothing loads until you call a data accessor method. After loading, the
/// data stays cached for later accesses.
///
/// # About Dataset
///
/// On April 15, 1912, during her maiden voyage, the widely considered
/// "unsinkable" RMS Titanic sank after it collided with an iceberg. The ship
/// did not have enough lifeboats for everyone on board. As a result, 1502 of
/// the 2224 passengers and crew died. Luck played some role in survival, but
/// some groups of people were more likely to survive than others.
///
/// # Feature columns
///
/// Features are split across two matrices: a `(891, 5)` string matrix and a
/// `(891, 6)` numeric `f64` matrix (numeric entries are `NaN` when missing in
/// the source).
///
/// String features (`Array2<String>`), by 0-based column:
///
/// | Columns | Attributes | Unit |
/// |---------|------------|------|
/// | `0`     | `Name`     |      |
/// | `1`     | `Sex`      |      |
/// | `2`     | `Ticket`   |      |
/// | `3`     | `Cabin`    |      |
/// | `4`     | `Embarked` |      |
///
/// Numeric features (`Array2<f64>`), by 0-based column:
///
/// | Columns | Attributes    | Unit  |
/// |---------|---------------|-------|
/// | `0`     | `PassengerId` |       |
/// | `1`     | `Pclass`      |       |
/// | `2`     | `Age`         | years |
/// | `3`     | `SibSp`       |       |
/// | `4`     | `Parch`       |       |
/// | `5`     | `Fare`        |       |
///
/// # Labels
///
/// - `Survived` (shape `(891,)`): `0.0` (died), `1.0` (survived), or `NaN` if
///   missing in the source
///
/// Missing values:
/// - The loader parses missing numeric fields as `NaN`.
/// - The loader parses missing string fields as empty strings.
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
/// Every field implements `Send` and `Sync`, so this struct implements them too. It is safe
/// to share across threads.
/// The internal [`Dataset`] makes initialization thread-safe and lazy.
///
/// # Example
/// ```no_run
/// use dataset_ml::titanic::Titanic;
///
/// let download_dir = "./titanic"; // creates the directory if it is missing
///
/// let mut dataset = Titanic::new(download_dir);
/// let (string_features, numeric_features) = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (string_features, numeric_features, labels) = dataset.data().unwrap(); // also gets all data
/// assert_eq!(string_features.shape(), &[891, 5]);
/// assert_eq!(numeric_features.shape(), &[891, 6]);
/// assert_eq!(labels.len(), 891);
///
/// // `get_data()` borrows the cached arrays without reloading. `get_data_mut()`
/// // edits them in place. It needs no clone and no reload, and the change
/// // stays cached. Prefer this method over cloning with `.to_owned()` when
/// // you only need to change values.
/// if let Some((_strings, numerics, labels)) = dataset.get_data_mut() {
///     numerics[[0, 0]] = 1.0;
///     labels[0] = 1.0;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out (no `to_owned()` clone) and leaves
/// // the instance reusable. The next access reloads from the cached file.
/// let (owned_strings, owned_numerics, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_strings.shape(), &[891, 5]);
/// assert_eq!(owned_numerics.shape(), &[891, 6]);
/// assert_eq!(owned_labels.len(), 891);
///
/// // `into_data()` also returns the owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_strings, owned_numerics, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_strings.shape(), &[891, 5]);
/// assert_eq!(owned_numerics.shape(), &[891, 6]);
/// assert_eq!(owned_labels.len(), 891);
/// ```
#[derive(Debug)]
pub struct Titanic {
    dataset: Dataset<TitanicData, DatasetError>,
}

impl Titanic {
    /// Create a new Titanic instance without loading data.
    ///
    /// This does not load the dataset. The dataset loads on the first call to a
    /// data accessor method. This is a lightweight operation: it only stores the
    /// storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory used to store the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - `Titanic` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Titanic {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Titanic dataset.
    fn load_data(dir: &str) -> Result<TitanicData, DatasetError> {
        // Prepare the dataset file
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

        // csv deserializes into the struct
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut string_features = Vec::new();
        let mut numeric_features = Vec::new();
        let mut labels = Vec::new();

        for result in rdr.deserialize::<TitanicRecord>().skip(1) {
            let TitanicRecord {
                passenger_id,
                survived,
                pclass,
                name,
                sex,
                age,
                sib_sp,
                parch,
                ticket,
                fare,
                cabin,
                embarked,
            } = result.map_err(|e| DatasetError::csv_read_error(TITANIC_DATASET_NAME, e))?;

            // Missing numeric fields (`None`) become `NaN`.
            // Label: Survived.
            labels.push(survived.unwrap_or(f64::NAN));

            // Numeric features, in column order:
            // PassengerId, Pclass, Age, SibSp, Parch, Fare.
            numeric_features.extend_from_slice(&[
                passenger_id.unwrap_or(f64::NAN),
                pclass.unwrap_or(f64::NAN),
                age.unwrap_or(f64::NAN),
                sib_sp.unwrap_or(f64::NAN),
                parch.unwrap_or(f64::NAN),
                fare.unwrap_or(f64::NAN),
            ]);

            // String features, in column order: Name, Sex, Ticket, Cabin, Embarked.
            string_features.push(name);
            string_features.push(sex);
            string_features.push(ticket);
            string_features.push(cabin);
            string_features.push(embarked);
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(TITANIC_DATASET_NAME));
        }

        // Titanic has a fixed schema of 5 string and 6 numeric features per sample.
        let string_array =
            Array2::from_shape_vec((n_samples, 5), string_features).map_err(|e| {
                DatasetError::array_shape_error(TITANIC_DATASET_NAME, "string_features", e)
            })?;

        let numeric_array =
            Array2::from_shape_vec((n_samples, 6), numeric_features).map_err(|e| {
                DatasetError::array_shape_error(TITANIC_DATASET_NAME, "numeric_features", e)
            })?;

        let labels_array = Array1::from_vec(labels);

        Ok((string_array, numeric_array, labels_array))
    }

    /// Get a reference to both string and numeric feature matrices.
    ///
    /// This method triggers lazy loading on first call. Later calls return the
    /// cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<String>` - Reference to string feature matrix with shape `(891, 5)` containing:
    ///     - `Name`
    ///     - `Sex`
    ///     - `Ticket`
    ///     - `Cabin`
    ///     - `Embarked`
    ///
    ///   (empty string if missing in source)
    ///
    /// - `&Array2<f64>` - Reference to numeric feature matrix with shape `(891, 6)` containing:
    ///     - `PassengerId`
    ///     - `Pclass`
    ///     - `Age`
    ///     - `SibSp`
    ///     - `Parch`
    ///     - `Fare`
    ///
    ///   (`NaN` if missing in source)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (891 samples)
    pub fn features(&self) -> Result<(&Array2<String>, &Array2<f64>), DatasetError> {
        let data = self.dataset.load()?;
        Ok((&data.0, &data.1))
    }

    /// Get a reference to the label vector.
    ///
    /// This method triggers lazy loading on first call. Later calls return the
    /// cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<f64>` - Reference to label vector with shape `(891,)`
    ///   containing `Survived` values (`0.0` or `1.0`, `NaN` if missing in
    ///   source)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (891 samples)
    pub fn labels(&self) -> Result<&Array1<f64>, DatasetError> {
        Ok(&self.dataset.load()?.2)
    }

    /// Get string features, numeric features and labels as references.
    ///
    /// This method triggers lazy loading on first call. Later calls return the
    /// cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&TitanicData` - reference to the cached `(string features, numeric
    ///   features, labels)` tuple: string feature matrix `(891, 5)` (Name, Sex,
    ///   Ticket, Cabin, Embarked), numeric feature matrix `(891, 6)`
    ///   (PassengerId, Pclass, Age, SibSp, Parch, Fare), and label vector
    ///   `(891,)` (Survived).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size does not match the expected dimensions (891 samples)
    pub fn data(&self) -> Result<&TitanicData, DatasetError> {
        self.dataset.load()
    }

    /// Get string features, numeric features and labels as references, without
    /// triggering loading.
    ///
    /// Unlike [`Titanic::data`], which loads the dataset on first call, this never
    /// runs the loader. If the data has not been loaded yet, it returns `None`
    /// instead of downloading and parsing.
    ///
    /// Use this method when you want the data only if it is already cached. This
    /// avoids the download and parse cost when the data is not cached.
    ///
    /// # Returns
    ///
    /// - `Some(&TitanicData)` - reference to the cached `(string features, numeric
    ///   features, labels)` tuple (`(891, 5)`, `(891, 6)`, `(891,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&TitanicData> {
        self.dataset.get()
    }

    /// Get mutable references to string features, numeric features, and labels
    /// for **in-place** editing.
    ///
    /// This lets you change the cached arrays directly (e.g. normalize numeric
    /// features, replace missing values). It needs no `to_owned()` clone, and
    /// the arrays stay in the cache. The changes persist, so later calls to
    /// [`Titanic::features`], [`Titanic::data`], or [`Titanic::get_data`] see
    /// them.
    ///
    /// Like [`Titanic::get_data`], this does **not** trigger loading. It returns
    /// `None` if the dataset has not been loaded. If you need to make sure the
    /// data is present, call a loading accessor first (e.g. [`Titanic::data`]).
    ///
    /// # Returns
    ///
    /// - `Some(&mut TitanicData)` - mutable reference to the cached `(string
    ///   features, numeric features, labels)` tuple (`(891, 5)`, `(891, 6)`,
    ///   `(891,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut TitanicData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** string features, numeric features,
    /// and labels.
    ///
    /// Unlike [`Titanic::data`], which borrows the cached data, this moves it out
    /// and returns owned arrays directly. It needs no `to_owned()` clone. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`Titanic::take_data`] instead. It takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array2<f64>, Array1<f64>)` - owned string feature matrix
    ///   `(891, 5)`, owned numeric feature matrix `(891, 6)`, and owned label vector
    ///   `(891,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn into_data(self) -> Result<TitanicData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** string features, numeric features, and labels out of the
    /// dataset. The instance stays reusable.
    ///
    /// Like [`Titanic::into_data`], this returns owned arrays with no `to_owned()`
    /// clone. But instead of consuming the instance, it takes `&mut self` and moves
    /// the cached data out. This resets the instance to its unloaded state. The
    /// next accessor call (e.g. [`Titanic::features`] or [`Titanic::data`]) loads
    /// the dataset again.
    ///
    /// If you are done with the instance, use [`Titanic::into_data`] instead.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array2<f64>, Array1<f64>)` - owned string feature matrix
    ///   `(891, 5)`, owned numeric feature matrix `(891, 6)`, and owned label vector
    ///   `(891,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// dimension mismatch).
    pub fn take_data(&mut self) -> Result<TitanicData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(Titanic, TitanicData, "titanic");
