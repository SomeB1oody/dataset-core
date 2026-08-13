//! Mushroom dataset.
//!
//! The dataset holds records from *The Audubon Society Field Guide to North
//! American Mushrooms* (1981). The records describe 23 species of gilled
//! mushrooms in the Agaricus and Lepiota family. The task is to predict whether
//! a mushroom is edible or poisonous. Every value is a single-letter code.
//!
//! **Columns (23):**
//!
//! | Name                        | Type      | Description                          |
//! |-----------------------------|-----------|----------------------------------------|
//! | `class`                     | `String`  | `e` = edible, `p` = poisonous        |
//! | `cap-shape`                 | `String`  | cap shape code                       |
//! | `cap-surface`               | `String`  | cap surface code                     |
//! | `cap-color`                 | `String`  | cap color code                       |
//! | `bruises`                   | `String`  | `t` = bruises, `f` = no bruises      |
//! | `odor`                      | `String`  | odor code                            |
//! | `gill-attachment`           | `String`  | gill attachment code                 |
//! | `gill-spacing`              | `String`  | gill spacing code                    |
//! | `gill-size`                 | `String`  | `b` = broad, `n` = narrow            |
//! | `gill-color`                | `String`  | gill color code                      |
//! | `stalk-shape`               | `String`  | stalk shape code                     |
//! | `stalk-root`                | `String`  | stalk root code, empty when missing  |
//! | `stalk-surface-above-ring`  | `String`  | stalk surface above the ring         |
//! | `stalk-surface-below-ring`  | `String`  | stalk surface below the ring         |
//! | `stalk-color-above-ring`    | `String`  | stalk color above the ring           |
//! | `stalk-color-below-ring`    | `String`  | stalk color below the ring           |
//! | `veil-type`                 | `String`  | veil type code                       |
//! | `veil-color`                | `String`  | veil color code                      |
//! | `ring-number`               | `String`  | ring number code                     |
//! | `ring-type`                 | `String`  | ring type code                       |
//! | `spore-print-color`         | `String`  | spore print color code               |
//! | `population`                | `String`  | population code                      |
//! | `habitat`                   | `String`  | habitat code                         |
//!
//! The source designates the 22 attributes as the inputs
//! ([`Mushroom::FEATURE_NAMES`](crate::Mushroom::FEATURE_NAMES)) and `class` as the label
//! ([`Mushroom::TARGET`](crate::Mushroom::TARGET)).
//!
//! **Samples:** 8,124
//! **Application:** Binary classification / edibility prediction
//!
//! **Missing values:** the source marks a missing value with `?`. Only
//! `stalk-root` holds this token, in 2,480 samples. The loader stores an empty
//! string for it.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://archive.ics.uci.edu/dataset/73/mushroom>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
use std::fs::File;

/// The URL for the Mushroom dataset (the `agaricus-lepiota.data` file).
const MUSHROOM_DATA_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data";

/// The name of the cached Mushroom dataset file.
const MUSHROOM_FILENAME: &str = "mushroom.csv";

/// The SHA256 hash of the cached Mushroom dataset file (`agaricus-lepiota.data`'s bytes).
const MUSHROOM_SHA256: &str = "e65d082030501a3ebcbcd7c9f7c71aa9d28fdfff463bf4cf4716a3fe13ac360e";

/// The name of the dataset.
const MUSHROOM_DATASET_NAME: &str = "mushroom";

/// Number of samples.
const N_SAMPLES: usize = 8_124;

/// Number of categorical features.
const N_FEATURES: usize = 22;

/// Number of columns per record (1 label + 22 features).
const N_COLUMNS: usize = 23;

/// Source column index of the label (`class`). The label is the **first** column.
const LABEL_COLUMN: usize = 0;

/// Categorical feature columns, as `(source column index, name)`, in output order.
/// All 22 features follow the leading `class` label column.
const FEATURE_COLUMNS: [(usize, &str); N_FEATURES] = [
    (1, "cap-shape"),
    (2, "cap-surface"),
    (3, "cap-color"),
    (4, "bruises"),
    (5, "odor"),
    (6, "gill-attachment"),
    (7, "gill-spacing"),
    (8, "gill-size"),
    (9, "gill-color"),
    (10, "stalk-shape"),
    (11, "stalk-root"),
    (12, "stalk-surface-above-ring"),
    (13, "stalk-surface-below-ring"),
    (14, "stalk-color-above-ring"),
    (15, "stalk-color-below-ring"),
    (16, "veil-type"),
    (17, "veil-color"),
    (18, "ring-number"),
    (19, "ring-type"),
    (20, "spore-print-color"),
    (21, "population"),
    (22, "habitat"),
];

/// The token marking a missing categorical value in the source (only in `stalk-root`).
const MISSING_TOKEN: &str = "?";

/// A struct that represents the Mushroom dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The Mushroom dataset describes hypothetical samples that correspond to 23
/// species of gilled mushrooms in the Agaricus and Lepiota family. The records
/// come from *The Audubon Society Field Guide to North American Mushrooms*
/// (1981). The guide labels each species edible or poisonous. The poisonous label
/// also covers species of unknown edibility and species not recommended for
/// eating. The classification task is to predict edibility from 22 categorical
/// attributes. No simple rule determines the edibility of a mushroom, and this
/// makes the dataset a difficult classification problem.
///
/// # Columns
///
/// | Name                        | Type      | Description                          |
/// |-----------------------------|-----------|----------------------------------------|
/// | `class`                     | `String`  | `e` = edible, `p` = poisonous        |
/// | `cap-shape`                 | `String`  | cap shape code                       |
/// | `cap-surface`               | `String`  | cap surface code                     |
/// | `cap-color`                 | `String`  | cap color code                       |
/// | `bruises`                   | `String`  | `t` = bruises, `f` = no bruises      |
/// | `odor`                      | `String`  | odor code                            |
/// | `gill-attachment`           | `String`  | gill attachment code                 |
/// | `gill-spacing`              | `String`  | gill spacing code                    |
/// | `gill-size`                 | `String`  | `b` = broad, `n` = narrow            |
/// | `gill-color`                | `String`  | gill color code                      |
/// | `stalk-shape`               | `String`  | stalk shape code                     |
/// | `stalk-root`                | `String`  | stalk root code, empty when missing  |
/// | `stalk-surface-above-ring`  | `String`  | stalk surface above the ring         |
/// | `stalk-surface-below-ring`  | `String`  | stalk surface below the ring         |
/// | `stalk-color-above-ring`    | `String`  | stalk color above the ring           |
/// | `stalk-color-below-ring`    | `String`  | stalk color below the ring           |
/// | `veil-type`                 | `String`  | veil type code                       |
/// | `veil-color`                | `String`  | veil color code                      |
/// | `ring-number`               | `String`  | ring number code                     |
/// | `ring-type`                 | `String`  | ring type code                       |
/// | `spore-print-color`         | `String`  | spore print color code               |
/// | `population`                | `String`  | population code                      |
/// | `habitat`                   | `String`  | habitat code                         |
///
/// Every value is a single-letter code. The source designates the 22
/// attributes as the inputs ([`Mushroom::FEATURE_NAMES`]) and `class` as the
/// label ([`Mushroom::TARGET`]).
///
/// Missing values:
/// - The source marks a missing value with `?`. Only `stalk-root` holds this
///   token, in 2,480 samples. The loader stores an empty string `""` for it.
///
/// See more information at <https://archive.ics.uci.edu/dataset/73/mushroom>.
///
/// # Citation
///
/// Mushroom (1987). UCI Machine Learning Repository.
/// <https://doi.org/10.24432/C5959T>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Mushroom;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./mushroom";
///
/// let mut dataset = Mushroom::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 8124);
/// assert_eq!(table.n_columns(), 23);
///
/// // Every feature is a string, so reach each one by name.
/// let cap_shape = table.column("cap-shape").unwrap().as_string().unwrap();
/// assert_eq!(cap_shape.len(), 8124);
///
/// // Reach the label column by name.
/// let class = table.column(Mushroom::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(class.len(), 8124);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("cap-shape") {
///         if let dataset_ml::ColumnData::String(values) = column.data_mut() {
///             values[0] = "x".to_string();
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 8124);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 8124);
/// ```
#[derive(Debug)]
pub struct Mushroom {
    dataset: Dataset<Table, DatasetError>,
}

impl Mushroom {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "class";

    /// Create a new Mushroom instance without loading data.
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
    /// - `Self` - a `Mushroom` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Mushroom {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Mushroom dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        // The source file is `agaricus-lepiota.data`. The code caches it as
        // `mushroom.csv`.
        let file_path = acquire_dataset(
            dir,
            MUSHROOM_FILENAME,
            MUSHROOM_DATASET_NAME,
            Some(MUSHROOM_SHA256),
            |temp_path| {
                download_to_with_retries(
                    MUSHROOM_DATA_URL,
                    temp_path,
                    Some(MUSHROOM_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                Ok(temp_path.join(MUSHROOM_FILENAME))
            },
        )?;

        // The source is plain comma-separated with no header and single-letter codes.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features: Vec<Vec<String>> = FEATURE_COLUMNS
            .iter()
            .map(|_| Vec::with_capacity(N_SAMPLES))
            .collect();
        let mut labels: Vec<String> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(MUSHROOM_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            // The source file can end with a trailing newline. The check below
            // skips the resulting blank line.
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    MUSHROOM_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // Categorical features, mapping the `?` missing token to an empty string.
            for (values, &(col, _name)) in features.iter_mut().zip(FEATURE_COLUMNS.iter()) {
                let value = &record[col];
                if value == MISSING_TOKEN {
                    values.push(String::new());
                } else {
                    values.push(value.to_string());
                }
            }

            // Label, kept verbatim (`e` or `p`).
            let label = &record[LABEL_COLUMN];
            if label.is_empty() {
                return Err(DatasetError::invalid_value(
                    MUSHROOM_DATASET_NAME,
                    "class",
                    label,
                    line_num,
                ));
            }
            labels.push(label.to_string());
        }

        // The columns follow the source order: `class` first, then the 22 features.
        let mut columns = Vec::with_capacity(N_COLUMNS);
        columns.push(Column::new(
            Self::TARGET,
            ColumnData::String(Array1::from_vec(labels)),
        ));
        for (values, &(_col, name)) in features.into_iter().zip(FEATURE_COLUMNS.iter()) {
            columns.push(Column::new(
                name,
                ColumnData::String(Array1::from_vec(values)),
            ));
        }

        Table::new(MUSHROOM_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 8,124 samples and 23
    ///   columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, an empty label)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`Mushroom::data`], this method never runs the loader. If the data
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
    /// changes stay in the cache. Later calls to [`Mushroom::data`] or
    /// [`Mushroom::get_data`] see them.
    ///
    /// Like [`Mushroom::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`Mushroom::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 8,124 samples and 23 columns.
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
    /// - `Table` - the owned table of 8,124 samples and 23 columns.
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

impl_ml_dataset!(Mushroom, "mushroom");
