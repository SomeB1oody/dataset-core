//! Ionosphere dataset.
//!
//! A radar system in Goose Bay, Labrador, collected these returns. The system
//! aimed at free electrons in the ionosphere. "Good" (`g`) radar returns show
//! evidence of some type of structure in the ionosphere. "Bad" (`b`) returns
//! pass through the ionosphere instead. The task is to predict the quality of a
//! return from 34 continuous features.
//!
//! The system processed each received signal with an autocorrelation function.
//! It used 17 pulse numbers. Each pulse gives two attributes: the real part and
//! the imaginary part of the complex electromagnetic signal. The source
//! normalizes the values to about `-1..=1`.
//!
//! **Columns (35):**
//!
//! | Name | Type | Description |
//! |------|------|-------------|
//! | `attribute_1` | `Numeric` | pulse 1, real part (`0` or `1`) |
//! | `attribute_2` | `Numeric` | pulse 1, imaginary part (constant `0` here) |
//! | `attribute_3` | `Numeric` | pulse 2, real part |
//! | `attribute_4` | `Numeric` | pulse 2, imaginary part |
//! | `attribute_5` | `Numeric` | pulse 3, real part |
//! | `attribute_6` | `Numeric` | pulse 3, imaginary part |
//! | `attribute_7` | `Numeric` | pulse 4, real part |
//! | `attribute_8` | `Numeric` | pulse 4, imaginary part |
//! | `attribute_9` | `Numeric` | pulse 5, real part |
//! | `attribute_10` | `Numeric` | pulse 5, imaginary part |
//! | `attribute_11` | `Numeric` | pulse 6, real part |
//! | `attribute_12` | `Numeric` | pulse 6, imaginary part |
//! | `attribute_13` | `Numeric` | pulse 7, real part |
//! | `attribute_14` | `Numeric` | pulse 7, imaginary part |
//! | `attribute_15` | `Numeric` | pulse 8, real part |
//! | `attribute_16` | `Numeric` | pulse 8, imaginary part |
//! | `attribute_17` | `Numeric` | pulse 9, real part |
//! | `attribute_18` | `Numeric` | pulse 9, imaginary part |
//! | `attribute_19` | `Numeric` | pulse 10, real part |
//! | `attribute_20` | `Numeric` | pulse 10, imaginary part |
//! | `attribute_21` | `Numeric` | pulse 11, real part |
//! | `attribute_22` | `Numeric` | pulse 11, imaginary part |
//! | `attribute_23` | `Numeric` | pulse 12, real part |
//! | `attribute_24` | `Numeric` | pulse 12, imaginary part |
//! | `attribute_25` | `Numeric` | pulse 13, real part |
//! | `attribute_26` | `Numeric` | pulse 13, imaginary part |
//! | `attribute_27` | `Numeric` | pulse 14, real part |
//! | `attribute_28` | `Numeric` | pulse 14, imaginary part |
//! | `attribute_29` | `Numeric` | pulse 15, real part |
//! | `attribute_30` | `Numeric` | pulse 15, imaginary part |
//! | `attribute_31` | `Numeric` | pulse 16, real part |
//! | `attribute_32` | `Numeric` | pulse 16, imaginary part |
//! | `attribute_33` | `Numeric` | pulse 17, real part |
//! | `attribute_34` | `Numeric` | pulse 17, imaginary part |
//! | `class` | `String` | `good` or `bad` |
//!
//! The source designates the 34 attributes as the inputs
//! ([`Ionosphere::FEATURE_NAMES`](crate::Ionosphere::FEATURE_NAMES)) and `class` as the label
//! ([`Ionosphere::TARGET`](crate::Ionosphere::TARGET)).
//!
//! The first two attributes are degenerate in this collection. `attribute_1`
//! holds `0` or `1`. `attribute_2` is always `0`. The loader keeps both
//! verbatim, so the schema matches the source exactly.
//!
//! **Samples:** 351 total (225 good, 126 bad)
//! **Application:** Binary classification / radar return quality
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5W01B>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
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

/// A struct that represents the Ionosphere dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// A system in Goose Bay, Labrador, collected this radar data. The system used
/// a phased array of 16 high-frequency antennas with a total transmitted power
/// of about 6.4 kilowatts. Its targets were free electrons in the ionosphere.
/// "Good" radar returns show evidence of some type of structure in the
/// ionosphere. "Bad" returns do not show this evidence. Their signals pass
/// through the ionosphere instead.
///
/// The system processed received signals with an autocorrelation function.
/// The function took the pulse time and the pulse number as arguments. The
/// Goose Bay system used 17 pulse numbers. Each pulse has two attributes, the
/// real and imaginary parts of the complex electromagnetic signal, for a
/// total of 34 continuous features.
///
/// # Columns
///
/// | Name | Type | Description |
/// |------|------|-------------|
/// | `attribute_1` | `Numeric` | pulse 1, real part (`0` or `1`) |
/// | `attribute_2` | `Numeric` | pulse 1, imaginary part (constant `0` here) |
/// | `attribute_3` | `Numeric` | pulse 2, real part |
/// | `attribute_4` | `Numeric` | pulse 2, imaginary part |
/// | `attribute_5` | `Numeric` | pulse 3, real part |
/// | `attribute_6` | `Numeric` | pulse 3, imaginary part |
/// | `attribute_7` | `Numeric` | pulse 4, real part |
/// | `attribute_8` | `Numeric` | pulse 4, imaginary part |
/// | `attribute_9` | `Numeric` | pulse 5, real part |
/// | `attribute_10` | `Numeric` | pulse 5, imaginary part |
/// | `attribute_11` | `Numeric` | pulse 6, real part |
/// | `attribute_12` | `Numeric` | pulse 6, imaginary part |
/// | `attribute_13` | `Numeric` | pulse 7, real part |
/// | `attribute_14` | `Numeric` | pulse 7, imaginary part |
/// | `attribute_15` | `Numeric` | pulse 8, real part |
/// | `attribute_16` | `Numeric` | pulse 8, imaginary part |
/// | `attribute_17` | `Numeric` | pulse 9, real part |
/// | `attribute_18` | `Numeric` | pulse 9, imaginary part |
/// | `attribute_19` | `Numeric` | pulse 10, real part |
/// | `attribute_20` | `Numeric` | pulse 10, imaginary part |
/// | `attribute_21` | `Numeric` | pulse 11, real part |
/// | `attribute_22` | `Numeric` | pulse 11, imaginary part |
/// | `attribute_23` | `Numeric` | pulse 12, real part |
/// | `attribute_24` | `Numeric` | pulse 12, imaginary part |
/// | `attribute_25` | `Numeric` | pulse 13, real part |
/// | `attribute_26` | `Numeric` | pulse 13, imaginary part |
/// | `attribute_27` | `Numeric` | pulse 14, real part |
/// | `attribute_28` | `Numeric` | pulse 14, imaginary part |
/// | `attribute_29` | `Numeric` | pulse 15, real part |
/// | `attribute_30` | `Numeric` | pulse 15, imaginary part |
/// | `attribute_31` | `Numeric` | pulse 16, real part |
/// | `attribute_32` | `Numeric` | pulse 16, imaginary part |
/// | `attribute_33` | `Numeric` | pulse 17, real part |
/// | `attribute_34` | `Numeric` | pulse 17, imaginary part |
/// | `class` | `String` | `good` or `bad` |
///
/// The source designates the 34 attributes as the inputs
/// ([`Ionosphere::FEATURE_NAMES`]) and `class` as the label
/// ([`Ionosphere::TARGET`]).
///
/// The source normalizes the values to about `-1..=1`. The first two attributes
/// are degenerate in this collection: `attribute_1` is `0` or `1`, and
/// `attribute_2` is always `0`. The loader keeps them verbatim, so the schema
/// matches the source exactly. The `class` column maps the source's
/// single-letter codes to readable names, `g` → `good` and `b` → `bad`.
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
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Ionosphere;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./ionosphere";
///
/// let mut dataset = Ionosphere::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 351);
/// assert_eq!(table.n_columns(), 35);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&Ionosphere::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[351, 34]);
///
/// // Reach the label column by name.
/// let class = table.column(Ionosphere::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(class.len(), 351);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("attribute_1") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 0.5;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 351);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 351);
/// ```
#[derive(Debug)]
pub struct Ionosphere {
    dataset: Dataset<Table, DatasetError>,
}

impl Ionosphere {
    /// The columns the source designates as the model inputs, in source order.
    /// The source numbers its attributes from 1.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "attribute_1",
        "attribute_2",
        "attribute_3",
        "attribute_4",
        "attribute_5",
        "attribute_6",
        "attribute_7",
        "attribute_8",
        "attribute_9",
        "attribute_10",
        "attribute_11",
        "attribute_12",
        "attribute_13",
        "attribute_14",
        "attribute_15",
        "attribute_16",
        "attribute_17",
        "attribute_18",
        "attribute_19",
        "attribute_20",
        "attribute_21",
        "attribute_22",
        "attribute_23",
        "attribute_24",
        "attribute_25",
        "attribute_26",
        "attribute_27",
        "attribute_28",
        "attribute_29",
        "attribute_30",
        "attribute_31",
        "attribute_32",
        "attribute_33",
        "attribute_34",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "class";

    /// Create a new Ionosphere instance without loading data.
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
    /// - `Self` - an `Ionosphere` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Ionosphere {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Ionosphere dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        // The source file is `ionosphere.data`. The cache stores it as
        // `ionosphere.csv`.
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
        let mut classes: Vec<String> = Vec::with_capacity(N_SAMPLES);

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
            classes.push(label.to_string());
        }

        // The source lists the 34 attributes first and the class last.
        let mut columns = Vec::with_capacity(N_COLUMNS);
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
        columns.push(Column::new(
            Self::TARGET,
            ColumnData::String(Array1::from_vec(classes)),
        ));

        Table::new(IONOSPHERE_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 351 samples and 35 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, an
    ///   unknown class)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`Ionosphere::data`], this method never runs the loader. If the
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
    /// changes stay in the cache. Later calls to [`Ionosphere::data`] or
    /// [`Ionosphere::get_data`] see them.
    ///
    /// Like [`Ionosphere::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`Ionosphere::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 351 samples and 35 columns.
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
    /// - `Table` - the owned table of 351 samples and 35 columns.
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

impl_ml_dataset!(Ionosphere, "ionosphere");
