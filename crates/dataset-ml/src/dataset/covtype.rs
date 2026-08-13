//! Forest Cover Type dataset.
//!
//! The Forest CoverType dataset supports multi-class classification. It matches
//! the dataset that scikit-learn exposes through `fetch_covtype`. Each of the
//! 581,012 samples describes a 30×30 meter cell of wilderness in the Roosevelt
//! National Forest of northern Colorado. The task is to predict the forest cover
//! type of the cell, one of seven classes, from 54 cartographic features.
//!
//! **Columns (55):**
//!
//! | Name                                        | Type      | Description                                                            |
//! |---------------------------------------------|-----------|------------------------------------------------------------------------|
//! | `Elevation`                                 | `Numeric` | elevation in meters                                                    |
//! | `Aspect`                                    | `Numeric` | aspect in azimuth degrees                                              |
//! | `Slope`                                     | `Numeric` | slope in degrees                                                       |
//! | `Horizontal_Distance_To_Hydrology`          | `Numeric` | horizontal distance to the nearest surface water                       |
//! | `Vertical_Distance_To_Hydrology`            | `Numeric` | vertical distance to the nearest surface water, which can be negative  |
//! | `Horizontal_Distance_To_Roadways`           | `Numeric` | horizontal distance to the nearest roadway                             |
//! | `Hillshade_9am`                             | `Numeric` | hillshade index at 9am, in `0..=255`                                   |
//! | `Hillshade_Noon`                            | `Numeric` | hillshade index at noon, in `0..=255`                                  |
//! | `Hillshade_3pm`                             | `Numeric` | hillshade index at 3pm, in `0..=255`                                   |
//! | `Horizontal_Distance_To_Fire_Points`        | `Numeric` | horizontal distance to the nearest wildfire ignition point             |
//! | `Wilderness_Area_0` to `Wilderness_Area_3`  | `Numeric` | one-hot block of 4 columns for the wilderness area, exactly one column holds `1.0` |
//! | `Soil_Type_0` to `Soil_Type_39`             | `Numeric` | one-hot block of 40 columns for the soil type, exactly one column holds `1.0` |
//! | `Cover_Type`                                | `Integer` | the forest cover type, one of `1`-`7`                                  |
//!
//! The source designates the 54 cartographic columns as the inputs
//! ([`Covtype::FEATURE_NAMES`](crate::Covtype::FEATURE_NAMES)) and `Cover_Type` as the label
//! ([`Covtype::TARGET`](crate::Covtype::TARGET)).
//!
//! The 54 feature columns encode 12 logical attributes. Ten of them are
//! quantitative, and two of them are categorical. The source ships the two
//! categorical attributes **one-hot expanded**, so each one-hot block sums to
//! `1` per row.
//!
//! The `Cover_Type` codes run from `1` to `7`.
//! [`Covtype::CLASS_NAMES`](crate::Covtype::CLASS_NAMES) names each class:
//!
//! - `1` = Spruce/Fir
//! - `2` = Lodgepole Pine
//! - `3` = Ponderosa Pine
//! - `4` = Cottonwood/Willow
//! - `5` = Aspen
//! - `6` = Douglas-fir
//! - `7` = Krummholz
//!
//! **Samples:** 581,012 total
//! **Application:** Multi-class classification / forest cover type prediction
//!
//! **Source:** UCI Machine Learning Repository, via the gzip-compressed
//! `covtype.data.gz` mirror that scikit-learn's `fetch_covtype` downloads.
//! <https://archive.ics.uci.edu/dataset/31/covertype>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, gunzip};
use ndarray::Array1;
use std::fs::File;

/// The URL for the Forest Cover Type dataset.
///
/// This is the gzip-compressed `covtype.data.gz` mirror that scikit-learn's
/// `fetch_covtype` uses. The URL has no filename segment, so the code saves the
/// download under an explicit name ([`COVTYPE_GZ_FILENAME`]).
///
/// # Citation
///
/// J. A. Blackard and D. J. Dean. "Covertype," UCI Machine Learning Repository,
/// \[Online\]. Available: <https://doi.org/10.24432/C50K5N>
const COVTYPE_DATA_URL: &str = "https://ndownloader.figshare.com/files/5976039";

/// The filename for the downloaded gzip archive inside the temp directory.
const COVTYPE_GZ_FILENAME: &str = "covtype.data.gz";

/// The name of the final cached (decompressed) Cover Type dataset file.
const COVTYPE_FILENAME: &str = "covtype.csv";

/// The SHA256 hash of the **decompressed** Cover Type dataset file (`covtype.csv`).
///
/// This hash covers the uncompressed comma-separated data, not the downloaded
/// `covtype.data.gz`, because the cached file is the decompressed one.
const COVTYPE_SHA256: &str = "0a9371cef7c964b5475d6053cc3e0894a5aa6f65ad1ed3ecb01c45aa96217945";

/// The name of the dataset
const COVTYPE_DATASET_NAME: &str = "covtype";

/// The number of cartographic features per sample.
const N_FEATURES: usize = 54;

/// The number of columns per CSV record (54 features and 1 cover-type label).
const N_COLUMNS: usize = N_FEATURES + 1;

/// The number of cover-type classes.
const N_CLASSES: usize = 7;

/// The expected number of samples, used only to pre-allocate the parse buffers.
const N_SAMPLES: usize = 581_012;

/// A struct that represents the Forest Cover Type dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// The Forest CoverType dataset contains 581,012 cartographic samples, each
/// describing a 30×30 meter cell of the Roosevelt National Forest in northern
/// Colorado. The 54 features combine 10 quantitative measurements (elevation,
/// slope, distances to hydrology, roadways, and fire points, and hillshade
/// indices) with two one-hot blocks. The one-hot blocks are 4 `Wilderness_Area`
/// columns and 40 `Soil_Type` columns. The target is the forest cover type
/// (`1`-`7`).
///
/// This is the same data that scikit-learn exposes through `fetch_covtype`.
///
/// # Columns
///
/// | Name                                        | Type      | Description                                                            |
/// |---------------------------------------------|-----------|------------------------------------------------------------------------|
/// | `Elevation`                                 | `Numeric` | elevation in meters                                                    |
/// | `Aspect`                                    | `Numeric` | aspect in azimuth degrees                                              |
/// | `Slope`                                     | `Numeric` | slope in degrees                                                       |
/// | `Horizontal_Distance_To_Hydrology`          | `Numeric` | horizontal distance to the nearest surface water                       |
/// | `Vertical_Distance_To_Hydrology`            | `Numeric` | vertical distance to the nearest surface water, which can be negative  |
/// | `Horizontal_Distance_To_Roadways`           | `Numeric` | horizontal distance to the nearest roadway                             |
/// | `Hillshade_9am`                             | `Numeric` | hillshade index at 9am, in `0..=255`                                   |
/// | `Hillshade_Noon`                            | `Numeric` | hillshade index at noon, in `0..=255`                                  |
/// | `Hillshade_3pm`                             | `Numeric` | hillshade index at 3pm, in `0..=255`                                   |
/// | `Horizontal_Distance_To_Fire_Points`        | `Numeric` | horizontal distance to the nearest wildfire ignition point             |
/// | `Wilderness_Area_0` to `Wilderness_Area_3`  | `Numeric` | one-hot block of 4 columns for the wilderness area, exactly one column holds `1.0` |
/// | `Soil_Type_0` to `Soil_Type_39`             | `Numeric` | one-hot block of 40 columns for the soil type, exactly one column holds `1.0` |
/// | `Cover_Type`                                | `Integer` | the forest cover type, one of `1`-`7`                                  |
///
/// The source designates the 54 cartographic columns as the inputs
/// ([`Covtype::FEATURE_NAMES`]) and `Cover_Type` as the label
/// ([`Covtype::TARGET`]).
///
/// The 54 feature columns are **not** 54 independent variables. They encode 12
/// logical attributes: 10 numeric and 2 categorical. The source ships the two
/// categorical attributes **one-hot expanded** into binary indicator columns.
/// The 4 `Wilderness_Area` columns jointly answer "which of 4 wilderness
/// areas". The 40 `Soil_Type` columns jointly answer "which of 40 soil types".
/// In each block, `1.0` marks the active category and `0.0` marks every other
/// column of the block. Each block sums to `1`.
///
/// The `Cover_Type` codes run from `1` to `7`. [`Covtype::CLASS_NAMES`] names
/// each class: `1` = Spruce/Fir, `2` = Lodgepole Pine, `3` = Ponderosa Pine,
/// `4` = Cottonwood/Willow, `5` = Aspen, `6` = Douglas-fir, `7` = Krummholz.
///
/// See more information at
/// <https://archive.ics.uci.edu/dataset/31/covertype>
///
/// # Citation
///
/// J. A. Blackard and D. J. Dean. "Covertype," UCI Machine Learning Repository,
/// \[Online\]. Available: <https://doi.org/10.24432/C50K5N>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Covtype;
///
/// let download_dir = "./covtype"; // the loader creates the directory if it does not exist
///
/// let mut dataset = Covtype::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 581012);
/// assert_eq!(table.n_columns(), 55);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&Covtype::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[581012, 54]);
///
/// // Reach one column by name, whatever its position.
/// let elevation = table.column("Elevation").unwrap().as_numeric().unwrap();
/// assert_eq!(elevation.len(), 581012);
///
/// let cover_type = table.column(Covtype::TARGET).unwrap().as_integer().unwrap();
/// assert_eq!(cover_type.len(), 581012);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("Elevation") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 2596.0;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable. The next access reloads the data from the cached file.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 581012);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 581012);
/// ```
#[derive(Debug)]
pub struct Covtype {
    dataset: Dataset<Table, DatasetError>,
}

impl Covtype {
    /// The columns the source designates as the model inputs, in source order.
    ///
    /// The first 10 names are the quantitative attributes. The next 4 are the
    /// one-hot `Wilderness_Area` block. The last 40 are the one-hot `Soil_Type`
    /// block.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
        "Wilderness_Area_0",
        "Wilderness_Area_1",
        "Wilderness_Area_2",
        "Wilderness_Area_3",
        "Soil_Type_0",
        "Soil_Type_1",
        "Soil_Type_2",
        "Soil_Type_3",
        "Soil_Type_4",
        "Soil_Type_5",
        "Soil_Type_6",
        "Soil_Type_7",
        "Soil_Type_8",
        "Soil_Type_9",
        "Soil_Type_10",
        "Soil_Type_11",
        "Soil_Type_12",
        "Soil_Type_13",
        "Soil_Type_14",
        "Soil_Type_15",
        "Soil_Type_16",
        "Soil_Type_17",
        "Soil_Type_18",
        "Soil_Type_19",
        "Soil_Type_20",
        "Soil_Type_21",
        "Soil_Type_22",
        "Soil_Type_23",
        "Soil_Type_24",
        "Soil_Type_25",
        "Soil_Type_26",
        "Soil_Type_27",
        "Soil_Type_28",
        "Soil_Type_29",
        "Soil_Type_30",
        "Soil_Type_31",
        "Soil_Type_32",
        "Soil_Type_33",
        "Soil_Type_34",
        "Soil_Type_35",
        "Soil_Type_36",
        "Soil_Type_37",
        "Soil_Type_38",
        "Soil_Type_39",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "Cover_Type";

    /// The name of each cover-type class.
    ///
    /// The source numbers the classes from `1`, so a code of `3` names
    /// `CLASS_NAMES[2]`, which is `"Ponderosa Pine"`. Subtract `1` from the code
    /// to index this array. The order is the one the source defines.
    ///
    /// # Example
    /// ```
    /// use dataset_ml::Covtype;
    ///
    /// assert_eq!(Covtype::CLASS_NAMES[0], "Spruce/Fir");
    /// assert_eq!(Covtype::CLASS_NAMES[6], "Krummholz");
    /// ```
    pub const CLASS_NAMES: [&'static str; N_CLASSES] = [
        "Spruce/Fir",
        "Lodgepole Pine",
        "Ponderosa Pine",
        "Cottonwood/Willow",
        "Aspen",
        "Douglas-fir",
        "Krummholz",
    ];

    /// Create a new Covtype instance without loading data.
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
    /// - `Self` - a `Covtype` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Covtype {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Forest Cover Type dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        // Download the gzip-compressed `covtype.data.gz` file, then decompress it
        // into the plain comma-separated `covtype.csv` file.
        let file_path = acquire_dataset(
            dir,
            COVTYPE_FILENAME,
            COVTYPE_DATASET_NAME,
            Some(COVTYPE_SHA256),
            |temp_path| {
                download_to_with_retries(
                    COVTYPE_DATA_URL,
                    temp_path,
                    Some(COVTYPE_GZ_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                let gz_path = temp_path.join(COVTYPE_GZ_FILENAME);
                let csv_path = temp_path.join(COVTYPE_FILENAME);
                gunzip(&gz_path, &csv_path)?;
                Ok(csv_path)
            },
        )?;

        // `covtype.data` is a headerless comma-separated file: every line is a
        // record of 54 cartographic features followed by the cover-type label.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        // Pre-allocate for the known sample count. Parsing still works for any
        // actual row count.
        let mut feature_columns: Vec<Vec<f64>> = (0..N_FEATURES)
            .map(|_| Vec::with_capacity(N_SAMPLES))
            .collect();
        let mut labels = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(COVTYPE_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    COVTYPE_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            for (col, field) in record.iter().take(N_FEATURES).enumerate() {
                let value: f64 = field.trim().parse().map_err(|e| {
                    DatasetError::parse_failed(
                        COVTYPE_DATASET_NAME,
                        &format!("feature_{}", col),
                        line_num,
                        e,
                    )
                })?;
                feature_columns[col].push(value);
            }

            let raw_label = record[N_FEATURES].trim();
            let label: u8 = raw_label.parse().map_err(|e| {
                DatasetError::parse_failed(COVTYPE_DATASET_NAME, "Cover_Type", line_num, e)
            })?;
            if !(1..=N_CLASSES as u8).contains(&label) {
                return Err(DatasetError::invalid_value(
                    COVTYPE_DATASET_NAME,
                    "Cover_Type",
                    raw_label,
                    line_num,
                ));
            }
            labels.push(i64::from(label));
        }

        let mut columns = Vec::with_capacity(N_COLUMNS);
        for (name, values) in Self::FEATURE_NAMES.into_iter().zip(feature_columns) {
            columns.push(Column::new(
                name,
                ColumnData::Numeric(Array1::from_vec(values)),
            ));
        }
        columns.push(Column::new(
            Self::TARGET,
            ColumnData::Integer(Array1::from_vec(labels)),
        ));

        Table::new(COVTYPE_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 581,012 samples and 55
    ///   columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File decompression or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`Covtype::data`], which loads the dataset on the first call, this
    /// method never runs the loader. If the data has not loaded yet, this method
    /// returns `None` instead of downloading and parsing it. Use this method only
    /// when you want data that is already cached. This avoids the download and
    /// parse cost if the dataset is not cached yet.
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
    /// changes persist, so later calls to [`Covtype::data`] or
    /// [`Covtype::get_data`] see them.
    ///
    /// Like [`Covtype::get_data`], this method does **not** trigger loading. It
    /// returns `None` if the dataset has not loaded yet. If you need the data to
    /// be present, call a loading accessor first, for example [`Covtype::data`].
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
    /// This method **consumes** `self`, so you cannot use the instance afterward.
    /// If you want owned data but need to keep using the instance, use
    /// [`Covtype::take_data`] instead. That method takes `&mut self` and leaves
    /// the instance reusable.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 581,012 samples and 55 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or
    /// invalid labels).
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
    /// This resets the instance to its unloaded state. The next accessor call,
    /// for example [`Covtype::data`], loads the dataset again.
    ///
    /// If you are done with the instance, use [`Covtype::into_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 581,012 samples and 55 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or
    /// invalid labels).
    pub fn take_data(&mut self) -> Result<Table, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(Covtype, "covtype");
