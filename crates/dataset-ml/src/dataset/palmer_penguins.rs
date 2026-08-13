//! Palmer Penguins dataset.
//!
//! Size measurements for three penguin species, observed on three islands in
//! the Palmer Archipelago, Antarctica. The task is multi-class classification.
//! It holds both numeric and text columns, and some values are missing.
//!
//! **Columns (8):**
//!
//! | Name                | Type      | Description                        |
//! |----------------------|-----------|-------------------------------------|
//! | `species`           | `String`  | `Adelie`, `Chinstrap`, or `Gentoo`  |
//! | `island`            | `String`  | `Biscoe`, `Dream`, or `Torgersen`   |
//! | `bill_length_mm`    | `Numeric` | bill length in mm                   |
//! | `bill_depth_mm`     | `Numeric` | bill depth in mm                    |
//! | `flipper_length_mm` | `Numeric` | flipper length in mm                |
//! | `body_mass_g`       | `Numeric` | body mass in g                      |
//! | `sex`               | `String`  | `male` or `female`                  |
//! | `year`              | `Numeric` | study year: 2007, 2008, or 2009     |
//!
//! The source designates four measurements, `island`, and `sex` as the inputs
//! ([`PalmerPenguins::FEATURE_NAMES`](crate::PalmerPenguins::FEATURE_NAMES)) and `species` as the label
//! ([`PalmerPenguins::TARGET`](crate::PalmerPenguins::TARGET)).
//!
//! **Samples:** 344 total (152 Adelie, 68 Chinstrap, 124 Gentoo)
//! **Application:** Multi-class classification / species recognition
//!
//! **Missing values:** the source encodes them as the literal string `NA`. A
//! numeric field becomes `NaN`. A text field becomes an empty string. `species`
//! is never missing.
//!
//! **Source:** Horst AM, Hill AP, Gorman KB (2020). palmerpenguins R package.
//! <https://allisonhorst.github.io/palmerpenguins/>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::Array1;
use serde::Deserialize;
use std::fs::File;

/// The URL for the Palmer Penguins dataset.
///
/// # Citation
///
/// Horst AM, Hill AP, Gorman KB (2020). "palmerpenguins: Palmer Archipelago
/// (Antarctica) penguin data." R package version 0.1.0. \[Online\].
/// Available: <https://allisonhorst.github.io/palmerpenguins/>
const PENGUINS_DATA_URL: &str =
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv";

/// The name of the Palmer Penguins dataset file.
const PENGUINS_FILENAME: &str = "penguins.csv";

/// The SHA256 hash of the Palmer Penguins dataset file.
const PENGUINS_SHA256: &str = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93";

/// The name of the dataset
const PENGUINS_DATASET_NAME: &str = "palmer_penguins";

/// The three species the `species` column names.
const SPECIES: [&str; 3] = ["Adelie", "Chinstrap", "Gentoo"];

/// One CSV record of the Palmer Penguins dataset, with fields in source column
/// order: `species`, `island`, `bill_length_mm`, `bill_depth_mm`,
/// `flipper_length_mm`, `body_mass_g`, `sex`, `year`.
///
/// This loader deserializes every field as a raw `String`. The source encodes
/// missing values as the literal token `NA`, not as an empty field.
///
/// This struct declares its fields in CSV column order, and the loader
/// deserializes them **positionally**. The loader disables the `csv` crate's
/// header handling, so this struct does not depend on the exact header
/// spelling.
#[derive(Deserialize)]
struct PenguinRecord {
    species: String,
    island: String,
    bill_length_mm: String,
    bill_depth_mm: String,
    flipper_length_mm: String,
    body_mass_g: String,
    sex: String,
    year: String,
}

/// Parse a numeric cell, mapping the source's missing-value token (`NA`) and any
/// empty field to `NaN`.
fn parse_numeric(value: &str, field_name: &str, line_num: usize) -> Result<f64, DatasetError> {
    if value == "NA" || value.is_empty() {
        Ok(f64::NAN)
    } else {
        value.parse::<f64>().map_err(|_| {
            DatasetError::invalid_value(PENGUINS_DATASET_NAME, field_name, value, line_num)
        })
    }
}

/// Normalize a categorical cell. The missing-value token (`NA`) becomes an empty
/// string.
fn clean_categorical(value: String) -> String {
    if value == "NA" { String::new() } else { value }
}

/// A struct that represents the Palmer Penguins dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the
/// first load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// Dr. Kristen Gorman and the Palmer Station Long Term Ecological Research (LTER)
/// program collected the data and made it available. The data hold size
/// measurements for adult foraging penguins of three species: Adelie, Chinstrap,
/// and Gentoo. Researchers observed the penguins on three islands in the Palmer
/// Archipelago, Antarctica: Biscoe, Dream, and Torgersen. It is a popular,
/// another multi-class classification task, next to Iris.
///
/// # Columns
///
/// | Name                | Type      | Description                        |
/// |----------------------|-----------|-------------------------------------|
/// | `species`           | `String`  | `Adelie`, `Chinstrap`, or `Gentoo`  |
/// | `island`            | `String`  | `Biscoe`, `Dream`, or `Torgersen`   |
/// | `bill_length_mm`    | `Numeric` | bill length in mm                   |
/// | `bill_depth_mm`     | `Numeric` | bill depth in mm                    |
/// | `flipper_length_mm` | `Numeric` | flipper length in mm                |
/// | `body_mass_g`       | `Numeric` | body mass in g                      |
/// | `sex`               | `String`  | `male` or `female`                  |
/// | `year`              | `Numeric` | study year: 2007, 2008, or 2009     |
///
/// The source designates four measurements, `island`, and `sex` as the inputs
/// ([`PalmerPenguins::FEATURE_NAMES`]) and `species` as the label
/// ([`PalmerPenguins::TARGET`]).
///
/// Missing values: the source encodes them as the literal token `NA`. A numeric
/// cell becomes `NaN`. A text cell becomes an empty string. The `species`
/// column is never missing.
///
/// See more information at <https://allisonhorst.github.io/palmerpenguins/>
///
/// # Citation
///
/// Horst AM, Hill AP, Gorman KB (2020). "palmerpenguins: Palmer Archipelago
/// (Antarctica) penguin data." R package version 0.1.0. \[Online\].
/// Available: <https://allisonhorst.github.io/palmerpenguins/>
///
/// Original data: Gorman KB, Williams TD, Fraser WR (2014). "Ecological sexual
/// dimorphism and environmental variability within a community of Antarctic
/// penguins (genus *Pygoscelis*)." PLoS ONE 9(3): e90081.
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::PalmerPenguins;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./palmer_penguins";
///
/// let mut dataset = PalmerPenguins::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 344);
/// assert_eq!(table.n_columns(), 8);
///
/// // Name the numeric feature columns you want in the matrix. `FEATURE_NAMES`
/// // also holds text columns, so a numeric matrix call must name a subset.
/// let measurements = table
///     .numeric_matrix(&[
///         "bill_length_mm",
///         "bill_depth_mm",
///         "flipper_length_mm",
///         "body_mass_g",
///     ])
///     .unwrap();
/// assert_eq!(measurements.shape(), &[344, 4]);
///
/// // Reach the label column by name.
/// let species = table.column(PalmerPenguins::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(species.len(), 344);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("bill_length_mm") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 40.0;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 344);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 344);
/// ```
#[derive(Debug)]
pub struct PalmerPenguins {
    dataset: Dataset<Table, DatasetError>,
}

impl PalmerPenguins {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; 6] = [
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "species";

    /// Create a new PalmerPenguins instance without loading data.
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
    /// - `Self` - a `PalmerPenguins` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        PalmerPenguins {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Palmer Penguins dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            PENGUINS_FILENAME,
            PENGUINS_DATASET_NAME,
            Some(PENGUINS_SHA256),
            |temp_path| {
                download_to_with_retries(PENGUINS_DATA_URL, temp_path, None, DOWNLOAD_RETRIES)?;
                Ok(temp_path.join(PENGUINS_FILENAME))
            },
        )?;

        // The `csv` crate deserializes each row into the struct. The file has a
        // header row, so the loader skips it explicitly.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut species = Vec::new();
        let mut island = Vec::new();
        let mut bill_length_mm = Vec::new();
        let mut bill_depth_mm = Vec::new();
        let mut flipper_length_mm = Vec::new();
        let mut body_mass_g = Vec::new();
        let mut sex = Vec::new();
        let mut year = Vec::new();

        for (idx, result) in rdr.deserialize::<PenguinRecord>().skip(1).enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(PENGUINS_DATASET_NAME, e))?;
            let line_num = idx + 2; // +1 for 0-indexed, +1 for header

            if !SPECIES.contains(&record.species.as_str()) {
                return Err(DatasetError::invalid_value(
                    PENGUINS_DATASET_NAME,
                    "species",
                    &record.species,
                    line_num,
                ));
            }
            species.push(record.species);

            // Missing (`NA`) values become empty strings.
            island.push(clean_categorical(record.island));

            // Missing (`NA`) values become `NaN`.
            bill_length_mm.push(parse_numeric(
                &record.bill_length_mm,
                "bill_length_mm",
                line_num,
            )?);
            bill_depth_mm.push(parse_numeric(
                &record.bill_depth_mm,
                "bill_depth_mm",
                line_num,
            )?);
            flipper_length_mm.push(parse_numeric(
                &record.flipper_length_mm,
                "flipper_length_mm",
                line_num,
            )?);
            body_mass_g.push(parse_numeric(&record.body_mass_g, "body_mass_g", line_num)?);
            sex.push(clean_categorical(record.sex));
            year.push(parse_numeric(&record.year, "year", line_num)?);
        }

        Table::new(
            PENGUINS_DATASET_NAME,
            vec![
                Column::new(Self::TARGET, ColumnData::String(Array1::from_vec(species))),
                Column::new(
                    Self::FEATURE_NAMES[0],
                    ColumnData::String(Array1::from_vec(island)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[1],
                    ColumnData::Numeric(Array1::from_vec(bill_length_mm)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[2],
                    ColumnData::Numeric(Array1::from_vec(bill_depth_mm)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[3],
                    ColumnData::Numeric(Array1::from_vec(flipper_length_mm)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[4],
                    ColumnData::Numeric(Array1::from_vec(body_mass_g)),
                ),
                Column::new(
                    Self::FEATURE_NAMES[5],
                    ColumnData::String(Array1::from_vec(sex)),
                ),
                Column::new("year", ColumnData::Numeric(Array1::from_vec(year))),
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
    /// - `&Table` - reference to the cached table of 344 samples and 8 columns.
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
    /// Unlike [`PalmerPenguins::data`], this method never runs the loader. If
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
    /// changes stay in the cache. Later calls to [`PalmerPenguins::data`] or
    /// [`PalmerPenguins::get_data`] see them.
    ///
    /// Like [`PalmerPenguins::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`PalmerPenguins::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 344 samples and 8 columns.
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
    /// - `Table` - the owned table of 344 samples and 8 columns.
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

impl_ml_dataset!(PalmerPenguins, "palmer_penguins");
