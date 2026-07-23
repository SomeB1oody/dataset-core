//! Palmer Penguins dataset.
//!
//! Size measurements for three penguin species observed on three islands in the
//! Palmer Archipelago, Antarctica. A modern, approachable alternative to Iris
//! for multi-class classification, with both numeric and categorical features
//! and some missing values.
//!
//! **Features (7, mixed):**
//! - String features: `island`, `sex`
//! - Numeric features: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`,
//!   `body_mass_g`, `year`
//!
//! **Target:** `species` - one of `Adelie`, `Chinstrap`, or `Gentoo`
//!
//! **Samples:** 344 total (152 Adelie, 68 Chinstrap, 124 Gentoo)
//! **Application:** Multi-class classification / species recognition
//!
//! **Missing values:** the source encodes them as the literal string `NA`.
//! Numeric fields become `NaN`; string fields become empty strings. `species`
//! is never missing.
//!
//! **Source:** Horst AM, Hill AP, Gorman KB (2020). palmerpenguins R package.
//! <https://allisonhorst.github.io/palmerpenguins/>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries};
use ndarray::{Array1, Array2};
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

/// The number of string (categorical) features per sample (`island`, `sex`).
const N_STRING_FEATURES: usize = 2;

/// The number of numeric features per sample (`bill_length_mm`, `bill_depth_mm`,
/// `flipper_length_mm`, `body_mass_g`, `year`).
const N_NUMERIC_FEATURES: usize = 5;

/// Type alias for the Palmer Penguins dataset: (string features, numeric features, labels).
type PenguinsData = (Array2<String>, Array2<f64>, Array1<&'static str>);

/// One CSV record of the Palmer Penguins dataset, with fields in source column
/// order: `species`, `island`, `bill_length_mm`, `bill_depth_mm`,
/// `flipper_length_mm`, `body_mass_g`, `sex`, `year`.
///
/// Every field is deserialized as a raw `String` because the source encodes
/// missing values as the literal token `NA` (not an empty field), so the numeric
/// columns are parsed and `NA`-handled manually rather than via `Option<f64>`.
/// Fields are declared in CSV column order and deserialized **positionally**
/// (the loader disables csv's header handling), so this struct is independent of
/// the exact header spelling.
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

/// Normalize a categorical cell: the missing-value token (`NA`) becomes an empty
/// string, mirroring how the other mixed dataset (Titanic) represents missing
/// text fields.
fn clean_categorical(value: String) -> String {
    if value == "NA" { String::new() } else { value }
}

/// A struct representing the Palmer Penguins dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The data were collected and made available by Dr. Kristen Gorman and the
/// Palmer Station Long Term Ecological Research (LTER) program. They contain
/// size measurements for adult foraging penguins of three species (Adelie,
/// Chinstrap, Gentoo) observed on three islands (Biscoe, Dream, Torgersen) in
/// the Palmer Archipelago, Antarctica. It is a popular, beginner-friendly
/// alternative to the Iris dataset.
///
/// # Feature columns
///
/// The features are split across two matrices: a string (categorical) matrix of
/// shape `(344, 2)` and a numeric matrix of shape `(344, 5)`. The source encodes
/// missing values as the literal token `NA`; numeric cells become `NaN` and
/// string cells become empty strings (`""`).
///
/// String features (shape `(344, 2)`), in column order:
///
/// | Columns | Attributes | Unit |
/// |---------|------------|------|
/// | `0`     | `island` (Biscoe, Dream, or Torgersen; `""` if missing) | |
/// | `1`     | `sex` (male or female; `""` if missing) | |
///
/// Numeric features (shape `(344, 5)`), in column order (may be `NaN` if missing
/// in the source):
///
/// | Columns | Attributes | Unit |
/// |---------|------------|------|
/// | `0`     | `bill_length_mm` | mm |
/// | `1`     | `bill_depth_mm` | mm |
/// | `2`     | `flipper_length_mm` | mm |
/// | `3`     | `body_mass_g` | g |
/// | `4`     | `year` (the study year: 2007, 2008, or 2009) | |
///
/// # Labels
///
/// - `species` (shape `(344,)`, in `&str`): `"Adelie"`, `"Chinstrap"`, `"Gentoo"`
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
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::palmer_penguins::PalmerPenguins;
///
/// let download_dir = "./palmer_penguins"; // the code will create the directory if it doesn't exist
///
/// let mut dataset = PalmerPenguins::new(download_dir);
/// let (string_features, numeric_features) = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (string_features, numeric_features, labels) = dataset.data().unwrap(); // this is also a way to get all data
/// assert_eq!(string_features.shape(), &[344, 2]);
/// assert_eq!(numeric_features.shape(), &[344, 5]);
/// assert_eq!(labels.len(), 344);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((_strings, numerics, labels)) = dataset.get_data_mut() {
///     numerics[[0, 0]] = 40.0;
///     labels[0] = "Gentoo";
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out (no `to_owned()` clone) and leaves
/// // the instance reusable — the next access reloads from the cached file.
/// let (owned_strings, owned_numerics, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_strings.shape(), &[344, 2]);
/// assert_eq!(owned_numerics.shape(), &[344, 5]);
/// assert_eq!(owned_labels.len(), 344);
///
/// // `into_data()` also returns the owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_strings, owned_numerics, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_strings.shape(), &[344, 2]);
/// assert_eq!(owned_numerics.shape(), &[344, 5]);
/// assert_eq!(owned_labels.len(), 344);
/// ```
#[derive(Debug)]
pub struct PalmerPenguins {
    dataset: Dataset<PenguinsData, DatasetError>,
}

impl PalmerPenguins {
    /// Create a new PalmerPenguins instance without loading data.
    ///
    /// The dataset will be loaded lazily when you first call any data accessor method.
    /// This is a lightweight operation that only stores the storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory where the dataset will be stored.
    ///
    /// # Returns
    ///
    /// - `Self` - `PalmerPenguins` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        PalmerPenguins {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Acquire and parse the Palmer Penguins dataset.
    fn load_data(dir: &str) -> Result<PenguinsData, DatasetError> {
        // Prepare the dataset file
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

        // csv deserializes into the struct. The file has a header row, so skip it.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut string_features = Vec::new();
        let mut numeric_features = Vec::new();
        let mut labels = Vec::new();

        for (idx, result) in rdr.deserialize::<PenguinRecord>().skip(1).enumerate() {
            let PenguinRecord {
                species,
                island,
                bill_length_mm,
                bill_depth_mm,
                flipper_length_mm,
                body_mass_g,
                sex,
                year,
            } = result.map_err(|e| DatasetError::csv_read_error(PENGUINS_DATASET_NAME, e))?;
            let line_num = idx + 2; // +1 for 0-indexed, +1 for header

            // Label: species (always present in the source).
            labels.push(match species.as_str() {
                "Adelie" => "Adelie",
                "Chinstrap" => "Chinstrap",
                "Gentoo" => "Gentoo",
                other => {
                    return Err(DatasetError::invalid_value(
                        PENGUINS_DATASET_NAME,
                        "species",
                        other,
                        line_num,
                    ));
                }
            });

            // Numeric features, in column order. Missing (`NA`) values become `NaN`.
            numeric_features.push(parse_numeric(&bill_length_mm, "bill_length_mm", line_num)?);
            numeric_features.push(parse_numeric(&bill_depth_mm, "bill_depth_mm", line_num)?);
            numeric_features.push(parse_numeric(
                &flipper_length_mm,
                "flipper_length_mm",
                line_num,
            )?);
            numeric_features.push(parse_numeric(&body_mass_g, "body_mass_g", line_num)?);
            numeric_features.push(parse_numeric(&year, "year", line_num)?);

            // String features, in column order: island, sex. Missing (`NA`) values
            // become empty strings.
            string_features.push(clean_categorical(island));
            string_features.push(clean_categorical(sex));
        }

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(PENGUINS_DATASET_NAME));
        }

        // Palmer Penguins has a fixed schema of 2 string and 5 numeric features per sample.
        let string_array = Array2::from_shape_vec((n_samples, N_STRING_FEATURES), string_features)
            .map_err(|e| {
                DatasetError::array_shape_error(PENGUINS_DATASET_NAME, "string_features", e)
            })?;

        let numeric_array =
            Array2::from_shape_vec((n_samples, N_NUMERIC_FEATURES), numeric_features).map_err(
                |e| DatasetError::array_shape_error(PENGUINS_DATASET_NAME, "numeric_features", e),
            )?;

        let labels_array = Array1::from_vec(labels);

        Ok((string_array, numeric_array, labels_array))
    }

    /// Get a reference to both string and numeric feature matrices.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<String>` - Reference to string feature matrix with shape `(344, 2)` containing:
    ///     - `island`
    ///     - `sex`
    ///
    ///   (empty string if missing in source)
    ///
    /// - `&Array2<f64>` - Reference to numeric feature matrix with shape `(344, 5)` containing:
    ///     - `bill_length_mm`
    ///     - `bill_depth_mm`
    ///     - `flipper_length_mm`
    ///     - `body_mass_g`
    ///     - `year`
    ///
    ///   (`NaN` if missing in source)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (344 samples)
    pub fn features(&self) -> Result<(&Array2<String>, &Array2<f64>), DatasetError> {
        let data = self.dataset.load()?;
        Ok((&data.0, &data.1))
    }

    /// Get a reference to the labels vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<&'static str>` - Reference to label vector with shape `(344,)` containing species names (`"Adelie"`, `"Chinstrap"`, `"Gentoo"`)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (344 samples)
    pub fn labels(&self) -> Result<&Array1<&'static str>, DatasetError> {
        Ok(&self.dataset.load()?.2)
    }

    /// Get string features, numeric features and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&PenguinsData` - reference to the cached `(string features, numeric
    ///   features, labels)` tuple: string feature matrix `(344, 2)` (island,
    ///   sex), numeric feature matrix `(344, 5)` (bill_length_mm, bill_depth_mm,
    ///   flipper_length_mm, body_mass_g, year), and label vector `(344,)`
    ///   (species).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
    /// - Dataset size doesn't match expected dimensions (344 samples)
    pub fn data(&self) -> Result<&PenguinsData, DatasetError> {
        self.dataset.load()
    }

    /// Get string features, numeric features and labels as references
    /// **without** triggering loading.
    ///
    /// Unlike [`PalmerPenguins::data`], which loads the dataset on first call,
    /// this never runs the loader: if the data has not been loaded yet, it returns
    /// `None` instead of downloading and parsing. Use it when you only want the
    /// data if it is already cached and want to avoid paying the download/parse
    /// cost otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&PenguinsData)` - reference to the cached `(string features, numeric
    ///   features, labels)` tuple (`(344, 2)`, `(344, 5)`, `(344,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&PenguinsData> {
        self.dataset.get()
    }

    /// Get mutable references to string features, numeric features, and labels
    /// for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize numeric
    /// features, replace missing values) with no `to_owned()` clone and without
    /// removing them from the cache: the changes persist, so later
    /// [`PalmerPenguins::features`], [`PalmerPenguins::data`], or
    /// [`PalmerPenguins::get_data`] calls observe them.
    ///
    /// Like [`PalmerPenguins::get_data`], this does **not** trigger loading: it
    /// returns `None` if the dataset has not been loaded. Call a loading accessor
    /// (e.g. [`PalmerPenguins::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut PenguinsData)` - mutable reference to the cached `(string
    ///   features, numeric features, labels)` tuple (`(344, 2)`, `(344, 5)`,
    ///   `(344,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut PenguinsData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** string features, numeric features,
    /// and labels.
    ///
    /// Unlike [`PalmerPenguins::data`], which borrows the cached data, this moves
    /// it out and returns owned arrays directly — no `to_owned()` clone needed. The
    /// dataset is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`PalmerPenguins::take_data`] instead — it takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array2<f64>, Array1<&'static str>)` - owned string
    ///   feature matrix `(344, 2)`, owned numeric feature matrix `(344, 5)`, and
    ///   owned label vector `(344,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn into_data(self) -> Result<PenguinsData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** string features, numeric features, and labels out of the
    /// dataset, leaving it reusable.
    ///
    /// Like [`PalmerPenguins::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. But instead of consuming the instance, it takes
    /// `&mut self` and moves the cached data out, resetting the instance to its
    /// unloaded state: the next accessor call (e.g. [`PalmerPenguins::features`] or
    /// [`PalmerPenguins::data`]) loads the dataset again.
    ///
    /// Use [`PalmerPenguins::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array2<f64>, Array1<&'static str>)` - owned string
    ///   feature matrix `(344, 2)`, owned numeric feature matrix `(344, 5)`, and
    ///   owned label vector `(344,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, invalid
    /// labels, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<PenguinsData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(PalmerPenguins, PenguinsData, "palmer_penguins");
