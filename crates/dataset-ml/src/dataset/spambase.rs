//! Spambase dataset.
//!
//! Researchers gathered this collection of 4,601 e-mails at Hewlett-Packard
//! Labs in 1999. The dataset summarizes each e-mail with 57 hand-crafted
//! frequency statistics, not its raw text. The spam came from a postmaster
//! and from individuals who had filed spam. The non-spam came from filed
//! work and personal e-mail. The task is to predict whether an e-mail is
//! spam from those statistics.
//!
//! A `word_freq_WORD` column is `100 * (times WORD appears) / (total words)`. A
//! `char_freq_CHAR` column is `100 * (occurrences of CHAR) / (total
//! characters)`. A "word" is any string of alphanumeric characters bounded by
//! non-alphanumeric characters or the end of the string. The
//! `capital_run_length_*` columns measure uninterrupted sequences of capital
//! letters. All 57 features are non-negative. The frequency columns lie in
//! `0..=100`.
//!
//! **Columns (58):**
//!
//! | Name | Type | Description |
//! |------|------|-------------|
//! | `word_freq_make` | `Numeric` | percentage of words that are `make` |
//! | `word_freq_address` | `Numeric` | percentage of words that are `address` |
//! | `word_freq_all` | `Numeric` | percentage of words that are `all` |
//! | `word_freq_3d` | `Numeric` | percentage of words that are `3d` |
//! | `word_freq_our` | `Numeric` | percentage of words that are `our` |
//! | `word_freq_over` | `Numeric` | percentage of words that are `over` |
//! | `word_freq_remove` | `Numeric` | percentage of words that are `remove` |
//! | `word_freq_internet` | `Numeric` | percentage of words that are `internet` |
//! | `word_freq_order` | `Numeric` | percentage of words that are `order` |
//! | `word_freq_mail` | `Numeric` | percentage of words that are `mail` |
//! | `word_freq_receive` | `Numeric` | percentage of words that are `receive` |
//! | `word_freq_will` | `Numeric` | percentage of words that are `will` |
//! | `word_freq_people` | `Numeric` | percentage of words that are `people` |
//! | `word_freq_report` | `Numeric` | percentage of words that are `report` |
//! | `word_freq_addresses` | `Numeric` | percentage of words that are `addresses` |
//! | `word_freq_free` | `Numeric` | percentage of words that are `free` |
//! | `word_freq_business` | `Numeric` | percentage of words that are `business` |
//! | `word_freq_email` | `Numeric` | percentage of words that are `email` |
//! | `word_freq_you` | `Numeric` | percentage of words that are `you` |
//! | `word_freq_credit` | `Numeric` | percentage of words that are `credit` |
//! | `word_freq_your` | `Numeric` | percentage of words that are `your` |
//! | `word_freq_font` | `Numeric` | percentage of words that are `font` |
//! | `word_freq_000` | `Numeric` | percentage of words that are `000` |
//! | `word_freq_money` | `Numeric` | percentage of words that are `money` |
//! | `word_freq_hp` | `Numeric` | percentage of words that are `hp` |
//! | `word_freq_hpl` | `Numeric` | percentage of words that are `hpl` |
//! | `word_freq_george` | `Numeric` | percentage of words that are `george` |
//! | `word_freq_650` | `Numeric` | percentage of words that are `650` |
//! | `word_freq_lab` | `Numeric` | percentage of words that are `lab` |
//! | `word_freq_labs` | `Numeric` | percentage of words that are `labs` |
//! | `word_freq_telnet` | `Numeric` | percentage of words that are `telnet` |
//! | `word_freq_857` | `Numeric` | percentage of words that are `857` |
//! | `word_freq_data` | `Numeric` | percentage of words that are `data` |
//! | `word_freq_415` | `Numeric` | percentage of words that are `415` |
//! | `word_freq_85` | `Numeric` | percentage of words that are `85` |
//! | `word_freq_technology` | `Numeric` | percentage of words that are `technology` |
//! | `word_freq_1999` | `Numeric` | percentage of words that are `1999` |
//! | `word_freq_parts` | `Numeric` | percentage of words that are `parts` |
//! | `word_freq_pm` | `Numeric` | percentage of words that are `pm` |
//! | `word_freq_direct` | `Numeric` | percentage of words that are `direct` |
//! | `word_freq_cs` | `Numeric` | percentage of words that are `cs` |
//! | `word_freq_meeting` | `Numeric` | percentage of words that are `meeting` |
//! | `word_freq_original` | `Numeric` | percentage of words that are `original` |
//! | `word_freq_project` | `Numeric` | percentage of words that are `project` |
//! | `word_freq_re` | `Numeric` | percentage of words that are `re` |
//! | `word_freq_edu` | `Numeric` | percentage of words that are `edu` |
//! | `word_freq_table` | `Numeric` | percentage of words that are `table` |
//! | `word_freq_conference` | `Numeric` | percentage of words that are `conference` |
//! | `char_freq_;` | `Numeric` | percentage of characters that are `;` |
//! | `char_freq_(` | `Numeric` | percentage of characters that are `(` |
//! | `char_freq_[` | `Numeric` | percentage of characters that are `[` |
//! | `char_freq_!` | `Numeric` | percentage of characters that are `!` |
//! | `char_freq_$` | `Numeric` | percentage of characters that are `$` |
//! | `char_freq_#` | `Numeric` | percentage of characters that are `#` |
//! | `capital_run_length_average` | `Numeric` | average length of the capital-letter runs |
//! | `capital_run_length_longest` | `Numeric` | length of the longest capital-letter run |
//! | `capital_run_length_total` | `Numeric` | total number of capital letters |
//! | `class` | `String` | `ham` (not spam) or `spam` |
//!
//! The source designates the 57 word, character, and capital-run statistics as
//! the inputs ([`Spambase::FEATURE_NAMES`](crate::Spambase::FEATURE_NAMES)) and `class` as the label
//! ([`Spambase::TARGET`](crate::Spambase::TARGET)).
//!
//! **Samples:** 4,601 total (2,788 ham, 1,813 spam)
//! **Application:** Binary classification / spam detection
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C53G6X>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::Array1;
use std::fs::File;

use csv::ReaderBuilder;

/// The URL for the Spambase dataset.
///
/// This is the UCI static package. It is a ZIP archive that contains
/// `spambase.DOCUMENTATION`, `spambase.data`, and `spambase.names`. The loader
/// uses only the `spambase.data` file.
///
/// # Citation
///
/// M. Hopkins, E. Reeber, G. Forman, and J. Suermondt. "Spambase," UCI Machine
/// Learning Repository, \[Online\]. Available: <https://doi.org/10.24432/C53G6X>
const SPAMBASE_DATA_URL: &str = "https://archive.ics.uci.edu/static/public/94/spambase.zip";

/// The filename for the downloaded ZIP archive inside the temp directory.
const SPAMBASE_ZIP_FILENAME: &str = "spambase.zip";

/// The name of the file inside the archive that holds the records.
const SPAMBASE_SOURCE_FILENAME: &str = "spambase.data";

/// The name of the final cached Spambase dataset file.
const SPAMBASE_FILENAME: &str = "spambase.csv";

/// The SHA256 hash of the Spambase dataset file (`spambase.data`).
const SPAMBASE_SHA256: &str = "b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12";

/// The name of the dataset.
const SPAMBASE_DATASET_NAME: &str = "spambase";

/// Number of samples.
const N_SAMPLES: usize = 4601;

/// The number of numeric features per sample (48 word frequencies, 6 char
/// frequencies, and 3 capital-run-length statistics).
const N_FEATURES: usize = 57;

/// The number of columns per CSV record (57 features and 1 label).
const N_COLUMNS: usize = N_FEATURES + 1;

/// Source column index of the label (`class`). The label is the **last** column.
const LABEL_COLUMN: usize = N_FEATURES;

/// A struct that represents the Spambase dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// Mark Hopkins, Erik Reeber, George Forman, and Jaap Suermondt generated the
/// Spambase collection at Hewlett-Packard Labs in June–July 1999. The "spam"
/// concept is diverse: advertisements for products or web sites, make-money-fast
/// schemes, chain letters, and pornography. The spam e-mails came from the lab's
/// postmaster and from individuals who had filed spam. The non-spam e-mails came
/// from filed work and personal e-mail. This is why the word `george` and the
/// area code `650` are strong non-spam indicators here. They are useful for a
/// personalized filter, but a general purpose filter would need to blind them.
///
/// The dataset reduces each e-mail to 57 continuous statistics: how often
/// selected words and characters occur, and how long its runs of capital letters
/// are.
///
/// A `word_freq_WORD` column is `100 * (times WORD appears) / (total words)`. A
/// `char_freq_CHAR` column is `100 * (occurrences of CHAR) / (total
/// characters)`. A "word" is any string of alphanumeric characters bounded by
/// non-alphanumeric characters or the end of the string. The
/// `capital_run_length_*` columns measure uninterrupted sequences of capital
/// letters. They report the average, the maximum, and the sum, that is, the
/// total number of capital letters in the e-mail.
///
/// # Columns
///
/// | Name | Type | Description |
/// |------|------|-------------|
/// | `word_freq_make` | `Numeric` | percentage of words that are `make` |
/// | `word_freq_address` | `Numeric` | percentage of words that are `address` |
/// | `word_freq_all` | `Numeric` | percentage of words that are `all` |
/// | `word_freq_3d` | `Numeric` | percentage of words that are `3d` |
/// | `word_freq_our` | `Numeric` | percentage of words that are `our` |
/// | `word_freq_over` | `Numeric` | percentage of words that are `over` |
/// | `word_freq_remove` | `Numeric` | percentage of words that are `remove` |
/// | `word_freq_internet` | `Numeric` | percentage of words that are `internet` |
/// | `word_freq_order` | `Numeric` | percentage of words that are `order` |
/// | `word_freq_mail` | `Numeric` | percentage of words that are `mail` |
/// | `word_freq_receive` | `Numeric` | percentage of words that are `receive` |
/// | `word_freq_will` | `Numeric` | percentage of words that are `will` |
/// | `word_freq_people` | `Numeric` | percentage of words that are `people` |
/// | `word_freq_report` | `Numeric` | percentage of words that are `report` |
/// | `word_freq_addresses` | `Numeric` | percentage of words that are `addresses` |
/// | `word_freq_free` | `Numeric` | percentage of words that are `free` |
/// | `word_freq_business` | `Numeric` | percentage of words that are `business` |
/// | `word_freq_email` | `Numeric` | percentage of words that are `email` |
/// | `word_freq_you` | `Numeric` | percentage of words that are `you` |
/// | `word_freq_credit` | `Numeric` | percentage of words that are `credit` |
/// | `word_freq_your` | `Numeric` | percentage of words that are `your` |
/// | `word_freq_font` | `Numeric` | percentage of words that are `font` |
/// | `word_freq_000` | `Numeric` | percentage of words that are `000` |
/// | `word_freq_money` | `Numeric` | percentage of words that are `money` |
/// | `word_freq_hp` | `Numeric` | percentage of words that are `hp` |
/// | `word_freq_hpl` | `Numeric` | percentage of words that are `hpl` |
/// | `word_freq_george` | `Numeric` | percentage of words that are `george` |
/// | `word_freq_650` | `Numeric` | percentage of words that are `650` |
/// | `word_freq_lab` | `Numeric` | percentage of words that are `lab` |
/// | `word_freq_labs` | `Numeric` | percentage of words that are `labs` |
/// | `word_freq_telnet` | `Numeric` | percentage of words that are `telnet` |
/// | `word_freq_857` | `Numeric` | percentage of words that are `857` |
/// | `word_freq_data` | `Numeric` | percentage of words that are `data` |
/// | `word_freq_415` | `Numeric` | percentage of words that are `415` |
/// | `word_freq_85` | `Numeric` | percentage of words that are `85` |
/// | `word_freq_technology` | `Numeric` | percentage of words that are `technology` |
/// | `word_freq_1999` | `Numeric` | percentage of words that are `1999` |
/// | `word_freq_parts` | `Numeric` | percentage of words that are `parts` |
/// | `word_freq_pm` | `Numeric` | percentage of words that are `pm` |
/// | `word_freq_direct` | `Numeric` | percentage of words that are `direct` |
/// | `word_freq_cs` | `Numeric` | percentage of words that are `cs` |
/// | `word_freq_meeting` | `Numeric` | percentage of words that are `meeting` |
/// | `word_freq_original` | `Numeric` | percentage of words that are `original` |
/// | `word_freq_project` | `Numeric` | percentage of words that are `project` |
/// | `word_freq_re` | `Numeric` | percentage of words that are `re` |
/// | `word_freq_edu` | `Numeric` | percentage of words that are `edu` |
/// | `word_freq_table` | `Numeric` | percentage of words that are `table` |
/// | `word_freq_conference` | `Numeric` | percentage of words that are `conference` |
/// | `char_freq_;` | `Numeric` | percentage of characters that are `;` |
/// | `char_freq_(` | `Numeric` | percentage of characters that are `(` |
/// | `char_freq_[` | `Numeric` | percentage of characters that are `[` |
/// | `char_freq_!` | `Numeric` | percentage of characters that are `!` |
/// | `char_freq_$` | `Numeric` | percentage of characters that are `$` |
/// | `char_freq_#` | `Numeric` | percentage of characters that are `#` |
/// | `capital_run_length_average` | `Numeric` | average length of the capital-letter runs |
/// | `capital_run_length_longest` | `Numeric` | length of the longest capital-letter run |
/// | `capital_run_length_total` | `Numeric` | total number of capital letters |
/// | `class` | `String` | `ham` (not spam) or `spam` |
///
/// The source designates the 57 word, character, and capital-run statistics as
/// the inputs ([`Spambase::FEATURE_NAMES`]) and `class` as the label
/// ([`Spambase::TARGET`]).
///
/// The `class` column maps the source's nominal codes to readable names:
/// `0` → `ham` and `1` → `spam`.
///
/// See more information at <https://archive.ics.uci.edu/dataset/94/spambase>.
///
/// # Citation
///
/// M. Hopkins, E. Reeber, G. Forman, and J. Suermondt. "Spambase," UCI Machine
/// Learning Repository, \[Online\]. Available: <https://doi.org/10.24432/C53G6X>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Spambase;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./spambase";
///
/// let mut dataset = Spambase::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 4601);
/// assert_eq!(table.n_columns(), 58);
///
/// // Ask for the feature matrix when you want it.
/// let features = table.numeric_matrix(&Spambase::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[4601, 57]);
///
/// // Reach one column by name.
/// let class = table.column(Spambase::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(class.len(), 4601);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("word_freq_make") {
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
/// assert_eq!(owned.n_samples(), 4601);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 4601);
/// ```
#[derive(Debug)]
pub struct Spambase {
    dataset: Dataset<Table, DatasetError>,
}

impl Spambase {
    /// The columns the source designates as the model inputs, in source order.
    /// The names come from `spambase.names`.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "word_freq_make",
        "word_freq_address",
        "word_freq_all",
        "word_freq_3d",
        "word_freq_our",
        "word_freq_over",
        "word_freq_remove",
        "word_freq_internet",
        "word_freq_order",
        "word_freq_mail",
        "word_freq_receive",
        "word_freq_will",
        "word_freq_people",
        "word_freq_report",
        "word_freq_addresses",
        "word_freq_free",
        "word_freq_business",
        "word_freq_email",
        "word_freq_you",
        "word_freq_credit",
        "word_freq_your",
        "word_freq_font",
        "word_freq_000",
        "word_freq_money",
        "word_freq_hp",
        "word_freq_hpl",
        "word_freq_george",
        "word_freq_650",
        "word_freq_lab",
        "word_freq_labs",
        "word_freq_telnet",
        "word_freq_857",
        "word_freq_data",
        "word_freq_415",
        "word_freq_85",
        "word_freq_technology",
        "word_freq_1999",
        "word_freq_parts",
        "word_freq_pm",
        "word_freq_direct",
        "word_freq_cs",
        "word_freq_meeting",
        "word_freq_original",
        "word_freq_project",
        "word_freq_re",
        "word_freq_edu",
        "word_freq_table",
        "word_freq_conference",
        "char_freq_;",
        "char_freq_(",
        "char_freq_[",
        "char_freq_!",
        "char_freq_$",
        "char_freq_#",
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "class";

    /// Create a new Spambase instance without loading data.
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
    /// - `Self` - a `Spambase` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Spambase {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the Spambase dataset.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            SPAMBASE_FILENAME,
            SPAMBASE_DATASET_NAME,
            Some(SPAMBASE_SHA256),
            |temp_path| {
                download_to_with_retries(
                    SPAMBASE_DATA_URL,
                    temp_path,
                    Some(SPAMBASE_ZIP_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                unzip(&temp_path.join(SPAMBASE_ZIP_FILENAME), temp_path)?;
                Ok(temp_path.join(SPAMBASE_SOURCE_FILENAME))
            },
        )?;

        // `spambase.data` is a headerless comma-separated file: every line is a
        // record of 57 numeric features followed by the class code.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        let mut features: Vec<f64> = Vec::with_capacity(N_SAMPLES * N_FEATURES);
        let mut classes: Vec<String> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(SPAMBASE_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            // Skip blank lines, for example a trailing newline.
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    SPAMBASE_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // 57 numeric features.
            for (col, field) in record.iter().take(N_FEATURES).enumerate() {
                let value: f64 = field.trim().parse().map_err(|e| {
                    DatasetError::parse_failed(
                        SPAMBASE_DATASET_NAME,
                        &format!("feature_{}", col),
                        line_num,
                        e,
                    )
                })?;
                features.push(value);
            }

            // Map the source's nominal code to a readable label.
            let label = match record[LABEL_COLUMN].trim() {
                "0" => "ham",
                "1" => "spam",
                other => {
                    return Err(DatasetError::invalid_value(
                        SPAMBASE_DATASET_NAME,
                        "class",
                        other,
                        line_num,
                    ));
                }
            };
            classes.push(label.to_string());
        }

        // The source lists the 57 statistics first and the class last.
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

        Table::new(SPAMBASE_DATASET_NAME, columns)
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 4601 samples and 58
    ///   columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, an
    ///   unknown class)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`Spambase::data`], this method never runs the loader. If the data
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
    /// changes stay in the cache. Later calls to [`Spambase::data`] or
    /// [`Spambase::get_data`] see them.
    ///
    /// Like [`Spambase::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`Spambase::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 4601 samples and 58 columns.
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
    /// - `Table` - the owned table of 4601 samples and 58 columns.
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

impl_ml_dataset!(Spambase, "spambase");
