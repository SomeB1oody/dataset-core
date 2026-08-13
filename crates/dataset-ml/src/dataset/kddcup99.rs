//! KDD Cup 1999 network-intrusion dataset.
//!
//! The KDD Cup 1999 dataset is for network-intrusion detection. It is identical
//! to what scikit-learn exposes through `fetch_kddcup99`. Each sample is a single
//! network connection described by 41 features. The task is to classify the
//! connection as `normal.` traffic or as one of 22 attack types.
//!
//! Like scikit-learn, this loader has two partitions:
//! - [`Kddcup99::new`](crate::Kddcup99::new): the **10% subset** (494,021
//!   samples), scikit-learn's default (`fetch_kddcup99(percent10=True)`).
//! - [`Kddcup99::new_full`](crate::Kddcup99::new_full): the **full set**
//!   (4,898,431 samples), `fetch_kddcup99(percent10=False)`.
//!
//! Both partitions share the same 42 columns and the same 23 connection
//! classes. They differ only in sample count and in the upstream source file.
//!
//! **Columns (42):**
//!
//! | Name | Type | Description |
//! |------|------|-------------|
//! | `duration` | `Numeric` | length of the connection in seconds |
//! | `protocol_type` | `String` | `tcp`, `udp`, or `icmp` |
//! | `service` | `String` | destination service, such as `http` or `smtp` |
//! | `flag` | `String` | connection status flag, such as `SF`, `S0`, or `REJ` |
//! | `src_bytes` | `Numeric` | bytes sent from the source to the destination |
//! | `dst_bytes` | `Numeric` | bytes sent from the destination to the source |
//! | `land` | `Numeric` | `1` if the source and destination address and port are equal |
//! | `wrong_fragment` | `Numeric` | number of wrong fragments |
//! | `urgent` | `Numeric` | number of urgent packets |
//! | `hot` | `Numeric` | number of hot indicators |
//! | `num_failed_logins` | `Numeric` | number of failed login attempts |
//! | `logged_in` | `Numeric` | `1` if the login succeeded |
//! | `num_compromised` | `Numeric` | number of compromised conditions |
//! | `root_shell` | `Numeric` | `1` if the connection obtained a root shell |
//! | `su_attempted` | `Numeric` | `1` if the connection attempted the `su root` command |
//! | `num_root` | `Numeric` | number of root accesses |
//! | `num_file_creations` | `Numeric` | number of file creation operations |
//! | `num_shells` | `Numeric` | number of shell prompts |
//! | `num_access_files` | `Numeric` | number of operations on access control files |
//! | `num_outbound_cmds` | `Numeric` | number of outbound commands in an FTP session |
//! | `is_host_login` | `Numeric` | `1` if the login belongs to the host list |
//! | `is_guest_login` | `Numeric` | `1` if the login is a guest login |
//! | `count` | `Numeric` | connections to the same host in the past two seconds |
//! | `srv_count` | `Numeric` | connections to the same service in the past two seconds |
//! | `serror_rate` | `Numeric` | fraction of those connections with a SYN error |
//! | `srv_serror_rate` | `Numeric` | fraction of the same-service connections with a SYN error |
//! | `rerror_rate` | `Numeric` | fraction of those connections with a REJ error |
//! | `srv_rerror_rate` | `Numeric` | fraction of the same-service connections with a REJ error |
//! | `same_srv_rate` | `Numeric` | fraction of those connections to the same service |
//! | `diff_srv_rate` | `Numeric` | fraction of those connections to a different service |
//! | `srv_diff_host_rate` | `Numeric` | fraction of the same-service connections to a different host |
//! | `dst_host_count` | `Numeric` | connections to the same destination host |
//! | `dst_host_srv_count` | `Numeric` | connections to the same destination host and service |
//! | `dst_host_same_srv_rate` | `Numeric` | fraction of the destination-host connections to the same service |
//! | `dst_host_diff_srv_rate` | `Numeric` | fraction of the destination-host connections to a different service |
//! | `dst_host_same_src_port_rate` | `Numeric` | fraction of the destination-host connections from the same source port |
//! | `dst_host_srv_diff_host_rate` | `Numeric` | fraction of the same-service connections to a different host |
//! | `dst_host_serror_rate` | `Numeric` | fraction of the destination-host connections with a SYN error |
//! | `dst_host_srv_serror_rate` | `Numeric` | fraction of the same-service connections with a SYN error |
//! | `dst_host_rerror_rate` | `Numeric` | fraction of the destination-host connections with a REJ error |
//! | `dst_host_srv_rerror_rate` | `Numeric` | fraction of the same-service connections with a REJ error |
//! | `label` | `String` | connection class, with the trailing period, such as `normal.` |
//!
//! The source designates the 41 connection features as the inputs
//! ([`Kddcup99::FEATURE_NAMES`](crate::Kddcup99::FEATURE_NAMES)) and `label` as the label ([`Kddcup99::TARGET`](crate::Kddcup99::TARGET)).
//!
//! **Samples:** 494,021 (10% subset, default) or 4,898,431 (full set)
//! **Application:** Multi-class classification / network-intrusion detection
//!
//! **Missing values:** none.
//!
//! **Source:** UCI KDD Archive, via the gzip-compressed mirrors that
//! scikit-learn's `fetch_kddcup99` downloads (`kddcup.data_10_percent.gz` for the
//! subset, `kddcup.data.gz` for the full set).
//! <https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html>
//!
//! **Note on size:** the full set is large. Its decompressed source file is about
//! 743 MB. The parsed table takes several gigabytes of memory. Loading it with
//! [`Kddcup99::new_full`](crate::Kddcup99::new_full) takes noticeable time and
//! memory. The default 10% subset is about 10 times smaller.

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, gunzip};
use ndarray::Array1;
use std::fs::File;

/// Which KDD Cup 1999 partition to load.
///
/// This mirrors scikit-learn's `fetch_kddcup99(percent10=…)` switch: the default
/// is the 10% subset ([`Kddcup99::new`]), and the full set is opt-in
/// ([`Kddcup99::new_full`]). The two variants are distinct upstream files, each
/// with its own URL, sample count, and pinned SHA-256. The loader caches each
/// under a distinct filename, so both can exist in the same storage directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kddcup99Subset {
    /// The 10% subset (494,021 samples), scikit-learn's default.
    Percent10,
    /// The full set (4,898,431 samples).
    Full,
}

impl Kddcup99Subset {
    /// The gzip-compressed source URL (the figshare mirror scikit-learn uses).
    const fn url(self) -> &'static str {
        match self {
            // ARCHIVE_10_PERCENT in scikit-learn's `_kddcup99.py`.
            Kddcup99Subset::Percent10 => "https://ndownloader.figshare.com/files/5976042",
            // ARCHIVE in scikit-learn's `_kddcup99.py`.
            Kddcup99Subset::Full => "https://ndownloader.figshare.com/files/5976045",
        }
    }

    /// The name used for the downloaded gzip archive inside the temp directory.
    const fn gz_filename(self) -> &'static str {
        match self {
            Kddcup99Subset::Percent10 => "kddcup99_10_percent.data.gz",
            Kddcup99Subset::Full => "kddcup99.data.gz",
        }
    }

    /// The name of the final cached (decompressed) dataset file. It is distinct
    /// for each variant, so the 10% subset and the full set never collide in one
    /// directory.
    const fn filename(self) -> &'static str {
        match self {
            Kddcup99Subset::Percent10 => "kddcup99_10_percent.csv",
            Kddcup99Subset::Full => "kddcup99.csv",
        }
    }

    /// The SHA256 hash of the **decompressed** dataset file (not the `.gz`,
    /// because the cached file is the decompressed one).
    const fn sha256(self) -> &'static str {
        match self {
            Kddcup99Subset::Percent10 => {
                "f8c8267ebcd9c0ed1fd7d6277fe5bfff8732e9b7db8e61b873542b2a534b6f9a"
            }
            Kddcup99Subset::Full => {
                "3ec2301a9a5d81b40937ba155b4713a77b60e85b89f0423257e58d566aa979fb"
            }
        }
    }

    /// The expected number of samples, used only to pre-allocate the parse buffers.
    const fn n_samples(self) -> usize {
        match self {
            Kddcup99Subset::Percent10 => 494_021,
            Kddcup99Subset::Full => 4_898_431,
        }
    }
}

/// The name of the dataset.
const KDDCUP99_DATASET_NAME: &str = "kddcup99";

/// The number of categorical columns (`protocol_type`, `service`, `flag`).
const N_STRING_FEATURES: usize = 3;

/// The number of numeric columns.
const N_NUMERIC_FEATURES: usize = 38;

/// The number of columns per CSV record (41 features + 1 label).
const N_COLUMNS: usize = N_STRING_FEATURES + N_NUMERIC_FEATURES + 1;

/// The categorical columns, as `(source column index, name)`.
const STRING_COLUMNS: [(usize, &str); N_STRING_FEATURES] =
    [(1, "protocol_type"), (2, "service"), (3, "flag")];

/// The number of feature columns (3 string features and 38 numeric features).
const N_FEATURES: usize = N_STRING_FEATURES + N_NUMERIC_FEATURES;

/// The 0-based source-column index of the label.
const LABEL_COLUMN: usize = 41;

/// The numeric columns, as `(source column index, name)`. These are every column
/// except the three categorical ones (indices 1, 2, 3) and the label (index 41).
const NUMERIC_COLUMNS: [(usize, &str); N_NUMERIC_FEATURES] = [
    (0, "duration"),
    (4, "src_bytes"),
    (5, "dst_bytes"),
    (6, "land"),
    (7, "wrong_fragment"),
    (8, "urgent"),
    (9, "hot"),
    (10, "num_failed_logins"),
    (11, "logged_in"),
    (12, "num_compromised"),
    (13, "root_shell"),
    (14, "su_attempted"),
    (15, "num_root"),
    (16, "num_file_creations"),
    (17, "num_shells"),
    (18, "num_access_files"),
    (19, "num_outbound_cmds"),
    (20, "is_host_login"),
    (21, "is_guest_login"),
    (22, "count"),
    (23, "srv_count"),
    (24, "serror_rate"),
    (25, "srv_serror_rate"),
    (26, "rerror_rate"),
    (27, "srv_rerror_rate"),
    (28, "same_srv_rate"),
    (29, "diff_srv_rate"),
    (30, "srv_diff_host_rate"),
    (31, "dst_host_count"),
    (32, "dst_host_srv_count"),
    (33, "dst_host_same_srv_rate"),
    (34, "dst_host_diff_srv_rate"),
    (35, "dst_host_same_src_port_rate"),
    (36, "dst_host_srv_diff_host_rate"),
    (37, "dst_host_serror_rate"),
    (38, "dst_host_srv_serror_rate"),
    (39, "dst_host_rerror_rate"),
    (40, "dst_host_srv_rerror_rate"),
];

/// A struct that represents the KDD Cup 1999 dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// Construct it with [`Kddcup99::new`] for scikit-learn's default 10% subset
/// (494,021 samples), or with [`Kddcup99::new_full`] for the full set
/// (4,898,431 samples). Both share the columns below and differ only in sample
/// count.
///
/// # About Dataset
///
/// The KDD Cup 1999 dataset was built from the DARPA 1998 intrusion-detection
/// evaluation. Each sample is a network connection summarized by 41 features.
/// The source labels each sample `normal.` or as one of 22 attack types. The
/// attack types fall into four categories: DoS, R2L, U2R, and probing. This is
/// the same data scikit-learn exposes through `fetch_kddcup99`.
///
/// # Columns
///
/// | Name | Type | Description |
/// |------|------|-------------|
/// | `duration` | `Numeric` | length of the connection in seconds |
/// | `protocol_type` | `String` | `tcp`, `udp`, or `icmp` |
/// | `service` | `String` | destination service, one of about 70 values, such as `http` |
/// | `flag` | `String` | connection status flag, one of 11 values, such as `SF` |
/// | `src_bytes` | `Numeric` | bytes sent from the source to the destination |
/// | `dst_bytes` | `Numeric` | bytes sent from the destination to the source |
/// | `land` | `Numeric` | `1` if the source and destination address and port are equal |
/// | `wrong_fragment` | `Numeric` | number of wrong fragments |
/// | `urgent` | `Numeric` | number of urgent packets |
/// | `hot` | `Numeric` | number of hot indicators |
/// | `num_failed_logins` | `Numeric` | number of failed login attempts |
/// | `logged_in` | `Numeric` | `1` if the login succeeded |
/// | `num_compromised` | `Numeric` | number of compromised conditions |
/// | `root_shell` | `Numeric` | `1` if the connection obtained a root shell |
/// | `su_attempted` | `Numeric` | `1` if the connection attempted the `su root` command |
/// | `num_root` | `Numeric` | number of root accesses |
/// | `num_file_creations` | `Numeric` | number of file creation operations |
/// | `num_shells` | `Numeric` | number of shell prompts |
/// | `num_access_files` | `Numeric` | number of operations on access control files |
/// | `num_outbound_cmds` | `Numeric` | number of outbound commands in an FTP session |
/// | `is_host_login` | `Numeric` | `1` if the login belongs to the host list |
/// | `is_guest_login` | `Numeric` | `1` if the login is a guest login |
/// | `count` | `Numeric` | connections to the same host in the past two seconds |
/// | `srv_count` | `Numeric` | connections to the same service in the past two seconds |
/// | `serror_rate` | `Numeric` | fraction of those connections with a SYN error |
/// | `srv_serror_rate` | `Numeric` | fraction of the same-service connections with a SYN error |
/// | `rerror_rate` | `Numeric` | fraction of those connections with a REJ error |
/// | `srv_rerror_rate` | `Numeric` | fraction of the same-service connections with a REJ error |
/// | `same_srv_rate` | `Numeric` | fraction of those connections to the same service |
/// | `diff_srv_rate` | `Numeric` | fraction of those connections to a different service |
/// | `srv_diff_host_rate` | `Numeric` | fraction of the same-service connections to a different host |
/// | `dst_host_count` | `Numeric` | connections to the same destination host |
/// | `dst_host_srv_count` | `Numeric` | connections to the same destination host and service |
/// | `dst_host_same_srv_rate` | `Numeric` | fraction of the destination-host connections to the same service |
/// | `dst_host_diff_srv_rate` | `Numeric` | fraction of the destination-host connections to a different service |
/// | `dst_host_same_src_port_rate` | `Numeric` | fraction of the destination-host connections from the same source port |
/// | `dst_host_srv_diff_host_rate` | `Numeric` | fraction of the same-service connections to a different host |
/// | `dst_host_serror_rate` | `Numeric` | fraction of the destination-host connections with a SYN error |
/// | `dst_host_srv_serror_rate` | `Numeric` | fraction of the same-service connections with a SYN error |
/// | `dst_host_rerror_rate` | `Numeric` | fraction of the destination-host connections with a REJ error |
/// | `dst_host_srv_rerror_rate` | `Numeric` | fraction of the same-service connections with a REJ error |
/// | `label` | `String` | connection class, with the trailing period, such as `normal.` |
///
/// The source designates the 41 connection features as the inputs
/// ([`Kddcup99::FEATURE_NAMES`]) and `label` as the label ([`Kddcup99::TARGET`]).
///
/// The columns keep the source column order. The `label` column holds the class
/// exactly as the source distributes it, **including the trailing period** (for
/// example `"normal."`, `"smurf."`, `"neptune."`). This matches scikit-learn's
/// `fetch_kddcup99` target. There are 23 distinct values (`normal.` plus 22
/// attack types).
///
/// Missing values: none.
///
/// See more information at
/// <https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html>
///
/// # Citation
///
/// "KDD Cup 1999 Data," UCI KDD Archive, 1999. \[Online\].
/// Available: <https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Kddcup99;
///
/// let download_dir = "./kddcup99"; // the loader creates the directory if it does not exist
///
/// // `new` loads the 10% subset (494,021 samples). Use `new_full` for the
/// // full 4,898,431-sample set with the same columns.
/// let mut dataset = Kddcup99::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 494021);
/// assert_eq!(table.n_columns(), 42);
///
/// // The 41 features mix types. Read a string feature by name.
/// let protocol_type = table.column("protocol_type").unwrap().as_string().unwrap();
/// assert_eq!(protocol_type.len(), 494021);
///
/// // Reach one column by name.
/// let duration = table.column("duration").unwrap().as_numeric().unwrap();
/// assert_eq!(duration.len(), 494021);
/// let label = table.column(Kddcup99::TARGET).unwrap().as_string().unwrap();
/// assert_eq!(label.len(), 494021);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("duration") {
///         if let dataset_ml::ColumnData::Numeric(values) = column.data_mut() {
///             values[0] = 0.0;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 494021);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 494021);
/// ```
#[derive(Debug)]
pub struct Kddcup99 {
    dataset: Dataset<Table, DatasetError>,
}

impl Kddcup99 {
    /// The columns the source designates as the model inputs, in source order.
    pub const FEATURE_NAMES: [&'static str; N_FEATURES] = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "label";

    /// Create a new Kddcup99 instance for the **10% subset** without loading data.
    ///
    /// This matches scikit-learn's default `fetch_kddcup99(percent10=True)`: the
    /// 494,021-sample subset. For the full 4,898,431-sample set, use
    /// [`Kddcup99::new_full`].
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
    /// - `Self` - `Kddcup99` instance (10% subset) ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, Kddcup99Subset::Percent10)
    }

    /// Create a new Kddcup99 instance for the **full set** without loading data.
    ///
    /// This matches scikit-learn's `fetch_kddcup99(percent10=False)`: the full
    /// 4,898,431-sample set. For the smaller default subset, use [`Kddcup99::new`].
    ///
    /// The dataset loads lazily, on your first call to a data accessor method.
    /// This is a lightweight operation that only stores the storage directory.
    ///
    /// **Note:** the full set is large. The decompressed source is about 743 MB.
    /// The parsed table takes several gigabytes of memory. Loading it takes
    /// noticeable time and memory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - The directory that stores the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - `Kddcup99` instance (full set) ready for lazy loading.
    pub fn new_full(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, Kddcup99Subset::Full)
    }

    /// Create a Kddcup99 instance bound to a specific source partition. The loader
    /// closure captures the chosen subset, so `load_data` knows which upstream
    /// file (URL, cached filename, SHA-256, sample count) to use.
    fn with_subset(storage_dir: &str, subset: Kddcup99Subset) -> Self {
        Kddcup99 {
            dataset: Dataset::new(storage_dir, move |dir| Self::load_data(dir, subset)),
        }
    }

    /// Get and parse the chosen KDD Cup 1999 partition.
    fn load_data(dir: &str, subset: Kddcup99Subset) -> Result<Table, DatasetError> {
        let gz_filename = subset.gz_filename();
        let filename = subset.filename();

        // The closure downloads the gzip-compressed source and decompresses
        // it into the plain comma-separated cached file.
        let file_path = acquire_dataset(
            dir,
            filename,
            KDDCUP99_DATASET_NAME,
            Some(subset.sha256()),
            |temp_path| {
                download_to_with_retries(
                    subset.url(),
                    temp_path,
                    Some(gz_filename),
                    DOWNLOAD_RETRIES,
                )?;
                let gz_path = temp_path.join(gz_filename);
                let csv_path = temp_path.join(filename);
                gunzip(&gz_path, &csv_path)?;
                Ok(csv_path)
            },
        )?;

        // `kddcup.data` is a headerless comma-separated file: every line is a
        // record of 41 features (3 categorical + 38 numeric) followed by the label.
        // The schema mixes string and numeric columns, so the loader parses raw
        // positional `StringRecord`s instead of deserializing into a named struct.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        // The code pre-allocates the buffers for the known sample count to avoid
        // repeated growth. Parsing still works for any actual row count.
        let n_expected = subset.n_samples();
        let mut string_values: Vec<Vec<String>> = STRING_COLUMNS
            .iter()
            .map(|_| Vec::with_capacity(n_expected))
            .collect();
        let mut numeric_values: Vec<Vec<f64>> = NUMERIC_COLUMNS
            .iter()
            .map(|_| Vec::with_capacity(n_expected))
            .collect();
        let mut labels: Vec<String> = Vec::with_capacity(n_expected);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(KDDCUP99_DATASET_NAME, e))?;
            let line_num = idx + 1; // headerless file, lines are 1-indexed

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    KDDCUP99_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            // Categorical columns.
            for (values, &(col, _name)) in string_values.iter_mut().zip(STRING_COLUMNS.iter()) {
                values.push(record[col].trim().to_string());
            }

            // Numeric columns.
            for (values, &(col, name)) in numeric_values.iter_mut().zip(NUMERIC_COLUMNS.iter()) {
                let value: f64 = record[col].trim().parse().map_err(|e| {
                    DatasetError::parse_failed(KDDCUP99_DATASET_NAME, name, line_num, e)
                })?;
                values.push(value);
            }

            // Label: kept verbatim (including the trailing period).
            let raw_label = record[LABEL_COLUMN].trim();
            if raw_label.is_empty() {
                return Err(DatasetError::invalid_value(
                    KDDCUP99_DATASET_NAME,
                    "label",
                    raw_label,
                    line_num,
                ));
            }
            labels.push(raw_label.to_string());
        }

        // Each entry keeps its source column index. The sort then restores the
        // source column order.
        let mut columns: Vec<(usize, Column)> = Vec::with_capacity(N_COLUMNS);
        for (values, &(col, name)) in string_values.into_iter().zip(STRING_COLUMNS.iter()) {
            columns.push((
                col,
                Column::new(name, ColumnData::String(Array1::from_vec(values))),
            ));
        }
        for (values, &(col, name)) in numeric_values.into_iter().zip(NUMERIC_COLUMNS.iter()) {
            columns.push((
                col,
                Column::new(name, ColumnData::Numeric(Array1::from_vec(values))),
            ));
        }
        columns.push((
            LABEL_COLUMN,
            Column::new(Self::TARGET, ColumnData::String(Array1::from_vec(labels))),
        ));
        columns.sort_by_key(|entry| entry.0);

        Table::new(
            KDDCUP99_DATASET_NAME,
            columns.into_iter().map(|(_, column)| column).collect(),
        )
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 42 columns. It holds 494,021
    ///   samples for the 10% subset, or 4,898,431 samples for the full set.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File decompression or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or empty label)
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`Kddcup99::data`], this method never runs the loader. If the data
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
    /// changes stay in the cache. Later calls to [`Kddcup99::data`] or
    /// [`Kddcup99::get_data`] see them.
    ///
    /// Like [`Kddcup99::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`Kddcup99::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 42 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or an
    /// empty label).
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
    /// - `Table` - the owned table of 42 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or an
    /// empty label).
    pub fn take_data(&mut self) -> Result<Table, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(Kddcup99, "kddcup99");
