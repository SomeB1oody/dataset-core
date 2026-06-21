//! KDD Cup 1999 network-intrusion dataset.
//!
//! The KDD Cup 1999 dataset for network-intrusion detection, identical to what
//! scikit-learn exposes through `fetch_kddcup99`. Each sample is a single network
//! connection described by 41 features, and the task is to classify the connection
//! as either `normal.` traffic or one of 22 attack types.
//!
//! Like scikit-learn, this loader has two partitions:
//! - [`Kddcup99::new`] — the **10% subset** (494,021 samples), scikit-learn's
//!   default (`fetch_kddcup99(percent10=True)`).
//! - [`Kddcup99::new_full`] — the **full set** (4,898,431 samples),
//!   `fetch_kddcup99(percent10=False)`.
//!
//! Both partitions share the same 41-feature schema and the same 23 connection
//! classes; they differ only in sample count (and the upstream source file).
//!
//! **Features (41, mixed):**
//! - String features (3): `protocol_type` (`tcp`/`udp`/`icmp`), `service` (~70
//!   network services, e.g. `http`, `smtp`), `flag` (11 connection-status flags,
//!   e.g. `SF`, `S0`, `REJ`)
//! - Numeric features (38): `duration`, `src_bytes`, `dst_bytes`, the various
//!   per-connection counters and rates (see the struct docs for the full list)
//!
//! **Target:** `label` - the connection class, one of 23 values including the
//! trailing period exactly as the source distributes them (e.g. `normal.`,
//! `smurf.`, `neptune.`), matching scikit-learn's `fetch_kddcup99` target.
//!
//! **Samples:** 494,021 (10% subset, default) or 4,898,431 (full set)
//! **Application:** Multi-class classification / network-intrusion detection
//!
//! **Source:** UCI KDD Archive, via the gzip-compressed mirrors that
//! scikit-learn's `fetch_kddcup99` downloads (`kddcup.data_10_percent.gz` for the
//! subset, `kddcup.data.gz` for the full set).
//! <https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html>
//!
//! **Note on size:** the full set is large — its decompressed source file is
//! ~743 MB and the parsed in-memory representation is several gigabytes (the
//! `(4898431, 38)` numeric matrix alone is ~1.5 GB), so [`Kddcup99::new_full`]
//! takes noticeable time and memory. The default 10% subset is ~10× smaller.

use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to, gunzip};
use ndarray::{Array1, Array2};
use std::fs::File;

/// Which KDD Cup 1999 partition to load.
///
/// Mirrors scikit-learn's `fetch_kddcup99(percent10=…)` switch: the default is
/// the 10% subset ([`Kddcup99::new`]), and the full set is opt-in
/// ([`Kddcup99::new_full`]). The two variants are distinct upstream files (with
/// their own URL, sample count, and pinned SHA-256), cached under distinct
/// filenames so both can live in the same storage directory.
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

    /// The name the downloaded gzip archive is saved under in the temp directory.
    const fn gz_filename(self) -> &'static str {
        match self {
            Kddcup99Subset::Percent10 => "kddcup99_10_percent.data.gz",
            Kddcup99Subset::Full => "kddcup99.data.gz",
        }
    }

    /// The name of the final cached (decompressed) dataset file. Distinct per
    /// variant so the 10% subset and the full set never collide in one directory.
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

/// The name of the dataset
const KDDCUP99_DATASET_NAME: &str = "kddcup99";

/// The number of string (categorical) features per sample (`protocol_type`,
/// `service`, `flag`).
const N_STRING_FEATURES: usize = 3;

/// The number of numeric features per sample.
const N_NUMERIC_FEATURES: usize = 38;

/// The number of columns per CSV record (41 features + 1 label).
const N_COLUMNS: usize = N_STRING_FEATURES + N_NUMERIC_FEATURES + 1;

/// The 0-based source-column indices of the three string (categorical) features,
/// in output column order: `protocol_type`, `service`, `flag`.
const STRING_COLUMNS: [usize; N_STRING_FEATURES] = [1, 2, 3];

/// The 0-based source-column index of the label.
const LABEL_COLUMN: usize = 41;

/// The 0-based source-column index and name of each numeric feature, in output
/// column order. These are every column except the three categorical ones
/// (indices 1, 2, 3) and the label (index 41).
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

/// Type alias for the KDD Cup 1999 dataset: (string features, numeric features, labels).
type Kddcup99Data = (Array2<String>, Array2<f64>, Array1<String>);

/// A struct representing the KDD Cup 1999 dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// Construct it with [`Kddcup99::new`] for scikit-learn's default 10% subset
/// (494,021 samples) or [`Kddcup99::new_full`] for the full set (4,898,431
/// samples); both share the schema below and differ only in sample count. The
/// shapes in this documentation use `n_samples` for that row count.
///
/// # About Dataset
///
/// The KDD Cup 1999 dataset was built from the DARPA 1998 intrusion-detection
/// evaluation: each sample is a network connection summarized by 41 features,
/// labelled either `normal.` or as one of 22 attack types (grouped into DoS, R2L,
/// U2R, and probing categories). This is the same data scikit-learn exposes
/// through `fetch_kddcup99`.
///
/// # Feature columns
///
/// The 41 features are mixed-type and split across two matrices: a string
/// (categorical) matrix of shape `(n_samples, 3)` and a numeric matrix of shape
/// `(n_samples, 38)`. The dataset has no missing values.
///
/// String features (`Array2<String>`), by 0-based column:
///
/// | Column | Attribute       | Values                                            |
/// |--------|-----------------|---------------------------------------------------|
/// | `0`    | `protocol_type` | `tcp`, `udp`, `icmp`                               |
/// | `1`    | `service`       | ~70 network services (e.g. `http`, `smtp`, `ftp`) |
/// | `2`    | `flag`          | 11 status flags (e.g. `SF`, `S0`, `REJ`, `RSTR`)  |
///
/// Numeric features (`Array2<f64>`), by 0-based column:
///
/// | Columns   | Attributes                                                        |
/// |-----------|-------------------------------------------------------------------|
/// | `0`       | `duration`                                                        |
/// | `1..=2`   | `src_bytes`, `dst_bytes`                                          |
/// | `3..=5`   | `land`, `wrong_fragment`, `urgent`                               |
/// | `6..=18`  | `hot`, `num_failed_logins`, `logged_in`, `num_compromised`, `root_shell`, `su_attempted`, `num_root`, `num_file_creations`, `num_shells`, `num_access_files`, `num_outbound_cmds`, `is_host_login`, `is_guest_login` |
/// | `19..=27` | `count`, `srv_count`, `serror_rate`, `srv_serror_rate`, `rerror_rate`, `srv_rerror_rate`, `same_srv_rate`, `diff_srv_rate`, `srv_diff_host_rate` |
/// | `28..=37` | `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`, `dst_host_diff_srv_rate`, `dst_host_same_src_port_rate`, `dst_host_srv_diff_host_rate`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `dst_host_rerror_rate`, `dst_host_srv_rerror_rate` |
///
/// The numeric columns are the 38 source columns that are neither categorical
/// (`protocol_type`, `service`, `flag`) nor the label, kept in their original
/// source order.
///
/// # Labels
///
/// - `label` (shape `(n_samples,)`, in `String`): the connection class, kept exactly
///   as distributed **including the trailing period** (e.g. `"normal."`,
///   `"smurf."`, `"neptune."`), matching scikit-learn's `fetch_kddcup99` target.
///   There are 23 distinct values (`normal.` plus 22 attack types).
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
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```no_run
/// use dataset_ml::kddcup99::Kddcup99;
///
/// let download_dir = "./kddcup99"; // the code will create the directory if it doesn't exist
///
/// // `new` loads the 10% subset (494,021 samples); use `new_full` for the full
/// // 4,898,431-sample set with the same schema.
/// let mut dataset = Kddcup99::new(download_dir);
/// let (string_features, numeric_features) = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// let (string_features, numeric_features, labels) = dataset.data().unwrap(); // this is also a way to get all data
/// assert_eq!(string_features.shape(), &[494021, 3]);
/// assert_eq!(numeric_features.shape(), &[494021, 38]);
/// assert_eq!(labels.len(), 494021);
///
/// // `get_data()` borrows the cached arrays without reloading; `get_data_mut()`
/// // edits them in place — no clone, no reload, the change stays cached. Prefer
/// // this over cloning with `.to_owned()` when you only need to tweak values.
/// if let Some((_strings, numerics, labels)) = dataset.get_data_mut() {
///     numerics[[0, 0]] = 0.0;
///     labels[0] = "normal.".to_string();
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out (no `to_owned()` clone) and leaves
/// // the instance reusable — the next access reloads from the cached file.
/// let (owned_strings, owned_numerics, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_strings.shape(), &[494021, 3]);
/// assert_eq!(owned_numerics.shape(), &[494021, 38]);
/// assert_eq!(owned_labels.len(), 494021);
///
/// // `into_data()` also returns the owned arrays with no clone, but consumes the
/// // instance (use it when you are done with the dataset).
/// let (owned_strings, owned_numerics, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_strings.shape(), &[494021, 3]);
/// assert_eq!(owned_numerics.shape(), &[494021, 38]);
/// assert_eq!(owned_labels.len(), 494021);
/// ```
#[derive(Debug)]
pub struct Kddcup99 {
    dataset: Dataset<Kddcup99Data, DatasetError>,
}

impl Kddcup99 {
    /// Create a new Kddcup99 instance for the **10% subset** without loading data.
    ///
    /// This matches scikit-learn's default `fetch_kddcup99(percent10=True)`: the
    /// 494,021-sample subset. For the full 4,898,431-sample set, use
    /// [`Kddcup99::new_full`].
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
    /// - `Self` - `Kddcup99` instance (10% subset) ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, Kddcup99Subset::Percent10)
    }

    /// Create a new Kddcup99 instance for the **full set** without loading data.
    ///
    /// This matches scikit-learn's `fetch_kddcup99(percent10=False)`: the full
    /// 4,898,431-sample set. For the smaller default subset, use [`Kddcup99::new`].
    ///
    /// The dataset will be loaded lazily when you first call any data accessor method.
    /// This is a lightweight operation that only stores the storage directory.
    ///
    /// **Heads-up:** the full set is large — the decompressed source is ~743 MB and
    /// the parsed in-memory arrays are several GB (the `(4898431, 38)` numeric
    /// matrix alone is ~1.5 GB), so loading takes noticeable time and memory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory where the dataset will be stored.
    ///
    /// # Returns
    ///
    /// - `Self` - `Kddcup99` instance (full set) ready for lazy loading.
    pub fn new_full(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, Kddcup99Subset::Full)
    }

    /// Create a Kddcup99 instance bound to a specific source partition. The chosen
    /// subset is captured by the loader closure so `load_data` knows which upstream
    /// file (URL, cached filename, SHA-256, sample count) to use.
    fn with_subset(storage_dir: &str, subset: Kddcup99Subset) -> Self {
        Kddcup99 {
            dataset: Dataset::new(storage_dir, move |dir| Self::load_data(dir, subset)),
        }
    }

    /// Acquire and parse the chosen KDD Cup 1999 partition.
    fn load_data(dir: &str, subset: Kddcup99Subset) -> Result<Kddcup99Data, DatasetError> {
        let gz_filename = subset.gz_filename();
        let filename = subset.filename();

        // Prepare the dataset file: download the gzip-compressed source and
        // decompress it into the plain comma-separated cached file.
        let file_path = acquire_dataset(
            dir,
            filename,
            KDDCUP99_DATASET_NAME,
            Some(subset.sha256()),
            |temp_path| {
                download_to(subset.url(), temp_path, Some(gz_filename))?;
                let gz_path = temp_path.join(gz_filename);
                let csv_path = temp_path.join(filename);
                gunzip(&gz_path, &csv_path)?;
                Ok(csv_path)
            },
        )?;

        // `kddcup.data` is a headerless comma-separated file: every line is a
        // record of 41 features (3 categorical + 38 numeric) followed by the label.
        // The schema mixes string and numeric columns, so parse raw positional
        // `StringRecord`s rather than deserializing into a named struct.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

        // Pre-allocate for the known sample count to avoid repeatedly growing the
        // large buffers; parsing still works for any actual row count.
        let n_expected = subset.n_samples();
        let mut string_features = Vec::with_capacity(n_expected * N_STRING_FEATURES);
        let mut numeric_features = Vec::with_capacity(n_expected * N_NUMERIC_FEATURES);
        let mut labels = Vec::with_capacity(n_expected);

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

            // String (categorical) features, in output column order.
            for &col in STRING_COLUMNS.iter() {
                string_features.push(record[col].trim().to_string());
            }

            // Numeric features, in output column order.
            for &(col, name) in NUMERIC_COLUMNS.iter() {
                let value: f64 = record[col].trim().parse().map_err(|e| {
                    DatasetError::parse_failed(KDDCUP99_DATASET_NAME, name, line_num, e)
                })?;
                numeric_features.push(value);
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

        let n_samples = labels.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(KDDCUP99_DATASET_NAME));
        }

        // KDD Cup 1999 has a fixed schema of 3 string and 38 numeric features per sample.
        let string_array = Array2::from_shape_vec((n_samples, N_STRING_FEATURES), string_features)
            .map_err(|e| {
                DatasetError::array_shape_error(KDDCUP99_DATASET_NAME, "string_features", e)
            })?;

        let numeric_array =
            Array2::from_shape_vec((n_samples, N_NUMERIC_FEATURES), numeric_features).map_err(
                |e| DatasetError::array_shape_error(KDDCUP99_DATASET_NAME, "numeric_features", e),
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
    /// - `&Array2<String>` - Reference to string feature matrix with shape `(n_samples, 3)` containing:
    ///     - `protocol_type`
    ///     - `service`
    ///     - `flag`
    ///
    /// - `&Array2<f64>` - Reference to numeric feature matrix with shape `(n_samples, 38)` containing
    ///   the 38 numeric connection features (`duration`, `src_bytes`, …,
    ///   `dst_host_srv_rerror_rate`) in source order.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File decompression or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or empty label)
    /// - Dataset size doesn't match expected dimensions (n_samples samples)
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
    /// - `&Array1<String>` - Reference to label vector with shape `(n_samples,)`
    ///   containing the connection class kept verbatim including the trailing period
    ///   (e.g. `"normal."`, `"smurf."`).
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File decompression or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or empty label)
    /// - Dataset size doesn't match expected dimensions (n_samples samples)
    pub fn labels(&self) -> Result<&Array1<String>, DatasetError> {
        Ok(&self.dataset.load()?.2)
    }

    /// Get string features, numeric features and labels as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Kddcup99Data` - reference to the cached `(string features, numeric
    ///   features, labels)` tuple: string feature matrix `(n_samples, 3)`
    ///   (protocol_type, service, flag), numeric feature matrix `(n_samples, 38)`,
    ///   and label vector `(n_samples,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File decompression or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values, or empty label)
    /// - Dataset size doesn't match expected dimensions (n_samples samples)
    pub fn data(&self) -> Result<&Kddcup99Data, DatasetError> {
        self.dataset.load()
    }

    /// Get string features, numeric features and labels as references
    /// **without** triggering loading.
    ///
    /// Unlike [`Kddcup99::data`], which loads the dataset on first call, this never
    /// runs the loader: if the data has not been loaded yet, it returns `None`
    /// instead of downloading and parsing. Use it when you only want the data if
    /// it is already cached and want to avoid paying the download/parse cost
    /// otherwise.
    ///
    /// # Returns
    ///
    /// - `Some(&Kddcup99Data)` - reference to the cached `(string features, numeric
    ///   features, labels)` tuple (`(n_samples, 3)`, `(n_samples, 38)`, `(n_samples,)`),
    ///   if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data(&self) -> Option<&Kddcup99Data> {
        self.dataset.get()
    }

    /// Get mutable references to string features, numeric features, and labels
    /// for **in-place** editing.
    ///
    /// This lets you modify the cached arrays directly (e.g. normalize numeric
    /// features, encode categorical columns) with no `to_owned()` clone and without
    /// removing them from the cache: the changes persist, so later
    /// [`Kddcup99::features`], [`Kddcup99::data`], or [`Kddcup99::get_data`] calls
    /// observe them.
    ///
    /// Like [`Kddcup99::get_data`], this does **not** trigger loading: it returns
    /// `None` if the dataset has not been loaded. Call a loading accessor (e.g.
    /// [`Kddcup99::data`]) first if you need to ensure the data is present.
    ///
    /// # Returns
    ///
    /// - `Some(&mut Kddcup99Data)` - mutable reference to the cached `(string
    ///   features, numeric features, labels)` tuple (`(n_samples, 3)`,
    ///   `(n_samples, 38)`, `(n_samples,)`), if loaded.
    /// - `None` - if the dataset has not been loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut Kddcup99Data> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** string features, numeric features,
    /// and labels.
    ///
    /// Unlike [`Kddcup99::data`], which borrows the cached data, this moves it out
    /// and returns owned arrays directly — no `to_owned()` clone needed. The dataset
    /// is loaded on first access if it has not been loaded yet.
    ///
    /// This **consumes** `self`, so the instance cannot be used afterwards. If you
    /// want owned data but need to keep using the instance, use
    /// [`Kddcup99::take_data`] instead — it takes `&mut self` and leaves the instance
    /// reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array2<f64>, Array1<String>)` - owned string feature
    ///   matrix `(n_samples, 3)`, owned numeric feature matrix `(n_samples, 38)`, and
    ///   owned label vector `(n_samples,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, empty
    /// label, or a dimension mismatch).
    pub fn into_data(self) -> Result<Kddcup99Data, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** string features, numeric features, and labels out of the
    /// dataset, leaving it reusable.
    ///
    /// Like [`Kddcup99::into_data`], this returns owned arrays with no `to_owned()`
    /// clone. But instead of consuming the instance, it takes `&mut self` and moves
    /// the cached data out, resetting the instance to its unloaded state: the next
    /// accessor call (e.g. [`Kddcup99::features`] or [`Kddcup99::data`]) loads the
    /// dataset again.
    ///
    /// Use [`Kddcup99::into_data`] instead if you are done with the instance.
    ///
    /// # Returns
    ///
    /// - `(Array2<String>, Array2<f64>, Array1<String>)` - owned string feature
    ///   matrix `(n_samples, 3)`, owned numeric feature matrix `(n_samples, 38)`, and
    ///   owned label vector `(n_samples,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, empty
    /// label, or a dimension mismatch).
    pub fn take_data(&mut self) -> Result<Kddcup99Data, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}
