//! MNIST database of handwritten digits.
//!
//! 70,000 grayscale images of handwritten digits, each 28×28 pixels, split into
//! a 60,000-image training partition and a 10,000-image test partition. The task
//! is to recognize which digit (`0`-`9`) an image shows. MNIST is the standard
//! entry benchmark for image classification.
//! [`digits`](crate::dataset::digits) holds the same task at 8×8.
//!
//! **Columns (2):**
//!
//! | Name     | Type      | Description                                                                        |
//! |----------|-----------|------------------------------------------------------------------------------------|
//! | `pixels` | `Bytes`   | 784 pixel intensities per image, one 28×28 image flattened in row-major order, each value in `0..=255` |
//! | `digit`  | `Integer` | the digit the image shows, one of `0`-`9`                                          |
//!
//! The source designates `pixels` as the input ([`Mnist::FEATURE_NAMES`](crate::Mnist::FEATURE_NAMES)) and
//! `digit` as the label ([`Mnist::TARGET`](crate::Mnist::TARGET)).
//!
//! **Samples:**
//! - Training partition: 60,000
//! - Test partition: 10,000
//! - Both: 70,000
//!
//! **Application:** Multi-class image classification / handwritten digit recognition
//!
//! **Missing values:** none.
//!
//! **Source:** LeCun, Y., Cortes, C., and Burges, C. J. C. The MNIST database of
//! handwritten digits. This loader reads the four canonical IDX files from the
//! `ossci-datasets` mirror. <http://yann.lecun.com/exdb/mnist/>

use super::idx::{self, Partition};
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError};

/// The name of the dataset.
const MNIST_DATASET_NAME: &str = "mnist";

/// Number of samples in the training partition.
const N_TRAIN_SAMPLES: usize = 60_000;

/// Number of samples in the test partition.
const N_TEST_SAMPLES: usize = 10_000;

/// The training partition: 60,000 images.
static TRAIN_PARTITION: Partition = Partition {
    images_url: "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    images_filename: "train-images-idx3-ubyte",
    images_sha256: "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
    labels_url: "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    labels_filename: "train-labels-idx1-ubyte",
    labels_sha256: "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
    n_samples: N_TRAIN_SAMPLES,
};

/// The test partition: 10,000 images.
static TEST_PARTITION: Partition = Partition {
    images_url: "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    images_filename: "t10k-images-idx3-ubyte",
    images_sha256: "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
    labels_url: "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
    labels_filename: "t10k-labels-idx1-ubyte",
    labels_sha256: "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
    n_samples: N_TEST_SAMPLES,
};

/// Subset selector: the training partition (60,000 images).
const SUBSET_TRAIN: &[&Partition] = &[&TRAIN_PARTITION];

/// Subset selector: the test partition (10,000 images).
const SUBSET_TEST: &[&Partition] = &[&TEST_PARTITION];

/// Subset selector: both partitions (70,000 images, train followed by test).
const SUBSET_ALL: &[&Partition] = &[&TRAIN_PARTITION, &TEST_PARTITION];

/// A struct that represents the MNIST dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// MNIST holds 70,000 grayscale images of handwritten digits, each 28×28 pixels,
/// with the digit each one shows. The source built it from two NIST databases,
/// so the digits of the training partition and the test partition come from
/// disjoint groups of writers. The task is to recognize the digit. It is the
/// standard entry benchmark for image classification.
///
/// # Subsets
///
/// The source ships two partitions, and three constructors select them:
///
/// - [`Mnist::new`]: the training partition, 60,000 images
/// - [`Mnist::new_test`]: the test partition, 10,000 images
/// - [`Mnist::new_all`]: both, 70,000 images, train followed by test
///
/// Keep the two partitions apart to compare a result with published work. The
/// standard protocol trains on the 60,000 and reports on the 10,000.
///
/// Each partition caches its own two files, so an instance downloads only what
/// its subset needs.
///
/// # Columns
///
/// | Name     | Type      | Description                                                                        |
/// |----------|-----------|------------------------------------------------------------------------------------|
/// | `pixels` | `Bytes`   | 784 pixel intensities per image, one 28×28 image flattened in row-major order, each value in `0..=255` |
/// | `digit`  | `Integer` | the digit the image shows, one of `0`-`9`                                          |
///
/// The source designates `pixels` as the input ([`Mnist::FEATURE_NAMES`]) and
/// `digit` as the label ([`Mnist::TARGET`]).
///
/// Missing values: none.
///
/// In the `pixels` column, `0` is the background and `255` is the darkest ink.
/// The column holds one row of 784 bytes per image. A view of that row shaped
/// `(28, 28)` reads the same bytes, at no copy.
///
/// The `digit` classes are close to balanced. The training partition ranges from
/// 5,421 images of `5` to 6,742 images of `1`.
///
/// # Source format
///
/// The source ships four gzip-compressed IDX files, one image file and one label
/// file per partition. IDX is a binary format: a big-endian header of 4-byte
/// integers, then the raw bytes. The storage directory holds each file
/// **decompressed**, under the name the source gives it.
///
/// See more information at <http://yann.lecun.com/exdb/mnist/>.
///
/// # Citation
///
/// LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). "Gradient-based
/// learning applied to document recognition." *Proceedings of the IEEE*, 86(11),
/// 2278-2324. <https://doi.org/10.1109/5.726791>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::Mnist;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./mnist";
///
/// let mut dataset = Mnist::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 60000);
/// assert_eq!(table.n_columns(), 2);
///
/// // The `pixels` column holds one 784-byte row per image.
/// let pixels = table.column("pixels").unwrap().as_bytes().unwrap();
/// assert_eq!(pixels.shape(), &[60000, 784]);
///
/// // A (n_samples, 28, 28) view reads the same bytes, at no copy.
/// let images = pixels.view().into_shape_with_order((60000, 28, 28)).unwrap();
/// assert_eq!(images.shape(), &[60000, 28, 28]);
///
/// // Ask for the feature matrix when you want it. The pixels become `f64`.
/// let features = table.numeric_matrix(&Mnist::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[60000, 784]);
///
/// // Scale the pixels to [0, 1] for a model.
/// let scaled = features.mapv(|pixel| pixel / 255.0);
/// assert_eq!(scaled.shape(), &[60000, 784]);
///
/// // The `digit` column holds the label of each image.
/// let digits = table.column(Mnist::TARGET).unwrap().as_integer().unwrap();
/// assert_eq!(digits.len(), 60000);
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut("pixels") {
///         if let dataset_ml::ColumnData::Bytes(values) = column.data_mut() {
///             values[[0, 0]] = 255;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 60000);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 60000);
/// ```
#[derive(Debug)]
pub struct Mnist {
    dataset: Dataset<Table, DatasetError>,
}

impl Mnist {
    /// The column the source designates as the model input.
    pub const FEATURE_NAMES: [&'static str; 1] = ["pixels"];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "digit";

    /// Create a new Mnist instance for the **training** partition (60,000
    /// images) without loading data.
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
    /// - `Self` - a `Mnist` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, SUBSET_TRAIN)
    }

    /// Create a new Mnist instance for the **test** partition (10,000 images)
    /// without loading data.
    ///
    /// See [`Mnist::new`] for the loading semantics.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - The directory that stores the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - a `Mnist` instance ready for lazy loading.
    pub fn new_test(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, SUBSET_TEST)
    }

    /// Create a new Mnist instance for **all** 70,000 images (the training
    /// partition followed by the test partition) without loading data.
    ///
    /// See [`Mnist::new`] for the loading semantics.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - The directory that stores the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - a `Mnist` instance ready for lazy loading.
    pub fn new_all(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, SUBSET_ALL)
    }

    /// Construct an instance whose loader reads the given partitions.
    fn with_subset(storage_dir: &str, subset: &'static [&'static Partition]) -> Self {
        Mnist {
            dataset: Dataset::new(storage_dir, move |dir| Self::load_data(dir, subset)),
        }
    }

    /// Get and parse the MNIST dataset for the requested subset.
    fn load_data(dir: &str, subset: &'static [&'static Partition]) -> Result<Table, DatasetError> {
        let (pixels, labels) = idx::load_partitions(dir, MNIST_DATASET_NAME, subset)?;

        Table::new(
            MNIST_DATASET_NAME,
            vec![
                Column::new(Self::FEATURE_NAMES[0], ColumnData::Bytes(pixels)),
                Column::new(Self::TARGET, ColumnData::Integer(labels.mapv(i64::from))),
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
    /// - `&Table` - reference to the cached table of 2 columns. It holds 60,000,
    ///   10,000, or 70,000 samples, by the constructor you used.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File decompression or I/O operations fail
    /// - The IDX header holds an unexpected magic number or image size
    /// - The file holds a different number of images, pixels, or labels than its
    ///   header states
    /// - A label falls outside `0..=9`
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`Mnist::data`], this method never runs the loader. If the data has
    /// not loaded yet, it returns `None` instead of downloading and parsing it.
    /// Use this method when you want the data only if it is already cached. This
    /// skips the cost of a download and a parse.
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
    /// changes stay in the cache. Later calls to [`Mnist::data`] or
    /// [`Mnist::get_data`] see them.
    ///
    /// Like [`Mnist::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`Mnist::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 2 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, or a header
    /// or length check).
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
    /// - `Table` - the owned table of 2 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, or a header
    /// or length check).
    pub fn take_data(&mut self) -> Result<Table, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(Mnist, "mnist");
