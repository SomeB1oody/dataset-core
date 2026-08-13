//! Fashion-MNIST dataset of clothing images.
//!
//! 70,000 grayscale images of clothing articles from the Zalando catalog, each
//! 28×28 pixels, split into a 60,000-image training partition and a
//! 10,000-image test partition. The task is to recognize which of 10 garment
//! classes an image shows.
//!
//! Fashion-MNIST matches [`mnist`](crate::dataset::mnist) in image size, class
//! count, partition sizes, and file format. Its classes overlap more, so it is
//! the harder task of the two.
//!
//! **Columns (2):**
//!
//! | Name     | Type      | Description                                                                        |
//! |----------|-----------|------------------------------------------------------------------------------------|
//! | `pixels` | `Bytes`   | 784 pixel intensities per image, one 28×28 image flattened in row-major order, each value in `0..=255` |
//! | `class`  | `Integer` | the garment class, one of `0`-`9`                                                  |
//!
//! The source designates `pixels` as the input
//! ([`FashionMnist::FEATURE_NAMES`](crate::FashionMnist::FEATURE_NAMES)) and `class` as the label
//! ([`FashionMnist::TARGET`](crate::FashionMnist::TARGET)). See
//! [`FashionMnist::CLASS_NAMES`](crate::FashionMnist::CLASS_NAMES) for the
//! name of each class code.
//!
//! **Samples:**
//! - Training partition: 60,000 (exactly 6,000 per class)
//! - Test partition: 10,000 (exactly 1,000 per class)
//! - Both: 70,000
//!
//! **Application:** Multi-class image classification / garment recognition
//!
//! **Missing values:** none.
//!
//! **Source:** Xiao, H., Rasul, K., and Vollgraf, R. (2017). Fashion-MNIST.
//! Zalando Research, released under the MIT license.
//! <https://github.com/zalandoresearch/fashion-mnist>

use super::idx::{self, N_CLASSES, Partition};
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError};

/// The name of the dataset.
const FASHION_MNIST_DATASET_NAME: &str = "fashion_mnist";

/// Number of samples in the training partition.
const N_TRAIN_SAMPLES: usize = 60_000;

/// Number of samples in the test partition.
const N_TEST_SAMPLES: usize = 10_000;

/// The training partition: 60,000 images.
static TRAIN_PARTITION: Partition = Partition {
    images_url: "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
    images_filename: "fashion-train-images-idx3-ubyte",
    images_sha256: "c59f468a2f672dc815687fe0f83887768d799fd8a3f3276145d20f83aa44d888",
    labels_url: "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
    labels_filename: "fashion-train-labels-idx1-ubyte",
    labels_sha256: "bad3541b69d912435c50bb6ba87bec294ff4f6a2e1246121d8633921760443d9",
    n_samples: N_TRAIN_SAMPLES,
};

/// The test partition: 10,000 images.
static TEST_PARTITION: Partition = Partition {
    images_url: "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
    images_filename: "fashion-t10k-images-idx3-ubyte",
    images_sha256: "5b4141f0afbad91edebe8549f8fcffe087ea10ca49f1dbef5c9a5cd8815ce37b",
    labels_url: "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
    labels_filename: "fashion-t10k-labels-idx1-ubyte",
    labels_sha256: "0402a96d92fd2663957122ceb108a494c5af83dab82d92729df917d7dec38c34",
    n_samples: N_TEST_SAMPLES,
};

/// Subset selector: the training partition (60,000 images).
const SUBSET_TRAIN: &[&Partition] = &[&TRAIN_PARTITION];

/// Subset selector: the test partition (10,000 images).
const SUBSET_TEST: &[&Partition] = &[&TEST_PARTITION];

/// Subset selector: both partitions (70,000 images, train followed by test).
const SUBSET_ALL: &[&Partition] = &[&TRAIN_PARTITION, &TEST_PARTITION];

/// A struct that represents the Fashion-MNIST dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// Fashion-MNIST holds 70,000 grayscale images of clothing articles from the
/// Zalando catalog, each 28×28 pixels, with the garment class each one shows.
/// It matches [`Mnist`](crate::Mnist) in image size, class count, partition
/// sizes, and file format, so the two loaders present the same interface.
///
/// Fashion-MNIST is the harder task of the two. The garment classes overlap far
/// more than the handwritten digits do, and the pullover, coat, and shirt
/// classes are the ones that confuse a model.
///
/// # Subsets
///
/// The source ships two partitions, and three constructors select them:
///
/// - [`FashionMnist::new`]: the training partition, 60,000 images
/// - [`FashionMnist::new_test`]: the test partition, 10,000 images
/// - [`FashionMnist::new_all`]: both, 70,000 images, train followed by test
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
/// | `class`  | `Integer` | the garment class, one of `0`-`9`                                                  |
///
/// The source designates `pixels` as the input
/// ([`FashionMnist::FEATURE_NAMES`]) and `class` as the label
/// ([`FashionMnist::TARGET`]).
///
/// Missing values: none.
///
/// In the `pixels` column, `0` is the background. The column holds one row of
/// 784 bytes per image. A view of that row shaped `(28, 28)` reads the same
/// bytes, at no copy. A garment can reach the edge of its frame, so a border
/// pixel is not always background.
///
/// [`FashionMnist::CLASS_NAMES`] maps a `class` code to its name:
///
/// | Code | Class       | Code | Class      |
/// |------|-------------|------|------------|
/// | `0`  | T-shirt/top | `5`  | Sandal     |
/// | `1`  | Trouser     | `6`  | Shirt      |
/// | `2`  | Pullover    | `7`  | Sneaker    |
/// | `3`  | Dress       | `8`  | Bag        |
/// | `4`  | Coat        | `9`  | Ankle boot |
///
/// The classes are **exactly** balanced: 6,000 images per class in the training
/// partition and 1,000 per class in the test partition. A plain
/// [`train_test_split`](crate::preprocessing::train_test_split) therefore needs
/// no stratification to keep the classes even.
///
/// # Source format
///
/// The source ships four gzip-compressed IDX files, one image file and one label
/// file per partition. IDX is a binary format: a big-endian header of 4-byte
/// integers, then the raw bytes. The storage directory holds each file
/// **decompressed**, under the source name prefixed with `fashion-`. The
/// upstream files carry the same names as the MNIST files, and the prefix keeps
/// them apart. [`Mnist`](crate::Mnist) and `FashionMnist` can therefore share
/// one storage directory.
///
/// See more information at <https://github.com/zalandoresearch/fashion-mnist>.
///
/// # License
///
/// Zalando Research releases Fashion-MNIST under the MIT license.
///
/// # Citation
///
/// Xiao, H., Rasul, K., and Vollgraf, R. (2017). "Fashion-MNIST: a Novel Image
/// Dataset for Benchmarking Machine Learning Algorithms." arXiv:1708.07747.
/// <https://arxiv.org/abs/1708.07747>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::FashionMnist;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./fashion_mnist";
///
/// let mut dataset = FashionMnist::new(download_dir);
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
/// let features = table.numeric_matrix(&FashionMnist::FEATURE_NAMES).unwrap();
/// assert_eq!(features.shape(), &[60000, 784]);
///
/// // Scale the pixels to [0, 1] for a model.
/// let scaled = features.mapv(|pixel| pixel / 255.0);
/// assert_eq!(scaled.shape(), &[60000, 784]);
///
/// // Name the class of the first image.
/// let classes = table.column(FashionMnist::TARGET).unwrap().as_integer().unwrap();
/// let name = FashionMnist::CLASS_NAMES[classes[0] as usize];
/// println!("the first image shows a {name}");
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
pub struct FashionMnist {
    dataset: Dataset<Table, DatasetError>,
}

impl FashionMnist {
    /// The column the source designates as the model input.
    pub const FEATURE_NAMES: [&'static str; 1] = ["pixels"];

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "class";

    /// The name of each garment class, indexed by its class code.
    ///
    /// A code of `3` names `CLASS_NAMES[3]`, which is `"Dress"`. The order is
    /// the one the source defines.
    ///
    /// # Example
    /// ```
    /// use dataset_ml::FashionMnist;
    ///
    /// assert_eq!(FashionMnist::CLASS_NAMES[0], "T-shirt/top");
    /// assert_eq!(FashionMnist::CLASS_NAMES[9], "Ankle boot");
    /// ```
    pub const CLASS_NAMES: [&'static str; N_CLASSES] = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ];

    /// Create a new FashionMnist instance for the **training** partition (60,000
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
    /// - `Self` - a `FashionMnist` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, SUBSET_TRAIN)
    }

    /// Create a new FashionMnist instance for the **test** partition (10,000
    /// images) without loading data.
    ///
    /// See [`FashionMnist::new`] for the loading semantics.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - The directory that stores the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - a `FashionMnist` instance ready for lazy loading.
    pub fn new_test(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, SUBSET_TEST)
    }

    /// Create a new FashionMnist instance for **all** 70,000 images (the training
    /// partition followed by the test partition) without loading data.
    ///
    /// See [`FashionMnist::new`] for the loading semantics.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - The directory that stores the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - a `FashionMnist` instance ready for lazy loading.
    pub fn new_all(storage_dir: &str) -> Self {
        Self::with_subset(storage_dir, SUBSET_ALL)
    }

    /// Construct an instance whose loader reads the given partitions.
    fn with_subset(storage_dir: &str, subset: &'static [&'static Partition]) -> Self {
        FashionMnist {
            dataset: Dataset::new(storage_dir, move |dir| Self::load_data(dir, subset)),
        }
    }

    /// Get and parse the Fashion-MNIST dataset for the requested subset.
    fn load_data(dir: &str, subset: &'static [&'static Partition]) -> Result<Table, DatasetError> {
        let (pixels, labels) = idx::load_partitions(dir, FASHION_MNIST_DATASET_NAME, subset)?;

        Table::new(
            FASHION_MNIST_DATASET_NAME,
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
    /// Unlike [`FashionMnist::data`], this method never runs the loader. If the
    /// data has not loaded yet, it returns `None` instead of downloading and
    /// parsing it. Use this method when you want the data only if it is already
    /// cached. This skips the cost of a download and a parse.
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
    /// changes stay in the cache. Later calls to [`FashionMnist::data`] or
    /// [`FashionMnist::get_data`] see them.
    ///
    /// Like [`FashionMnist::get_data`], this does **not** trigger loading.
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
    /// the instance, use [`FashionMnist::take_data`] instead.
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

impl_ml_dataset!(FashionMnist, "fashion_mnist");
