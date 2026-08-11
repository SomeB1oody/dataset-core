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
//! **Images:** an `Array2<u8>` of shape `(n_samples, 784)`. Each row is one 28×28
//! image, flattened in row-major order. Each value is a pixel intensity in
//! `0..=255`, where `0` is the background.
//! [`FashionMnist::images`](crate::FashionMnist::images) returns the same buffer
//! as a `(n_samples, 28, 28)` view, at no copy.
//!
//! **Labels:** an `Array1<u8>`, the garment class, one of `0`-`9`. See
//! [`FashionMnist::CLASS_NAMES`](crate::FashionMnist::CLASS_NAMES) for the name
//! of each code.
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
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError};
use ndarray::{Array1, Array2, ArrayView3};

/// Type alias for the Fashion-MNIST dataset: (images, labels).
pub type FashionMnistData = (Array2<u8>, Array1<u8>);

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
/// # Images
///
/// [`FashionMnist::features`] returns an `Array2<u8>` of shape
/// `(n_samples, 784)`. Each row is one image, flattened in row-major order, and
/// each value is a pixel intensity in `0..=255`. `0` is the background.
///
/// [`FashionMnist::images`] returns the same buffer shaped
/// `(n_samples, 28, 28)`. It is a view over that buffer, not a second copy.
///
/// A garment can reach the edge of its frame, so a border pixel is not always
/// background.
///
/// [`preprocessing`](crate::preprocessing) takes `&Array2<f64>`. Convert the
/// pixels with `features.mapv(f64::from)`. To scale them to `[0, 1]` in the same
/// pass, use `features.mapv(|p| f64::from(p) / 255.0)`.
///
/// # Labels
///
/// [`FashionMnist::labels`] returns an `Array1<u8>`, the garment class, one of
/// `0`-`9`. [`FashionMnist::CLASS_NAMES`] maps a code to its name:
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
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// // data() also returns both at once
/// let (features, labels) = dataset.data().unwrap();
/// assert_eq!(features.shape(), &[60000, 784]);
/// assert_eq!(labels.len(), 60000);
///
/// // Name the class of the first image.
/// let name = FashionMnist::CLASS_NAMES[labels[0] as usize];
/// println!("the first image shows a {name}");
///
/// // images() reshapes the same buffer to 28x28, with no copy.
/// let images = dataset.images().unwrap();
/// assert_eq!(images.shape(), &[60000, 28, 28]);
///
/// // Scale the pixels to [0, 1] for a model. This allocates the f64 copy.
/// let scaled = features.mapv(|pixel| f64::from(pixel) / 255.0);
/// assert_eq!(scaled.shape(), &[60000, 784]);
///
/// // `get_data_mut()` edits the arrays in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some((features, _labels)) = dataset.get_data_mut() {
///     features[[0, 0]] = 255;
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned arrays out with no `to_owned()` clone. This
/// // leaves the instance reusable.
/// let (owned_features, owned_labels) = dataset.take_data().unwrap();
/// assert_eq!(owned_features.shape(), &[60000, 784]);
/// assert_eq!(owned_labels.len(), 60000);
///
/// // `into_data()` also returns the owned arrays with no clone, but it
/// // consumes the instance.
/// let (owned_features, owned_labels) = dataset.into_data().unwrap();
/// assert_eq!(owned_features.shape(), &[60000, 784]);
/// assert_eq!(owned_labels.len(), 60000);
/// ```
#[derive(Debug)]
pub struct FashionMnist {
    dataset: Dataset<FashionMnistData, DatasetError>,
}

impl FashionMnist {
    /// The name of each garment class, indexed by its label code.
    ///
    /// A label of `3` names `CLASS_NAMES[3]`, which is `"Dress"`. The order is
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
    fn load_data(
        dir: &str,
        subset: &'static [&'static Partition],
    ) -> Result<FashionMnistData, DatasetError> {
        idx::load_partitions(dir, FASHION_MNIST_DATASET_NAME, subset)
    }

    /// Get a reference to the flattened image matrix.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array2<u8>` - Reference to the image matrix with shape
    ///   `(n_samples, 784)`. Each row is one 28×28 image, flattened in row-major
    ///   order. Each value is a pixel intensity in `0..=255`. `n_samples` is
    ///   60,000, 10,000, or 70,000, by the constructor you used.
    ///
    /// For the 28×28 shape, use [`FashionMnist::images`]. For
    /// [`preprocessing`](crate::preprocessing), convert with
    /// `features.mapv(f64::from)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File decompression or I/O operations fail
    /// - The IDX header holds an unexpected magic number or image size
    /// - The file holds a different number of images or pixels than its header states
    pub fn features(&self) -> Result<&Array2<u8>, DatasetError> {
        Ok(&self.dataset.load()?.0)
    }

    /// Get the images as a `(n_samples, 28, 28)` view.
    ///
    /// This reshapes the buffer that [`FashionMnist::features`] returns. It is a
    /// view over the same memory, not a second copy. Use it when a model wants
    /// the spatial layout instead of a flat row.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `ArrayView3<u8>` - View of the images with shape `(n_samples, 28, 28)`,
    ///   indexed as `[image, row, column]`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, or a header
    /// or length check), or if the cached matrix does not reshape to 28×28.
    pub fn images(&self) -> Result<ArrayView3<'_, u8>, DatasetError> {
        let features = &self.dataset.load()?.0;
        let n_samples = features.nrows();
        features
            .view()
            .into_shape_with_order((n_samples, idx::IMAGE_ROWS, idx::IMAGE_COLS))
            .map_err(|e| DatasetError::array_shape_error(FASHION_MNIST_DATASET_NAME, "images", e))
    }

    /// Get a reference to the label vector.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array1<u8>` - Reference to the label vector with shape `(n_samples,)`.
    ///   Each value is the garment class of the matching image, one of `0`-`9`.
    ///   [`FashionMnist::CLASS_NAMES`] names each code.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File decompression or I/O operations fail
    /// - The IDX header holds an unexpected magic number
    /// - The file holds a different number of labels than its header states
    /// - A label falls outside `0..=9`
    pub fn labels(&self) -> Result<&Array1<u8>, DatasetError> {
        Ok(&self.dataset.load()?.1)
    }

    /// Get images and labels as references.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&FashionMnistData` - reference to the cached `(images, labels)` tuple:
    ///   image matrix `(n_samples, 784)` and label vector `(n_samples,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, or a header
    /// or length check).
    pub fn data(&self) -> Result<&FashionMnistData, DatasetError> {
        self.dataset.load()
    }

    /// Get images and labels as references **without** triggering loading.
    ///
    /// Unlike [`FashionMnist::data`], this method never runs the loader. If the
    /// data has not loaded yet, it returns `None` instead of downloading and
    /// parsing it. Use this method when you want the data only if it is already
    /// cached. This skips the cost of a download and a parse.
    ///
    /// # Returns
    ///
    /// - `Some(&FashionMnistData)` - reference to the cached `(images, labels)`
    ///   tuple, if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&FashionMnistData> {
        self.dataset.get()
    }

    /// Get mutable references to images and labels for **in-place** editing.
    ///
    /// This lets you change the cached arrays directly. For example, you can
    /// binarize the pixels. This needs no `.to_owned()` clone, and it does not
    /// remove the data from the cache. The changes stay in the cache. Later calls
    /// to [`FashionMnist::features`], [`FashionMnist::data`], or
    /// [`FashionMnist::get_data`] see the changes.
    ///
    /// Like [`FashionMnist::get_data`], this does **not** trigger loading. It
    /// returns `None` if the dataset has not loaded yet. If you need the data to
    /// be present, call a loading accessor first, for example
    /// [`FashionMnist::data`].
    ///
    /// # Returns
    ///
    /// - `Some(&mut FashionMnistData)` - mutable reference to the cached
    ///   `(images, labels)` tuple, if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut FashionMnistData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** images and labels.
    ///
    /// Unlike [`FashionMnist::data`], which borrows the cached data, this moves
    /// the data out and returns owned arrays directly. It needs no `to_owned()`
    /// clone. If the dataset has not loaded yet, the first access loads it.
    ///
    /// This **consumes** `self`. After the call, you cannot use the instance
    /// again. If you want owned data but need to keep using the instance, use
    /// [`FashionMnist::take_data`] instead. It takes `&mut self` and leaves the
    /// instance reusable.
    ///
    /// # Returns
    ///
    /// - `(Array2<u8>, Array1<u8>)` - owned image matrix `(n_samples, 784)` and
    ///   owned label vector `(n_samples,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, or a header or
    /// length check).
    pub fn into_data(self) -> Result<FashionMnistData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** images and labels out of the dataset. This leaves the
    /// instance reusable.
    ///
    /// Like [`FashionMnist::into_data`], this returns owned arrays with no
    /// `to_owned()` clone. Instead of consuming the instance, it takes `&mut self`
    /// and moves the cached data out. This resets the instance to its unloaded
    /// state. The next accessor call, for example [`FashionMnist::features`] or
    /// [`FashionMnist::data`], loads the dataset again.
    ///
    /// If you are done with the instance, use [`FashionMnist::into_data`] instead.
    ///
    /// # Returns
    ///
    /// - `(Array2<u8>, Array1<u8>)` - owned image matrix `(n_samples, 784)` and
    ///   owned label vector `(n_samples,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, or a header or
    /// length check).
    pub fn take_data(&mut self) -> Result<FashionMnistData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(FashionMnist, FashionMnistData, "fashion_mnist");
