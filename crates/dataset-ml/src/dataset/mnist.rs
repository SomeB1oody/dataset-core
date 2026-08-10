//! MNIST database of handwritten digits.
//!
//! 70,000 grayscale images of handwritten digits, each 28×28 pixels, split into
//! a 60,000-image training partition and a 10,000-image test partition. The task
//! is to recognize which digit (`0`-`9`) an image shows. MNIST is the standard
//! entry benchmark for image classification.
//!
//! This is the crate's first loader for a **binary** source format and its first
//! dataset that gives raw pixels at a useful size. [`digits`](crate::dataset::digits)
//! holds the same task at 8×8.
//!
//! **Images:** an `Array2<u8>` of shape `(n_samples, 784)`. Each row is one 28×28
//! image, flattened in row-major order. Each value is a pixel intensity in
//! `0..=255`, where `0` is the background. [`Mnist::images`](crate::Mnist::images)
//! returns the same buffer as a `(n_samples, 28, 28)` view, at no copy.
//!
//! **Labels:** an `Array1<u8>`, the digit each image shows, one of `0`-`9`.
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
//! **Element type:** the loader stores pixels as `u8`, not as the `f64` that the
//! other loaders use for features. The IDX source stores one unsigned byte per
//! pixel, so `u8` is lossless. It also costs 52 MiB for all 70,000 images, where
//! `f64` would cost 419 MiB. See the struct docs for how to feed
//! [`preprocessing`](crate::preprocessing), which takes `f64`.
//!
//! **Source:** LeCun, Y., Cortes, C., and Burges, C. J. C. The MNIST database of
//! handwritten digits. This loader reads the four canonical IDX files from the
//! `ossci-datasets` mirror. <http://yann.lecun.com/exdb/mnist/>

use crate::DOWNLOAD_RETRIES;
use crate::traits::impl_ml_dataset;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, gunzip};
use ndarray::{Array1, Array2, ArrayView3};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Type alias for the MNIST dataset: (images, labels).
pub type MnistData = (Array2<u8>, Array1<u8>);

/// The name of the dataset.
const MNIST_DATASET_NAME: &str = "mnist";

/// Height of one image, in pixels.
const IMAGE_ROWS: usize = 28;

/// Width of one image, in pixels.
const IMAGE_COLS: usize = 28;

/// Number of pixels per image, which is the column count of the flattened
/// image matrix.
const N_PIXELS: usize = IMAGE_ROWS * IMAGE_COLS;

/// Magic number that starts an IDX file of 3-dimensional unsigned bytes. The
/// image files use it.
const IDX_IMAGES_MAGIC: u32 = 2051;

/// Magic number that starts an IDX file of 1-dimensional unsigned bytes. The
/// label files use it.
const IDX_LABELS_MAGIC: u32 = 2049;

/// Header length of an IDX image file: the magic number and 3 dimensions, each
/// a 4-byte big-endian integer.
const IDX_IMAGES_HEADER_LEN: usize = 16;

/// Header length of an IDX label file: the magic number and 1 dimension, each a
/// 4-byte big-endian integer.
const IDX_LABELS_HEADER_LEN: usize = 8;

/// Number of samples in the training partition.
const N_TRAIN_SAMPLES: usize = 60_000;

/// Number of samples in the test partition.
const N_TEST_SAMPLES: usize = 10_000;

/// One MNIST partition and the two IDX files that hold it.
///
/// The source ships each partition as an image file and a label file, gzip
/// compressed and downloaded separately. Every SHA256 hash here is the hash of
/// the **decompressed** file, which is what the loader caches.
struct Partition {
    /// URL of the gzip-compressed image file.
    images_url: &'static str,
    /// Cache filename of the decompressed image file.
    images_filename: &'static str,
    /// SHA256 hash of the decompressed image file.
    images_sha256: &'static str,
    /// URL of the gzip-compressed label file.
    labels_url: &'static str,
    /// Cache filename of the decompressed label file.
    labels_filename: &'static str,
    /// SHA256 hash of the decompressed label file.
    labels_sha256: &'static str,
    /// Number of samples the partition holds.
    n_samples: usize,
}

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

/// Read a big-endian `u32` out of an IDX header, at a 4-byte offset.
///
/// The caller must pass a header slice long enough to hold `offset + 4` bytes.
fn read_be_u32(header: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        header[offset],
        header[offset + 1],
        header[offset + 2],
        header[offset + 3],
    ])
}

/// Report a header field that holds the wrong value.
///
/// An IDX file is binary, and its header starts at offset 0. The error type
/// carries a line number, so this reports line 1 for every header field.
fn header_error(field_name: &str, value: u32) -> DatasetError {
    DatasetError::invalid_value(MNIST_DATASET_NAME, field_name, &value.to_string(), 1)
}

/// Download one gzip-compressed IDX file and cache it decompressed.
///
/// The cached file is the decompressed IDX file, and `expected_sha256` is that
/// file's hash. A later run reuses the cache and downloads nothing.
///
/// # Parameters
///
/// - `dir` - The directory that stores the dataset.
/// - `url` - URL of the gzip-compressed source file.
/// - `filename` - Cache filename of the decompressed file.
/// - `expected_sha256` - SHA256 hash of the decompressed file.
///
/// # Returns
///
/// - `PathBuf` - Path to the cached IDX file.
///
/// # Errors
///
/// Returns `DatasetError` if the download, the decompression, or the hash check
/// fails.
fn acquire_idx_file(
    dir: &str,
    url: &str,
    filename: &str,
    expected_sha256: &str,
) -> Result<PathBuf, DatasetError> {
    let gz_filename = format!("{filename}.gz");
    acquire_dataset(
        dir,
        filename,
        MNIST_DATASET_NAME,
        Some(expected_sha256),
        |temp_path| {
            download_to_with_retries(url, temp_path, Some(&gz_filename), DOWNLOAD_RETRIES)?;
            let idx_path = temp_path.join(filename);
            gunzip(&temp_path.join(&gz_filename), &idx_path)?;
            Ok(idx_path)
        },
    )
}

/// Read an IDX image file and append its pixels to `pixels`.
///
/// The function checks the header before it reads any pixel: the magic number
/// must be [`IDX_IMAGES_MAGIC`], the image count must be `n_samples`, and each
/// image must be 28×28. It then appends `n_samples * 784` bytes, in the file's
/// own order, so the images stay in their source order.
///
/// # Parameters
///
/// - `file_path` - Path to the decompressed IDX image file.
/// - `n_samples` - Number of images the file must hold.
/// - `pixels` - Buffer that receives the pixels.
///
/// # Errors
///
/// Returns `DatasetError` if the file cannot be read, a header field holds an
/// unexpected value, or the pixel count does not match the header.
fn read_idx_images(
    file_path: &Path,
    n_samples: usize,
    pixels: &mut Vec<u8>,
) -> Result<(), DatasetError> {
    let mut file = File::open(file_path)?;

    let mut header = [0u8; IDX_IMAGES_HEADER_LEN];
    file.read_exact(&mut header)?;

    let magic = read_be_u32(&header, 0);
    if magic != IDX_IMAGES_MAGIC {
        return Err(header_error("idx_images_magic", magic));
    }

    let count = read_be_u32(&header, 4) as usize;
    if count != n_samples {
        return Err(DatasetError::length_mismatch(
            MNIST_DATASET_NAME,
            "images",
            n_samples,
            count,
        ));
    }

    let rows = read_be_u32(&header, 8);
    let cols = read_be_u32(&header, 12);
    if rows as usize != IMAGE_ROWS {
        return Err(header_error("image_rows", rows));
    }
    if cols as usize != IMAGE_COLS {
        return Err(header_error("image_cols", cols));
    }

    // `read_to_end` appends to the buffer, so the pixels of every partition
    // land end to end with no second copy.
    let before = pixels.len();
    file.read_to_end(pixels)?;
    let read = pixels.len() - before;

    let expected = n_samples * N_PIXELS;
    if read != expected {
        return Err(DatasetError::length_mismatch(
            MNIST_DATASET_NAME,
            "pixels",
            expected,
            read,
        ));
    }

    Ok(())
}

/// Read an IDX label file and append its labels to `labels`.
///
/// The function checks the header before it reads any label: the magic number
/// must be [`IDX_LABELS_MAGIC`] and the label count must be `n_samples`.
///
/// # Parameters
///
/// - `file_path` - Path to the decompressed IDX label file.
/// - `n_samples` - Number of labels the file must hold.
/// - `labels` - Buffer that receives the labels.
///
/// # Errors
///
/// Returns `DatasetError` if the file cannot be read, a header field holds an
/// unexpected value, the label count does not match the header, or a label
/// falls outside `0..=9`.
fn read_idx_labels(
    file_path: &Path,
    n_samples: usize,
    labels: &mut Vec<u8>,
) -> Result<(), DatasetError> {
    let mut file = File::open(file_path)?;

    let mut header = [0u8; IDX_LABELS_HEADER_LEN];
    file.read_exact(&mut header)?;

    let magic = read_be_u32(&header, 0);
    if magic != IDX_LABELS_MAGIC {
        return Err(header_error("idx_labels_magic", magic));
    }

    let count = read_be_u32(&header, 4) as usize;
    if count != n_samples {
        return Err(DatasetError::length_mismatch(
            MNIST_DATASET_NAME,
            "labels",
            n_samples,
            count,
        ));
    }

    let before = labels.len();
    file.read_to_end(labels)?;
    let read = labels.len() - before;

    if read != n_samples {
        return Err(DatasetError::length_mismatch(
            MNIST_DATASET_NAME,
            "labels",
            n_samples,
            read,
        ));
    }

    for (offset, &label) in labels[before..].iter().enumerate() {
        if label > 9 {
            return Err(DatasetError::invalid_value(
                MNIST_DATASET_NAME,
                "label",
                &label.to_string(),
                before + offset + 1,
            ));
        }
    }

    Ok(())
}

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
/// # Images
///
/// [`Mnist::features`] returns an `Array2<u8>` of shape `(n_samples, 784)`. Each
/// row is one image, flattened in row-major order, and each value is a pixel
/// intensity in `0..=255`. `0` is the background and 255 is the darkest ink.
///
/// [`Mnist::images`] returns the same buffer shaped `(n_samples, 28, 28)`. This
/// is a view, not a second copy, so it costs no memory and no time.
///
/// # Labels
///
/// [`Mnist::labels`] returns an `Array1<u8>`, the digit each image shows, one of
/// `0`-`9`. The classes are close to balanced. The training partition ranges
/// from 5,421 images of `5` to 6,742 images of `1`.
///
/// # Element type and memory
///
/// This loader stores pixels as `u8`, where the other loaders return `f64`
/// features. The IDX source stores one unsigned byte per pixel, so `u8` loses
/// nothing. It also decides how much memory a loaded instance holds:
///
/// | Type  | 60,000 images | 70,000 images |
/// |-------|---------------|---------------|
/// | `u8`  | 44.9 MiB      | 52.3 MiB      |
/// | `f64` | 358.9 MiB     | 418.7 MiB     |
///
/// [`preprocessing`](crate::preprocessing) takes `&Array2<f64>`. Convert first
/// with `features.mapv(f64::from)`, which allocates the larger matrix once, when
/// you ask for it. Most models want the pixels scaled to `[0, 1]` anyway, and
/// `features.mapv(|p| f64::from(p) / 255.0)` does both steps in one pass.
///
/// # Source format
///
/// The source ships four gzip-compressed IDX files, one image file and one label
/// file per partition. IDX is a binary format: a big-endian header of 4-byte
/// integers, then the raw bytes. This loader caches each file **decompressed**
/// and pins the SHA256 hash of the decompressed file, so a later load reads the
/// bytes with no download and no decompression.
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
/// let features = dataset.features().unwrap();
/// let labels = dataset.labels().unwrap();
///
/// // data() also returns both at once
/// let (features, labels) = dataset.data().unwrap();
/// assert_eq!(features.shape(), &[60000, 784]);
/// assert_eq!(labels.len(), 60000);
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
pub struct Mnist {
    dataset: Dataset<MnistData, DatasetError>,
}

impl Mnist {
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
    fn load_data(
        dir: &str,
        subset: &'static [&'static Partition],
    ) -> Result<MnistData, DatasetError> {
        let n_samples: usize = subset.iter().map(|partition| partition.n_samples).sum();

        // Reserve the full buffers once. The training images alone are 47 MB, so
        // growing them step by step would copy tens of megabytes.
        let mut pixels: Vec<u8> = Vec::with_capacity(n_samples * N_PIXELS);
        let mut labels: Vec<u8> = Vec::with_capacity(n_samples);

        for partition in subset {
            let images_path = acquire_idx_file(
                dir,
                partition.images_url,
                partition.images_filename,
                partition.images_sha256,
            )?;
            let labels_path = acquire_idx_file(
                dir,
                partition.labels_url,
                partition.labels_filename,
                partition.labels_sha256,
            )?;

            read_idx_images(&images_path, partition.n_samples, &mut pixels)?;
            read_idx_labels(&labels_path, partition.n_samples, &mut labels)?;
        }

        if labels.is_empty() {
            return Err(DatasetError::empty_dataset(MNIST_DATASET_NAME));
        }

        let features_array = Array2::from_shape_vec((n_samples, N_PIXELS), pixels)
            .map_err(|e| DatasetError::array_shape_error(MNIST_DATASET_NAME, "features", e))?;
        let labels_array = Array1::from_vec(labels);

        Ok((features_array, labels_array))
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
    /// For the 28×28 shape, use [`Mnist::images`]. To feed
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
    /// This reshapes the buffer that [`Mnist::features`] returns. It is a view
    /// over the same memory, so it copies no pixel and allocates nothing. Use it
    /// when a model wants the spatial layout instead of a flat row.
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
            .into_shape_with_order((n_samples, IMAGE_ROWS, IMAGE_COLS))
            .map_err(|e| DatasetError::array_shape_error(MNIST_DATASET_NAME, "images", e))
    }

    /// Get a reference to the label vector.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Array1<u8>` - Reference to the label vector with shape `(n_samples,)`.
    ///   Each value is the digit the matching image shows, one of `0`-`9`.
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
    /// - `&MnistData` - reference to the cached `(images, labels)` tuple: image
    ///   matrix `(n_samples, 784)` and label vector `(n_samples,)`.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, or a header
    /// or length check).
    pub fn data(&self) -> Result<&MnistData, DatasetError> {
        self.dataset.load()
    }

    /// Get images and labels as references **without** triggering loading.
    ///
    /// Unlike [`Mnist::data`], this method never runs the loader. If the data has
    /// not loaded yet, it returns `None` instead of downloading and parsing it.
    /// Use this method when you want the data only if it is already cached. This
    /// skips the cost of a download and a parse.
    ///
    /// # Returns
    ///
    /// - `Some(&MnistData)` - reference to the cached `(images, labels)` tuple,
    ///   if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&MnistData> {
        self.dataset.get()
    }

    /// Get mutable references to images and labels for **in-place** editing.
    ///
    /// This lets you change the cached arrays directly. For example, you can
    /// binarize the pixels. This needs no `.to_owned()` clone, and it does not
    /// remove the data from the cache. The changes stay in the cache. Later calls
    /// to [`Mnist::features`], [`Mnist::data`], or [`Mnist::get_data`] see the
    /// changes.
    ///
    /// Like [`Mnist::get_data`], this does **not** trigger loading. It returns
    /// `None` if the dataset has not loaded yet. If you need the data to be
    /// present, call a loading accessor first, for example [`Mnist::data`].
    ///
    /// # Returns
    ///
    /// - `Some(&mut MnistData)` - mutable reference to the cached
    ///   `(images, labels)` tuple, if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut MnistData> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return **owned** images and labels.
    ///
    /// Unlike [`Mnist::data`], which borrows the cached data, this moves the data
    /// out and returns owned arrays directly. It needs no `to_owned()` clone. If
    /// the dataset has not loaded yet, the first access loads it. For a 47 MB
    /// image matrix, that saved clone matters.
    ///
    /// This **consumes** `self`. After the call, you cannot use the instance
    /// again. If you want owned data but need to keep using the instance, use
    /// [`Mnist::take_data`] instead. It takes `&mut self` and leaves the instance
    /// reusable.
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
    pub fn into_data(self) -> Result<MnistData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take **owned** images and labels out of the dataset. This leaves the
    /// instance reusable.
    ///
    /// Like [`Mnist::into_data`], this returns owned arrays with no `to_owned()`
    /// clone. Instead of consuming the instance, it takes `&mut self` and moves
    /// the cached data out. This resets the instance to its unloaded state. The
    /// next accessor call, for example [`Mnist::features`] or [`Mnist::data`],
    /// loads the dataset again.
    ///
    /// If you are done with the instance, use [`Mnist::into_data`] instead.
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
    pub fn take_data(&mut self) -> Result<MnistData, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(Mnist, MnistData, "mnist");
