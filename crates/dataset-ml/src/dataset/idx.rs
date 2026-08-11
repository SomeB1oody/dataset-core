//! Reader for the IDX binary format.
//!
//! IDX holds an array of unsigned bytes. A file starts with a big-endian header
//! of 4-byte integers: a magic number, then one size per dimension. The raw
//! bytes follow, with no separator and no padding.
//!
//! [`mnist`](super::mnist) and [`fashion_mnist`](super::fashion_mnist) read
//! their sources through this module. Both ship four gzip-compressed files: an
//! image file and a label file for each of a training partition and a test
//! partition. Both hold 28×28 grayscale images and 10 classes.
//!
//! This module is internal to the crate.

use crate::DOWNLOAD_RETRIES;
use dataset_core::{DatasetError, acquire_dataset, download_to_with_retries, gunzip};
use ndarray::{Array1, Array2};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Height of one image, in pixels.
pub(super) const IMAGE_ROWS: usize = 28;

/// Width of one image, in pixels.
pub(super) const IMAGE_COLS: usize = 28;

/// Number of pixels per image, which is the column count of the flattened image
/// matrix.
pub(super) const N_PIXELS: usize = IMAGE_ROWS * IMAGE_COLS;

/// Number of classes both datasets label their images with.
pub(super) const N_CLASSES: usize = 10;

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

/// The parsed shape of an IDX-sourced image dataset: (images, labels).
///
/// The images hold one row per image, flattened to [`N_PIXELS`] columns. The
/// labels hold one class code per image.
pub(super) type IdxImageData = (Array2<u8>, Array1<u8>);

/// One partition of an IDX-sourced dataset and the two files that hold it.
///
/// The source ships each partition as an image file and a label file, gzip
/// compressed and downloaded separately. Every SHA256 hash here is the hash of
/// the **decompressed** file, which is what the loader caches.
pub(super) struct Partition {
    /// URL of the gzip-compressed image file.
    pub images_url: &'static str,
    /// Cache filename of the decompressed image file.
    pub images_filename: &'static str,
    /// SHA256 hash of the decompressed image file.
    pub images_sha256: &'static str,
    /// URL of the gzip-compressed label file.
    pub labels_url: &'static str,
    /// Cache filename of the decompressed label file.
    pub labels_filename: &'static str,
    /// SHA256 hash of the decompressed label file.
    pub labels_sha256: &'static str,
    /// Number of samples the partition holds.
    pub n_samples: usize,
}

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
fn header_error(dataset_name: &str, field_name: &str, value: u32) -> DatasetError {
    DatasetError::invalid_value(dataset_name, field_name, &value.to_string(), 1)
}

/// Download one gzip-compressed IDX file and cache it decompressed.
///
/// The cached file is the decompressed IDX file, and `expected_sha256` is that
/// file's hash. A later run reuses the cache and downloads nothing.
///
/// # Parameters
///
/// - `dir` - The directory that stores the dataset.
/// - `dataset_name` - The dataset name for error messages.
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
    dataset_name: &str,
    url: &str,
    filename: &str,
    expected_sha256: &str,
) -> Result<PathBuf, DatasetError> {
    let gz_filename = format!("{filename}.gz");
    acquire_dataset(
        dir,
        filename,
        dataset_name,
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
/// image must be [`IMAGE_ROWS`]×[`IMAGE_COLS`]. It then appends
/// `n_samples * N_PIXELS` bytes, in the file's own order, so the images stay in
/// their source order.
///
/// # Parameters
///
/// - `dataset_name` - The dataset name for error messages.
/// - `file_path` - Path to the decompressed IDX image file.
/// - `n_samples` - Number of images the file must hold.
/// - `pixels` - Buffer that receives the pixels.
///
/// # Errors
///
/// Returns `DatasetError` if the file cannot be read, a header field holds an
/// unexpected value, or the pixel count does not match the header.
fn read_idx_images(
    dataset_name: &str,
    file_path: &Path,
    n_samples: usize,
    pixels: &mut Vec<u8>,
) -> Result<(), DatasetError> {
    let mut file = File::open(file_path)?;

    let mut header = [0u8; IDX_IMAGES_HEADER_LEN];
    file.read_exact(&mut header)?;

    let magic = read_be_u32(&header, 0);
    if magic != IDX_IMAGES_MAGIC {
        return Err(header_error(dataset_name, "idx_images_magic", magic));
    }

    let count = read_be_u32(&header, 4) as usize;
    if count != n_samples {
        return Err(DatasetError::length_mismatch(
            dataset_name,
            "images",
            n_samples,
            count,
        ));
    }

    let rows = read_be_u32(&header, 8);
    let cols = read_be_u32(&header, 12);
    if rows as usize != IMAGE_ROWS {
        return Err(header_error(dataset_name, "image_rows", rows));
    }
    if cols as usize != IMAGE_COLS {
        return Err(header_error(dataset_name, "image_cols", cols));
    }

    let before = pixels.len();
    file.read_to_end(pixels)?;
    let read = pixels.len() - before;

    let expected = n_samples * N_PIXELS;
    if read != expected {
        return Err(DatasetError::length_mismatch(
            dataset_name,
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
/// must be [`IDX_LABELS_MAGIC`] and the label count must be `n_samples`. Every
/// label must name one of the [`N_CLASSES`] classes.
///
/// # Parameters
///
/// - `dataset_name` - The dataset name for error messages.
/// - `file_path` - Path to the decompressed IDX label file.
/// - `n_samples` - Number of labels the file must hold.
/// - `labels` - Buffer that receives the labels.
///
/// # Errors
///
/// Returns `DatasetError` if the file cannot be read, a header field holds an
/// unexpected value, the label count does not match the header, or a label
/// names no class.
fn read_idx_labels(
    dataset_name: &str,
    file_path: &Path,
    n_samples: usize,
    labels: &mut Vec<u8>,
) -> Result<(), DatasetError> {
    let mut file = File::open(file_path)?;

    let mut header = [0u8; IDX_LABELS_HEADER_LEN];
    file.read_exact(&mut header)?;

    let magic = read_be_u32(&header, 0);
    if magic != IDX_LABELS_MAGIC {
        return Err(header_error(dataset_name, "idx_labels_magic", magic));
    }

    let count = read_be_u32(&header, 4) as usize;
    if count != n_samples {
        return Err(DatasetError::length_mismatch(
            dataset_name,
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
            dataset_name,
            "labels",
            n_samples,
            read,
        ));
    }

    for (offset, &label) in labels[before..].iter().enumerate() {
        if label as usize >= N_CLASSES {
            return Err(DatasetError::invalid_value(
                dataset_name,
                "label",
                &label.to_string(),
                before + offset + 1,
            ));
        }
    }

    Ok(())
}

/// Load the given partitions into one image matrix and one label vector.
///
/// The function reads the partitions in the order the caller lists them and
/// concatenates the result. A two-partition subset therefore holds the first
/// partition followed by the second. Each partition downloads and caches its own
/// two files. A one-partition subset never touches the other partition's files.
///
/// # Parameters
///
/// - `dir` - The directory that stores the dataset.
/// - `dataset_name` - The dataset name for error messages.
/// - `subset` - The partitions to read, in output order.
///
/// # Returns
///
/// - `Array2<u8>` - Image matrix with shape `(n_samples, 784)`.
/// - `Array1<u8>` - Label vector with length `n_samples`.
///
/// # Errors
///
/// Returns `DatasetError` if a download, a decompression, a hash check, or a
/// header or length check fails.
pub(super) fn load_partitions(
    dir: &str,
    dataset_name: &str,
    subset: &'static [&'static Partition],
) -> Result<IdxImageData, DatasetError> {
    let n_samples: usize = subset.iter().map(|partition| partition.n_samples).sum();

    let mut pixels: Vec<u8> = Vec::with_capacity(n_samples * N_PIXELS);
    let mut labels: Vec<u8> = Vec::with_capacity(n_samples);

    for partition in subset {
        let images_path = acquire_idx_file(
            dir,
            dataset_name,
            partition.images_url,
            partition.images_filename,
            partition.images_sha256,
        )?;
        let labels_path = acquire_idx_file(
            dir,
            dataset_name,
            partition.labels_url,
            partition.labels_filename,
            partition.labels_sha256,
        )?;

        read_idx_images(dataset_name, &images_path, partition.n_samples, &mut pixels)?;
        read_idx_labels(dataset_name, &labels_path, partition.n_samples, &mut labels)?;
    }

    if labels.is_empty() {
        return Err(DatasetError::empty_dataset(dataset_name));
    }

    let features_array = Array2::from_shape_vec((n_samples, N_PIXELS), pixels)
        .map_err(|e| DatasetError::array_shape_error(dataset_name, "features", e))?;
    let labels_array = Array1::from_vec(labels);

    Ok((features_array, labels_array))
}
