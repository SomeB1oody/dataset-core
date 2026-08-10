#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::mnist::Mnist;
use ndarray::{Array1, Array2};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached (decompressed) test-partition image file.
const TEST_IMAGES_SHA256: &str = "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7";

/// Cache filename of the decompressed test-partition image file.
const TEST_IMAGES_FILENAME: &str = "t10k-images-idx3-ubyte";

/// Cache filename of the decompressed test-partition label file.
const TEST_LABELS_FILENAME: &str = "t10k-labels-idx1-ubyte";

/// Number of images in the training partition.
const N_TRAIN: usize = 60_000;

/// Number of images in the test partition.
const N_TEST: usize = 10_000;

/// Number of images in both partitions together.
const N_ALL: usize = 70_000;

/// Number of pixels per image (28 × 28).
const N_PIXELS: usize = 784;

/// Images per digit class in the training partition, for classes `0` to `9`.
const TRAIN_CLASS_COUNTS: [usize; 10] =
    [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949];

/// Images per digit class in the test partition, for classes `0` to `9`.
const TEST_CLASS_COUNTS: [usize; 10] = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009];

/// Sum of every pixel of the training partition.
const TRAIN_PIXEL_SUM: u64 = 1_567_298_545;

/// Sum of every pixel of the test partition.
const TEST_PIXEL_SUM: u64 = 264_923_200;

/// Count the images of each digit class.
fn class_counts(labels: &Array1<u8>) -> [usize; 10] {
    let mut counts = [0usize; 10];
    for &label in labels.iter() {
        counts[label as usize] += 1;
    }
    counts
}

/// Sum every pixel of the image matrix.
fn pixel_sum(features: &Array2<u8>) -> u64 {
    features.iter().map(|&pixel| u64::from(pixel)).sum()
}

/// Assert the invariants that hold for every MNIST subset: the shapes, the
/// label domain, and a background pixel in the top-left corner of every image.
fn assert_mnist_shape(features: &Array2<u8>, labels: &Array1<u8>, n_samples: usize) {
    assert_eq!(features.shape(), &[n_samples, N_PIXELS]);
    assert_eq!(labels.len(), n_samples);

    // Every label names a digit.
    for (row, &label) in labels.iter().enumerate() {
        assert!(label <= 9, "label[{}] = {} is not a digit", row, label);
    }

    // The digits sit centered in the frame, so the corner pixel is background in
    // every image. This catches a row/column transposition of the flattening.
    for row in 0..features.nrows() {
        assert_eq!(
            features[[row, 0]],
            0,
            "image {} has ink in its top-left corner",
            row
        );
    }
}

#[test]
// Verifies that the MNIST training partition loads with the correct shapes,
// class balance, and pinned pixel values.
fn test_load_mnist() {
    let download_dir = "./test_load_mnist"; // the loader creates this directory if it is missing

    let dataset = Mnist::new(download_dir);
    let (features, labels) = dataset.data().unwrap();

    assert_mnist_shape(features, labels, N_TRAIN);
    assert_eq!(
        class_counts(labels),
        TRAIN_CLASS_COUNTS,
        "the training class balance should match the published counts"
    );
    assert_eq!(
        pixel_sum(features),
        TRAIN_PIXEL_SUM,
        "the training pixel sum should match"
    );

    // The partition keeps its source order: the first image is a 5, the last an 8.
    assert_eq!(labels[0], 5);
    assert_eq!(labels[N_TRAIN - 1], 8);

    // The first image, pinned. It holds 166 inked pixels that sum to 27,525, and
    // its first inked pixel is at flat index 152.
    let first = features.row(0);
    assert_eq!(first.iter().filter(|&&pixel| pixel != 0).count(), 166);
    assert_eq!(first.iter().map(|&p| u64::from(p)).sum::<u64>(), 27_525);
    assert_eq!(first[152], 3);
    assert_eq!(
        first.iter().position(|&pixel| pixel != 0),
        Some(152),
        "the first inked pixel should be at flat index 152"
    );

    // Pixels use the full 0..=255 range.
    assert_eq!(*features.iter().min().unwrap(), 0);
    assert_eq!(*features.iter().max().unwrap(), 255);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that images() reshapes the same buffer to 28x28 without copying.
fn test_mnist_images_view() {
    let download_dir = "./test_mnist_images_view";

    let dataset = Mnist::new_test(download_dir);
    let images = dataset.images().unwrap();
    let features = dataset.features().unwrap();

    assert_eq!(images.shape(), &[N_TEST, 28, 28]);

    // The view indexes the same bytes as the flat matrix: `[image, row, col]`
    // maps to flat index `row * 28 + col`.
    for image in [0usize, 1, N_TEST - 1] {
        for row in [0usize, 5, 14, 27] {
            for col in [0usize, 12, 27] {
                assert_eq!(
                    images[[image, row, col]],
                    features[[image, row * 28 + col]],
                    "image {} pixel ({}, {}) disagrees between the two shapes",
                    image,
                    row,
                    col
                );
            }
        }
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that new_test() loads the 10,000-image test partition.
fn test_mnist_test_subset() {
    let download_dir = "./test_mnist_test_subset";

    let dataset = Mnist::new_test(download_dir);
    let (features, labels) = dataset.data().unwrap();

    assert_mnist_shape(features, labels, N_TEST);
    assert_eq!(
        class_counts(labels),
        TEST_CLASS_COUNTS,
        "the test class balance should match the published counts"
    );
    assert_eq!(
        pixel_sum(features),
        TEST_PIXEL_SUM,
        "the test pixel sum should match"
    );

    // The partition keeps its source order: the first image is a 7, the last a 6.
    assert_eq!(labels[0], 7);
    assert_eq!(labels[N_TEST - 1], 6);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that new_all() concatenates the training partition and then the test
// partition, in that order.
fn test_mnist_all_subset() {
    let download_dir = "./test_mnist_all_subset";

    let dataset = Mnist::new_all(download_dir);
    let (features, labels) = dataset.data().unwrap();

    assert_mnist_shape(features, labels, N_ALL);

    // Train comes first, then test. The boundary sits at index 60,000: the last
    // training label is an 8 and the first test label is a 7.
    assert_eq!(labels[0], 5);
    assert_eq!(labels[N_TRAIN - 1], 8);
    assert_eq!(labels[N_TRAIN], 7);
    assert_eq!(labels[N_ALL - 1], 6);

    // Every class count is the sum of the two partitions' counts.
    let counts = class_counts(labels);
    for digit in 0..10 {
        assert_eq!(
            counts[digit],
            TRAIN_CLASS_COUNTS[digit] + TEST_CLASS_COUNTS[digit],
            "class {} should hold both partitions",
            digit
        );
    }
    assert_eq!(pixel_sum(features), TRAIN_PIXEL_SUM + TEST_PIXEL_SUM);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that MNIST reuses a cached file instead of a new download.
fn test_mnist_no_need_download() {
    let download_dir = "./test_mnist_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load primes the cache. The second instance then reuses it.
    Mnist::new_test(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join(TEST_IMAGES_FILENAME),
            TEST_IMAGES_SHA256
        )
        .unwrap(),
        "the cached image file should match the expected SHA256"
    );

    let dataset = Mnist::new_test(download_dir);
    let (_features, labels) = dataset.data().unwrap();
    assert_eq!(labels.len(), N_TEST);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake IDX file and overwrites it
// with the real one.
fn test_mnist_overwrite() {
    let download_dir = "./test_mnist_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        // A short fake file cannot even hold an IDX header. The SHA256 check
        // must reject it before the parser reads it.
        let mut fake_images = File::create(download_dir_path.join(TEST_IMAGES_FILENAME)).unwrap();
        fake_images.write_all(b"fake data").unwrap();
        let mut fake_labels = File::create(download_dir_path.join(TEST_LABELS_FILENAME)).unwrap();
        fake_labels.write_all(b"fake data").unwrap();
    }

    let dataset = Mnist::new_test(download_dir);
    let (_features, labels) = dataset.data().unwrap();
    assert_eq!(labels.len(), N_TEST);

    assert!(
        file_sha256_matches(
            &download_dir_path.join(TEST_IMAGES_FILENAME),
            TEST_IMAGES_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays and consumes the dataset.
fn test_mnist_into_data() {
    let download_dir = "./test_mnist_into_data";

    let dataset = Mnist::new_test(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. Both arrays are now fully owned.

    assert_eq!(features.shape(), &[N_TEST, N_PIXELS]);
    assert_eq!(labels.len(), N_TEST);

    // The caller can mutate the owned data directly, with no `to_owned()` clone.
    features[[0, 0]] = 255;
    assert_eq!(features[[0, 0]], 255);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned arrays and leaves the instance reusable.
fn test_mnist_take_data() {
    let download_dir = "./test_mnist_take_data";

    let mut dataset = Mnist::new_test(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[N_TEST, N_PIXELS]);
    assert_eq!(labels.len(), N_TEST);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file and yields the same shapes.
    let (features, labels) = dataset.data().unwrap();
    assert_eq!(features.shape(), &[N_TEST, N_PIXELS]);
    assert_eq!(labels.len(), N_TEST);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_mnist_get_data() {
    let download_dir = "./test_mnist_get_data";

    let dataset = Mnist::new_test(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[N_TEST, N_PIXELS]);
    assert_eq!(labels.len(), N_TEST);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached arrays in place.
fn test_mnist_get_data_mut() {
    let download_dir = "./test_mnist_get_data_mut";

    let mut dataset = Mnist::new_test(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() binarizes the first image in place, with no clone and no
    // reload.
    dataset.data().unwrap();
    if let Some((features, _labels)) = dataset.get_data_mut() {
        for col in 0..N_PIXELS {
            features[[0, col]] = u8::from(features[[0, col]] > 127);
        }
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _labels) = dataset.data().unwrap();
    assert!(
        features.row(0).iter().all(|&pixel| pixel <= 1),
        "the first image should hold only 0 and 1 after binarization"
    );
    assert!(
        features.row(1).iter().any(|&pixel| pixel > 1),
        "the second image should stay untouched"
    );

    remove_dir_all(download_dir).unwrap();
}
