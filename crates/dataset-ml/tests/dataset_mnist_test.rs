#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::mnist::Mnist;
use dataset_ml::table::{ColumnData, Table};
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

/// The two column names, in source order.
const COLUMN_NAMES: [&str; 2] = ["pixels", "digit"];

/// Images per digit class in the training partition, for classes `0` to `9`.
const TRAIN_CLASS_COUNTS: [usize; 10] =
    [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949];

/// Images per digit class in the test partition, for classes `0` to `9`.
const TEST_CLASS_COUNTS: [usize; 10] = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009];

/// Sum of every pixel of the training partition.
const TRAIN_PIXEL_SUM: u64 = 1_567_298_545;

/// Sum of every pixel of the test partition.
const TEST_PIXEL_SUM: u64 = 264_923_200;

/// Borrow the `pixels` column of the table.
fn pixels_of(table: &Table) -> &Array2<u8> {
    table.column("pixels").unwrap().as_bytes().unwrap()
}

/// Borrow the `digit` column of the table.
fn digits_of(table: &Table) -> &Array1<i64> {
    table.column("digit").unwrap().as_integer().unwrap()
}

/// Count the images of each digit class.
fn class_counts(digits: &Array1<i64>) -> [usize; 10] {
    let mut counts = [0usize; 10];
    for &digit in digits.iter() {
        counts[digit as usize] += 1;
    }
    counts
}

/// Sum every pixel of the image matrix.
fn pixel_sum(pixels: &Array2<u8>) -> u64 {
    pixels.iter().map(|&pixel| u64::from(pixel)).sum()
}

/// Assert the column layout the documentation claims: the names, the named
/// constants, and the column types.
fn assert_mnist_columns(table: &Table) {
    assert_eq!(table.n_columns(), 2);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    assert_eq!(Mnist::FEATURE_NAMES, ["pixels"]);
    assert_eq!(Mnist::TARGET, "digit");

    let pixels = table.column(Mnist::FEATURE_NAMES[0]).unwrap();
    assert!(
        matches!(pixels.data(), ColumnData::Bytes(_)),
        "pixels should be a bytes column"
    );

    let digit = table.column(Mnist::TARGET).unwrap();
    assert!(
        matches!(digit.data(), ColumnData::Integer(_)),
        "digit should be an integer column"
    );
}

/// Assert the invariants that hold for every MNIST subset: the shapes, the
/// label domain, and a background pixel in the top-left corner of every image.
fn assert_mnist_shape(table: &Table, n_samples: usize) {
    assert_mnist_columns(table);
    assert_eq!(table.n_samples(), n_samples);

    let pixels = pixels_of(table);
    let digits = digits_of(table);
    assert_eq!(pixels.shape(), &[n_samples, N_PIXELS]);
    assert_eq!(digits.len(), n_samples);

    // Every label names a digit.
    for (row, &digit) in digits.iter().enumerate() {
        assert!(
            (0..=9).contains(&digit),
            "digit[{}] = {} is not a digit",
            row,
            digit
        );
    }

    // The digits sit centered in the frame, so the corner pixel is background in
    // every image. This catches a row/column transposition of the flattening.
    for row in 0..pixels.nrows() {
        assert_eq!(
            pixels[[row, 0]],
            0,
            "image {} has ink in its top-left corner",
            row
        );
    }
}

#[test]
// Verifies that the MNIST training partition loads with the correct column
// layout, shapes, class balance, and pinned pixel values.
fn test_load_mnist() {
    let download_dir = "./test_load_mnist"; // the loader creates this directory if it is missing

    let dataset = Mnist::new(download_dir);
    let table = dataset.data().unwrap();

    assert_mnist_shape(table, N_TRAIN);

    let pixels = pixels_of(table);
    let digits = digits_of(table);

    assert_eq!(
        class_counts(digits),
        TRAIN_CLASS_COUNTS,
        "the training class balance should match the published counts"
    );
    assert_eq!(
        pixel_sum(pixels),
        TRAIN_PIXEL_SUM,
        "the training pixel sum should match"
    );

    // The partition keeps its source order: the first image is a 5, the last an 8.
    assert_eq!(digits[0], 5);
    assert_eq!(digits[N_TRAIN - 1], 8);

    // The first image, pinned. It holds 166 inked pixels that sum to 27,525, and
    // its first inked pixel is at flat index 152.
    let first = pixels.row(0);
    assert_eq!(first.iter().filter(|&&pixel| pixel != 0).count(), 166);
    assert_eq!(first.iter().map(|&p| u64::from(p)).sum::<u64>(), 27_525);
    assert_eq!(first[152], 3);
    assert_eq!(
        first.iter().position(|&pixel| pixel != 0),
        Some(152),
        "the first inked pixel should be at flat index 152"
    );

    // Pixels use the full 0..=255 range.
    assert_eq!(*pixels.iter().min().unwrap(), 0);
    assert_eq!(*pixels.iter().max().unwrap(), 255);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a 28x28 view of the pixels column reads the same bytes as the
// flat rows, and that the feature matrix holds the same pixels as `f64`.
fn test_mnist_images_view() {
    let download_dir = "./test_mnist_images_view";

    let dataset = Mnist::new_test(download_dir);
    let table = dataset.data().unwrap();
    let pixels = pixels_of(table);

    let images = pixels
        .view()
        .into_shape_with_order((N_TEST, 28, 28))
        .unwrap();
    assert_eq!(images.shape(), &[N_TEST, 28, 28]);

    // The view indexes the same bytes as the flat matrix: `[image, row, col]`
    // maps to flat index `row * 28 + col`.
    for image in [0usize, 1, N_TEST - 1] {
        for row in [0usize, 5, 14, 27] {
            for col in [0usize, 12, 27] {
                assert_eq!(
                    images[[image, row, col]],
                    pixels[[image, row * 28 + col]],
                    "image {} pixel ({}, {}) disagrees between the two shapes",
                    image,
                    row,
                    col
                );
            }
        }
    }

    // The bytes column contributes its full width to the feature matrix.
    let features = table.numeric_matrix(&Mnist::FEATURE_NAMES).unwrap();
    assert_eq!(features.shape(), &[N_TEST, N_PIXELS]);
    for image in [0usize, 1, N_TEST - 1] {
        for col in [0usize, 152, N_PIXELS - 1] {
            assert_eq!(features[[image, col]], f64::from(pixels[[image, col]]));
        }
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that new_test() loads the 10,000-image test partition.
fn test_mnist_test_subset() {
    let download_dir = "./test_mnist_test_subset";

    let dataset = Mnist::new_test(download_dir);
    let table = dataset.data().unwrap();

    assert_mnist_shape(table, N_TEST);

    let pixels = pixels_of(table);
    let digits = digits_of(table);

    assert_eq!(
        class_counts(digits),
        TEST_CLASS_COUNTS,
        "the test class balance should match the published counts"
    );
    assert_eq!(
        pixel_sum(pixels),
        TEST_PIXEL_SUM,
        "the test pixel sum should match"
    );

    // The partition keeps its source order: the first image is a 7, the last a 6.
    assert_eq!(digits[0], 7);
    assert_eq!(digits[N_TEST - 1], 6);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that new_all() concatenates the training partition and then the test
// partition, in that order.
fn test_mnist_all_subset() {
    let download_dir = "./test_mnist_all_subset";

    let dataset = Mnist::new_all(download_dir);
    let table = dataset.data().unwrap();

    assert_mnist_shape(table, N_ALL);

    let pixels = pixels_of(table);
    let digits = digits_of(table);

    // Train comes first, then test. The boundary sits at index 60,000: the last
    // training label is an 8 and the first test label is a 7.
    assert_eq!(digits[0], 5);
    assert_eq!(digits[N_TRAIN - 1], 8);
    assert_eq!(digits[N_TRAIN], 7);
    assert_eq!(digits[N_ALL - 1], 6);

    // Every class count is the sum of the two partitions' counts.
    let counts = class_counts(digits);
    for digit in 0..10 {
        assert_eq!(
            counts[digit],
            TRAIN_CLASS_COUNTS[digit] + TEST_CLASS_COUNTS[digit],
            "class {} should hold both partitions",
            digit
        );
    }
    assert_eq!(pixel_sum(pixels), TRAIN_PIXEL_SUM + TEST_PIXEL_SUM);

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
    assert_eq!(dataset.data().unwrap().n_samples(), N_TEST);

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
    assert_eq!(dataset.data().unwrap().n_samples(), N_TEST);

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
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_mnist_into_data() {
    let download_dir = "./test_mnist_into_data";

    let dataset = Mnist::new_test(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The table is now fully owned.

    assert_mnist_columns(&table);
    assert_eq!(table.n_samples(), N_TEST);
    assert_eq!(pixels_of(&table).shape(), &[N_TEST, N_PIXELS]);
    assert_eq!(digits_of(&table).len(), N_TEST);

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Bytes(values)) = table.column_mut("pixels").map(|c| c.data_mut()) {
        values[[0, 0]] = 255;
    }
    assert_eq!(pixels_of(&table)[[0, 0]], 255);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the instance reusable.
fn test_mnist_take_data() {
    let download_dir = "./test_mnist_take_data";

    let mut dataset = Mnist::new_test(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(pixels_of(&table).shape(), &[N_TEST, N_PIXELS]);
    assert_eq!(digits_of(&table).len(), N_TEST);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file and yields the same shapes.
    let table = dataset.data().unwrap();
    assert_eq!(pixels_of(table).shape(), &[N_TEST, N_PIXELS]);
    assert_eq!(digits_of(table).len(), N_TEST);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_mnist_get_data() {
    let download_dir = "./test_mnist_get_data";

    let dataset = Mnist::new_test(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(pixels_of(table).shape(), &[N_TEST, N_PIXELS]);
    assert_eq!(digits_of(table).len(), N_TEST);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place.
fn test_mnist_get_data_mut() {
    let download_dir = "./test_mnist_get_data_mut";

    let mut dataset = Mnist::new_test(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() binarizes the first image in place, with no clone and no
    // reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Bytes(values)) = table.column_mut("pixels").map(|c| c.data_mut())
    {
        for col in 0..N_PIXELS {
            values[[0, col]] = u8::from(values[[0, col]] > 127);
        }
    }

    // The change persisted in the cache: a later access observes it.
    let pixels = pixels_of(dataset.data().unwrap());
    assert!(
        pixels.row(0).iter().all(|&pixel| pixel <= 1),
        "the first image should hold only 0 and 1 after binarization"
    );
    assert!(
        pixels.row(1).iter().any(|&pixel| pixel > 1),
        "the second image should stay untouched"
    );

    remove_dir_all(download_dir).unwrap();
}
