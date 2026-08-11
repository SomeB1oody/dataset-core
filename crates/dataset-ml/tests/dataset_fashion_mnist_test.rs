#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::Mnist;
use dataset_ml::dataset::fashion_mnist::FashionMnist;
use ndarray::{Array1, Array2};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached (decompressed) test-partition image file.
const TEST_IMAGES_SHA256: &str = "5b4141f0afbad91edebe8549f8fcffe087ea10ca49f1dbef5c9a5cd8815ce37b";

/// Cache filename of the decompressed test-partition image file.
const TEST_IMAGES_FILENAME: &str = "fashion-t10k-images-idx3-ubyte";

/// Cache filename of the decompressed test-partition label file.
const TEST_LABELS_FILENAME: &str = "fashion-t10k-labels-idx1-ubyte";

/// Number of images in the training partition.
const N_TRAIN: usize = 60_000;

/// Number of images in the test partition.
const N_TEST: usize = 10_000;

/// Number of images in both partitions together.
const N_ALL: usize = 70_000;

/// Number of pixels per image (28 × 28).
const N_PIXELS: usize = 784;

/// Images per class in the training partition. The source balances the classes
/// exactly.
const N_TRAIN_PER_CLASS: usize = 6_000;

/// Images per class in the test partition.
const N_TEST_PER_CLASS: usize = 1_000;

/// Sum of every pixel of the training partition.
const TRAIN_PIXEL_SUM: u64 = 3_431_114_169;

/// Sum of every pixel of the test partition.
const TEST_PIXEL_SUM: u64 = 573_469_082;

/// Count the images of each garment class.
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

/// Assert the invariants that hold for every Fashion-MNIST subset: the shapes
/// and the label domain.
fn assert_fashion_shape(features: &Array2<u8>, labels: &Array1<u8>, n_samples: usize) {
    assert_eq!(features.shape(), &[n_samples, N_PIXELS]);
    assert_eq!(labels.len(), n_samples);

    for (row, &label) in labels.iter().enumerate() {
        assert!(
            (label as usize) < FashionMnist::CLASS_NAMES.len(),
            "label[{}] = {} names no class",
            row,
            label
        );
    }
}

#[test]
// Verifies that CLASS_NAMES names the 10 garment classes in the source's order.
fn test_fashion_mnist_class_names() {
    assert_eq!(FashionMnist::CLASS_NAMES.len(), 10);
    assert_eq!(
        FashionMnist::CLASS_NAMES,
        [
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
        ]
    );
}

#[test]
// Verifies that the Fashion-MNIST training partition loads with the correct
// shapes, exact class balance, and pinned pixel values.
fn test_load_fashion_mnist() {
    let download_dir = "./test_load_fashion_mnist"; // the loader creates this directory if it is missing

    let dataset = FashionMnist::new(download_dir);
    let (features, labels) = dataset.data().unwrap();

    assert_fashion_shape(features, labels, N_TRAIN);
    assert_eq!(
        class_counts(labels),
        [N_TRAIN_PER_CLASS; 10],
        "the training partition should hold exactly 6,000 images per class"
    );
    assert_eq!(
        pixel_sum(features),
        TRAIN_PIXEL_SUM,
        "the training pixel sum should match"
    );

    // The partition keeps its source order: the first image is an ankle boot and
    // the last a sandal.
    assert_eq!(labels[0], 9);
    assert_eq!(FashionMnist::CLASS_NAMES[labels[0] as usize], "Ankle boot");
    assert_eq!(labels[N_TRAIN - 1], 5);

    // The first image, pinned. It holds 433 inked pixels that sum to 76,247, and
    // its first inked pixel is at flat index 96.
    let first = features.row(0);
    assert_eq!(first.iter().filter(|&&pixel| pixel != 0).count(), 433);
    assert_eq!(first.iter().map(|&p| u64::from(p)).sum::<u64>(), 76_247);
    assert_eq!(
        first.iter().position(|&pixel| pixel != 0),
        Some(96),
        "the first inked pixel should be at flat index 96"
    );
    assert_eq!(first[96], 1);

    // Pixels use the full 0..=255 range.
    assert_eq!(*features.iter().min().unwrap(), 0);
    assert_eq!(*features.iter().max().unwrap(), 255);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that images() reshapes the same buffer to 28x28 without copying.
fn test_fashion_mnist_images_view() {
    let download_dir = "./test_fashion_mnist_images_view";

    let dataset = FashionMnist::new_test(download_dir);
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
fn test_fashion_mnist_test_subset() {
    let download_dir = "./test_fashion_mnist_test_subset";

    let dataset = FashionMnist::new_test(download_dir);
    let (features, labels) = dataset.data().unwrap();

    assert_fashion_shape(features, labels, N_TEST);
    assert_eq!(
        class_counts(labels),
        [N_TEST_PER_CLASS; 10],
        "the test partition should hold exactly 1,000 images per class"
    );
    assert_eq!(
        pixel_sum(features),
        TEST_PIXEL_SUM,
        "the test pixel sum should match"
    );

    assert_eq!(labels[0], 9);
    assert_eq!(labels[N_TEST - 1], 5);

    // The first image, pinned.
    let first = features.row(0);
    assert_eq!(first.iter().filter(|&&pixel| pixel != 0).count(), 267);
    assert_eq!(first.iter().map(|&p| u64::from(p)).sum::<u64>(), 33_456);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that new_all() concatenates the training partition and then the test
// partition, in that order.
fn test_fashion_mnist_all_subset() {
    let download_dir = "./test_fashion_mnist_all_subset";

    let dataset = FashionMnist::new_all(download_dir);
    let (features, labels) = dataset.data().unwrap();

    assert_fashion_shape(features, labels, N_ALL);

    // Train comes first, then test. The boundary sits at index 60,000.
    assert_eq!(labels[N_TRAIN - 1], 5);
    assert_eq!(labels[N_TRAIN], 9);
    assert_eq!(labels[N_ALL - 1], 5);

    // Both partitions are exactly balanced, so their sum is too.
    assert_eq!(
        class_counts(labels),
        [N_TRAIN_PER_CLASS + N_TEST_PER_CLASS; 10]
    );
    assert_eq!(pixel_sum(features), TRAIN_PIXEL_SUM + TEST_PIXEL_SUM);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Fashion-MNIST and MNIST can share one storage directory. Their
// upstream files carry the same names, so this loader prefixes its cached names
// with `fashion-`.
fn test_fashion_mnist_shares_dir_with_mnist() {
    let download_dir = "./test_fashion_mnist_shares_dir_with_mnist";
    let download_dir_path = Path::new(download_dir);

    let fashion = FashionMnist::new_test(download_dir);
    let (fashion_features, fashion_labels) = fashion.data().unwrap();

    let mnist = Mnist::new_test(download_dir);
    let (mnist_features, mnist_labels) = mnist.data().unwrap();

    // Both cache files sit in the one directory, under different names.
    assert!(download_dir_path.join(TEST_IMAGES_FILENAME).exists());
    assert!(download_dir_path.join("t10k-images-idx3-ubyte").exists());

    // Neither load overwrote the other: the two datasets hold different bytes.
    assert_eq!(pixel_sum(fashion_features), TEST_PIXEL_SUM);
    assert_ne!(pixel_sum(mnist_features), TEST_PIXEL_SUM);
    assert_eq!(fashion_labels[0], 9);
    assert_eq!(mnist_labels[0], 7);

    // A reload of the first dataset still reads its own cached file.
    let fashion_again = FashionMnist::new_test(download_dir);
    assert_eq!(pixel_sum(fashion_again.features().unwrap()), TEST_PIXEL_SUM);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Fashion-MNIST reuses a cached file instead of a new download.
fn test_fashion_mnist_no_need_download() {
    let download_dir = "./test_fashion_mnist_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load primes the cache. The second instance then reuses it.
    FashionMnist::new_test(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join(TEST_IMAGES_FILENAME),
            TEST_IMAGES_SHA256
        )
        .unwrap(),
        "the cached image file should match the expected SHA256"
    );

    let dataset = FashionMnist::new_test(download_dir);
    let (_features, labels) = dataset.data().unwrap();
    assert_eq!(labels.len(), N_TEST);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake IDX file and overwrites it
// with the real one.
fn test_fashion_mnist_overwrite() {
    let download_dir = "./test_fashion_mnist_overwrite";
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

    let dataset = FashionMnist::new_test(download_dir);
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
fn test_fashion_mnist_into_data() {
    let download_dir = "./test_fashion_mnist_into_data";

    let dataset = FashionMnist::new_test(download_dir);
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
fn test_fashion_mnist_take_data() {
    let download_dir = "./test_fashion_mnist_take_data";

    let mut dataset = FashionMnist::new_test(download_dir);
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
fn test_fashion_mnist_get_data() {
    let download_dir = "./test_fashion_mnist_get_data";

    let dataset = FashionMnist::new_test(download_dir);
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
fn test_fashion_mnist_get_data_mut() {
    let download_dir = "./test_fashion_mnist_get_data_mut";

    let mut dataset = FashionMnist::new_test(download_dir);
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
