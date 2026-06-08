mod common;

use common::file_sha256_matches;
use dataset_ml::digits::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Digits dataset file (`optdigits.tes`).
const DIGITS_SHA256: &str = "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8";

#[test]
// Verifies that the Digits dataset loads with the correct feature shape and label count.
fn test_load_digits() {
    let download_dir = "./test_load_digits"; // the code will create the directory if it doesn't exist

    let dataset = Digits::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_eq!(features.shape(), &[1797, 64]);
    assert_eq!(labels.len(), 1797);

    let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels

    // Semantic assertions: labels are exactly the ten digit classes 0..=9.
    let unique_labels: std::collections::HashSet<_> = labels.iter().copied().collect();
    assert_eq!(
        unique_labels.len(),
        10,
        "Digits should have exactly 10 unique classes"
    );
    for digit in 0u8..=9 {
        assert!(
            unique_labels.contains(&digit),
            "labels must contain the digit {}",
            digit
        );
    }

    // Semantic assertions: every pixel is a finite integer-valued intensity in 0..=16.
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let val = features[[row, col]];
            assert!(
                val.is_finite(),
                "feature[{}, {}] = {} is not finite",
                row,
                col,
                val
            );
            assert!(
                (0.0..=16.0).contains(&val) && val.fract() == 0.0,
                "feature[{}, {}] = {} is not an integer in 0..=16",
                row,
                col,
                val
            );
        }
    }

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Digits loading uses a pre-existing cached file without re-downloading.
fn test_digits_no_need_download() {
    let download_dir = "./test_load_digits_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Prime the cache by loading once, then confirm a second instance reuses it.
    Digits::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(&download_dir_path.join("digits.csv"), DIGITS_SHA256).unwrap(),
        "cached digits.csv should match the expected SHA256"
    );

    let dataset = Digits::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Digits data file is detected and overwritten with the real dataset.
fn test_digits_overwrite() {
    let download_dir = "./test_load_digits_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Digits dataset in advance
    {
        let digits_path = download_dir_path.join("digits.csv");
        let mut fake_digits = File::create(digits_path).unwrap();
        fake_digits.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Digits dataset
    let dataset = Digits::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(file_sha256_matches(&download_dir_path.join("digits.csv"), DIGITS_SHA256).unwrap());

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and labels, consuming the dataset.
fn test_digits_into_data() {
    let download_dir = "./test_digits_into_data";

    let dataset = Digits::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; `features`/`labels` are fully owned.

    assert_eq!(features.shape(), &[1797, 64]);
    assert_eq!(labels.len(), 1797);

    // Owned labels are correct: exactly the ten digit classes.
    let unique_labels: std::collections::HashSet<_> = labels.iter().copied().collect();
    assert_eq!(
        unique_labels.len(),
        10,
        "Digits should have exactly 10 unique classes"
    );

    // Owned data can be mutated directly, with no `to_owned()` clone.
    features[[0, 0]] = 5.0;
    assert_eq!(features[[0, 0]], 5.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_digits_take_data() {
    let download_dir = "./test_digits_take_data";

    let mut dataset = Digits::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[1797, 64]);
    assert_eq!(labels.len(), 1797);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[1797, 64]);
    assert_eq!(reloaded_labels.len(), 1797);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_digits_get_data() {
    let download_dir = "./test_digits_get_data";

    let dataset = Digits::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[1797, 64]);
    assert_eq!(labels.len(), 1797);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_digits_get_data_mut() {
    let download_dir = "./test_digits_get_data_mut";

    let mut dataset = Digits::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load, then mutate the cached features in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((features, _labels)) = dataset.get_data_mut() {
        features[[0, 0]] = 9.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _labels) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 9.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
