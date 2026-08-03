mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::dataset::breast_cancer::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/breast_cancer.rs`.
const BREAST_CANCER_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data";
const BREAST_CANCER_SHA256: &str =
    "d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a";

#[test]
// Verifies that the Breast Cancer dataset loads with the correct feature shape and label count.
fn test_load_breast_cancer() {
    let download_dir = "./test_load_breast_cancer"; // the loader creates the directory if it does not exist

    let dataset = BreastCancer::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_eq!(features.shape(), &[569, 30]);
    assert_eq!(labels.len(), 569);

    let (features, labels) = dataset.data().unwrap(); // data() also returns the features and labels

    // Semantic assertions: labels must be one of the two known diagnoses, and
    // both classes must be present.
    let mut has_malignant = false;
    let mut has_benign = false;
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label == "malignant" || label == "benign",
            "labels[{}] = {:?} is not a known diagnosis",
            i,
            label
        );
        if label == "malignant" {
            has_malignant = true;
        }
        if label == "benign" {
            has_benign = true;
        }
    }
    assert!(has_malignant, "labels must contain at least one malignant");
    assert!(has_benign, "labels must contain at least one benign");

    // Semantic assertions: all feature values must be finite and non-negative
    // (every measurement is a size/shape statistic, so none can be negative).
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let val = features[[row, col]];
            assert!(
                val.is_finite() && val >= 0.0,
                "feature[{}, {}] = {} is not a finite non-negative value",
                row,
                col,
                val
            );
        }
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Breast Cancer loading uses a pre-downloaded cached file without re-downloading.
fn test_breast_cancer_no_need_download() {
    let download_dir = "./test_breast_cancer_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download the Breast Cancer dataset in advance, under the filename the loader expects
    download_to(
        BREAST_CANCER_URL,
        download_dir_path,
        Some("breast_cancer.csv"),
    )
    .unwrap();

    // should use the cached Breast Cancer dataset
    let dataset = BreastCancer::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Breast Cancer data file and
// overwrites it with the real dataset.
fn test_breast_cancer_overwrite() {
    let download_dir = "./test_breast_cancer_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Breast Cancer dataset in advance
    {
        let path = download_dir_path.join("breast_cancer.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Breast Cancer dataset
    let dataset = BreastCancer::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // check that the loader overwrote the fake file
    assert!(
        file_sha256_matches(
            &download_dir_path.join("breast_cancer.csv"),
            BREAST_CANCER_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and labels and consumes the dataset.
fn test_breast_cancer_into_data() {
    let download_dir = "./test_breast_cancer_into_data";

    let dataset = BreastCancer::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // into_data() consumes `dataset`. The returned `features` and `labels` are fully owned.

    assert_eq!(features.shape(), &[569, 30]);
    assert_eq!(labels.len(), 569);

    // Owned labels are correct: one of the two known diagnoses.
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label == "malignant" || label == "benign",
            "labels[{}] = {:?} is not a known diagnosis",
            i,
            label
        );
    }

    // The caller can mutate owned data directly, with no `to_owned()` clone.
    features[[0, 0]] = 15.0;
    assert_eq!(features[[0, 0]], 15.0);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_breast_cancer_take_data() {
    let download_dir = "./test_breast_cancer_take_data";

    let mut dataset = BreastCancer::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[569, 30]);
    assert_eq!(labels.len(), 569);

    // take_data() resets the instance to unloaded, but it stays usable. The next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[569, 30]);
    assert_eq!(reloaded_labels.len(), 569);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_breast_cancer_get_data() {
    let download_dir = "./test_breast_cancer_get_data";

    let dataset = BreastCancer::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading. Then get_data() returns the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[569, 30]);
    assert_eq!(labels.len(), 569);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_breast_cancer_get_data_mut() {
    let download_dir = "./test_breast_cancer_get_data_mut";

    let mut dataset = BreastCancer::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load the dataset. Then mutate the cached features in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((features, _labels)) = dataset.get_data_mut() {
        features[[0, 0]] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _labels) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 99.0);

    remove_dir_all(download_dir).unwrap();
}
