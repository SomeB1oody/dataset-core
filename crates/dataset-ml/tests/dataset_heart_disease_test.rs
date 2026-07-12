mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::heart_disease::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/heart_disease.rs`.
const HEART_DISEASE_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data";
const HEART_DISEASE_SHA256: &str =
    "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8";

/// The Heart Disease dataset has this many samples.
const N_SAMPLES: usize = 303;

/// Assert the Heart Disease dataset invariants: the schema shape, the diagnosis
/// target domain, and the numeric feature domain including the `?` → `NaN` mapping.
fn assert_heart_disease_semantics(features: &ndarray::Array2<f64>, labels: &ndarray::Array1<u8>) {
    assert_eq!(features.shape(), &[N_SAMPLES, 13]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Diagnosis is 0..=4, and both absence (0) and presence (>0) are present.
    let mut has_absence = false;
    let mut has_presence = false;
    for (i, &y) in labels.iter().enumerate() {
        assert!(
            y <= 4,
            "labels[{i}] = {y} is outside the 0..=4 diagnosis range"
        );
        if y == 0 {
            has_absence = true;
        } else {
            has_presence = true;
        }
    }
    assert!(has_absence, "labels must contain at least one absence (0)");
    assert!(
        has_presence,
        "labels must contain at least one presence (>0)"
    );

    // Every non-missing feature value is finite; NaN marks the source's `?`. Only
    // `ca` (column 11) and `thal` (column 12) carry missing values.
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let v = features[[row, col]];
            if v.is_nan() {
                assert!(
                    col == 11 || col == 12,
                    "unexpected NaN at feature[{row}, {col}] (only ca/thal may be missing)"
                );
            } else {
                assert!(v.is_finite(), "feature[{row}, {col}] = {v} is not finite");
            }
        }
    }

    // `age` (column 0) is always present and positive.
    for row in 0..features.nrows() {
        let age = features[[row, 0]];
        assert!(
            age.is_finite() && age > 0.0,
            "age at row {row} = {age} is not a positive finite value"
        );
    }

    // The exact missing-value counts: 4 in `ca`, 2 in `thal`.
    let ca_missing = (0..features.nrows())
        .filter(|&row| features[[row, 11]].is_nan())
        .count();
    let thal_missing = (0..features.nrows())
        .filter(|&row| features[[row, 12]].is_nan())
        .count();
    assert_eq!(ca_missing, 4, "expected 4 missing `ca` values");
    assert_eq!(thal_missing, 2, "expected 2 missing `thal` values");
}

#[test]
// Verifies that the Heart Disease dataset loads with the correct shape, target
// domain, and numeric feature domain (including the ? -> NaN mapping).
fn test_load_heart_disease() {
    let download_dir = "./test_load_heart_disease"; // the code will create the directory if it doesn't exist

    let dataset = HeartDisease::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_heart_disease_semantics(features, labels);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Heart Disease loading uses a pre-downloaded cached file without re-downloading.
fn test_heart_disease_no_need_download() {
    let download_dir = "./test_heart_disease_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Heart Disease dataset in advance, under the filename the loader expects
    download_to(
        HEART_DISEASE_URL,
        download_dir_path,
        Some("heart_disease.csv"),
    )
    .unwrap();

    // should use cached Heart Disease dataset
    let dataset = HeartDisease::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Heart Disease data file is detected and overwritten with the real dataset.
fn test_heart_disease_overwrite() {
    let download_dir = "./test_heart_disease_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Heart Disease dataset in advance
    {
        let path = download_dir_path.join("heart_disease.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Heart Disease dataset
    let dataset = HeartDisease::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("heart_disease.csv"),
            HEART_DISEASE_SHA256
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and labels, consuming the dataset.
fn test_heart_disease_into_data() {
    let download_dir = "./test_heart_disease_into_data";

    let dataset = HeartDisease::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; `features`/`labels` are fully owned.

    assert_eq!(features.shape(), &[N_SAMPLES, 13]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Owned labels are correct: within the 0..=4 diagnosis range.
    for (i, &y) in labels.iter().enumerate() {
        assert!(
            y <= 4,
            "labels[{i}] = {y} is outside the 0..=4 diagnosis range"
        );
    }

    // Owned data can be mutated directly, with no `to_owned()` clone.
    features[[0, 0]] = 60.0;
    assert_eq!(features[[0, 0]], 60.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_heart_disease_take_data() {
    let download_dir = "./test_heart_disease_take_data";

    let mut dataset = HeartDisease::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[N_SAMPLES, 13]);
    assert_eq!(labels.len(), N_SAMPLES);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[N_SAMPLES, 13]);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_heart_disease_get_data() {
    let download_dir = "./test_heart_disease_get_data";

    let dataset = HeartDisease::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 13]);
    assert_eq!(labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_heart_disease_get_data_mut() {
    let download_dir = "./test_heart_disease_get_data_mut";

    let mut dataset = HeartDisease::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load, then mutate the cached features in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((features, _labels)) = dataset.get_data_mut() {
        features[[0, 0]] = 42.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _labels) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 42.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
