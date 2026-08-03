mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::dataset::heart_disease::*;
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

    // Every non-missing feature value is finite. NaN marks the source's `?` value.
    // Only `ca` (column 11) and `thal` (column 12) can have missing values.
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
fn test_load_heart_disease() {
    let download_dir = "./test_load_heart_disease"; // the code creates the directory if it does not exist

    let dataset = HeartDisease::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_heart_disease_semantics(features, labels);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_heart_disease_no_need_download() {
    let download_dir = "./test_heart_disease_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download the dataset before the test, using the file name the loader expects
    download_to(
        HEART_DISEASE_URL,
        download_dir_path,
        Some("heart_disease.csv"),
    )
    .unwrap();

    // this call uses the cached dataset instead of downloading it again
    let dataset = HeartDisease::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_heart_disease_overwrite() {
    let download_dir = "./test_heart_disease_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("heart_disease.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // this call replaces the fake file with the real dataset
    let dataset = HeartDisease::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    assert!(
        file_sha256_matches(
            &download_dir_path.join("heart_disease.csv"),
            HEART_DISEASE_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_heart_disease_into_data() {
    let download_dir = "./test_heart_disease_into_data";

    let dataset = HeartDisease::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // `into_data` consumes `dataset`. `features` and `labels` are now fully owned.

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

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_heart_disease_take_data() {
    let download_dir = "./test_heart_disease_take_data";

    let mut dataset = HeartDisease::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[N_SAMPLES, 13]);
    assert_eq!(labels.len(), N_SAMPLES);

    // After `take_data`, the instance resets to unloaded, but remains usable. The
    // next access reloads the data from the cached file and returns the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[N_SAMPLES, 13]);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_heart_disease_get_data() {
    let download_dir = "./test_heart_disease_get_data";

    let dataset = HeartDisease::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, `get_data` returns the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 13]);
    assert_eq!(labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_heart_disease_get_data_mut() {
    let download_dir = "./test_heart_disease_get_data_mut";

    let mut dataset = HeartDisease::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // This loads the dataset, then mutates the cached features in place. No clone
    // or reload occurs.
    dataset.data().unwrap();
    if let Some((features, _labels)) = dataset.get_data_mut() {
        features[[0, 0]] = 42.0;
    }

    // The change persists in the cache. A later access observes it.
    let (features, _labels) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 42.0);

    remove_dir_all(download_dir).unwrap();
}
