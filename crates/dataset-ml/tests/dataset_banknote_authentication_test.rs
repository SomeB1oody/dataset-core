mod common;

use common::file_sha256_matches;
use dataset_core::utils::{download_to, unzip};
use dataset_ml::banknote_authentication::*;
use std::fs::{File, copy, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/banknote_authentication.rs`.
const BANKNOTE_AUTHENTICATION_URL: &str =
    "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip";
const BANKNOTE_AUTHENTICATION_SHA256: &str =
    "d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9";

/// The Banknote Authentication dataset has this many samples.
const N_SAMPLES: usize = 1372;

/// Assert the Banknote Authentication dataset invariants: the schema shape, the
/// two `class` codes with their exact counts, and the finite numeric features.
fn assert_banknote_authentication_semantics(
    features: &ndarray::Array2<f64>,
    labels: &ndarray::Array1<u8>,
) {
    assert_eq!(features.shape(), &[N_SAMPLES, 4]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Labels are the raw `0`/`1` codes, and both classes are present with the
    // documented per-class counts.
    let mut zeros = 0usize;
    let mut ones = 0usize;
    for (i, &label) in labels.iter().enumerate() {
        match label {
            0 => zeros += 1,
            1 => ones += 1,
            other => panic!("labels[{i}] = {other} is not a known class"),
        }
    }
    assert_eq!(zeros, 762, "expected 762 samples of class 0");
    assert_eq!(ones, 610, "expected 610 samples of class 1");

    // Every feature value is finite.
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
        }
    }
}

#[test]
// Verifies that the Banknote Authentication dataset loads with the correct shape,
// label values, and finite feature domain.
fn test_load_banknote_authentication() {
    let download_dir = "./test_load_banknote_authentication"; // the code will create the directory if it doesn't exist

    let dataset = BanknoteAuthentication::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_banknote_authentication_semantics(features, labels);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Banknote Authentication loading uses a pre-downloaded cached file without re-downloading.
fn test_banknote_authentication_no_need_download() {
    let download_dir = "./test_banknote_authentication_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download the Banknote Authentication ZIP in advance, extract it, and place
    // the single data file under the filename the loader expects
    {
        let temp_dir_path = download_dir_path.join("temp");
        create_dir_all(&temp_dir_path).unwrap();
        download_to(
            BANKNOTE_AUTHENTICATION_URL,
            &temp_dir_path,
            Some("banknote_authentication.zip"),
        )
        .unwrap();
        unzip(
            &temp_dir_path.join("banknote_authentication.zip"),
            &temp_dir_path,
        )
        .unwrap();
        copy(
            temp_dir_path.join("data_banknote_authentication.txt"),
            download_dir_path.join("banknote_authentication.csv"),
        )
        .unwrap();
        remove_dir_all(&temp_dir_path).unwrap();
    }

    // should use cached Banknote Authentication dataset
    let dataset = BanknoteAuthentication::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Banknote Authentication data file is detected and overwritten with the real dataset.
fn test_banknote_authentication_overwrite() {
    let download_dir = "./test_banknote_authentication_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Banknote Authentication dataset in advance
    {
        let path = download_dir_path.join("banknote_authentication.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Banknote Authentication dataset
    let dataset = BanknoteAuthentication::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("banknote_authentication.csv"),
            BANKNOTE_AUTHENTICATION_SHA256
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and labels, consuming the dataset.
fn test_banknote_authentication_into_data() {
    let download_dir = "./test_banknote_authentication_into_data";

    let dataset = BanknoteAuthentication::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; `features`/`labels` are fully owned.

    assert_eq!(features.shape(), &[N_SAMPLES, 4]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Owned labels are correct: one of the two known class codes.
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label == 0 || label == 1,
            "labels[{}] = {} is not a known class",
            i,
            label
        );
    }

    // Owned data can be mutated directly, with no `to_owned()` clone.
    features[[0, 0]] = 0.5;
    assert_eq!(features[[0, 0]], 0.5);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_banknote_authentication_take_data() {
    let download_dir = "./test_banknote_authentication_take_data";

    let mut dataset = BanknoteAuthentication::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[N_SAMPLES, 4]);
    assert_eq!(labels.len(), N_SAMPLES);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[N_SAMPLES, 4]);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_banknote_authentication_get_data() {
    let download_dir = "./test_banknote_authentication_get_data";

    let dataset = BanknoteAuthentication::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 4]);
    assert_eq!(labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_banknote_authentication_get_data_mut() {
    let download_dir = "./test_banknote_authentication_get_data_mut";

    let mut dataset = BanknoteAuthentication::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load, then mutate the cached features in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((features, _labels)) = dataset.get_data_mut() {
        features[[0, 0]] = 0.25;
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _labels) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 0.25);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
