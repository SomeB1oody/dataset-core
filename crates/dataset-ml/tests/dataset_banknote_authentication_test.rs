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
fn test_load_banknote_authentication() {
    let download_dir = "./test_load_banknote_authentication"; // the code creates the directory if it does not exist

    let dataset = BanknoteAuthentication::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_banknote_authentication_semantics(features, labels);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_banknote_authentication_no_need_download() {
    let download_dir = "./test_banknote_authentication_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Download the ZIP file before the test. Extract it and place the single data
    // file under the file name the loader expects.
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

    // this call uses the cached dataset instead of downloading it again
    let dataset = BanknoteAuthentication::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_banknote_authentication_overwrite() {
    let download_dir = "./test_banknote_authentication_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("banknote_authentication.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // this call replaces the fake file with the real dataset
    let dataset = BanknoteAuthentication::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    assert!(
        file_sha256_matches(
            &download_dir_path.join("banknote_authentication.csv"),
            BANKNOTE_AUTHENTICATION_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_banknote_authentication_into_data() {
    let download_dir = "./test_banknote_authentication_into_data";

    let dataset = BanknoteAuthentication::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // `into_data` consumes `dataset`. `features` and `labels` are now fully owned.

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

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_banknote_authentication_take_data() {
    let download_dir = "./test_banknote_authentication_take_data";

    let mut dataset = BanknoteAuthentication::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[N_SAMPLES, 4]);
    assert_eq!(labels.len(), N_SAMPLES);

    // After `take_data`, the instance resets to unloaded, but remains usable. The
    // next access reloads the data from the cached file and returns the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[N_SAMPLES, 4]);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_banknote_authentication_get_data() {
    let download_dir = "./test_banknote_authentication_get_data";

    let dataset = BanknoteAuthentication::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, `get_data` returns the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 4]);
    assert_eq!(labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_banknote_authentication_get_data_mut() {
    let download_dir = "./test_banknote_authentication_get_data_mut";

    let mut dataset = BanknoteAuthentication::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // This loads the dataset, then mutates the cached features in place. No clone
    // or reload occurs.
    dataset.data().unwrap();
    if let Some((features, _labels)) = dataset.get_data_mut() {
        features[[0, 0]] = 0.25;
    }

    // The change persists in the cache. A later access observes it.
    let (features, _labels) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 0.25);

    remove_dir_all(download_dir).unwrap();
}
