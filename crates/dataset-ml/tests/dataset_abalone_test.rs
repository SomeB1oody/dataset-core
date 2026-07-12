mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::abalone::*;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/abalone.rs`.
const ABALONE_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data";
const ABALONE_SHA256: &str = "de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f";

/// The Abalone dataset has this many samples.
const N_SAMPLES: usize = 4_177;

/// Assert the Abalone dataset invariants: the schema shapes, the three `sex`
/// categories, the positive-finite numeric measurements, and the integer-valued
/// regression target.
fn assert_abalone_semantics(
    string_features: &ndarray::Array2<String>,
    numeric_features: &ndarray::Array2<f64>,
    targets: &ndarray::Array1<f64>,
) {
    assert_eq!(string_features.shape(), &[N_SAMPLES, 1]);
    assert_eq!(numeric_features.shape(), &[N_SAMPLES, 7]);
    assert_eq!(targets.len(), N_SAMPLES);

    // `sex` is exactly the three recorded categories.
    let unique_sex: HashSet<&str> = string_features.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        unique_sex,
        HashSet::from(["M", "F", "I"]),
        "Abalone `sex` should be exactly M/F/I"
    );

    // Every numeric measurement is finite and non-negative.
    for row in 0..numeric_features.nrows() {
        for col in 0..numeric_features.ncols() {
            let v = numeric_features[[row, col]];
            assert!(
                v.is_finite() && v >= 0.0,
                "numeric_features[{}, {}] = {} is not a finite non-negative value",
                row,
                col,
                v
            );
        }
    }

    // The target is an integer ring count in 1..=29.
    for (i, &y) in targets.iter().enumerate() {
        assert!(
            y.is_finite() && (1.0..=29.0).contains(&y) && y.fract() == 0.0,
            "targets[{i}] = {y} is not an integer ring count in 1..=29"
        );
    }
}

#[test]
// Verifies that the Abalone dataset loads with the correct shapes, sex categories,
// numeric feature domain, and integer regression target.
fn test_load_abalone() {
    let download_dir = "./test_load_abalone"; // the code will create the directory if it doesn't exist

    let dataset = Abalone::new(download_dir);
    let (string_features, numeric_features) = dataset.features().unwrap();
    let targets = dataset.targets().unwrap();

    assert_abalone_semantics(string_features, numeric_features, targets);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Abalone loading uses a pre-downloaded cached file without re-downloading.
fn test_abalone_no_need_download() {
    let download_dir = "./test_abalone_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Abalone dataset in advance, under the filename the loader expects
    download_to(ABALONE_URL, download_dir_path, Some("abalone.csv")).unwrap();

    // should use cached Abalone dataset
    let dataset = Abalone::new(download_dir);
    let (_s, _n, _t) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Abalone data file is detected and overwritten with the real dataset.
fn test_abalone_overwrite() {
    let download_dir = "./test_abalone_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Abalone dataset in advance
    {
        let path = download_dir_path.join("abalone.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Abalone dataset
    let dataset = Abalone::new(download_dir);
    let (_s, _n, _t) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(file_sha256_matches(&download_dir_path.join("abalone.csv"), ABALONE_SHA256).unwrap());

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays, consuming the dataset.
fn test_abalone_into_data() {
    let download_dir = "./test_abalone_into_data";

    let dataset = Abalone::new(download_dir);
    let (string_features, mut numeric_features, targets) = dataset.into_data().unwrap();
    // `dataset` has been consumed; the arrays are fully owned.

    assert_eq!(string_features.shape(), &[N_SAMPLES, 1]);
    assert_eq!(numeric_features.shape(), &[N_SAMPLES, 7]);
    assert_eq!(targets.len(), N_SAMPLES);

    // Owned data can be mutated directly, with no `to_owned()` clone.
    numeric_features[[0, 0]] = 0.5;
    assert_eq!(numeric_features[[0, 0]], 0.5);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_abalone_take_data() {
    let download_dir = "./test_abalone_take_data";

    let mut dataset = Abalone::new(download_dir);
    let (string_features, numeric_features, targets) = dataset.take_data().unwrap();

    assert_eq!(string_features.shape(), &[N_SAMPLES, 1]);
    assert_eq!(numeric_features.shape(), &[N_SAMPLES, 7]);
    assert_eq!(targets.len(), N_SAMPLES);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (rs, rn, rt) = dataset.data().unwrap();
    assert_eq!(rs.shape(), &[N_SAMPLES, 1]);
    assert_eq!(rn.shape(), &[N_SAMPLES, 7]);
    assert_eq!(rt.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_abalone_get_data() {
    let download_dir = "./test_abalone_get_data";

    let dataset = Abalone::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (string_features, numeric_features, targets) = dataset.get_data().unwrap();
    assert_eq!(string_features.shape(), &[N_SAMPLES, 1]);
    assert_eq!(numeric_features.shape(), &[N_SAMPLES, 7]);
    assert_eq!(targets.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_abalone_get_data_mut() {
    let download_dir = "./test_abalone_get_data_mut";

    let mut dataset = Abalone::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load, then mutate the cached target in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((_s, _n, targets)) = dataset.get_data_mut() {
        targets[0] = 11.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (_s, _n, targets) = dataset.data().unwrap();
    assert_eq!(targets[0], 11.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
