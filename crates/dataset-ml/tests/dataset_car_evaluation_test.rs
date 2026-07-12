mod common;

use common::file_sha256_matches;
use dataset_ml::car_evaluation::*;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Car Evaluation dataset file (`car_evaluation.csv`).
const CAR_EVALUATION_SHA256: &str =
    "b703a9ac69f11e64ce8c223c0a40de4d2e9d769f7fb20be5f8f2e8a619893d83";

/// The Car Evaluation dataset has this many samples.
const N_SAMPLES: usize = 1_728;

/// Allowed value domain for each of the six categorical feature columns.
const FEATURE_DOMAINS: [[&str; 4]; 6] = [
    ["vhigh", "high", "med", "low"], // buying
    ["vhigh", "high", "med", "low"], // maint
    ["2", "3", "4", "5more"],        // doors
    ["2", "4", "more", ""],          // persons (only 3 levels; "" pads the row)
    ["small", "med", "big", ""],     // lug_boot (only 3 levels)
    ["low", "med", "high", ""],      // safety (only 3 levels)
];

/// Assert the Car Evaluation dataset invariants: the schema shape, the four
/// `class` classes, and the per-column categorical feature domains.
fn assert_car_evaluation_semantics(
    features: &ndarray::Array2<String>,
    labels: &ndarray::Array1<String>,
) {
    assert_eq!(features.shape(), &[N_SAMPLES, 6]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Exactly the four acceptability classes, kept verbatim.
    let unique_labels: HashSet<&str> = labels.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        unique_labels,
        HashSet::from(["unacc", "acc", "good", "vgood"]),
        "Car Evaluation should have exactly the four classes unacc/acc/good/vgood"
    );

    // Every feature value is non-empty and drawn from its column's known domain.
    for col in 0..features.ncols() {
        let domain: HashSet<&str> = FEATURE_DOMAINS[col]
            .iter()
            .copied()
            .filter(|s| !s.is_empty())
            .collect();
        for row in 0..features.nrows() {
            let v = features[[row, col]].as_str();
            assert!(!v.is_empty(), "feature[{row}, {col}] should not be empty");
            assert!(
                domain.contains(v),
                "feature[{}, {}] = {:?} is outside column {}'s domain",
                row,
                col,
                v,
                col
            );
        }
    }
}

#[test]
// Verifies that the Car Evaluation dataset loads with the correct shape, label
// values, and categorical feature domains.
fn test_load_car_evaluation() {
    let download_dir = "./test_load_car_evaluation"; // the code will create the directory if it doesn't exist

    let dataset = CarEvaluation::new(download_dir);
    let (features, labels) = dataset.data().unwrap();

    assert_car_evaluation_semantics(features, labels);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Car Evaluation loading uses a pre-existing cached file without re-downloading.
fn test_car_evaluation_no_need_download() {
    let download_dir = "./test_car_evaluation_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Prime the cache by loading once, then confirm a second instance reuses it.
    CarEvaluation::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join("car_evaluation.csv"),
            CAR_EVALUATION_SHA256
        )
        .unwrap(),
        "cached car_evaluation.csv should match the expected SHA256"
    );

    let dataset = CarEvaluation::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Car Evaluation data file is detected and overwritten with the real dataset.
fn test_car_evaluation_overwrite() {
    let download_dir = "./test_car_evaluation_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Car Evaluation dataset in advance
    {
        let path = download_dir_path.join("car_evaluation.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Car Evaluation dataset
    let dataset = CarEvaluation::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("car_evaluation.csv"),
            CAR_EVALUATION_SHA256
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays, consuming the dataset.
fn test_car_evaluation_into_data() {
    let download_dir = "./test_car_evaluation_into_data";

    let dataset = CarEvaluation::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; the arrays are fully owned.

    assert_eq!(features.shape(), &[N_SAMPLES, 6]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Owned data can be mutated directly, with no `to_owned()` clone.
    features[[0, 0]] = "low".to_string();
    assert_eq!(features[[0, 0]], "low");

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_car_evaluation_take_data() {
    let download_dir = "./test_car_evaluation_take_data";

    let mut dataset = CarEvaluation::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[N_SAMPLES, 6]);
    assert_eq!(labels.len(), N_SAMPLES);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[N_SAMPLES, 6]);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_car_evaluation_get_data() {
    let download_dir = "./test_car_evaluation_get_data";

    let dataset = CarEvaluation::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 6]);
    assert_eq!(labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_car_evaluation_get_data_mut() {
    let download_dir = "./test_car_evaluation_get_data_mut";

    let mut dataset = CarEvaluation::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load, then mutate the cached labels in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((_features, labels)) = dataset.get_data_mut() {
        labels[0] = "vgood".to_string();
    }

    // The change persisted in the cache: a later access observes it.
    let (_features, labels) = dataset.data().unwrap();
    assert_eq!(labels[0], "vgood");

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
