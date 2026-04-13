#![cfg(feature = "datasets")]

use dataset_core::datasets::titanic::*;
use dataset_core::utils::{download_to, file_sha256_matches};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
// Verifies that the Titanic dataset loads with the correct shapes for string/numeric features and labels.
fn test_load_titanic() {
    let download_dir = "./test_load_titanic"; // the code will create the directory if it doesn't exist

    let dataset = Titanic::new(download_dir);
    let (string_features, numeric_features) = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    // Accessor consistency: data() returns the same arrays as features() and labels()
    assert_eq!(string_features.shape(), &[891, 5]);
    assert_eq!(numeric_features.shape(), &[891, 6]);
    assert_eq!(labels.len(), 891);
    assert_eq!(string_features.nrows(), numeric_features.nrows());
    assert_eq!(numeric_features.nrows(), labels.len());

    let (string_features, numeric_features, labels) = dataset.data().unwrap(); // this is also a way to get all data
    // you can use `.to_owned()` to get owned copies of the data
    let mut string_features_owned = string_features.to_owned();
    let mut numeric_features_owned = numeric_features.to_owned();
    let mut labels_owned = labels.to_owned();

    // Semantic assertions: labels must be binary (0.0 or 1.0), NaN is allowed for missing values
    for i in 0..labels.len() {
        let val = labels[i];
        if !val.is_nan() {
            assert!(
                val == 0.0 || val == 1.0,
                "labels[{}] = {} is not a binary value (expected 0.0 or 1.0, or NaN)",
                i,
                val
            );
        }
    }

    // Semantic assertions: numeric features must be finite or NaN (no Inf)
    for row in 0..numeric_features.nrows() {
        for col in 0..numeric_features.ncols() {
            let val = numeric_features[[row, col]];
            assert!(
                val.is_finite() || val.is_nan(),
                "numeric_feature[{}, {}] = {} is not finite or NaN",
                row,
                col,
                val
            );
        }
    }

    // Semantic assertions: Titanic has known missing values — confirm NaN and empty strings exist
    let nan_count: usize = numeric_features
        .iter()
        .map(|&v| if v.is_nan() { 1 } else { 0 })
        .sum();
    assert!(nan_count > 0, "numeric features should contain at least one NaN (missing Age values)");

    let empty_string_count: usize = string_features
        .iter()
        .map(|s| if s.is_empty() { 1 } else { 0 })
        .sum();
    assert!(
        empty_string_count > 0,
        "string features should contain at least one empty string (missing Cabin/Embarked values)"
    );

    // Example: Modify feature values
    string_features_owned[[0, 0]] = "test".to_string();
    numeric_features_owned[[0, 0]] = 1.0;
    labels_owned[0] = 1.0;

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Titanic data file is detected and overwritten with the real dataset.
fn test_titanic_overwrite() {
    let download_dir = "./test_titanic_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Titanic dataset in advance
    {
        let titanic_path = download_dir_path.join("titanic.csv");
        let mut fake_titanic = File::create(titanic_path).unwrap();
        fake_titanic.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Titanic dataset
    let dataset = Titanic::new(download_dir);
    let (_string_features, _numeric_features, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("titanic.csv"),
            "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7"
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Titanic loading uses a pre-downloaded cached file without re-downloading.
fn test_titanic_no_need_download() {
    let download_dir = "./test_titanic_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Titanic dataset in advance
    {
        download_to(
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            download_dir_path,
            None,
        )
        .unwrap();
    }

    // should use cached Titanic dataset
    let dataset = Titanic::new(download_dir);
    let (_string_features, _numeric_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
