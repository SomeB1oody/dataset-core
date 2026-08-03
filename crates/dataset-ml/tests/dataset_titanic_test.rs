#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::dataset::titanic::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
fn test_load_titanic() {
    let download_dir = "./test_load_titanic"; // the code creates the directory if it does not exist

    let dataset = Titanic::new(download_dir);
    let (string_features, numeric_features) = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    // Accessor consistency: data() returns the same arrays as features() and labels()
    assert_eq!(string_features.shape(), &[891, 5]);
    assert_eq!(numeric_features.shape(), &[891, 6]);
    assert_eq!(labels.len(), 891);
    assert_eq!(string_features.nrows(), numeric_features.nrows());
    assert_eq!(numeric_features.nrows(), labels.len());

    let (string_features, numeric_features, labels) = dataset.data().unwrap(); // data() is another way to get all data
    // you can use `.to_owned()` to get owned copies of the data
    let mut string_features_owned = string_features.to_owned();
    let mut numeric_features_owned = numeric_features.to_owned();
    let mut labels_owned = labels.to_owned();

    // Semantic assertions: labels must be binary, 0.0 or 1.0. NaN marks a missing value.
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

    // Semantic assertions: Titanic is known to have missing values. The checks below
    // confirm at least one NaN and one empty string exist.
    let nan_count: usize = numeric_features
        .iter()
        .map(|&v| if v.is_nan() { 1 } else { 0 })
        .sum();
    assert!(
        nan_count > 0,
        "numeric features should contain at least one NaN (missing Age values)"
    );

    let empty_string_count: usize = string_features
        .iter()
        .map(|s| if s.is_empty() { 1 } else { 0 })
        .sum();
    assert!(
        empty_string_count > 0,
        "string features should contain at least one empty string (missing Cabin/Embarked values)"
    );

    // Example: Change feature values
    string_features_owned[[0, 0]] = "test".to_string();
    numeric_features_owned[[0, 0]] = 1.0;
    labels_owned[0] = 1.0;

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_titanic_overwrite() {
    let download_dir = "./test_titanic_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let titanic_path = download_dir_path.join("titanic.csv");
        let mut fake_titanic = File::create(titanic_path).unwrap();
        fake_titanic.write_all(b"fake data").unwrap();
    }

    // this call replaces the fake file with the real dataset
    let dataset = Titanic::new(download_dir);
    let (_string_features, _numeric_features, _labels) = dataset.data().unwrap();

    assert!(
        file_sha256_matches(
            &download_dir_path.join("titanic.csv"),
            "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7"
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_titanic_no_need_download() {
    let download_dir = "./test_titanic_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download the dataset before the test
    {
        download_to(
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            download_dir_path,
            None,
        )
        .unwrap();
    }

    // this call uses the cached dataset instead of downloading it again
    let dataset = Titanic::new(download_dir);
    let (_string_features, _numeric_features, _labels) = dataset.data().unwrap();

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_titanic_into_data() {
    let download_dir = "./test_titanic_into_data";

    let dataset = Titanic::new(download_dir);
    let (string_features, mut numeric_features, labels) = dataset.into_data().unwrap();
    // `into_data` consumes `dataset`. All three arrays are now fully owned.

    assert_eq!(string_features.shape(), &[891, 5]);
    assert_eq!(numeric_features.shape(), &[891, 6]);
    assert_eq!(labels.len(), 891);

    // Owned labels are correct: binary 0.0/1.0, or NaN for missing values.
    for i in 0..labels.len() {
        let val = labels[i];
        assert!(
            val.is_nan() || val == 0.0 || val == 1.0,
            "labels[{}] = {} is not binary or NaN",
            i,
            val
        );
    }

    // Owned data can be mutated directly, with no `to_owned()` clone.
    numeric_features[[0, 0]] = 999.0;
    assert_eq!(numeric_features[[0, 0]], 999.0);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_titanic_take_data() {
    let download_dir = "./test_titanic_take_data";

    let mut dataset = Titanic::new(download_dir);
    let (string_features, numeric_features, labels) = dataset.take_data().unwrap();

    assert_eq!(string_features.shape(), &[891, 5]);
    assert_eq!(numeric_features.shape(), &[891, 6]);
    assert_eq!(labels.len(), 891);

    // After `take_data`, the instance resets to unloaded, but remains usable. The
    // next access reloads the data from the cached file and returns the same shapes.
    let (re_string, re_numeric, re_labels) = dataset.data().unwrap();
    assert_eq!(re_string.shape(), &[891, 5]);
    assert_eq!(re_numeric.shape(), &[891, 6]);
    assert_eq!(re_labels.len(), 891);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_titanic_get_data() {
    let download_dir = "./test_titanic_get_data";

    let dataset = Titanic::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, `get_data` returns the cached references.
    dataset.data().unwrap();
    let (string_features, numeric_features, labels) = dataset.get_data().unwrap();
    assert_eq!(string_features.shape(), &[891, 5]);
    assert_eq!(numeric_features.shape(), &[891, 6]);
    assert_eq!(labels.len(), 891);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_titanic_get_data_mut() {
    let download_dir = "./test_titanic_get_data_mut";

    let mut dataset = Titanic::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // This loads the dataset, then mutates the cached numeric features in place. No
    // clone or reload occurs.
    dataset.data().unwrap();
    if let Some((_strings, numerics, _labels)) = dataset.get_data_mut() {
        numerics[[0, 0]] = 999.0;
    }

    // The change persists in the cache. A later access observes it.
    let (_strings, numerics, _labels) = dataset.data().unwrap();
    assert_eq!(numerics[[0, 0]], 999.0);

    remove_dir_all(download_dir).unwrap();
}
