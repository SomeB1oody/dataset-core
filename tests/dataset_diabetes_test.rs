#![cfg(feature = "datasets")]

use dataset_core::datasets::diabetes::*;
use dataset_core::utils::{download_to, file_sha256_matches};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
// Verifies that the Diabetes dataset loads with the correct feature shape and label count.
fn test_load_diabetes() {
    let download_dir = "./test_load_diabetes"; // the code will create the directory if it doesn't exist

    let dataset = Diabetes::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    // Accessor consistency: data() returns the same arrays as features() and labels()
    assert_eq!(features.shape(), &[768, 8]);
    assert_eq!(labels.len(), 768);

    let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
    // you can use `.to_owned()` to get owned copies of the data
    let mut features_owned = features.to_owned();
    let mut labels_owned = labels.to_owned();

    // Semantic assertions: labels must be binary (0.0 or 1.0)
    let mut has_class_0 = false;
    let mut has_class_1 = false;
    for i in 0..labels.len() {
        let val = labels[i];
        assert!(
            val == 0.0 || val == 1.0,
            "labels[{}] = {} is not a binary value (expected 0.0 or 1.0)",
            i,
            val
        );
        if val == 0.0 {
            has_class_0 = true;
        }
        if val == 1.0 {
            has_class_1 = true;
        }
    }
    assert!(has_class_0, "labels must contain at least one instance of class 0");
    assert!(has_class_1, "labels must contain at least one instance of class 1");

    // Semantic assertions: all feature values must be finite (no NaN or Inf)
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

    // Example: Modify feature values
    features_owned[[0, 0]] = 10.0;
    labels_owned[0] = 1.0;

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Diabetes loading uses a pre-downloaded cached file without re-downloading.
fn test_diabetes_no_need_download() {
    let download_dir = "./test_diabetes_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Diabetes dataset in advance
    {
        download_to(
            "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
            download_dir_path,
            None,
        )
        .unwrap();
    }

    // should use cached Diabetes dataset
    let dataset = Diabetes::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Diabetes data file is detected and overwritten with the real dataset.
fn test_diabetes_overwrite() {
    let download_dir = "./test_diabetes_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Diabetes dataset in advance
    {
        let diabetes_path = download_dir_path.join("diabetes.csv");
        let mut fake_diabetes = File::create(diabetes_path).unwrap();
        fake_diabetes.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Diabetes dataset
    let dataset = Diabetes::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("diabetes.csv"),
            "698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72"
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
