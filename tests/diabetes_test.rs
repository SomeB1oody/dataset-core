use rustyml_dataset::diabetes::*;
use rustyml_dataset::{download_to, file_sha256_matches};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
fn test_load_diabetes() {
    let download_dir = "./test_load_diabetes"; // the code will create the directory if it doesn't exist

    let dataset = Diabetes::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_eq!(features.shape(), &[768, 8]);
    assert_eq!(labels.len(), 768);

    let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
    // you can use `.to_owned()` to get owned copies of the data
    let mut features_owned = features.to_owned();
    let mut labels_owned = labels.to_owned();

    // Example: Modify feature values
    features_owned[[0, 0]] = 10.0;
    labels_owned[0] = 1.0;

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_diabetes_no_need_download() {
    let download_dir = "./test_diabetes_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Diabetes dataset in advance
    {
        download_to(
            "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
            download_dir_path,
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
