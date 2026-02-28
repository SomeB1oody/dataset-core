use rustyml_dataset::diabetes::*;
use std::fs::{create_dir_all, remove_dir_all, File};
use std::io::Write;
use std::path::Path;
use rustyml_dataset::{download_to, file_sha256_matches};

#[test]
fn test_load_diabetes() {
    let download_dir = "./test_load_diabetes"; // the code will create the directory if it doesn't exist
    
    let (features, labels) = load_diabetes(download_dir).unwrap();
    assert_eq!(features.shape(), &[768, 8]);
    assert_eq!(labels.len(), 768);
    
    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_diabetes_owned() {
    let download_dir = "./test_load_diabetes_owned"; // the code will create the directory if it doesn't exist
    
    let (mut features, mut labels) = load_diabetes_owned(download_dir).unwrap();
    
    assert_eq!(features.shape(), &[768, 8]);
    assert_eq!(labels.len(), 768);
    
    // Example: Modify feature values (not possible with references)
    features[[0, 0]] = 10.0;
    labels[0] = 1.0;
    
    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();   
}

#[test]
fn test_load_diabetes_no_need_download() {
    let download_dir = "./test_load_diabetes_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download in advance
    download_to("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv", download_dir_path).unwrap();

    // should use cached data
    let (_features, _labels) = load_diabetes(download_dir).unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_diabetes_overwrite() {
    let download_dir = "./test_load_diabetes_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create fake data in the download directory
    {
        let diabetes_path = download_dir_path.join("diabetes.csv");
        let mut fake_diabetes = File::create(diabetes_path).unwrap();
        fake_diabetes.write_all(b"fake data").unwrap();
    }
    // should overwrite data
    let (_features, _labels) = load_diabetes(download_dir).unwrap();

    // check the fake file is overwritten
    assert!(file_sha256_matches(
        &download_dir_path.join("diabetes.csv"),
        "698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72"
    ).unwrap());

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}