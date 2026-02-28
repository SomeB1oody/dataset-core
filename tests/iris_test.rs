use rustyml_dataset::iris::*;
use std::fs::{create_dir_all, remove_dir_all, File};
use std::io::Write;
use std::path::Path;
use rustyml_dataset::{download_to, file_sha256_matches, unzip};

#[test]
fn test_load_iris() {
    let download_dir = "./test_load_iris"; // the code will create the directory if it doesn't exist
    
    let (features, labels) = load_iris(download_dir).unwrap();
    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);
    
    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_iris_owned() {
    let download_dir = "./test_load_iris_owned"; // the code will create the directory if it doesn't exist
    
    let (mut features, labels) = load_iris_owned(download_dir).unwrap();
    
    // You can now modify the data since these are owned copies
    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);
    
    // Example: Modify feature values (not possible with references)
    features[[0, 0]] = 5.5;
    
    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_iris_no_need_download() {
    let download_dir = "./test_load_iris_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Boston Housing dataset in advance
    {
        download_to(
            "https://archive.ics.uci.edu/static/public/53/iris.zip",
            download_dir_path
        ).unwrap();
        unzip(&download_dir_path.join("iris.zip"), download_dir_path).unwrap();
    }

    // should use cached Iris dataset
    let (_features, _labels) = load_iris(download_dir).unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_iris_overwrite() {
    let download_dir = "./test_load_iris_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Iris dataset in advance
    {
        let iris_path = download_dir_path.join("iris.data");
        let mut fake_iris = File::create(iris_path).unwrap();
        fake_iris.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Iris dataset
    let (_features, _labels) = load_iris(download_dir).unwrap();

    // check the fake file is overwritten
    assert!(file_sha256_matches(
        &download_dir_path.join("iris.data"),
        "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0"
    ).unwrap());

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}