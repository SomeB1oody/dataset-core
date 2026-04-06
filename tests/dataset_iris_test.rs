use rustyml_dataset::datasets::iris::*;
use rustyml_dataset::utils::{download_to, file_sha256_matches, unzip};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
// Verifies that the Iris dataset loads with the correct feature shape and label count.
fn test_load_iris() {
    let download_dir = "./test_load_iris"; // the code will create the directory if it doesn't exist

    let dataset = Iris::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);

    let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
    // you can use `.to_owned()` to get owned copies of the data
    let mut features_owned = features.to_owned();
    let mut labels_owned = labels.to_owned();

    // Example: Modify feature values
    features_owned[[0, 0]] = 5.5;
    labels_owned[0] = "setosa-modified";

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Iris loading uses a pre-downloaded cached file without re-downloading.
fn test_iris_no_need_download() {
    let download_dir = "./test_load_iris_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Iris dataset in advance
    {
        download_to(
            "https://archive.ics.uci.edu/static/public/53/iris.zip",
            download_dir_path,
        )
        .unwrap();
        unzip(&download_dir_path.join("iris.zip"), download_dir_path).unwrap();
    }

    // should use cached Iris dataset
    let dataset = Iris::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Iris data file is detected and overwritten with the real dataset.
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
    let dataset = Iris::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("iris.data"),
            "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0"
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
