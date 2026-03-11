use rustyml_dataset::titanic::*;
use rustyml_dataset::{download_to, file_sha256_matches};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
fn test_load_titanic() {
    let download_dir = "./test_load_titanic"; // the code will create the directory if it doesn't exist

    let dataset = Titanic::new(download_dir);
    let (string_features, numeric_features) = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_eq!(string_features.shape(), &[891, 5]);
    assert_eq!(numeric_features.shape(), &[891, 6]);
    assert_eq!(labels.len(), 891);

    let (string_features, numeric_features, labels) = dataset.data().unwrap(); // this is also a way to get all data
    // you can use `.to_owned()` to get owned copies of the data
    let mut string_features_owned = string_features.to_owned();
    let mut numeric_features_owned = numeric_features.to_owned();
    let mut labels_owned = labels.to_owned();

    // Example: Modify feature values
    string_features_owned[[0, 0]] = "test".to_string();
    numeric_features_owned[[0, 0]] = 1.0;
    labels_owned[0] = 1.0;

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
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
fn test_titanic_no_need_download() {
    let download_dir = "./test_titanic_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Titanic dataset in advance
    {
        download_to(
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            download_dir_path,
        )
        .unwrap();
    }

    // should use cached Titanic dataset
    let dataset = Titanic::new(download_dir);
    let (_string_features, _numeric_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
