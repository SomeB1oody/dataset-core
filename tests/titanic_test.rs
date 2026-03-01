use rustyml_dataset::titanic::*;
use std::fs::{create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;
use rustyml_dataset::{download_to, file_sha256_matches};

#[test]
fn test_load_titanic() {
    let download_dir = "./test_load_titanic"; // the code will create the directory if it doesn't exist

    let (string_features, num_features, labels) = load_titanic(download_dir).unwrap();

    assert_eq!(string_features.nrows(), 891); // 891 samples
    assert_eq!(string_features.ncols(), 5); // 5 features
    assert_eq!(num_features.nrows(), 891); // 891 samples
    assert_eq!(num_features.ncols(), 6); // 6 features
    assert_eq!(labels.len(), 891); // 891 samples

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_titanic_owned() {
    let download_dir = "./test_load_titanic_owned"; // the code will create the directory if it doesn't exist

    let (string_features, mut num_features, labels) = load_titanic_owned(download_dir).unwrap();

    assert_eq!(string_features.nrows(), 891); // 891 samples
    assert_eq!(string_features.ncols(), 5); // 5 features
    assert_eq!(num_features.nrows(), 891); // 891 samples
    assert_eq!(num_features.ncols(), 6); // 6 features
    assert_eq!(labels.len(), 891); // 891 samples

    // modify the data (not possible with references)
    num_features.mapv_inplace(|x| {
        if x.is_nan() { 0.0 } else { x }
    });

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_titanic_no_need_download() {
    let download_dir = "./test_load_titanic_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // download dataset in advance
    download_to(
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        download_dir_path
    ).unwrap();
    // should use cached data
    let (_string_features, _num_features, _labels) = load_titanic(download_dir).unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_titanic_overwrite() {
    let download_dir = "./test_load_titanic_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create fake data in the download directory
    {
        let titanic_path = download_dir_path.join("titanic.csv");
        let mut fake_titanic = std::fs::File::create(titanic_path).unwrap();
        fake_titanic.write_all(b"fake data").unwrap();
    }
    // should overwrite data
    let (_string_features, _num_features, _labels) = load_titanic(download_dir).unwrap();

    // check the fake file is overwritten
    assert!(file_sha256_matches(
        &download_dir_path.join("titanic.csv"),
        "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7"
    ).unwrap());

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}