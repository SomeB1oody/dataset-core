use rustyml_dataset::wine_quality::*;
use std::fs::{create_dir_all, remove_dir_all, File};
use std::io::Write;
use std::path::Path;
use rustyml_dataset::{download_to, file_sha256_matches, unzip};

#[test]
fn test_load_white_wine_quality() {
    let download_dir = "./test_load_white_wine_quality"; // the code will create this directory if it doesn't exist

    let (features, targets) = load_white_wine_quality(download_dir).unwrap();

    // Use the feature matrix for machine learning
    assert_eq!(features.shape(), &[4898, 11]);  // 4898 samples, 11 features
    assert_eq!(targets.len(), 4898); // 4898 samples

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_white_wine_quality_owned() {
    let download_dir = "./test_load_white_wine_quality_owned"; // the code will create this directory if it doesn't exist

    let (mut features, targets) = load_white_wine_quality_owned(download_dir).unwrap();

    // Use the feature matrix for machine learning
    assert_eq!(features.shape(), &[4898, 11]);  // 4898 samples, 11 features
    assert_eq!(targets.len(), 4898); // 4898 samples

    // Example: Modify feature values (not possible with references)
    features[[0, 0]] = 10.0;

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_red_wine_quality() {
    let download_dir = "./test_load_red_wine_quality"; // the code will create this directory if it doesn't exist

    let (features, targets) = load_red_wine_quality(download_dir).unwrap();

    // Use the feature matrix for machine learning
    assert_eq!(features.shape(), &[1599, 11]);  // 4898 samples, 11 features
    assert_eq!(targets.len(), 1599); // 4898 samples

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_red_wine_quality_owned() {
    let download_dir = "./test_load_red_wine_quality_owned"; // the code will create this directory if it doesn't exist

    let (mut features, targets) = load_red_wine_quality_owned(download_dir).unwrap();

    // Use the feature matrix for machine learning
    assert_eq!(features.shape(), &[1599, 11]);  // 4898 samples, 11 features
    assert_eq!(targets.len(), 1599); // 4898 samples

    // Example: Modify feature values (not possible with references)
    features[[0, 0]] = 10.0;

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_white_wine_quality_no_need_download() {
    let download_dir = "./test_load_white_wine_quality_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // download dataset in advance
    {
        download_to(
            "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
            download_dir_path
        ).unwrap();
        unzip(&download_dir_path.join("wine+quality.zip"), download_dir_path).unwrap();
    }
    // should use cached dataset
    let (_features, _targets) = load_white_wine_quality(download_dir).unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_red_wine_quality_no_need_download() {
    let download_dir = "./test_load_red_wine_quality_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // download dataset in advance
    {
        download_to(
            "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
            download_dir_path
        ).unwrap();
        unzip(&download_dir_path.join("wine+quality.zip"), download_dir_path).unwrap();
    }
    // should use cached dataset
    let (_features, _targets) = load_red_wine_quality(download_dir).unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_white_wine_quality_overwrite() {
    let download_dir = "./test_load_white_wine_quality_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create fake dataset
    {
        let fake_red_wine_dataset_path = download_dir_path.join("winequality-red.csv");
        let mut fake_red_wine = File::create(fake_red_wine_dataset_path).unwrap();
        fake_red_wine.write_all(b"fake data").unwrap();

        let fake_white_wine_dataset_path = download_dir_path.join("winequality-white.csv");
        let mut fake_white_wine = File::create(fake_white_wine_dataset_path).unwrap();
        fake_white_wine.write_all(b"fake data").unwrap();
    }
    // should overwrite files
    let (_features, _targets) = load_white_wine_quality(download_dir).unwrap();

    // check that the downloaded files are correct
    assert!(file_sha256_matches(
        &download_dir_path.join("winequality-white.csv"),
        "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836"
    ).unwrap());

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_red_wine_quality_overwrite() {
    let download_dir = "./test_load_red_wine_quality_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create fake dataset
    {
        let fake_red_wine_dataset_path = download_dir_path.join("winequality-red.csv");
        let mut fake_red_wine = File::create(fake_red_wine_dataset_path).unwrap();
        fake_red_wine.write_all(b"fake data").unwrap();

        let fake_white_wine_dataset_path = download_dir_path.join("winequality-white.csv");
        let mut fake_white_wine = File::create(fake_white_wine_dataset_path).unwrap();
        fake_white_wine.write_all(b"fake data").unwrap();
    }
    // should overwrite files
    let (_features, _targets) = load_red_wine_quality(download_dir).unwrap();

    // check that the downloaded files are correct
    assert!(file_sha256_matches(
        &download_dir_path.join("winequality-red.csv"),
        "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e"
    ).unwrap());

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}