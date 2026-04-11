#![cfg(feature = "datasets")]

use dataset_core::datasets::boston_housing::*;
use dataset_core::utils::{download_to, file_sha256_matches};
use std::fs::File;
use std::fs::{create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
// Verifies that the Boston Housing dataset loads with the correct feature shape and target count.
fn test_load_boston_housing() {
    let download_dir = "./test_load_boston_housing"; // the code will create the directory if it doesn't exist

    let dataset = BostonHousing::new(download_dir);
    let features = dataset.features().unwrap();
    let targets = dataset.targets().unwrap();

    assert_eq!(features.shape(), &[506, 13]);
    assert_eq!(targets.len(), 506);

    let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
    // you can use `.to_owned()` to get owned copies of the data
    let mut features_owned = features.to_owned();
    let mut targets_owned = targets.to_owned();

    // Example: Modify feature values
    features_owned[[0, 0]] = 0.1;
    targets_owned[0] = 25.5;

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Boston Housing loading uses a pre-downloaded cached file without re-downloading.
fn test_boston_housing_no_need_download() {
    let download_dir = "./test_boston_housing_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Boston Housing dataset in advance
    {
        download_to(
            "https://github.com/selva86/datasets/raw/master/BostonHousing.csv",
            download_dir_path,
            None,
        ).unwrap();
    }

    // should use cached Boston Housing dataset
    let dataset = BostonHousing::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Boston Housing data file is detected and overwritten with the real dataset.
fn test_boston_housing_overwrite() {
    let download_dir = "./test_boston_housing_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Boston Housing dataset in advance
    {
        let boston_housing_path = download_dir_path.join("BostonHousing.csv");
        let mut fake_boston_housing = File::create(boston_housing_path).unwrap();
        fake_boston_housing.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Boston Housing dataset
    let dataset = BostonHousing::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("BostonHousing.csv"),
            "ab16ba38fbbbbcc69fe930aab1293104f1442c8279c130d9eba03dd864bef675"
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
