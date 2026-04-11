#![cfg(feature = "datasets")]

use dataset_core::datasets::wine_quality::red_wine_quality::RedWineQuality;
use dataset_core::utils::{download_to, file_sha256_matches};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
// Verifies that the Red Wine Quality dataset loads with the correct feature shape and target count.
fn test_load_red_wine_quality() {
    let download_dir = "./test_load_red_wine_quality"; // the code will create this directory if it doesn't exist

    let dataset = RedWineQuality::new(download_dir);
    let features = dataset.features().unwrap();
    let targets = dataset.targets().unwrap();

    assert_eq!(features.shape(), &[1599, 11]);
    assert_eq!(targets.len(), 1599);

    let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
    // you can use `.to_owned()` to get owned copies of the data
    let mut features_owned = features.to_owned();
    let mut targets_owned = targets.to_owned();

    // Example: Modify feature values
    features_owned[[0, 0]] = 10.0;
    targets_owned[0] = 7.0;

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Red Wine Quality loading uses a pre-downloaded cached file without re-downloading.
fn test_red_wine_quality_no_need_download() {
    let download_dir = "./test_red_wine_quality_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download dataset in advance
    download_to(
        "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-red.csv",
        download_dir_path,
        None,
    )
    .unwrap();

    // should use cached dataset
    let dataset = RedWineQuality::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Red Wine Quality data file is detected and overwritten with the real dataset.
fn test_red_wine_quality_overwrite() {
    let download_dir = "./test_red_wine_quality_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // create fake dataset
    {
        let fake_red_wine_dataset_path = download_dir_path.join("winequality-red.csv");
        let mut fake_red_wine = File::create(fake_red_wine_dataset_path).unwrap();
        fake_red_wine.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake dataset
    let dataset = RedWineQuality::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    // check that the downloaded file is correct
    assert!(
        file_sha256_matches(
            &download_dir_path.join("winequality-red.csv"),
            "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e"
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
