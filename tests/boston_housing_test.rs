use rustyml_dataset::boston_housing::*;
use std::fs::{remove_dir_all, rename, create_dir_all};
use rustyml_dataset::{download_to, unzip};
use std::path::Path;
use std::fs::File;
use std::io::Write;

#[test]
fn test_load_boston_housing() {
    let download_dir = "./test_load_boston_housing"; // the code will create the directory if it doesn't exist

    let (features, targets) = load_boston_housing(download_dir).unwrap();
    assert_eq!(features.shape(), &[506, 13]);
    assert_eq!(targets.len(), 506);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_boston_housing_owned() {
    let download_dir = "./test_load_boston_housing_owned"; // the code will create the directory if it doesn't exist

    let (mut features, mut targets) = load_boston_housing_owned(download_dir).unwrap();

    // You can now modify the data since these are owned copies
    assert_eq!(features.shape(), &[506, 13]);
    assert_eq!(targets.len(), 506);

    // Example: Modify feature values (not possible with references)
    features[[0, 0]] = 0.1;
    targets[0] = 25.5;

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_boston_housing_no_need_download() {
    let download_dir = "./test_boston_housing_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Boston Housing dataset in advance
    {
        download_to(
            "https://gist.github.com/nnbphuong/def91b5553736764e8e08f6255390f37/archive/373a856a3c9c1119e34b344de9230ae2ea89569d.zip",
            download_dir_path
        ).unwrap();

        // Extract file
        unzip(&download_dir_path.join("373a856a3c9c1119e34b344de9230ae2ea89569d.zip"), download_dir_path).unwrap();

        let src = download_dir_path
            .join("def91b5553736764e8e08f6255390f37-373a856a3c9c1119e34b344de9230ae2ea89569d")
            .join("BostonHousing.csv");
        let dst = download_dir_path.join("BostonHousing.csv");
        // move boston_housing.csv out of the directory
        rename(src, &dst).unwrap();
    }

    // should use cached Boston Housing dataset
    let (_features, _targets) = load_boston_housing(download_dir).unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_boston_housing_overwrite() {
    let download_dir = "./test_boston_housing_overwrite";
    // create a fake Boston Housing dataset in advance
    {
        let download_dir_path = Path::new(download_dir);
        create_dir_all(download_dir_path).unwrap();
        let boston_housing_path = download_dir_path.join("BostonHousing.csv");
        let mut fake_boston_housing = File::create(boston_housing_path).unwrap();
        fake_boston_housing.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Boston Housing dataset
    let (_features, _targets) = load_boston_housing(download_dir).unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}