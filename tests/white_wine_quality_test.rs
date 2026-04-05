use rustyml_dataset::datasets::wine_quality::*;
use rustyml_dataset::utils::{download_to, file_sha256_matches, unzip};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
fn test_load_white_wine_quality() {
    let download_dir = "./test_load_white_wine_quality"; // the code will create this directory if it doesn't exist

    let dataset = WhiteWineQuality::new(download_dir);
    let features = dataset.features().unwrap();
    let targets = dataset.targets().unwrap();

    assert_eq!(features.shape(), &[4898, 11]);
    assert_eq!(targets.len(), 4898);

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
fn test_white_wine_quality_no_need_download() {
    let download_dir = "./test_white_wine_quality_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download dataset in advance
    {
        download_to(
            "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
            download_dir_path,
        )
        .unwrap();
        unzip(
            &download_dir_path.join("wine+quality.zip"),
            download_dir_path,
        )
        .unwrap();
    }

    // should use cached dataset
    let dataset = WhiteWineQuality::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_white_wine_quality_overwrite() {
    let download_dir = "./test_white_wine_quality_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // create fake dataset
    {
        let fake_white_wine_dataset_path = download_dir_path.join("winequality-white.csv");
        let mut fake_white_wine = File::create(fake_white_wine_dataset_path).unwrap();
        fake_white_wine.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake dataset
    let dataset = WhiteWineQuality::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    // check that the downloaded file is correct
    assert!(
        file_sha256_matches(
            &download_dir_path.join("winequality-white.csv"),
            "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836"
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
