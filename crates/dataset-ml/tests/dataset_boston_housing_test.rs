#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::dataset::boston_housing::*;
use std::fs::File;
use std::fs::{create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
// Verifies that the Boston Housing dataset loads with the correct feature shape and target count.
fn test_load_boston_housing() {
    let download_dir = "./test_load_boston_housing"; // the loader creates the directory if it does not exist

    let dataset = BostonHousing::new(download_dir);
    let features = dataset.features().unwrap();
    let targets = dataset.targets().unwrap();

    assert_eq!(features.shape(), &[506, 13]);
    assert_eq!(targets.len(), 506);

    // Accessor consistency: data() returns the same arrays as features() and targets()
    assert_eq!(features.shape(), &[506, 13]);
    assert_eq!(targets.len(), 506);

    let (features, targets) = dataset.data().unwrap(); // data() also returns the features and targets
    // `.to_owned()` returns owned copies of the data
    let mut features_owned = features.to_owned();
    let mut targets_owned = targets.to_owned();

    // Semantic assertions: all feature and target values must be finite (no NaN or Inf)
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let val = features[[row, col]];
            assert!(
                val.is_finite(),
                "feature[{}, {}] = {} is not finite",
                row,
                col,
                val
            );
        }
    }
    for i in 0..targets.len() {
        let val = targets[i];
        assert!(val.is_finite(), "target[{}] = {} is not finite", i, val);
    }

    // CHAS (Charles River dummy variable) is at column index 3
    // and must only contain 0.0 or 1.0
    for row in 0..features.nrows() {
        let chas_val = features[[row, 3]];
        assert!(
            chas_val == 0.0 || chas_val == 1.0,
            "CHAS[{}] = {} is not a binary value (expected 0.0 or 1.0)",
            row,
            chas_val
        );
    }

    // Example: Modify feature values
    features_owned[[0, 0]] = 0.1;
    targets_owned[0] = 25.5;

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Boston Housing loading uses a pre-downloaded cached file without re-downloading.
fn test_boston_housing_no_need_download() {
    let download_dir = "./test_boston_housing_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download the Boston Housing dataset in advance
    {
        download_to(
            "https://github.com/selva86/datasets/raw/master/BostonHousing.csv",
            download_dir_path,
            None,
        )
        .unwrap();
    }

    // should use the cached Boston Housing dataset
    let dataset = BostonHousing::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Boston Housing data file and
// overwrites it with the real dataset.
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

    // check that the loader overwrote the fake file
    assert!(
        file_sha256_matches(
            &download_dir_path.join("BostonHousing.csv"),
            "ab16ba38fbbbbcc69fe930aab1293104f1442c8279c130d9eba03dd864bef675"
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and targets and consumes the dataset.
fn test_boston_housing_into_data() {
    let download_dir = "./test_boston_housing_into_data";

    let dataset = BostonHousing::new(download_dir);
    let (mut features, targets) = dataset.into_data().unwrap();
    // into_data() consumes `dataset`. The returned `features` and `targets` are fully owned.

    assert_eq!(features.shape(), &[506, 13]);
    assert_eq!(targets.len(), 506);

    // Owned targets are correct: all finite.
    for i in 0..targets.len() {
        assert!(targets[i].is_finite(), "target[{}] is not finite", i);
    }

    // The caller can mutate owned data directly, with no `to_owned()` clone.
    features[[0, 0]] = 0.1;
    assert_eq!(features[[0, 0]], 0.1);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_boston_housing_take_data() {
    let download_dir = "./test_boston_housing_take_data";

    let mut dataset = BostonHousing::new(download_dir);
    let (features, targets) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[506, 13]);
    assert_eq!(targets.len(), 506);

    // take_data() resets the instance to unloaded, but it stays usable. The next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_features, reloaded_targets) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[506, 13]);
    assert_eq!(reloaded_targets.len(), 506);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_boston_housing_get_data() {
    let download_dir = "./test_boston_housing_get_data";

    let dataset = BostonHousing::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading. Then get_data() returns the cached references.
    dataset.data().unwrap();
    let (features, targets) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[506, 13]);
    assert_eq!(targets.len(), 506);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_boston_housing_get_data_mut() {
    let download_dir = "./test_boston_housing_get_data_mut";

    let mut dataset = BostonHousing::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load the dataset. Then mutate the cached features in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((features, _targets)) = dataset.get_data_mut() {
        features[[0, 0]] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _targets) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 99.0);

    remove_dir_all(download_dir).unwrap();
}
