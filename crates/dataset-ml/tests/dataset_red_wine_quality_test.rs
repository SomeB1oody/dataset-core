#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::dataset::wine_quality::red_wine_quality::RedWineQuality;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
fn test_load_red_wine_quality() {
    let download_dir = "./test_load_red_wine_quality"; // if this directory does not exist, the code creates it

    let dataset = RedWineQuality::new(download_dir);
    let features = dataset.features().unwrap();
    let targets = dataset.targets().unwrap();

    assert_eq!(features.shape(), &[1599, 11]);
    assert_eq!(targets.len(), 1599);

    let (features, targets) = dataset.data().unwrap();
    let mut features_owned = features.to_owned();
    let mut targets_owned = targets.to_owned();

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

    let mut unique_qualities = std::collections::HashSet::new();
    for i in 0..targets.len() {
        let val = targets[i];
        assert!(
            val.fract() == 0.0,
            "target[{}] = {} is not an integer value",
            i,
            val
        );
        assert!(
            (0.0..=10.0).contains(&val),
            "target[{}] = {} is outside the valid quality range [0, 10]",
            i,
            val
        );
        unique_qualities.insert(val as i32);
    }
    assert!(
        unique_qualities.len() > 1,
        "targets should contain more than one unique quality score"
    );

    features_owned[[0, 0]] = 10.0;
    targets_owned[0] = 7.0;

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_red_wine_quality_no_need_download() {
    let download_dir = "./test_red_wine_quality_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    download_to(
        "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-red.csv",
        download_dir_path,
        None,
    )
    .unwrap();

    let dataset = RedWineQuality::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_red_wine_quality_overwrite() {
    let download_dir = "./test_red_wine_quality_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    {
        let fake_red_wine_dataset_path = download_dir_path.join("winequality-red.csv");
        let mut fake_red_wine = File::create(fake_red_wine_dataset_path).unwrap();
        fake_red_wine.write_all(b"fake data").unwrap();
    }

    let dataset = RedWineQuality::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    assert!(
        file_sha256_matches(
            &download_dir_path.join("winequality-red.csv"),
            "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e"
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_red_wine_quality_into_data() {
    let download_dir = "./test_red_wine_quality_into_data";

    let dataset = RedWineQuality::new(download_dir);
    let (mut features, targets) = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. `features` and `targets` are now fully owned.

    assert_eq!(features.shape(), &[1599, 11]);
    assert_eq!(targets.len(), 1599);

    for i in 0..targets.len() {
        let val = targets[i];
        assert!(
            (0.0..=10.0).contains(&val),
            "target[{}] = {} is outside the valid quality range [0, 10]",
            i,
            val
        );
    }

    // Owned data allows direct mutation and needs no `to_owned()` clone.
    features[[0, 0]] = 10.0;
    assert_eq!(features[[0, 0]], 10.0);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_red_wine_quality_take_data() {
    let download_dir = "./test_red_wine_quality_take_data";

    let mut dataset = RedWineQuality::new(download_dir);
    let (features, targets) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[1599, 11]);
    assert_eq!(targets.len(), 1599);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file and yields the same shapes.
    let (reloaded_features, reloaded_targets) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[1599, 11]);
    assert_eq!(reloaded_targets.len(), 1599);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_red_wine_quality_get_data() {
    let download_dir = "./test_red_wine_quality_get_data";

    let dataset = RedWineQuality::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached references.
    dataset.data().unwrap();
    let (features, targets) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[1599, 11]);
    assert_eq!(targets.len(), 1599);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_red_wine_quality_get_data_mut() {
    let download_dir = "./test_red_wine_quality_get_data_mut";

    let mut dataset = RedWineQuality::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() mutates the cached features in place, with no clone or reload.
    dataset.data().unwrap();
    if let Some((features, _targets)) = dataset.get_data_mut() {
        features[[0, 0]] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _targets) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 99.0);

    remove_dir_all(download_dir).unwrap();
}
