mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::wine_recognition::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/wine_recognition.rs`.
const WINE_RECOGNITION_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data";
const WINE_RECOGNITION_SHA256: &str =
    "6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659";

#[test]
// Verifies that the Wine Recognition dataset loads with the correct feature shape and label count.
fn test_load_wine_recognition() {
    let download_dir = "./test_load_wine_recognition"; // the code will create the directory if it doesn't exist

    let dataset = WineRecognition::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_eq!(features.shape(), &[178, 13]);
    assert_eq!(labels.len(), 178);

    let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels

    // Semantic assertions: labels must be one of the three known classes, and
    // all three classes must be present.
    let mut has_class_1 = false;
    let mut has_class_2 = false;
    let mut has_class_3 = false;
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label == "class_1" || label == "class_2" || label == "class_3",
            "labels[{}] = {:?} is not a known class",
            i,
            label
        );
        match label {
            "class_1" => has_class_1 = true,
            "class_2" => has_class_2 = true,
            "class_3" => has_class_3 = true,
            _ => {}
        }
    }
    assert!(has_class_1, "labels must contain at least one class_1");
    assert!(has_class_2, "labels must contain at least one class_2");
    assert!(has_class_3, "labels must contain at least one class_3");

    // Semantic assertions: all feature values must be finite and positive (every
    // constituent is a physical quantity, so none can be zero or negative).
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let val = features[[row, col]];
            assert!(
                val.is_finite() && val > 0.0,
                "feature[{}, {}] = {} is not a finite positive value",
                row,
                col,
                val
            );
        }
    }

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Wine Recognition loading uses a pre-downloaded cached file without re-downloading.
fn test_wine_recognition_no_need_download() {
    let download_dir = "./test_wine_recognition_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Wine Recognition dataset in advance, under the filename the loader expects
    download_to(
        WINE_RECOGNITION_URL,
        download_dir_path,
        Some("wine_recognition.csv"),
    )
    .unwrap();

    // should use cached Wine Recognition dataset
    let dataset = WineRecognition::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Wine Recognition data file is detected and overwritten with the real dataset.
fn test_wine_recognition_overwrite() {
    let download_dir = "./test_wine_recognition_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Wine Recognition dataset in advance
    {
        let path = download_dir_path.join("wine_recognition.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Wine Recognition dataset
    let dataset = WineRecognition::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("wine_recognition.csv"),
            WINE_RECOGNITION_SHA256
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and labels, consuming the dataset.
fn test_wine_recognition_into_data() {
    let download_dir = "./test_wine_recognition_into_data";

    let dataset = WineRecognition::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; `features`/`labels` are fully owned.

    assert_eq!(features.shape(), &[178, 13]);
    assert_eq!(labels.len(), 178);

    // Owned labels are correct: one of the three known classes.
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label == "class_1" || label == "class_2" || label == "class_3",
            "labels[{}] = {:?} is not a known class",
            i,
            label
        );
    }

    // Owned data can be mutated directly, with no `to_owned()` clone.
    features[[0, 0]] = 13.5;
    assert_eq!(features[[0, 0]], 13.5);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_wine_recognition_take_data() {
    let download_dir = "./test_wine_recognition_take_data";

    let mut dataset = WineRecognition::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[178, 13]);
    assert_eq!(labels.len(), 178);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[178, 13]);
    assert_eq!(reloaded_labels.len(), 178);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_wine_recognition_get_data() {
    let download_dir = "./test_wine_recognition_get_data";

    let dataset = WineRecognition::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[178, 13]);
    assert_eq!(labels.len(), 178);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_wine_recognition_get_data_mut() {
    let download_dir = "./test_wine_recognition_get_data_mut";

    let mut dataset = WineRecognition::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load, then mutate the cached features in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((features, _labels)) = dataset.get_data_mut() {
        features[[0, 0]] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _labels) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 99.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
