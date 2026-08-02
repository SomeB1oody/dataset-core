mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::iris::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
// Verifies that the Iris dataset loads with the correct feature shape and label count.
fn test_load_iris() {
    let download_dir = "./test_load_iris"; // the loader creates the directory if it does not exist

    let dataset = Iris::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);

    // Accessor consistency: data() returns the same arrays as features() and labels()
    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);

    let (features, labels) = dataset.data().unwrap(); // data() also returns the features and labels
    // `.to_owned()` returns owned copies of the data
    let mut features_owned = features.to_owned();
    let mut labels_owned = labels.to_owned();

    // Semantic assertions: the labels must be valid Iris species
    let unique_labels: std::collections::HashSet<_> = labels.iter().copied().collect();
    assert_eq!(
        unique_labels.len(),
        3,
        "Iris should have exactly 3 unique species"
    );
    assert!(
        unique_labels.contains(&"setosa"),
        "labels must contain 'setosa'"
    );
    assert!(
        unique_labels.contains(&"versicolor"),
        "labels must contain 'versicolor'"
    );
    assert!(
        unique_labels.contains(&"virginica"),
        "labels must contain 'virginica'"
    );

    // Semantic assertions: all feature values must be finite (no NaN or Inf)
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

    // Example: Modify feature values
    features_owned[[0, 0]] = 5.5;
    labels_owned[0] = "setosa-modified";

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Iris loading uses a pre-downloaded cached file without re-downloading.
fn test_iris_no_need_download() {
    let download_dir = "./test_load_iris_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download the Iris dataset in advance
    download_to(
        "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv",
        download_dir_path,
        None,
    )
    .unwrap();

    // should use the cached Iris dataset
    let dataset = Iris::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Iris data file and
// overwrites it with the real dataset.
fn test_iris_overwrite() {
    let download_dir = "./test_load_iris_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Iris dataset in advance
    {
        let iris_path = download_dir_path.join("iris.csv");
        let mut fake_iris = File::create(iris_path).unwrap();
        fake_iris.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Iris dataset
    let dataset = Iris::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // check that the loader overwrote the fake file
    assert!(
        file_sha256_matches(
            &download_dir_path.join("iris.csv"),
            "c52742e50315a99f956a383faedf7575552675f6409ef0f9a47076dd08479930"
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and labels and consumes the dataset.
fn test_iris_into_data() {
    let download_dir = "./test_iris_into_data";

    let dataset = Iris::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // into_data() consumes `dataset`. The returned `features` and `labels` are fully owned.

    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);

    // Owned labels are correct: exactly the three Iris species.
    let unique_labels: std::collections::HashSet<_> = labels.iter().copied().collect();
    assert_eq!(
        unique_labels.len(),
        3,
        "Iris should have exactly 3 unique species"
    );

    // The caller can mutate owned data directly, with no `to_owned()` clone.
    features[[0, 0]] = 5.5;
    assert_eq!(features[[0, 0]], 5.5);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_iris_take_data() {
    let download_dir = "./test_iris_take_data";

    let mut dataset = Iris::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);

    // take_data() resets the instance to unloaded, but it stays usable. The next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[150, 4]);
    assert_eq!(reloaded_labels.len(), 150);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_iris_get_data() {
    let download_dir = "./test_iris_get_data";

    let dataset = Iris::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading. Then get_data() returns the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_iris_get_data_mut() {
    let download_dir = "./test_iris_get_data_mut";

    let mut dataset = Iris::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load the dataset. Then mutate the cached features in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((features, _labels)) = dataset.get_data_mut() {
        features[[0, 0]] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _labels) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 99.0);

    remove_dir_all(download_dir).unwrap();
}
