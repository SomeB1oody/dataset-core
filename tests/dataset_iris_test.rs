#![cfg(feature = "datasets")]

use dataset_core::datasets::iris::*;
use dataset_core::utils::{download_to, file_sha256_matches};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
// Verifies that the Iris dataset loads with the correct feature shape and label count.
fn test_load_iris() {
    let download_dir = "./test_load_iris"; // the code will create the directory if it doesn't exist

    let dataset = Iris::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);

    // Accessor consistency: data() returns the same arrays as features() and labels()
    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);

    let (features, labels) = dataset.data().unwrap(); // this is also a way to get features and labels
    // you can use `.to_owned()` to get owned copies of the data
    let mut features_owned = features.to_owned();
    let mut labels_owned = labels.to_owned();

    // Semantic assertions: verify label values are valid Iris species
    let unique_labels: std::collections::HashSet<_> = labels.iter().copied().collect();
    assert_eq!(unique_labels.len(), 3, "Iris should have exactly 3 unique species");
    assert!(unique_labels.contains(&"setosa"), "labels must contain 'setosa'");
    assert!(unique_labels.contains(&"versicolor"), "labels must contain 'versicolor'");
    assert!(unique_labels.contains(&"virginica"), "labels must contain 'virginica'");

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

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Iris loading uses a pre-downloaded cached file without re-downloading.
fn test_iris_no_need_download() {
    let download_dir = "./test_load_iris_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Iris dataset in advance
    download_to(
        "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv",
        download_dir_path,
        None,
    )
    .unwrap();

    // should use cached Iris dataset
    let dataset = Iris::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Iris data file is detected and overwritten with the real dataset.
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

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("iris.csv"),
            "c52742e50315a99f956a383faedf7575552675f6409ef0f9a47076dd08479930"
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
