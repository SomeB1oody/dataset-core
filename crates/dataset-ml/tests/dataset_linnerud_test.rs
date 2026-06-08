mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::linnerud::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// The pinned source URLs, filenames, and SHA-256 hashes of the two Linnerud files,
/// mirrored here so the cache-reuse and overwrite tests stay in sync with the loader.
const LINNERUD_EXERCISE_URL: &str = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/linnerud_exercise.csv";
const LINNERUD_PHYSIOLOGICAL_URL: &str = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/linnerud_physiological.csv";
const LINNERUD_EXERCISE_FILENAME: &str = "linnerud_exercise.csv";
const LINNERUD_PHYSIOLOGICAL_FILENAME: &str = "linnerud_physiological.csv";
const LINNERUD_EXERCISE_SHA256: &str =
    "cb8d8c24937643fa2459682efb86c5e667bcd6dd93109eef81964d9e9f11bf8c";
const LINNERUD_PHYSIOLOGICAL_SHA256: &str =
    "2bf7e05c1cd7d0adf0eca1e456941f624bed0a4fc96694d60d0ff7853ec5fcf7";

/// Assert the defining properties of the Linnerud feature/target matrices: shape
/// `(20, 3)` and every value finite.
fn assert_matrix_ok(matrix: &ndarray::Array2<f64>) {
    assert_eq!(matrix.shape(), &[20, 3]);
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            let val = matrix[[row, col]];
            assert!(
                val.is_finite(),
                "value[{}, {}] = {} is not finite",
                row,
                col,
                val
            );
        }
    }
}

#[test]
// Verifies that the Linnerud dataset loads with the correct shapes and reference values.
fn test_load_linnerud() {
    let download_dir = "./test_load_linnerud"; // the code will create the directory if it doesn't exist

    let dataset = Linnerud::new(download_dir);
    let features = dataset.features().unwrap();
    let targets = dataset.targets().unwrap();

    assert_eq!(features.shape(), &[20, 3]);
    assert_eq!(targets.shape(), &[20, 3]);

    let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets

    assert_matrix_ok(features);
    assert_matrix_ok(targets);

    // Pin against scikit-learn's published reference: the first exercise row is
    // (Chins, Situps, Jumps) = (5, 162, 60) and the first physiological row is
    // (Weight, Waist, Pulse) = (191, 36, 50).
    assert_eq!(features[[0, 0]], 5.0, "features[0, 0] (Chins) should be 5");
    assert_eq!(
        features[[0, 1]],
        162.0,
        "features[0, 1] (Situps) should be 162"
    );
    assert_eq!(
        features[[0, 2]],
        60.0,
        "features[0, 2] (Jumps) should be 60"
    );
    assert_eq!(
        targets[[0, 0]],
        191.0,
        "targets[0, 0] (Weight) should be 191"
    );
    assert_eq!(targets[[0, 1]], 36.0, "targets[0, 1] (Waist) should be 36");
    assert_eq!(targets[[0, 2]], 50.0, "targets[0, 2] (Pulse) should be 50");

    // Targets are continuous regression values across all three columns.
    for j in 0..targets.ncols() {
        let mut distinct: Vec<f64> = targets.column(j).to_vec();
        distinct.sort_by(|a, b| a.total_cmp(b));
        distinct.dedup();
        assert!(
            distinct.len() > 2,
            "target column {} should be continuous (got {} distinct values)",
            j,
            distinct.len()
        );
    }

    // You can use `.to_owned()` to get an owned, mutable copy of the data.
    let mut features_owned = features.to_owned();
    let mut targets_owned = targets.to_owned();
    features_owned[[0, 0]] = 6.0;
    targets_owned[[0, 0]] = 190.0;

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Linnerud loading uses pre-downloaded cached files without re-downloading.
fn test_linnerud_no_need_download() {
    let download_dir = "./test_linnerud_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download both Linnerud files in advance, under the loader's expected filenames
    {
        download_to(
            LINNERUD_EXERCISE_URL,
            download_dir_path,
            Some(LINNERUD_EXERCISE_FILENAME),
        )
        .unwrap();
        download_to(
            LINNERUD_PHYSIOLOGICAL_URL,
            download_dir_path,
            Some(LINNERUD_PHYSIOLOGICAL_FILENAME),
        )
        .unwrap();
    }

    // should use cached Linnerud files
    let dataset = Linnerud::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that corrupt or fake Linnerud files are detected and overwritten with the real data.
fn test_linnerud_overwrite() {
    let download_dir = "./test_linnerud_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create fake Linnerud files in advance
    {
        let mut fake_exercise =
            File::create(download_dir_path.join(LINNERUD_EXERCISE_FILENAME)).unwrap();
        fake_exercise.write_all(b"fake data").unwrap();
        let mut fake_physiological =
            File::create(download_dir_path.join(LINNERUD_PHYSIOLOGICAL_FILENAME)).unwrap();
        fake_physiological.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Linnerud files
    let dataset = Linnerud::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    // check the fake files are overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join(LINNERUD_EXERCISE_FILENAME),
            LINNERUD_EXERCISE_SHA256
        )
        .unwrap()
    );
    assert!(
        file_sha256_matches(
            &download_dir_path.join(LINNERUD_PHYSIOLOGICAL_FILENAME),
            LINNERUD_PHYSIOLOGICAL_SHA256
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and targets, consuming the dataset.
fn test_linnerud_into_data() {
    let download_dir = "./test_linnerud_into_data";

    let dataset = Linnerud::new(download_dir);
    let (mut features, targets) = dataset.into_data().unwrap();
    // `dataset` has been consumed; `features`/`targets` are fully owned.

    assert_matrix_ok(&features);
    assert_matrix_ok(&targets);

    // Owned data can be mutated directly, with no `to_owned()` clone.
    features[[0, 0]] = 6.0;
    assert_eq!(features[[0, 0]], 6.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_linnerud_take_data() {
    let download_dir = "./test_linnerud_take_data";

    let mut dataset = Linnerud::new(download_dir);
    let (features, targets) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[20, 3]);
    assert_eq!(targets.shape(), &[20, 3]);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached files) and yields the same shapes.
    let (reloaded_features, reloaded_targets) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[20, 3]);
    assert_eq!(reloaded_targets.shape(), &[20, 3]);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_linnerud_get_data() {
    let download_dir = "./test_linnerud_get_data";

    let dataset = Linnerud::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (features, targets) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[20, 3]);
    assert_eq!(targets.shape(), &[20, 3]);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_linnerud_get_data_mut() {
    let download_dir = "./test_linnerud_get_data_mut";

    let mut dataset = Linnerud::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load, then mutate the cached features in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((features, _targets)) = dataset.get_data_mut() {
        features[[0, 0]] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _targets) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 99.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
