mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::ionosphere::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/ionosphere.rs`.
const IONOSPHERE_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data";
const IONOSPHERE_SHA256: &str = "46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd";

/// The Ionosphere dataset has this many samples.
const N_SAMPLES: usize = 351;

/// Assert the Ionosphere dataset invariants: the schema shape, the two `class`
/// classes, and the normalized numeric feature domain.
fn assert_ionosphere_semantics(
    features: &ndarray::Array2<f64>,
    labels: &ndarray::Array1<&'static str>,
) {
    assert_eq!(features.shape(), &[N_SAMPLES, 34]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Labels are one of the two mapped names, and both classes are present.
    let mut has_good = false;
    let mut has_bad = false;
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label == "good" || label == "bad",
            "labels[{}] = {:?} is not a known class",
            i,
            label
        );
        if label == "good" {
            has_good = true;
        }
        if label == "bad" {
            has_bad = true;
        }
    }
    assert!(has_good, "labels must contain at least one good");
    assert!(has_bad, "labels must contain at least one bad");

    // Every feature value is finite and normalized to the [-1, 1] range.
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let val = features[[row, col]];
            assert!(
                val.is_finite() && val.abs() <= 1.0,
                "feature[{}, {}] = {} is not a finite value in [-1, 1]",
                row,
                col,
                val
            );
        }
    }

    // The second attribute (column 1) is constant 0 in this collection.
    for row in 0..features.nrows() {
        assert_eq!(
            features[[row, 1]],
            0.0,
            "column 1 should be constant 0 at row {row}"
        );
    }
}

#[test]
// Verifies that the Ionosphere dataset loads with the correct shape, label values,
// and normalized feature domain.
fn test_load_ionosphere() {
    let download_dir = "./test_load_ionosphere"; // the code will create the directory if it doesn't exist

    let dataset = Ionosphere::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_ionosphere_semantics(features, labels);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Ionosphere loading uses a pre-downloaded cached file without re-downloading.
fn test_ionosphere_no_need_download() {
    let download_dir = "./test_ionosphere_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Ionosphere dataset in advance, under the filename the loader expects
    download_to(IONOSPHERE_URL, download_dir_path, Some("ionosphere.csv")).unwrap();

    // should use cached Ionosphere dataset
    let dataset = Ionosphere::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Ionosphere data file is detected and overwritten with the real dataset.
fn test_ionosphere_overwrite() {
    let download_dir = "./test_ionosphere_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Ionosphere dataset in advance
    {
        let path = download_dir_path.join("ionosphere.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Ionosphere dataset
    let dataset = Ionosphere::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(&download_dir_path.join("ionosphere.csv"), IONOSPHERE_SHA256).unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and labels, consuming the dataset.
fn test_ionosphere_into_data() {
    let download_dir = "./test_ionosphere_into_data";

    let dataset = Ionosphere::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; `features`/`labels` are fully owned.

    assert_eq!(features.shape(), &[N_SAMPLES, 34]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Owned labels are correct: one of the two known classes.
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label == "good" || label == "bad",
            "labels[{}] = {:?} is not a known class",
            i,
            label
        );
    }

    // Owned data can be mutated directly, with no `to_owned()` clone.
    features[[0, 0]] = 0.5;
    assert_eq!(features[[0, 0]], 0.5);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_ionosphere_take_data() {
    let download_dir = "./test_ionosphere_take_data";

    let mut dataset = Ionosphere::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[N_SAMPLES, 34]);
    assert_eq!(labels.len(), N_SAMPLES);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[N_SAMPLES, 34]);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_ionosphere_get_data() {
    let download_dir = "./test_ionosphere_get_data";

    let dataset = Ionosphere::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 34]);
    assert_eq!(labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_ionosphere_get_data_mut() {
    let download_dir = "./test_ionosphere_get_data_mut";

    let mut dataset = Ionosphere::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load, then mutate the cached features in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((features, _labels)) = dataset.get_data_mut() {
        features[[0, 0]] = 0.25;
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _labels) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 0.25);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
