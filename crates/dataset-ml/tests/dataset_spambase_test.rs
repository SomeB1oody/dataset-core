mod common;

use common::file_sha256_matches;
use dataset_ml::spambase::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA-256 mirrors the constant in `src/spambase.rs` (`spambase.data`'s bytes).
const SPAMBASE_SHA256: &str = "b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12";

/// The Spambase dataset has this many samples.
const N_SAMPLES: usize = 4601;

/// The Spambase class distribution: 1,813 spam and 2,788 ham.
const N_SPAM: usize = 1813;
const N_HAM: usize = 2788;

/// Assert the Spambase dataset invariants: the schema shape, the two `class`
/// classes with their exact distribution, and the non-negative numeric feature
/// domain.
fn assert_spambase_semantics(
    features: &ndarray::Array2<f64>,
    labels: &ndarray::Array1<&'static str>,
) {
    assert_eq!(features.shape(), &[N_SAMPLES, 57]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Labels are one of the two mapped names, and both classes are present with
    // the documented counts.
    let mut n_spam = 0usize;
    let mut n_ham = 0usize;
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label == "ham" || label == "spam",
            "labels[{}] = {:?} is not a known class",
            i,
            label
        );
        if label == "spam" {
            n_spam += 1;
        }
        if label == "ham" {
            n_ham += 1;
        }
    }
    assert!(n_spam > 0, "labels must contain at least one spam");
    assert!(n_ham > 0, "labels must contain at least one ham");
    assert_eq!(n_spam, N_SPAM, "Spambase should have {} spam rows", N_SPAM);
    assert_eq!(n_ham, N_HAM, "Spambase should have {} ham rows", N_HAM);

    // Every feature is a finite, non-negative frequency or run-length statistic.
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let val = features[[row, col]];
            assert!(
                val.is_finite() && val >= 0.0,
                "feature[{}, {}] = {} is not a finite non-negative value",
                row,
                col,
                val
            );
        }
    }
}

#[test]
// Verifies that the Spambase dataset loads with the correct shape, label values,
// and non-negative feature domain.
fn test_load_spambase() {
    let download_dir = "./test_load_spambase"; // the code will create the directory if it doesn't exist

    let dataset = Spambase::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_spambase_semantics(features, labels);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Spambase loading uses a pre-existing cached file without re-downloading.
fn test_spambase_no_need_download() {
    let download_dir = "./test_spambase_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Prime the cache by loading once (downloads and extracts the ZIP), then
    // confirm a second instance reuses the extracted file.
    Spambase::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(&download_dir_path.join("spambase.csv"), SPAMBASE_SHA256).unwrap(),
        "cached spambase.csv should match the expected SHA256"
    );

    let dataset = Spambase::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Spambase data file is detected and overwritten with the real dataset.
fn test_spambase_overwrite() {
    let download_dir = "./test_spambase_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Spambase dataset in advance
    {
        let path = download_dir_path.join("spambase.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Spambase dataset
    let dataset = Spambase::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(file_sha256_matches(&download_dir_path.join("spambase.csv"), SPAMBASE_SHA256).unwrap());

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and labels, consuming the dataset.
fn test_spambase_into_data() {
    let download_dir = "./test_spambase_into_data";

    let dataset = Spambase::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; `features`/`labels` are fully owned.

    assert_eq!(features.shape(), &[N_SAMPLES, 57]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Owned labels are correct: one of the two known classes.
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label == "ham" || label == "spam",
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
fn test_spambase_take_data() {
    let download_dir = "./test_spambase_take_data";

    let mut dataset = Spambase::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[N_SAMPLES, 57]);
    assert_eq!(labels.len(), N_SAMPLES);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[N_SAMPLES, 57]);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_spambase_get_data() {
    let download_dir = "./test_spambase_get_data";

    let dataset = Spambase::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 57]);
    assert_eq!(labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_spambase_get_data_mut() {
    let download_dir = "./test_spambase_get_data_mut";

    let mut dataset = Spambase::new(download_dir);
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
