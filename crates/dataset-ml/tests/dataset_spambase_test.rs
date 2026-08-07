#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::spambase::*;
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

/// Checks the Spambase dataset invariants: the schema shape, the two `class`
/// classes with their exact distribution, and the non-negative numeric feature
/// domain.
fn assert_spambase_semantics(
    features: &ndarray::Array2<f64>,
    labels: &ndarray::Array1<&'static str>,
) {
    assert_eq!(features.shape(), &[N_SAMPLES, 57]);
    assert_eq!(labels.len(), N_SAMPLES);

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
fn test_load_spambase() {
    let download_dir = "./test_load_spambase"; // if the directory does not exist, the code creates it

    let dataset = Spambase::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_spambase_semantics(features, labels);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_spambase_no_need_download() {
    let download_dir = "./test_spambase_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load downloads and extracts the ZIP archive. This primes the cache.
    // The second instance then reuses the extracted file.
    Spambase::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(&download_dir_path.join("spambase.csv"), SPAMBASE_SHA256).unwrap(),
        "cached spambase.csv should match the expected SHA256"
    );

    let dataset = Spambase::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_spambase_overwrite() {
    let download_dir = "./test_spambase_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("spambase.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = Spambase::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    assert!(file_sha256_matches(&download_dir_path.join("spambase.csv"), SPAMBASE_SHA256).unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_spambase_into_data() {
    let download_dir = "./test_spambase_into_data";

    let dataset = Spambase::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. `features` and `labels` are now fully owned.

    assert_eq!(features.shape(), &[N_SAMPLES, 57]);
    assert_eq!(labels.len(), N_SAMPLES);

    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label == "ham" || label == "spam",
            "labels[{}] = {:?} is not a known class",
            i,
            label
        );
    }

    // Owned data allows direct mutation and needs no `to_owned()` clone.
    features[[0, 0]] = 0.5;
    assert_eq!(features[[0, 0]], 0.5);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_spambase_take_data() {
    let download_dir = "./test_spambase_take_data";

    let mut dataset = Spambase::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[N_SAMPLES, 57]);
    assert_eq!(labels.len(), N_SAMPLES);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file and yields the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[N_SAMPLES, 57]);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_spambase_get_data() {
    let download_dir = "./test_spambase_get_data";

    let dataset = Spambase::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 57]);
    assert_eq!(labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_spambase_get_data_mut() {
    let download_dir = "./test_spambase_get_data_mut";

    let mut dataset = Spambase::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() mutates the cached features in place, with no clone or reload.
    dataset.data().unwrap();
    if let Some((features, _labels)) = dataset.get_data_mut() {
        features[[0, 0]] = 0.25;
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _labels) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 0.25);

    remove_dir_all(download_dir).unwrap();
}
