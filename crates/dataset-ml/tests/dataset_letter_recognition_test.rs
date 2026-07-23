mod common;

use common::file_sha256_matches;
use dataset_ml::letter_recognition::*;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Letter Recognition dataset file (`letter-recognition.data`).
const LETTER_RECOGNITION_SHA256: &str =
    "2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2";

/// The Letter Recognition dataset has this many samples.
const N_SAMPLES: usize = 20_000;

/// The Letter Recognition dataset has this many features.
const N_FEATURES: usize = 16;

/// Assert the Letter Recognition dataset invariants: the schema shape, the 26
/// capital-letter classes, and the integer feature domain.
fn assert_letter_recognition_semantics(
    features: &ndarray::Array2<f64>,
    labels: &ndarray::Array1<char>,
) {
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Labels are capital letters, and all 26 classes are present.
    let mut unique_labels: HashSet<char> = HashSet::new();
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label.is_ascii_uppercase(),
            "labels[{}] = {:?} is not a capital letter in 'A'..='Z'",
            i,
            label
        );
        unique_labels.insert(label);
    }
    assert_eq!(
        unique_labels.len(),
        26,
        "Letter Recognition should have exactly 26 unique classes"
    );
    for letter in 'A'..='Z' {
        assert!(
            unique_labels.contains(&letter),
            "labels must contain the letter {}",
            letter
        );
    }

    // Every feature value is a finite integer-valued attribute in 0..=15.
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
            assert!(
                (0.0..=15.0).contains(&val) && val.fract() == 0.0,
                "feature[{}, {}] = {} is not an integer in 0..=15",
                row,
                col,
                val
            );
        }
    }
}

#[test]
// Verifies that the Letter Recognition dataset loads with the correct shape, label
// values, and integer feature domain.
fn test_load_letter_recognition() {
    let download_dir = "./test_load_letter_recognition"; // the code will create the directory if it doesn't exist

    let dataset = LetterRecognition::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_letter_recognition_semantics(features, labels);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Letter Recognition loading uses a pre-existing cached file without re-downloading.
fn test_letter_recognition_no_need_download() {
    let download_dir = "./test_letter_recognition_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Prime the cache by loading once (downloads and extracts the ZIP), then
    // confirm a second instance reuses the extracted file.
    LetterRecognition::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join("letter_recognition.csv"),
            LETTER_RECOGNITION_SHA256
        )
        .unwrap(),
        "cached letter_recognition.csv should match the expected SHA256"
    );

    let dataset = LetterRecognition::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Letter Recognition data file is detected and overwritten with the real dataset.
fn test_letter_recognition_overwrite() {
    let download_dir = "./test_letter_recognition_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Letter Recognition dataset in advance
    {
        let path = download_dir_path.join("letter_recognition.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Letter Recognition dataset
    let dataset = LetterRecognition::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("letter_recognition.csv"),
            LETTER_RECOGNITION_SHA256
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and labels, consuming the dataset.
fn test_letter_recognition_into_data() {
    let download_dir = "./test_letter_recognition_into_data";

    let dataset = LetterRecognition::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; `features`/`labels` are fully owned.

    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Owned labels are correct: exactly the 26 capital-letter classes.
    let unique_labels: HashSet<char> = labels.iter().copied().collect();
    assert_eq!(
        unique_labels.len(),
        26,
        "Letter Recognition should have exactly 26 unique classes"
    );

    // Owned data can be mutated directly, with no `to_owned()` clone.
    features[[0, 0]] = 5.0;
    assert_eq!(features[[0, 0]], 5.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_letter_recognition_take_data() {
    let download_dir = "./test_letter_recognition_take_data";

    let mut dataset = LetterRecognition::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(labels.len(), N_SAMPLES);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_letter_recognition_get_data() {
    let download_dir = "./test_letter_recognition_get_data";

    let dataset = LetterRecognition::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_letter_recognition_get_data_mut() {
    let download_dir = "./test_letter_recognition_get_data_mut";

    let mut dataset = LetterRecognition::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load, then mutate the cached features in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((features, _labels)) = dataset.get_data_mut() {
        features[[0, 0]] = 9.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _labels) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 9.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
