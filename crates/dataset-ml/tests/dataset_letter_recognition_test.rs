mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::letter_recognition::*;
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
fn test_load_letter_recognition() {
    let download_dir = "./test_load_letter_recognition"; // the code creates the directory if it does not exist

    let dataset = LetterRecognition::new(download_dir);
    let features = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_letter_recognition_semantics(features, labels);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_letter_recognition_no_need_download() {
    let download_dir = "./test_letter_recognition_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Load the dataset once to fill the cache. This downloads and extracts the ZIP
    // file. A second instance then reuses the extracted file.
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

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_letter_recognition_overwrite() {
    let download_dir = "./test_letter_recognition_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("letter_recognition.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // this call replaces the fake file with the real dataset
    let dataset = LetterRecognition::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    assert!(
        file_sha256_matches(
            &download_dir_path.join("letter_recognition.csv"),
            LETTER_RECOGNITION_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_letter_recognition_into_data() {
    let download_dir = "./test_letter_recognition_into_data";

    let dataset = LetterRecognition::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // `into_data` consumes `dataset`. `features` and `labels` are now fully owned.

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

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_letter_recognition_take_data() {
    let download_dir = "./test_letter_recognition_take_data";

    let mut dataset = LetterRecognition::new(download_dir);
    let (features, labels) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(labels.len(), N_SAMPLES);

    // After `take_data`, the instance resets to unloaded, but remains usable. The
    // next access reloads the data from the cached file and returns the same shapes.
    let (reloaded_features, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_letter_recognition_get_data() {
    let download_dir = "./test_letter_recognition_get_data";

    let dataset = LetterRecognition::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, `get_data` returns the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_letter_recognition_get_data_mut() {
    let download_dir = "./test_letter_recognition_get_data_mut";

    let mut dataset = LetterRecognition::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // This loads the dataset, then mutates the cached features in place. No clone
    // or reload occurs.
    dataset.data().unwrap();
    if let Some((features, _labels)) = dataset.get_data_mut() {
        features[[0, 0]] = 9.0;
    }

    // The change persists in the cache. A later access observes it.
    let (features, _labels) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 9.0);

    remove_dir_all(download_dir).unwrap();
}
