mod common;

use common::file_sha256_matches;
use dataset_ml::mushroom::*;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Mushroom dataset file (`mushroom.csv`).
const MUSHROOM_SHA256: &str = "e65d082030501a3ebcbcd7c9f7c71aa9d28fdfff463bf4cf4716a3fe13ac360e";

/// The Mushroom dataset has this many samples.
const N_SAMPLES: usize = 8_124;

/// Assert the Mushroom dataset invariants: the schema shape, the two `class`
/// classes, and the all-categorical feature domains.
fn assert_mushroom_semantics(features: &ndarray::Array2<String>, labels: &ndarray::Array1<String>) {
    assert_eq!(features.shape(), &[N_SAMPLES, 22]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Exactly two classes, kept verbatim.
    let unique_labels: HashSet<&str> = labels.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        unique_labels,
        HashSet::from(["e", "p"]),
        "Mushroom should have exactly the two classes `e` (edible) and `p` (poisonous)"
    );

    // Every feature value is either a single-letter code or the empty string (for
    // the missing `stalk-root` token), and the raw `?` token never survives.
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let v = &features[[row, col]];
            assert!(
                v.is_empty() || v.chars().count() == 1,
                "feature[{}, {}] = {:?} should be a single-letter code or empty",
                row,
                col,
                v
            );
            assert_ne!(
                v, "?",
                "the `?` missing token must not survive at [{row}, {col}]"
            );
        }
    }

    // `bruises` (column 3) is one of the two recorded codes.
    let valid_bruises: HashSet<&str> = ["t", "f"].into_iter().collect();
    // `gill-size` (column 7) is one of its two recorded codes.
    let valid_gill_size: HashSet<&str> = ["b", "n"].into_iter().collect();
    for row in 0..features.nrows() {
        assert!(
            valid_bruises.contains(features[[row, 3]].as_str()),
            "row {} bruises {:?} is unexpected",
            row,
            features[[row, 3]]
        );
        assert!(
            valid_gill_size.contains(features[[row, 7]].as_str()),
            "row {} gill-size {:?} is unexpected",
            row,
            features[[row, 7]]
        );
    }

    // The loader maps the missing `?` token (only in `stalk-root`, column 10) to
    // empty strings, so at least one empty value must be present there.
    let has_empty_stalk_root = (0..features.nrows()).any(|row| features[[row, 10]].is_empty());
    assert!(
        has_empty_stalk_root,
        "missing `stalk-root` values should be mapped to empty strings"
    );
}

#[test]
// Verifies that the Mushroom dataset loads with the correct shape, label values,
// and categorical feature domains.
fn test_load_mushroom() {
    let download_dir = "./test_load_mushroom"; // the loader creates this directory if it is missing

    let dataset = Mushroom::new(download_dir);
    let (features, labels) = dataset.data().unwrap();

    assert_mushroom_semantics(features, labels);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Mushroom reuses a cached file instead of a new download.
fn test_mushroom_no_need_download() {
    let download_dir = "./test_load_mushroom_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load primes the cache. The second instance then reuses it.
    Mushroom::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(&download_dir_path.join("mushroom.csv"), MUSHROOM_SHA256).unwrap(),
        "cached mushroom.csv should match the expected SHA256"
    );

    let dataset = Mushroom::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Mushroom data file and
// overwrites it with the real dataset.
fn test_mushroom_overwrite() {
    let download_dir = "./test_load_mushroom_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let mushroom_path = download_dir_path.join("mushroom.csv");
        let mut fake_mushroom = File::create(mushroom_path).unwrap();
        fake_mushroom.write_all(b"fake data").unwrap();
    }

    // The loader overwrites the fake file with the real dataset.
    let dataset = Mushroom::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    assert!(file_sha256_matches(&download_dir_path.join("mushroom.csv"), MUSHROOM_SHA256).unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays and consumes the dataset.
fn test_mushroom_into_data() {
    let download_dir = "./test_mushroom_into_data";

    let dataset = Mushroom::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // `into_data()` consumes `dataset`. The arrays are fully owned.

    assert_eq!(features.shape(), &[N_SAMPLES, 22]);
    assert_eq!(labels.len(), N_SAMPLES);

    // The caller can mutate the owned data directly, with no `to_owned()` clone.
    features[[0, 0]] = "z".to_string();
    assert_eq!(features[[0, 0]], "z");

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_mushroom_get_data() {
    let download_dir = "./test_mushroom_get_data";

    let dataset = Mushroom::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // This loads the dataset. get_data() then returns the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 22]);
    assert_eq!(labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}
