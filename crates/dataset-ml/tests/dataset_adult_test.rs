mod common;

use common::file_sha256_matches;
use dataset_ml::adult::*;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Adult dataset file (`adult.csv`).
const ADULT_SHA256: &str = "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d";

/// The `adult.data` partition has this many samples.
const N_SAMPLES: usize = 32_561;

/// Assert the Adult dataset invariants: the schema shapes, the two income classes,
/// and the per-feature domains.
fn assert_adult_semantics(
    strings: &ndarray::Array2<String>,
    numerics: &ndarray::Array2<f64>,
    labels: &ndarray::Array1<String>,
) {
    assert_eq!(strings.shape(), &[N_SAMPLES, 8]);
    assert_eq!(numerics.shape(), &[N_SAMPLES, 6]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Exactly two income classes, kept verbatim (no trailing period).
    let unique_labels: HashSet<&str> = labels.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        unique_labels,
        HashSet::from(["<=50K", ">50K"]),
        "Adult should have exactly the two income classes `<=50K` and `>50K`"
    );

    // `sex` (string column 6) is one of the two recorded values.
    let valid_sex: HashSet<&str> = ["Male", "Female"].into_iter().collect();
    for row in 0..strings.nrows() {
        assert!(
            valid_sex.contains(strings[[row, 6]].as_str()),
            "row {} sex {:?} is unexpected",
            row,
            strings[[row, 6]]
        );
    }

    // Every numeric feature is finite. `age` (col 0) and `hours-per-week` (col 5) are
    // positive. The byte and capital counters (cols 3, 4) are non-negative.
    for row in 0..numerics.nrows() {
        for col in 0..numerics.ncols() {
            assert!(
                numerics[[row, col]].is_finite(),
                "numeric[{}, {}] = {} is not finite",
                row,
                col,
                numerics[[row, col]]
            );
        }
        assert!(numerics[[row, 0]] > 0.0, "row {} age must be positive", row);
        assert!(
            numerics[[row, 5]] > 0.0,
            "row {} hours-per-week must be positive",
            row
        );
        assert!(
            numerics[[row, 3]] >= 0.0 && numerics[[row, 4]] >= 0.0,
            "row {} capital gain/loss must be non-negative",
            row
        );
    }

    // The loader maps the `?` missing token to empty strings. At least one missing
    // categorical value exists in the source (workclass/occupation/native-country),
    // so empty strings appear among the parsed features.
    let has_empty = strings.iter().any(|s| s.is_empty());
    assert!(
        has_empty,
        "missing `?` categorical values should be mapped to empty strings"
    );
    assert!(
        !strings.iter().any(|s| s == "?"),
        "the `?` missing token must not survive into the string features"
    );
}

#[test]
// Verifies that the Adult dataset loads with the correct shapes, label values,
// and feature-domain invariants.
fn test_load_adult() {
    let download_dir = "./test_load_adult"; // the loader creates this directory if it is missing

    let dataset = Adult::new(download_dir);
    let (strings, numerics, labels) = dataset.data().unwrap();

    assert_adult_semantics(strings, numerics, labels);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Adult reuses a cached file instead of a new download.
fn test_adult_no_need_download() {
    let download_dir = "./test_load_adult_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load primes the cache. The second instance then reuses it.
    Adult::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(&download_dir_path.join("adult.csv"), ADULT_SHA256).unwrap(),
        "cached adult.csv should match the expected SHA256"
    );

    let dataset = Adult::new(download_dir);
    let (_strings, _numerics, _labels) = dataset.data().unwrap();

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Adult data file and
// overwrites it with the real dataset.
fn test_adult_overwrite() {
    let download_dir = "./test_load_adult_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let adult_path = download_dir_path.join("adult.csv");
        let mut fake_adult = File::create(adult_path).unwrap();
        fake_adult.write_all(b"fake data").unwrap();
    }

    // The loader overwrites the fake file with the real dataset.
    let dataset = Adult::new(download_dir);
    let (_strings, _numerics, _labels) = dataset.data().unwrap();

    assert!(file_sha256_matches(&download_dir_path.join("adult.csv"), ADULT_SHA256).unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays and consumes the dataset.
fn test_adult_into_data() {
    let download_dir = "./test_adult_into_data";

    let dataset = Adult::new(download_dir);
    let (strings, mut numerics, labels) = dataset.into_data().unwrap();
    // `into_data()` consumes `dataset`. The arrays are fully owned.

    assert_eq!(strings.shape(), &[N_SAMPLES, 8]);
    assert_eq!(numerics.shape(), &[N_SAMPLES, 6]);
    assert_eq!(labels.len(), N_SAMPLES);

    // The caller can mutate the owned data directly, with no `to_owned()` clone.
    numerics[[0, 0]] = 1234.0;
    assert_eq!(numerics[[0, 0]], 1234.0);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_adult_get_data() {
    let download_dir = "./test_adult_get_data";

    let dataset = Adult::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // This loads the dataset. get_data() then returns the cached references.
    dataset.data().unwrap();
    let (strings, numerics, labels) = dataset.get_data().unwrap();
    assert_eq!(strings.shape(), &[N_SAMPLES, 8]);
    assert_eq!(numerics.shape(), &[N_SAMPLES, 6]);
    assert_eq!(labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}
