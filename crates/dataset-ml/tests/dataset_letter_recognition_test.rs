#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::letter_recognition::*;
use dataset_ml::table::{ColumnData, Table};
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

/// The 17 column names, in source order: the label, then the 16 attributes.
const COLUMN_NAMES: [&str; 17] = [
    "lettr", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar",
    "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx",
];

/// Assert the column layout the documentation claims: the names and the column
/// types.
fn assert_letter_recognition_schema(table: &Table) {
    assert_eq!(table.n_columns(), N_FEATURES + 1);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    // The named constants agree with the source column order.
    assert_eq!(LetterRecognition::FEATURE_NAMES.len(), N_FEATURES);
    assert_eq!(LetterRecognition::FEATURE_NAMES, COLUMN_NAMES[1..]);
    assert_eq!(LetterRecognition::TARGET, "lettr");
    assert_eq!(LetterRecognition::TARGET, COLUMN_NAMES[0]);

    let lettr = table.column(LetterRecognition::TARGET).unwrap();
    assert!(matches!(lettr.data(), ColumnData::String(_)));

    for name in &COLUMN_NAMES[1..] {
        let column = table.column(name).unwrap();
        assert!(
            matches!(column.data(), ColumnData::Numeric(_)),
            "feature {name} should be numeric"
        );
    }
}

/// Assert the Letter Recognition dataset invariants: the schema shape, the 26
/// capital-letter classes, and the integer feature domain.
fn assert_letter_recognition_semantics(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_letter_recognition_schema(table);

    let features = table
        .numeric_matrix(&LetterRecognition::FEATURE_NAMES)
        .unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);

    let labels = table
        .column(LetterRecognition::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    assert_eq!(labels.len(), N_SAMPLES);

    let mut unique_labels: HashSet<&str> = HashSet::new();
    for (i, label) in labels.iter().enumerate() {
        assert!(
            label.len() == 1 && label.as_bytes()[0].is_ascii_uppercase(),
            "labels[{}] = {:?} is not a capital letter in 'A'..='Z'",
            i,
            label
        );
        unique_labels.insert(label.as_str());
    }
    assert_eq!(
        unique_labels.len(),
        26,
        "Letter Recognition should have exactly 26 unique classes"
    );
    for letter in 'A'..='Z' {
        assert!(
            unique_labels.contains(letter.to_string().as_str()),
            "labels must contain the letter {}",
            letter
        );
    }

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
    assert_letter_recognition_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_letter_recognition_columns_agree_with_the_matrix() {
    let download_dir = "./test_letter_recognition_columns_agree_with_the_matrix";

    let dataset = LetterRecognition::new(download_dir);
    let table = dataset.data().unwrap();
    let features = table
        .numeric_matrix(&LetterRecognition::FEATURE_NAMES)
        .unwrap();

    for (col, name) in COLUMN_NAMES[1..].iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 10_000, N_SAMPLES - 1] {
            assert_eq!(
                column[row],
                features[[row, col]],
                "column {name} disagrees with matrix column {col} at row {row}"
            );
        }
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_letter_recognition_no_need_download() {
    let download_dir = "./test_letter_recognition_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The loader downloads and extracts a ZIP archive. A second instance reuses
    // the extracted file without downloading it again.
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
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

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
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

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
    let mut table = dataset.into_data().unwrap();
    // `into_data` consumes `dataset`. The table is now fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), N_FEATURES + 1);

    // Owned labels are correct: exactly the 26 capital-letter classes.
    let unique_labels: HashSet<&str> = table
        .column(LetterRecognition::TARGET)
        .unwrap()
        .as_string()
        .unwrap()
        .iter()
        .map(String::as_str)
        .collect();
    assert_eq!(
        unique_labels.len(),
        26,
        "Letter Recognition should have exactly 26 unique classes"
    );

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("x-box").map(|c| c.data_mut()) {
        values[0] = 5.0;
    }
    assert_eq!(table.column("x-box").unwrap().as_numeric().unwrap()[0], 5.0);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_letter_recognition_take_data() {
    let download_dir = "./test_letter_recognition_take_data";

    let mut dataset = LetterRecognition::new(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), N_FEATURES + 1);

    // After `take_data`, the instance resets to unloaded, but remains usable. The
    // next access reloads the data from the cached file and returns the same table.
    let reloaded = dataset.data().unwrap();
    assert_eq!(reloaded.n_samples(), N_SAMPLES);
    assert_eq!(reloaded.n_columns(), N_FEATURES + 1);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_letter_recognition_get_data() {
    let download_dir = "./test_letter_recognition_get_data";

    let dataset = LetterRecognition::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, `get_data` returns the cached table.
    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), N_FEATURES + 1);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_letter_recognition_get_data_mut() {
    let download_dir = "./test_letter_recognition_get_data_mut";

    let mut dataset = LetterRecognition::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // The mutation happens in place. It needs no clone and no reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) = table.column_mut("x-box").map(|c| c.data_mut())
    {
        values[0] = 9.0;
    }

    // The change persists in the cache. A later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(table.column("x-box").unwrap().as_numeric().unwrap()[0], 9.0);

    remove_dir_all(download_dir).unwrap();
}
