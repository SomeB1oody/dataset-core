#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::digits::*;
use dataset_ml::table::{ColumnData, Table};
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Digits dataset file (`optdigits.tes`).
const DIGITS_SHA256: &str = "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8";

/// The Digits dataset has this many samples.
const N_SAMPLES: usize = 1797;

/// The Digits dataset has this many pixel columns.
const N_PIXELS: usize = 64;

/// The 65 column names, in source order: the 64 pixels, then the digit.
fn column_names() -> Vec<String> {
    let mut names: Vec<String> = (0..N_PIXELS)
        .map(|col| format!("pixel_{}_{}", col / 8, col % 8))
        .collect();
    names.push("digit".to_string());
    names
}

/// Assert the column layout the documentation claims: the names, the named
/// constants, and the column types.
fn assert_digits_schema(table: &Table) {
    assert_eq!(table.n_columns(), N_PIXELS + 1);
    assert_eq!(table.names().collect::<Vec<_>>(), column_names());

    assert_eq!(Digits::FEATURE_NAMES.len(), N_PIXELS);
    assert_eq!(Digits::TARGET, "digit");

    for name in Digits::FEATURE_NAMES {
        assert!(
            matches!(table.column(name).unwrap().data(), ColumnData::Numeric(_)),
            "feature {name} should be numeric"
        );
    }

    let digit = table.column(Digits::TARGET).unwrap();
    assert!(matches!(digit.data(), ColumnData::Integer(_)));
}

/// Assert the Digits dataset invariants: the schema shape, the 10 digit classes,
/// and the `0..=16` integer pixel domain.
fn assert_digits_semantics(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_digits_schema(table);

    let features = table.numeric_matrix(&Digits::FEATURE_NAMES).unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, N_PIXELS]);

    let labels = table.column("digit").unwrap().as_integer().unwrap();
    assert_eq!(labels.len(), N_SAMPLES);

    let unique_labels: HashSet<i64> = labels.iter().copied().collect();
    assert_eq!(
        unique_labels.len(),
        10,
        "Digits should have exactly 10 unique classes"
    );
    for digit in 0i64..=9 {
        assert!(
            unique_labels.contains(&digit),
            "labels must contain the digit {}",
            digit
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
                (0.0..=16.0).contains(&val) && val.fract() == 0.0,
                "feature[{}, {}] = {} is not an integer in 0..=16",
                row,
                col,
                val
            );
        }
    }
}

#[test]
// Verifies that the Digits dataset loads with the correct column layout and
// label domain.
fn test_load_digits() {
    let download_dir = "./test_load_digits"; // the loader creates this directory if it is missing

    let dataset = Digits::new(download_dir);
    assert_digits_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a pixel column reached by name holds the same values as its
// position in the feature matrix.
fn test_digits_columns_agree_with_the_matrix() {
    let download_dir = "./test_digits_columns_agree_with_the_matrix";

    let dataset = Digits::new(download_dir);
    let table = dataset.data().unwrap();
    let features = table.numeric_matrix(&Digits::FEATURE_NAMES).unwrap();
    let names = column_names();

    for col in [0usize, 1, 32, N_PIXELS - 1] {
        let column = table.column(&names[col]).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 900, N_SAMPLES - 1] {
            assert_eq!(
                column[row],
                features[[row, col]],
                "column {} disagrees with matrix column {col} at row {row}",
                names[col]
            );
        }
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Digits reuses a cached file instead of a new download.
fn test_digits_no_need_download() {
    let download_dir = "./test_load_digits_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    Digits::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(&download_dir_path.join("digits.csv"), DIGITS_SHA256).unwrap(),
        "cached digits.csv should match the expected SHA256"
    );

    let dataset = Digits::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Digits data file and
// overwrites it with the real dataset.
fn test_digits_overwrite() {
    let download_dir = "./test_load_digits_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let digits_path = download_dir_path.join("digits.csv");
        let mut fake_digits = File::create(digits_path).unwrap();
        fake_digits.write_all(b"fake data").unwrap();
    }

    let dataset = Digits::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(file_sha256_matches(&download_dir_path.join("digits.csv"), DIGITS_SHA256).unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_digits_into_data() {
    let download_dir = "./test_digits_into_data";

    let dataset = Digits::new(download_dir);
    let mut table = dataset.into_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), N_PIXELS + 1);

    let unique_labels: HashSet<i64> = table
        .column("digit")
        .unwrap()
        .as_integer()
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert_eq!(
        unique_labels.len(),
        10,
        "Digits should have exactly 10 unique classes"
    );

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("pixel_0_0").map(|c| c.data_mut()) {
        values[0] = 5.0;
    }
    assert_eq!(
        table.column("pixel_0_0").unwrap().as_numeric().unwrap()[0],
        5.0
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the dataset reusable.
fn test_digits_take_data() {
    let download_dir = "./test_digits_take_data";

    let mut dataset = Digits::new(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), N_PIXELS + 1);

    // take_data() resets the instance to unloaded, but it stays usable. The next
    // access reloads it from the cached file and yields the same table.
    let reloaded = dataset.data().unwrap();
    assert_eq!(reloaded.n_samples(), N_SAMPLES);
    assert_eq!(reloaded.n_columns(), N_PIXELS + 1);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_digits_get_data() {
    let download_dir = "./test_digits_get_data";

    let dataset = Digits::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached table.
    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), N_PIXELS + 1);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place and the change persists.
fn test_digits_get_data_mut() {
    let download_dir = "./test_digits_get_data_mut";

    let mut dataset = Digits::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() mutates the cached table in place. It needs no clone or reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) =
            table.column_mut("pixel_0_0").map(|c| c.data_mut())
    {
        values[0] = 9.0;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table.column("pixel_0_0").unwrap().as_numeric().unwrap()[0],
        9.0
    );

    remove_dir_all(download_dir).unwrap();
}
