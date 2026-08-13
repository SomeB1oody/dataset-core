#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::dataset::abalone::*;
use dataset_ml::table::{ColumnData, Table};
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/abalone.rs`.
const ABALONE_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data";
const ABALONE_SHA256: &str = "de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f";

/// The Abalone dataset has this many samples.
const N_SAMPLES: usize = 4_177;

/// The Abalone dataset has this many numeric feature columns.
const N_NUMERIC_FEATURES: usize = 7;

/// The 9 column names, in source order.
const COLUMN_NAMES: [&str; 9] = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
    "rings",
];

/// Assert the column layout the documentation claims: the names and the types.
fn assert_abalone_schema(table: &Table) {
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    // The named constants agree with the source column order.
    assert_eq!(Abalone::FEATURE_NAMES.len(), 1 + N_NUMERIC_FEATURES);
    assert_eq!(
        Abalone::FEATURE_NAMES,
        COLUMN_NAMES[..1 + N_NUMERIC_FEATURES]
    );
    assert_eq!(Abalone::TARGET, COLUMN_NAMES[1 + N_NUMERIC_FEATURES]);

    // `sex` is the one string feature. The seven measurements are numeric
    // features. `rings` is the numeric target.
    let sex = table.column("sex").unwrap();
    assert!(matches!(sex.data(), ColumnData::String(_)));

    for name in &COLUMN_NAMES[1..1 + N_NUMERIC_FEATURES] {
        let column = table.column(name).unwrap();
        assert!(
            matches!(column.data(), ColumnData::Numeric(_)),
            "column {name} should be numeric"
        );
    }

    let rings = table.column(Abalone::TARGET).unwrap();
    assert!(matches!(rings.data(), ColumnData::Numeric(_)));
}

/// Assert the Abalone dataset invariants: the schema shape, the three `sex`
/// categories, the positive-finite numeric measurements, and the integer-valued
/// regression target.
fn assert_abalone_semantics(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_abalone_schema(table);

    let sex = table.column("sex").unwrap().as_string().unwrap();
    assert_eq!(sex.len(), N_SAMPLES);
    let unique_sex: HashSet<&str> = sex.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        unique_sex,
        HashSet::from(["M", "F", "I"]),
        "Abalone `sex` should be exactly M/F/I"
    );

    // The seven measurements are finite and non-negative.
    for name in &COLUMN_NAMES[1..1 + N_NUMERIC_FEATURES] {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        assert_eq!(column.len(), N_SAMPLES);
        for (row, &v) in column.iter().enumerate() {
            assert!(
                v.is_finite() && v >= 0.0,
                "{name}[{row}] = {v} is not a finite non-negative value"
            );
        }
    }

    let targets = table.column("rings").unwrap().as_numeric().unwrap();
    assert_eq!(targets.len(), N_SAMPLES);
    for (i, &y) in targets.iter().enumerate() {
        assert!(
            y.is_finite() && (1.0..=29.0).contains(&y) && y.fract() == 0.0,
            "targets[{i}] = {y} is not an integer ring count in 1..=29"
        );
    }
}

#[test]
// Verifies that the Abalone dataset loads with the correct column layout, sex
// categories, numeric feature domain, and integer regression target.
fn test_load_abalone() {
    let download_dir = "./test_load_abalone"; // the loader creates this directory if it is missing

    let dataset = Abalone::new(download_dir);
    assert_abalone_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name holds the same values as its position
// in the numeric feature matrix.
fn test_abalone_columns_agree_with_the_matrix() {
    let download_dir = "./test_abalone_columns_agree_with_the_matrix";

    let dataset = Abalone::new(download_dir);
    let table = dataset.data().unwrap();

    // `sex` has no numeric reading, so a numeric matrix over `FEATURE_NAMES`
    // fails.
    let features = table.numeric_matrix(&Abalone::FEATURE_NAMES);
    assert!(
        features.is_err(),
        "FEATURE_NAMES includes the string `sex` column, so no numeric matrix exists"
    );

    for (col, name) in COLUMN_NAMES[1..1 + N_NUMERIC_FEATURES].iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 2_000, N_SAMPLES - 1] {
            assert!(
                column[row].is_finite(),
                "column {name} (source column {col}) is not finite at row {row}"
            );
        }
    }

    // A numeric matrix over just `TARGET` holds exactly the `rings` column.
    let targets = table.numeric_matrix(&[Abalone::TARGET]).unwrap();
    assert_eq!(targets.shape(), &[N_SAMPLES, 1]);
    let rings = table.column(Abalone::TARGET).unwrap().as_numeric().unwrap();
    for row in [0usize, 1, 2_000, N_SAMPLES - 1] {
        assert_eq!(targets[[row, 0]], rings[row]);
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Abalone reuses a cached file instead of a new download.
fn test_abalone_no_need_download() {
    let download_dir = "./test_abalone_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // This downloads the Abalone dataset in advance, using the filename the loader expects.
    download_to(ABALONE_URL, download_dir_path, Some("abalone.csv")).unwrap();

    let dataset = Abalone::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Abalone data file and
// overwrites it with the real dataset.
fn test_abalone_overwrite() {
    let download_dir = "./test_abalone_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("abalone.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = Abalone::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(file_sha256_matches(&download_dir_path.join("abalone.csv"), ABALONE_SHA256).unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_abalone_into_data() {
    let download_dir = "./test_abalone_into_data";

    let dataset = Abalone::new(download_dir);
    let mut table = dataset.into_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("length").map(|c| c.data_mut()) {
        values[0] = 0.5;
    }
    assert_eq!(
        table.column("length").unwrap().as_numeric().unwrap()[0],
        0.5
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the dataset reusable.
fn test_abalone_take_data() {
    let download_dir = "./test_abalone_take_data";

    let mut dataset = Abalone::new(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());

    // take_data() resets the instance to unloaded, but it stays usable. The next
    // access reloads it from the cached file and yields the same layout.
    let reloaded = dataset.data().unwrap();
    assert_eq!(reloaded.n_samples(), N_SAMPLES);
    assert_eq!(reloaded.n_columns(), COLUMN_NAMES.len());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_abalone_get_data() {
    let download_dir = "./test_abalone_get_data";

    let dataset = Abalone::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place and the change persists.
fn test_abalone_get_data_mut() {
    let download_dir = "./test_abalone_get_data_mut";

    let mut dataset = Abalone::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Mutating the cached target needs no clone or reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) = table.column_mut("rings").map(|c| c.data_mut())
    {
        values[0] = 11.0;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table.column("rings").unwrap().as_numeric().unwrap()[0],
        11.0
    );

    remove_dir_all(download_dir).unwrap();
}
