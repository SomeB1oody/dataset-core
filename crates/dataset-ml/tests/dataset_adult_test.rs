#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::adult::*;
use dataset_ml::table::{ColumnData, Table};
use ndarray::Array1;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Adult dataset file (`adult.csv`).
const ADULT_SHA256: &str = "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d";

/// The `adult.data` partition has this many samples.
const N_SAMPLES: usize = 32_561;

/// The 15 columns the documentation lists, as `(name, type)`, in source
/// order.
const COLUMN_SCHEMA: [(&str, &str); 15] = [
    ("age", "numeric"),
    ("workclass", "string"),
    ("fnlwgt", "numeric"),
    ("education", "string"),
    ("education-num", "numeric"),
    ("marital-status", "string"),
    ("occupation", "string"),
    ("relationship", "string"),
    ("race", "string"),
    ("sex", "string"),
    ("capital-gain", "numeric"),
    ("capital-loss", "numeric"),
    ("hours-per-week", "numeric"),
    ("native-country", "string"),
    ("income", "string"),
];

/// The eight categorical feature columns, in source order.
const CATEGORICAL_NAMES: [&str; 8] = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
];

/// The six numeric feature columns, in source order.
const NUMERIC_NAMES: [&str; 6] = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
];

/// Assert that the column names and types match the documented table.
fn assert_adult_schema(table: &Table) {
    // Every name in the loader's constants reaches a real column.
    for name in Adult::FEATURE_NAMES {
        assert!(
            table.column(name).is_some(),
            "FEATURE_NAMES entry `{name}` names no column"
        );
    }
    assert!(
        table.column(Adult::TARGET).is_some(),
        "TARGET `{}` names no column",
        Adult::TARGET
    );
    assert!(
        !Adult::FEATURE_NAMES.contains(&Adult::TARGET),
        "the target must not also be a feature"
    );
    assert_eq!(table.n_columns(), COLUMN_SCHEMA.len());
    for (column, &(name, kind)) in table.columns().iter().zip(COLUMN_SCHEMA.iter()) {
        assert_eq!(column.name(), name, "column order differs from the source");
        assert_eq!(
            column.data().kind(),
            kind,
            "column {name} has an unexpected type"
        );
    }

    // FEATURE_NAMES and TARGET agree with the source column order.
    assert_eq!(Adult::FEATURE_NAMES.len(), 14);
    for (feature, &(name, _)) in Adult::FEATURE_NAMES.iter().zip(COLUMN_SCHEMA[..14].iter()) {
        assert_eq!(*feature, name);
    }
    assert_eq!(Adult::TARGET, COLUMN_SCHEMA[14].0);
}

/// Assert the Adult dataset invariants: the schema, the two income classes,
/// and the per-column domains.
fn assert_adult_semantics(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_adult_schema(table);

    // The eight categorical features keep their source order. The 14 features
    // mix types, so the numeric ones have no single feature matrix.
    assert_eq!(CATEGORICAL_NAMES.len(), 8);
    let categorical_columns: Vec<&Array1<String>> = CATEGORICAL_NAMES
        .iter()
        .map(|name| table.column(name).unwrap().as_string().unwrap())
        .collect();
    for column in &categorical_columns {
        assert_eq!(column.len(), N_SAMPLES);
    }

    // Six numeric feature columns, in the documented order.
    let numeric_names: Vec<&str> = Adult::FEATURE_NAMES
        .iter()
        .copied()
        .filter(|name| table.column(name).unwrap().as_numeric().is_some())
        .collect();
    assert_eq!(numeric_names, NUMERIC_NAMES);

    // Exactly two income classes, kept verbatim (no trailing period).
    let income = table.column(Adult::TARGET).unwrap().as_string().unwrap();
    assert_eq!(income.len(), N_SAMPLES);
    let unique_labels: HashSet<&str> = income.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        unique_labels,
        HashSet::from(["<=50K", ">50K"]),
        "Adult should have exactly the two income classes `<=50K` and `>50K`"
    );

    // `sex` is one of the two recorded values.
    let valid_sex: HashSet<&str> = ["Male", "Female"].into_iter().collect();
    let sex = table.column("sex").unwrap().as_string().unwrap();
    for (row, value) in sex.iter().enumerate() {
        assert!(
            valid_sex.contains(value.as_str()),
            "row {} sex {:?} is unexpected",
            row,
            value
        );
    }

    // Every numeric feature is finite.
    for name in NUMERIC_NAMES {
        let values = table.column(name).unwrap().as_numeric().unwrap();
        assert_eq!(values.len(), N_SAMPLES);
        for (row, value) in values.iter().enumerate() {
            assert!(
                value.is_finite(),
                "column {} row {} = {} is not finite",
                name,
                row,
                value
            );
        }
    }

    // `age` and `hours-per-week` are positive. The capital counters are
    // non-negative.
    let age = table.column("age").unwrap().as_numeric().unwrap();
    let hours = table
        .column("hours-per-week")
        .unwrap()
        .as_numeric()
        .unwrap();
    let gain = table.column("capital-gain").unwrap().as_numeric().unwrap();
    let loss = table.column("capital-loss").unwrap().as_numeric().unwrap();
    for row in 0..N_SAMPLES {
        assert!(age[row] > 0.0, "row {} age must be positive", row);
        assert!(
            hours[row] > 0.0,
            "row {} hours-per-week must be positive",
            row
        );
        assert!(
            gain[row] >= 0.0 && loss[row] >= 0.0,
            "row {} capital gain/loss must be non-negative",
            row
        );
    }

    // The loader maps the `?` missing token to empty strings. At least one missing
    // categorical value exists in the source (workclass/occupation/native-country),
    // so empty strings appear among the parsed features.
    let has_empty = categorical_columns
        .iter()
        .any(|column| column.iter().any(|s| s.is_empty()));
    assert!(
        has_empty,
        "missing `?` categorical values should be mapped to empty strings"
    );
    assert!(
        !categorical_columns
            .iter()
            .any(|column| column.iter().any(|s| s == "?")),
        "the `?` missing token must not survive into the categorical columns"
    );
}

#[test]
// Verifies that the Adult dataset loads with the documented columns, label
// values, and feature-domain invariants.
fn test_load_adult() {
    let download_dir = "./test_load_adult"; // the loader creates this directory if it is missing

    let dataset = Adult::new(download_dir);
    assert_adult_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name holds the same values as its position
// in the numeric feature matrix.
fn test_adult_columns_agree_with_the_matrix() {
    let download_dir = "./test_adult_columns_agree_with_the_matrix";

    let dataset = Adult::new(download_dir);
    let table = dataset.data().unwrap();
    let numeric = table.numeric_matrix(&NUMERIC_NAMES).unwrap();

    for (col, name) in NUMERIC_NAMES.iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 16_000, N_SAMPLES - 1] {
            assert_eq!(
                column[row],
                numeric[[row, col]],
                "column {name} disagrees with numeric matrix column {col} at row {row}"
            );
        }
    }

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
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

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

    let dataset = Adult::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(file_sha256_matches(&download_dir_path.join("adult.csv"), ADULT_SHA256).unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_adult_into_data() {
    let download_dir = "./test_adult_into_data";

    let dataset = Adult::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The table is now fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 15);

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("age").map(|c| c.data_mut()) {
        values[0] = 1234.0;
    }
    assert_eq!(
        table.column("age").unwrap().as_numeric().unwrap()[0],
        1234.0
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_adult_get_data() {
    let download_dir = "./test_adult_get_data";

    let dataset = Adult::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 15);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place.
fn test_adult_get_data_mut() {
    let download_dir = "./test_adult_get_data_mut";

    let mut dataset = Adult::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) = table.column_mut("age").map(|c| c.data_mut())
    {
        values[0] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(table.column("age").unwrap().as_numeric().unwrap()[0], 99.0);

    remove_dir_all(download_dir).unwrap();
}
