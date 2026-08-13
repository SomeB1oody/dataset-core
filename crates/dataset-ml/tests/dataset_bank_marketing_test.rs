#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::bank_marketing::*;
use dataset_ml::table::{ColumnData, Table};
use ndarray::Array1;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Bank Marketing dataset file (`bank_marketing.csv`).
const BANK_SHA256: &str = "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d";

/// The `bank-full.csv` partition has this many samples.
const N_SAMPLES: usize = 45_211;

/// The 17 columns the documentation lists, as `(name, type)`, in source
/// order.
const COLUMN_SCHEMA: [(&str, &str); 17] = [
    ("age", "numeric"),
    ("job", "string"),
    ("marital", "string"),
    ("education", "string"),
    ("default", "string"),
    ("balance", "numeric"),
    ("housing", "string"),
    ("loan", "string"),
    ("contact", "string"),
    ("day", "numeric"),
    ("month", "string"),
    ("duration", "numeric"),
    ("campaign", "numeric"),
    ("pdays", "numeric"),
    ("previous", "numeric"),
    ("poutcome", "string"),
    ("y", "string"),
];

/// The nine categorical feature columns, in source order.
const CATEGORICAL_NAMES: [&str; 9] = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
];

/// The seven numeric feature columns, in source order.
const NUMERIC_NAMES: [&str; 7] = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous",
];

/// Assert that the column names and types match the documented table.
fn assert_bank_schema(table: &Table) {
    // Every name in the loader's constants reaches a real column.
    for name in BankMarketing::FEATURE_NAMES {
        assert!(
            table.column(name).is_some(),
            "FEATURE_NAMES entry `{name}` names no column"
        );
    }
    assert!(
        table.column(BankMarketing::TARGET).is_some(),
        "TARGET `{}` names no column",
        BankMarketing::TARGET
    );
    assert!(
        !BankMarketing::FEATURE_NAMES.contains(&BankMarketing::TARGET),
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
    assert_eq!(BankMarketing::FEATURE_NAMES.len(), 16);
    for (feature, &(name, _)) in BankMarketing::FEATURE_NAMES
        .iter()
        .zip(COLUMN_SCHEMA[..16].iter())
    {
        assert_eq!(*feature, name);
    }
    assert_eq!(BankMarketing::TARGET, COLUMN_SCHEMA[16].0);
}

/// Assert the Bank Marketing dataset invariants: the schema, the two `y`
/// classes, and the per-column domains.
fn assert_bank_semantics(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_bank_schema(table);

    // The nine categorical features keep their source order. The 16 features
    // mix types, so the numeric ones have no single feature matrix.
    assert_eq!(CATEGORICAL_NAMES.len(), 9);
    let categorical_columns: Vec<&Array1<String>> = CATEGORICAL_NAMES
        .iter()
        .map(|name| table.column(name).unwrap().as_string().unwrap())
        .collect();
    for column in &categorical_columns {
        assert_eq!(column.len(), N_SAMPLES);
    }

    // Seven numeric feature columns, in the documented order.
    let numeric_names: Vec<&str> = BankMarketing::FEATURE_NAMES
        .iter()
        .copied()
        .filter(|name| table.column(name).unwrap().as_numeric().is_some())
        .collect();
    assert_eq!(numeric_names, NUMERIC_NAMES);

    // Exactly two target classes, kept verbatim.
    let y = table
        .column(BankMarketing::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    assert_eq!(y.len(), N_SAMPLES);
    let unique_labels: HashSet<&str> = y.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        unique_labels,
        HashSet::from(["yes", "no"]),
        "Bank Marketing should have exactly the two `y` classes `yes` and `no`"
    );

    // `marital` is one of the recorded values. `default`, `housing`, and `loan`
    // are binary yes/no values.
    let valid_marital: HashSet<&str> = ["married", "single", "divorced"].into_iter().collect();
    let yes_no: HashSet<&str> = ["yes", "no"].into_iter().collect();
    let marital = table.column("marital").unwrap().as_string().unwrap();
    for (row, value) in marital.iter().enumerate() {
        assert!(
            valid_marital.contains(value.as_str()),
            "row {} marital {:?} is unexpected",
            row,
            value
        );
    }
    for name in ["default", "housing", "loan"] {
        let column = table.column(name).unwrap().as_string().unwrap();
        for (row, value) in column.iter().enumerate() {
            assert!(
                yes_no.contains(value.as_str()),
                "row {} column {} = {:?} should be yes/no",
                row,
                name,
                value
            );
        }
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

    // `age` is positive. `duration`, `campaign`, and `previous` are non-negative.
    // `duration` can be 0 for a 0-second call.
    let age = table.column("age").unwrap().as_numeric().unwrap();
    let duration = table.column("duration").unwrap().as_numeric().unwrap();
    let campaign = table.column("campaign").unwrap().as_numeric().unwrap();
    let previous = table.column("previous").unwrap().as_numeric().unwrap();
    for row in 0..N_SAMPLES {
        assert!(age[row] > 0.0, "row {} age must be positive", row);
        assert!(
            duration[row] >= 0.0 && campaign[row] >= 0.0 && previous[row] >= 0.0,
            "row {} duration/campaign/previous must be non-negative",
            row
        );
    }

    // The loader keeps `unknown` verbatim as a category value. It does not map this
    // to an empty string. This value appears in the source, for example in
    // `poutcome` or `contact`.
    assert!(
        categorical_columns
            .iter()
            .any(|column| column.iter().any(|s| s == "unknown")),
        "the `unknown` category label should be preserved verbatim"
    );
    assert!(
        !categorical_columns
            .iter()
            .any(|column| column.iter().any(|s| s.is_empty())),
        "no categorical column should be empty (missing is encoded as `unknown`)"
    );
}

#[test]
// Verifies that the Bank Marketing dataset loads with the documented columns,
// label values, and feature-domain invariants.
fn test_load_bank_marketing() {
    let download_dir = "./test_load_bank_marketing"; // the loader creates this directory if it is missing

    let dataset = BankMarketing::new(download_dir);
    assert_bank_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name holds the same values as its position
// in the numeric feature matrix.
fn test_bank_marketing_columns_agree_with_the_matrix() {
    let download_dir = "./test_bank_marketing_columns_agree_with_the_matrix";

    let dataset = BankMarketing::new(download_dir);
    let table = dataset.data().unwrap();
    let numeric = table.numeric_matrix(&NUMERIC_NAMES).unwrap();

    for (col, name) in NUMERIC_NAMES.iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 22_000, N_SAMPLES - 1] {
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
// Verifies that Bank Marketing reuses a cached file instead of a new download.
fn test_bank_marketing_no_need_download() {
    let download_dir = "./test_load_bank_marketing_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load primes the cache. The second instance then reuses it.
    BankMarketing::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(&download_dir_path.join("bank_marketing.csv"), BANK_SHA256).unwrap(),
        "cached bank_marketing.csv should match the expected SHA256"
    );

    let dataset = BankMarketing::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Bank Marketing data file and
// overwrites it with the real dataset.
fn test_bank_marketing_overwrite() {
    let download_dir = "./test_load_bank_marketing_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let bank_path = download_dir_path.join("bank_marketing.csv");
        let mut fake_bank = File::create(bank_path).unwrap();
        fake_bank.write_all(b"fake data").unwrap();
    }

    let dataset = BankMarketing::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(&download_dir_path.join("bank_marketing.csv"), BANK_SHA256).unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_bank_marketing_into_data() {
    let download_dir = "./test_bank_marketing_into_data";

    let dataset = BankMarketing::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The table is now fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 17);

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
fn test_bank_marketing_get_data() {
    let download_dir = "./test_bank_marketing_get_data";

    let dataset = BankMarketing::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 17);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place.
fn test_bank_marketing_get_data_mut() {
    let download_dir = "./test_bank_marketing_get_data_mut";

    let mut dataset = BankMarketing::new(download_dir);
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
