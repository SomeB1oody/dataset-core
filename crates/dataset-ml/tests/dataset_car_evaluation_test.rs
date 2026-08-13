#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::car_evaluation::*;
use dataset_ml::table::{ColumnData, Table};
use ndarray::Array1;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Car Evaluation dataset file (`car_evaluation.csv`).
const CAR_EVALUATION_SHA256: &str =
    "b703a9ac69f11e64ce8c223c0a40de4d2e9d769f7fb20be5f8f2e8a619893d83";

/// The Car Evaluation dataset has this many samples.
const N_SAMPLES: usize = 1_728;

/// The Car Evaluation dataset has this many feature columns.
const N_FEATURES: usize = 6;

/// The 7 column names, in source order.
const COLUMN_NAMES: [&str; 7] = [
    "buying", "maint", "doors", "persons", "lug_boot", "safety", "class",
];

/// Allowed value domain for each of the six categorical feature columns.
const FEATURE_DOMAINS: [[&str; 4]; 6] = [
    ["vhigh", "high", "med", "low"], // buying
    ["vhigh", "high", "med", "low"], // maint
    ["2", "3", "4", "5more"],        // doors
    ["2", "4", "more", ""],          // persons (only 3 levels: "" pads the row)
    ["small", "med", "big", ""],     // lug_boot (only 3 levels)
    ["low", "med", "high", ""],      // safety (only 3 levels)
];

/// Assert the column layout the documentation claims: the names and the
/// types.
fn assert_car_evaluation_schema(table: &Table) {
    // Every name in the loader's constants reaches a real column.
    for name in CarEvaluation::FEATURE_NAMES {
        assert!(
            table.column(name).is_some(),
            "FEATURE_NAMES entry `{name}` names no column"
        );
    }
    assert!(
        table.column(CarEvaluation::TARGET).is_some(),
        "TARGET `{}` names no column",
        CarEvaluation::TARGET
    );
    assert!(
        !CarEvaluation::FEATURE_NAMES.contains(&CarEvaluation::TARGET),
        "the target must not also be a feature"
    );
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    for column in table.columns() {
        assert!(
            matches!(column.data(), ColumnData::String(_)),
            "column {} should be a string",
            column.name()
        );
    }

    // FEATURE_NAMES and TARGET agree with the source column order: the six
    // features, then `class`.
    assert_eq!(CarEvaluation::FEATURE_NAMES.len(), N_FEATURES);
    assert_eq!(CarEvaluation::FEATURE_NAMES, COLUMN_NAMES[..N_FEATURES]);
    assert_eq!(CarEvaluation::TARGET, COLUMN_NAMES[N_FEATURES]);
}

/// Checks the Car Evaluation dataset invariants: the schema shape, the four
/// `class` classes, and the per-column categorical feature domains.
fn assert_car_evaluation_semantics(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_car_evaluation_schema(table);

    // Every feature column is a string, reached individually by name (car
    // evaluation has no numeric feature, so there is no numeric_matrix for the
    // features).
    let feature_columns: Vec<&Array1<String>> = CarEvaluation::FEATURE_NAMES
        .iter()
        .map(|name| table.column(name).unwrap().as_string().unwrap())
        .collect();
    assert_eq!(feature_columns.len(), N_FEATURES);
    for column in &feature_columns {
        assert_eq!(column.len(), N_SAMPLES);
    }

    let labels = table
        .column(CarEvaluation::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    assert_eq!(labels.len(), N_SAMPLES);

    let unique_labels: HashSet<&str> = labels.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        unique_labels,
        HashSet::from(["unacc", "acc", "good", "vgood"]),
        "Car Evaluation should have exactly the four classes unacc/acc/good/vgood"
    );

    for (col, column) in feature_columns.iter().enumerate() {
        let domain: HashSet<&str> = FEATURE_DOMAINS[col]
            .iter()
            .copied()
            .filter(|s| !s.is_empty())
            .collect();
        for (row, v) in column.iter().enumerate() {
            assert!(!v.is_empty(), "feature[{row}, {col}] should not be empty");
            assert!(
                domain.contains(v.as_str()),
                "feature[{}, {}] = {:?} is outside column {}'s domain",
                row,
                col,
                v,
                col
            );
        }
    }
}

#[test]
// Verifies that the Car Evaluation dataset loads with the correct column layout,
// label values, and categorical feature domains.
fn test_load_car_evaluation() {
    let download_dir = "./test_load_car_evaluation"; // if the directory does not exist, the code creates it

    let dataset = CarEvaluation::new(download_dir);
    assert_car_evaluation_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name sits at the position the documented
// column order says it does.
fn test_car_evaluation_columns_agree_with_the_matrix() {
    let download_dir = "./test_car_evaluation_columns_agree_with_the_matrix";

    let dataset = CarEvaluation::new(download_dir);
    let table = dataset.data().unwrap();

    for (col, name) in COLUMN_NAMES[..N_FEATURES].iter().enumerate() {
        assert_eq!(
            table.columns()[col].name(),
            *name,
            "table position {col} should hold column {name}"
        );
        let by_name = table.column(name).unwrap().as_string().unwrap();
        let by_position = table.columns()[col].as_string().unwrap();
        for row in [0usize, 1, 864, N_SAMPLES - 1] {
            assert_eq!(
                by_name[row], by_position[row],
                "column {name} disagrees with its table position {col} at row {row}"
            );
        }
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Car Evaluation loading reuses an existing cached file instead of downloading it again.
fn test_car_evaluation_no_need_download() {
    let download_dir = "./test_car_evaluation_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    CarEvaluation::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join("car_evaluation.csv"),
            CAR_EVALUATION_SHA256
        )
        .unwrap(),
        "cached car_evaluation.csv should match the expected SHA256"
    );

    let dataset = CarEvaluation::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Car Evaluation data file and overwrites it with the real dataset.
fn test_car_evaluation_overwrite() {
    let download_dir = "./test_car_evaluation_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("car_evaluation.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = CarEvaluation::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(
            &download_dir_path.join("car_evaluation.csv"),
            CAR_EVALUATION_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_car_evaluation_into_data() {
    let download_dir = "./test_car_evaluation_into_data";

    let dataset = CarEvaluation::new(download_dir);
    let mut table = dataset.into_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());

    // Owned data allows direct mutation and needs no clone.
    if let Some(ColumnData::String(values)) = table.column_mut("buying").map(|c| c.data_mut()) {
        values[0] = "low".to_string();
    }
    assert_eq!(
        table.column("buying").unwrap().as_string().unwrap()[0],
        "low"
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the dataset reusable.
fn test_car_evaluation_take_data() {
    let download_dir = "./test_car_evaluation_take_data";

    let mut dataset = CarEvaluation::new(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file and yields the same layout.
    let reloaded = dataset.data().unwrap();
    assert_eq!(reloaded.n_samples(), N_SAMPLES);
    assert_eq!(reloaded.n_columns(), COLUMN_NAMES.len());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_car_evaluation_get_data() {
    let download_dir = "./test_car_evaluation_get_data";

    let dataset = CarEvaluation::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached reference.
    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place and the change persists.
fn test_car_evaluation_get_data_mut() {
    let download_dir = "./test_car_evaluation_get_data_mut";

    let mut dataset = CarEvaluation::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() mutates the cached `class` column in place. It needs no clone
    // or reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::String(values)) = table.column_mut("class").map(|c| c.data_mut())
    {
        values[0] = "vgood".to_string();
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table.column("class").unwrap().as_string().unwrap()[0],
        "vgood"
    );

    remove_dir_all(download_dir).unwrap();
}
