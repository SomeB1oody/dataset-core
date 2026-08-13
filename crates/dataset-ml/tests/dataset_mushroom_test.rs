#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::mushroom::*;
use dataset_ml::table::{ColumnData, Table};
use ndarray::Array1;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Mushroom dataset file (`mushroom.csv`).
const MUSHROOM_SHA256: &str = "e65d082030501a3ebcbcd7c9f7c71aa9d28fdfff463bf4cf4716a3fe13ac360e";

/// The Mushroom dataset has this many samples.
const N_SAMPLES: usize = 8_124;

/// The Mushroom dataset has this many feature columns.
const N_FEATURES: usize = 22;

/// The 23 column names, in source order.
const COLUMN_NAMES: [&str; 23] = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
];

/// Assert the column layout the documentation claims: the names and the
/// types.
fn assert_mushroom_schema(table: &Table) {
    // Every name in the loader's constants reaches a real column.
    for name in Mushroom::FEATURE_NAMES {
        assert!(
            table.column(name).is_some(),
            "FEATURE_NAMES entry `{name}` names no column"
        );
    }
    assert!(
        table.column(Mushroom::TARGET).is_some(),
        "TARGET `{}` names no column",
        Mushroom::TARGET
    );
    assert!(
        !Mushroom::FEATURE_NAMES.contains(&Mushroom::TARGET),
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

    // FEATURE_NAMES and TARGET agree with the source column order: `class`
    // leads, then the 22 features.
    assert_eq!(Mushroom::FEATURE_NAMES.len(), N_FEATURES);
    assert_eq!(Mushroom::FEATURE_NAMES, COLUMN_NAMES[1..]);
    assert_eq!(Mushroom::TARGET, COLUMN_NAMES[0]);
}

/// Assert the Mushroom dataset invariants: the schema shape, the two `class`
/// classes, and the all-categorical feature domains.
fn assert_mushroom_semantics(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_mushroom_schema(table);

    // Every feature column is a string, reached individually by name (mushroom
    // has no numeric feature, so there is no numeric_matrix for the features).
    let feature_columns: Vec<&Array1<String>> = Mushroom::FEATURE_NAMES
        .iter()
        .map(|name| table.column(name).unwrap().as_string().unwrap())
        .collect();
    assert_eq!(feature_columns.len(), N_FEATURES);
    for column in &feature_columns {
        assert_eq!(column.len(), N_SAMPLES);
    }

    let labels = table.column(Mushroom::TARGET).unwrap().as_string().unwrap();
    assert_eq!(labels.len(), N_SAMPLES);

    let unique_labels: HashSet<&str> = labels.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        unique_labels,
        HashSet::from(["e", "p"]),
        "Mushroom should have exactly the two classes `e` (edible) and `p` (poisonous)"
    );

    // Every feature value is either a single-letter code or the empty string (for
    // the missing `stalk-root` token), and the raw `?` token never survives.
    for (name, column) in Mushroom::FEATURE_NAMES.iter().zip(feature_columns.iter()) {
        for (row, v) in column.iter().enumerate() {
            assert!(
                v.is_empty() || v.chars().count() == 1,
                "feature {name}[{row}] = {v:?} should be a single-letter code or empty"
            );
            assert_ne!(
                v, "?",
                "the `?` missing token must not survive at {name}[{row}]"
            );
        }
    }

    // `bruises` is one of the two recorded codes.
    let bruises = table.column("bruises").unwrap().as_string().unwrap();
    let valid_bruises: HashSet<&str> = ["t", "f"].into_iter().collect();
    // `gill-size` is one of its two recorded codes.
    let gill_size = table.column("gill-size").unwrap().as_string().unwrap();
    let valid_gill_size: HashSet<&str> = ["b", "n"].into_iter().collect();
    for row in 0..N_SAMPLES {
        assert!(
            valid_bruises.contains(bruises[row].as_str()),
            "row {} bruises {:?} is unexpected",
            row,
            bruises[row]
        );
        assert!(
            valid_gill_size.contains(gill_size[row].as_str()),
            "row {} gill-size {:?} is unexpected",
            row,
            gill_size[row]
        );
    }

    // The loader maps the missing `?` token (only in `stalk-root`) to empty
    // strings, so at least one empty value must be present there.
    let stalk_root = table.column("stalk-root").unwrap().as_string().unwrap();
    let has_empty_stalk_root = stalk_root.iter().any(|value| value.is_empty());
    assert!(
        has_empty_stalk_root,
        "missing `stalk-root` values should be mapped to empty strings"
    );
}

#[test]
// Verifies that the Mushroom dataset loads with the correct column layout, label
// values, and categorical feature domains.
fn test_load_mushroom() {
    let download_dir = "./test_load_mushroom"; // the loader creates this directory if it is missing

    let dataset = Mushroom::new(download_dir);
    assert_mushroom_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name sits at the position the documented
// column order says it does.
fn test_mushroom_columns_agree_with_the_matrix() {
    let download_dir = "./test_mushroom_columns_agree_with_the_matrix";

    let dataset = Mushroom::new(download_dir);
    let table = dataset.data().unwrap();

    // `class` leads the source order, so feature column `col` sits at table
    // position `col + 1`.
    for (col, name) in COLUMN_NAMES[1..].iter().enumerate() {
        let position = col + 1;
        assert_eq!(
            table.columns()[position].name(),
            *name,
            "table position {position} should hold column {name}"
        );
        let by_name = table.column(name).unwrap().as_string().unwrap();
        let by_position = table.columns()[position].as_string().unwrap();
        for row in [0usize, 1, 4_000, N_SAMPLES - 1] {
            assert_eq!(
                by_name[row], by_position[row],
                "column {name} disagrees with its table position {position} at row {row}"
            );
        }
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Mushroom reuses a cached file instead of a new download.
fn test_mushroom_no_need_download() {
    let download_dir = "./test_load_mushroom_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    Mushroom::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(&download_dir_path.join("mushroom.csv"), MUSHROOM_SHA256).unwrap(),
        "cached mushroom.csv should match the expected SHA256"
    );

    let dataset = Mushroom::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

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

    let dataset = Mushroom::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(file_sha256_matches(&download_dir_path.join("mushroom.csv"), MUSHROOM_SHA256).unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_mushroom_into_data() {
    let download_dir = "./test_mushroom_into_data";

    let dataset = Mushroom::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // `into_data()` consumes `dataset`. The table is now fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::String(values)) = table.column_mut("cap-shape").map(|c| c.data_mut()) {
        values[0] = "z".to_string();
    }
    assert_eq!(
        table.column("cap-shape").unwrap().as_string().unwrap()[0],
        "z"
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_mushroom_get_data() {
    let download_dir = "./test_mushroom_get_data";

    let dataset = Mushroom::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached reference.
    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());

    remove_dir_all(download_dir).unwrap();
}
