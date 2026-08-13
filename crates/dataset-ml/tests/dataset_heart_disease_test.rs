#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::dataset::heart_disease::*;
use dataset_ml::table::{ColumnData, Table};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/heart_disease.rs`.
const HEART_DISEASE_URL: &str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data";
const HEART_DISEASE_SHA256: &str =
    "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8";

/// The Heart Disease dataset has this many samples.
const N_SAMPLES: usize = 303;

/// The 14 column names, in source order.
const COLUMN_NAMES: [&str; 14] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num",
];

/// Assert the column layout the documentation claims: the names and the column
/// types.
fn assert_heart_disease_schema(table: &Table) {
    assert_eq!(table.n_columns(), 14);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    assert_eq!(HeartDisease::FEATURE_NAMES.len(), 13);
    assert_eq!(HeartDisease::FEATURE_NAMES, COLUMN_NAMES[..13]);
    assert_eq!(HeartDisease::TARGET, COLUMN_NAMES[13]);

    for name in HeartDisease::FEATURE_NAMES {
        let column = table.column(name).unwrap();
        assert!(
            matches!(column.data(), ColumnData::Numeric(_)),
            "feature {name} should be numeric"
        );
    }

    let num = table.column(HeartDisease::TARGET).unwrap();
    assert!(matches!(num.data(), ColumnData::Integer(_)));
}

/// Assert the Heart Disease dataset invariants: the schema shape, the diagnosis
/// target domain, and the numeric feature domain including the `?` → `NaN` mapping.
fn assert_heart_disease_semantics(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_heart_disease_schema(table);

    let features = table.numeric_matrix(&HeartDisease::FEATURE_NAMES).unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 13]);

    // Diagnosis is 0..=4, and both absence (0) and presence (>0) are present.
    let labels = table
        .column(HeartDisease::TARGET)
        .unwrap()
        .as_integer()
        .unwrap();
    assert_eq!(labels.len(), N_SAMPLES);
    let mut has_absence = false;
    let mut has_presence = false;
    for (i, &y) in labels.iter().enumerate() {
        assert!(
            (0..=4).contains(&y),
            "labels[{i}] = {y} is outside the 0..=4 diagnosis range"
        );
        if y == 0 {
            has_absence = true;
        } else {
            has_presence = true;
        }
    }
    assert!(has_absence, "labels must contain at least one absence (0)");
    assert!(
        has_presence,
        "labels must contain at least one presence (>0)"
    );

    // Every non-missing feature value is finite. NaN marks the source's `?` value.
    // Only `ca` and `thal` can have missing values.
    for name in HeartDisease::FEATURE_NAMES {
        let values = table.column(name).unwrap().as_numeric().unwrap();
        for (row, &v) in values.iter().enumerate() {
            if v.is_nan() {
                assert!(
                    name == "ca" || name == "thal",
                    "unexpected NaN in column {name} at row {row} (only ca/thal may be missing)"
                );
            } else {
                assert!(v.is_finite(), "{name}[{row}] = {v} is not finite");
            }
        }
    }

    // `age` is always present and positive.
    let age = table.column("age").unwrap().as_numeric().unwrap();
    for (row, &value) in age.iter().enumerate() {
        assert!(
            value.is_finite() && value > 0.0,
            "age at row {row} = {value} is not a positive finite value"
        );
    }

    let ca_missing = table
        .column("ca")
        .unwrap()
        .as_numeric()
        .unwrap()
        .iter()
        .filter(|value| value.is_nan())
        .count();
    let thal_missing = table
        .column("thal")
        .unwrap()
        .as_numeric()
        .unwrap()
        .iter()
        .filter(|value| value.is_nan())
        .count();
    assert_eq!(ca_missing, 4, "expected 4 missing `ca` values");
    assert_eq!(thal_missing, 2, "expected 2 missing `thal` values");
}

#[test]
fn test_load_heart_disease() {
    let download_dir = "./test_load_heart_disease"; // the code creates the directory if it does not exist

    let dataset = HeartDisease::new(download_dir);
    assert_heart_disease_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_heart_disease_columns_agree_with_the_matrix() {
    let download_dir = "./test_heart_disease_columns_agree_with_the_matrix";

    let dataset = HeartDisease::new(download_dir);
    let table = dataset.data().unwrap();
    let features = table.numeric_matrix(&HeartDisease::FEATURE_NAMES).unwrap();

    for (col, name) in COLUMN_NAMES[..13].iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 150, N_SAMPLES - 1] {
            let (from_column, from_matrix) = (column[row], features[[row, col]]);
            if from_column.is_nan() {
                assert!(
                    from_matrix.is_nan(),
                    "column {name} disagrees with matrix column {col} at row {row}"
                );
            } else {
                assert_eq!(
                    from_column, from_matrix,
                    "column {name} disagrees with matrix column {col} at row {row}"
                );
            }
        }
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_heart_disease_no_need_download() {
    let download_dir = "./test_heart_disease_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download the dataset before the test, using the file name the loader expects
    download_to(
        HEART_DISEASE_URL,
        download_dir_path,
        Some("heart_disease.csv"),
    )
    .unwrap();

    // this call uses the cached dataset instead of downloading it again
    let dataset = HeartDisease::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_heart_disease_overwrite() {
    let download_dir = "./test_heart_disease_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("heart_disease.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // this call overwrites the fake file with the real dataset
    let dataset = HeartDisease::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(
            &download_dir_path.join("heart_disease.csv"),
            HEART_DISEASE_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_heart_disease_into_data() {
    let download_dir = "./test_heart_disease_into_data";

    let dataset = HeartDisease::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // `into_data` consumes `dataset`. The table is now fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 14);

    for (i, &y) in table
        .column(HeartDisease::TARGET)
        .unwrap()
        .as_integer()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert!(
            (0..=4).contains(&y),
            "labels[{i}] = {y} is outside the 0..=4 diagnosis range"
        );
    }

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("age").map(|c| c.data_mut()) {
        values[0] = 60.0;
    }
    assert_eq!(table.column("age").unwrap().as_numeric().unwrap()[0], 60.0);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_heart_disease_take_data() {
    let download_dir = "./test_heart_disease_take_data";

    let mut dataset = HeartDisease::new(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 14);

    // After `take_data`, the instance resets to unloaded, but remains usable. The
    // next access reloads the data from the cached file and returns the same table.
    let reloaded = dataset.data().unwrap();
    assert_eq!(reloaded.n_samples(), N_SAMPLES);
    assert_eq!(reloaded.n_columns(), 14);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_heart_disease_get_data() {
    let download_dir = "./test_heart_disease_get_data";

    let dataset = HeartDisease::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, `get_data` returns the cached table.
    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 14);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_heart_disease_get_data_mut() {
    let download_dir = "./test_heart_disease_get_data_mut";

    let mut dataset = HeartDisease::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() mutates the cached table in place. It needs no clone or reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) = table.column_mut("age").map(|c| c.data_mut())
    {
        values[0] = 42.0;
    }

    // The change persists in the cache. A later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(table.column("age").unwrap().as_numeric().unwrap()[0], 42.0);

    remove_dir_all(download_dir).unwrap();
}
