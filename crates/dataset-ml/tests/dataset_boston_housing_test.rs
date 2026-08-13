#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::dataset::boston_housing::*;
use dataset_ml::table::{ColumnData, Table};
use std::fs::File;
use std::fs::{create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// Number of samples.
const N_SAMPLES: usize = 506;

/// The 14 column names, in source order.
const COLUMN_NAMES: [&str; 14] = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B",
    "LSTAT", "MEDV",
];

/// Assert the column layout the documentation claims: 14 numeric columns in
/// source order, 13 features and one target named `MEDV`.
fn assert_columns_match_the_docs(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 14);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    // The named constants agree with the source column order.
    assert_eq!(BostonHousing::FEATURE_NAMES.len(), 13);
    assert_eq!(BostonHousing::FEATURE_NAMES, COLUMN_NAMES[..13]);
    assert_eq!(BostonHousing::TARGET, "MEDV");
    assert_eq!(BostonHousing::TARGET, COLUMN_NAMES[13]);

    for column in table.columns() {
        assert!(
            matches!(column.data(), ColumnData::Numeric(_)),
            "column {} should be numeric",
            column.name()
        );
    }
}

/// Assert the Boston Housing invariants: every value is finite, and `CHAS` is a
/// binary indicator.
fn assert_boston_housing_semantics(table: &Table) {
    assert_columns_match_the_docs(table);

    // Semantic assertions: all feature and target values must be finite (no NaN or Inf)
    let features = table.numeric_matrix(&BostonHousing::FEATURE_NAMES).unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 13]);
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
        }
    }

    let targets = table
        .column(BostonHousing::TARGET)
        .unwrap()
        .as_numeric()
        .unwrap();
    assert_eq!(targets.len(), N_SAMPLES);
    for i in 0..targets.len() {
        let val = targets[i];
        assert!(val.is_finite(), "target[{}] = {} is not finite", i, val);
    }

    // CHAS (Charles River dummy variable) must only contain 0.0 or 1.0
    let chas = table.column("CHAS").unwrap().as_numeric().unwrap();
    for row in 0..chas.len() {
        let chas_val = chas[row];
        assert!(
            chas_val == 0.0 || chas_val == 1.0,
            "CHAS[{}] = {} is not a binary value (expected 0.0 or 1.0)",
            row,
            chas_val
        );
    }
}

#[test]
// Verifies that the Boston Housing dataset loads with the correct column layout,
// roles, and value domains.
fn test_load_boston_housing() {
    let download_dir = "./test_load_boston_housing"; // the loader creates the directory if it does not exist

    let dataset = BostonHousing::new(download_dir);
    assert_boston_housing_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name holds the same values as its position
// in the feature matrix.
fn test_boston_housing_columns_agree_with_the_matrix() {
    let download_dir = "./test_boston_housing_columns_agree_with_the_matrix";

    let dataset = BostonHousing::new(download_dir);
    let table = dataset.data().unwrap();
    let features = table.numeric_matrix(&BostonHousing::FEATURE_NAMES).unwrap();

    for (col, name) in COLUMN_NAMES[..13].iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 253, N_SAMPLES - 1] {
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
// Verifies that Boston Housing loading uses a pre-downloaded cached file without re-downloading.
fn test_boston_housing_no_need_download() {
    let download_dir = "./test_boston_housing_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    {
        download_to(
            "https://github.com/selva86/datasets/raw/master/BostonHousing.csv",
            download_dir_path,
            None,
        )
        .unwrap();
    }

    let dataset = BostonHousing::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Boston Housing data file and
// overwrites it with the real dataset.
fn test_boston_housing_overwrite() {
    let download_dir = "./test_boston_housing_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let boston_housing_path = download_dir_path.join("BostonHousing.csv");
        let mut fake_boston_housing = File::create(boston_housing_path).unwrap();
        fake_boston_housing.write_all(b"fake data").unwrap();
    }

    let dataset = BostonHousing::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(
            &download_dir_path.join("BostonHousing.csv"),
            "ab16ba38fbbbbcc69fe930aab1293104f1442c8279c130d9eba03dd864bef675"
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_boston_housing_into_data() {
    let download_dir = "./test_boston_housing_into_data";

    let dataset = BostonHousing::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The table is now fully owned.

    assert_boston_housing_semantics(&table);

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("CRIM").map(|c| c.data_mut()) {
        values[0] = 0.1;
    }
    assert_eq!(table.column("CRIM").unwrap().as_numeric().unwrap()[0], 0.1);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the dataset reusable.
fn test_boston_housing_take_data() {
    let download_dir = "./test_boston_housing_take_data";

    let mut dataset = BostonHousing::new(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 14);

    // take_data() resets the instance to unloaded, but it stays usable. The next
    // access reloads it (from the cached file) and yields the same layout.
    let reloaded = dataset.data().unwrap();
    assert_eq!(reloaded.n_samples(), N_SAMPLES);
    assert_eq!(reloaded.n_columns(), 14);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_boston_housing_get_data() {
    let download_dir = "./test_boston_housing_get_data";

    let dataset = BostonHousing::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 14);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place and the change persists.
fn test_boston_housing_get_data_mut() {
    let download_dir = "./test_boston_housing_get_data_mut";

    let mut dataset = BostonHousing::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Mutating the cached column needs no clone or reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) = table.column_mut("CRIM").map(|c| c.data_mut())
    {
        values[0] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(table.column("CRIM").unwrap().as_numeric().unwrap()[0], 99.0);

    remove_dir_all(download_dir).unwrap();
}
