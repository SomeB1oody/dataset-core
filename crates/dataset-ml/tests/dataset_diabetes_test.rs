#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::dataset::diabetes::*;
use dataset_ml::table::{ColumnData, Table};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// The pinned source URL and SHA-256 of the Diabetes dataset, mirrored here so the
/// cache-reuse and overwrite tests stay in sync with the loader.
const DIABETES_URL: &str = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt";
const DIABETES_FILENAME: &str = "diabetes.tab";
const DIABETES_SHA256: &str = "4733febee697862c22139cdac87478a300ce0d101593deb07ed6c0f3328a99cd";

/// Number of samples.
const N_SAMPLES: usize = 442;

/// The 11 column names, in scikit-learn column order.
const COLUMN_NAMES: [&str; 11] = [
    "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6", "target",
];

/// Assert the column layout the documentation claims: 11 numeric columns in
/// scikit-learn order, ten features and one target named `target`.
fn assert_columns_match_the_docs(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 11);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    // The named constants agree with the source column order.
    assert_eq!(Diabetes::FEATURE_NAMES.len(), 10);
    assert_eq!(Diabetes::FEATURE_NAMES, COLUMN_NAMES[..10]);
    assert_eq!(Diabetes::TARGET, "target");
    assert_eq!(Diabetes::TARGET, COLUMN_NAMES[10]);

    for column in table.columns() {
        assert!(
            matches!(column.data(), ColumnData::Numeric(_)),
            "column {} should be numeric",
            column.name()
        );
    }
}

/// Assert the defining properties of scikit-learn's standardized diabetes
/// features. Every value is finite. Each column has mean ~0, and each column's
/// sum of squares is ~1.
fn assert_features_standardized(table: &Table) {
    let features = table.numeric_matrix(&Diabetes::FEATURE_NAMES).unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 10]);

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

    let n = features.nrows() as f64;
    for j in 0..features.ncols() {
        let col = features.column(j);
        let mean = col.sum() / n;
        let sum_sq: f64 = col.iter().map(|&x| x * x).sum();
        assert!(
            mean.abs() < 1e-6,
            "column {} mean = {} is not ~0 (features must be mean-centered)",
            j,
            mean
        );
        assert!(
            (sum_sq - 1.0).abs() < 1e-6,
            "column {} sum-of-squares = {} is not ~1 (features must be L2-normalized)",
            j,
            sum_sq
        );
    }
}

/// Assert the regression target holds the unscaled disease-progression values:
/// finite and within the known range 25..=346.
fn assert_targets_in_range(table: &Table) {
    let targets = table
        .column(Diabetes::TARGET)
        .unwrap()
        .as_numeric()
        .unwrap();
    assert_eq!(targets.len(), N_SAMPLES);
    for i in 0..targets.len() {
        let val = targets[i];
        assert!(val.is_finite(), "targets[{}] = {} is not finite", i, val);
        assert!(
            (25.0..=346.0).contains(&val),
            "targets[{}] = {} is outside the expected range [25, 346]",
            i,
            val
        );
    }
}

#[test]
// Verifies that the Diabetes dataset loads with the correct column layout,
// standardized features, and unscaled regression target.
fn test_load_diabetes() {
    // If the directory does not exist, the code creates it.
    let download_dir = "./test_load_diabetes";

    let dataset = Diabetes::new(download_dir);
    let table = dataset.data().unwrap();

    assert_columns_match_the_docs(table);
    assert_features_standardized(table);
    assert_targets_in_range(table);

    // Check against scikit-learn's published reference: the first standardized
    // `age` value and the first (unscaled) target.
    let age = table.column("age").unwrap().as_numeric().unwrap();
    let targets = table
        .column(Diabetes::TARGET)
        .unwrap()
        .as_numeric()
        .unwrap();
    assert!(
        (age[0] - 0.0380759).abs() < 1e-4,
        "age[0] = {} does not match sklearn's reference 0.0380759",
        age[0]
    );
    assert_eq!(
        targets[0], 151.0,
        "targets[0] should be the unscaled Y = 151"
    );

    // Targets are continuous regression values, not a binary label: there must be
    // more than two distinct values.
    let mut distinct: Vec<f64> = targets.to_vec();
    distinct.sort_by(|a, b| a.total_cmp(b));
    distinct.dedup();
    assert!(
        distinct.len() > 2,
        "targets should be continuous (got {} distinct values)",
        distinct.len()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name holds the same values as its position
// in the feature matrix.
fn test_diabetes_columns_agree_with_the_matrix() {
    let download_dir = "./test_diabetes_columns_agree_with_the_matrix";

    let dataset = Diabetes::new(download_dir);
    let table = dataset.data().unwrap();
    let features = table.numeric_matrix(&Diabetes::FEATURE_NAMES).unwrap();

    for (col, name) in COLUMN_NAMES[..10].iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 221, N_SAMPLES - 1] {
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
// Verifies that Diabetes loading uses a pre-downloaded cached file without re-downloading.
fn test_diabetes_no_need_download() {
    let download_dir = "./test_diabetes_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download the Diabetes dataset in advance, under the loader's expected filename
    {
        download_to(DIABETES_URL, download_dir_path, Some(DIABETES_FILENAME)).unwrap();
    }

    let dataset = Diabetes::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Diabetes data file and
// overwrites it with the real dataset.
fn test_diabetes_overwrite() {
    let download_dir = "./test_diabetes_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let diabetes_path = download_dir_path.join(DIABETES_FILENAME);
        let mut fake_diabetes = File::create(diabetes_path).unwrap();
        fake_diabetes.write_all(b"fake data").unwrap();
    }

    let dataset = Diabetes::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(&download_dir_path.join(DIABETES_FILENAME), DIABETES_SHA256).unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table, consuming the dataset.
fn test_diabetes_into_data() {
    let download_dir = "./test_diabetes_into_data";

    let dataset = Diabetes::new(download_dir);
    let mut table = dataset.into_data().unwrap();

    assert_columns_match_the_docs(&table);
    assert_features_standardized(&table);
    assert_targets_in_range(&table);

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("age").map(|c| c.data_mut()) {
        values[0] = 0.05;
    }
    assert_eq!(table.column("age").unwrap().as_numeric().unwrap()[0], 0.05);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the dataset reusable.
fn test_diabetes_take_data() {
    let download_dir = "./test_diabetes_take_data";

    let mut dataset = Diabetes::new(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 11);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it (from the cached file) and yields the same layout.
    let reloaded = dataset.data().unwrap();
    assert_eq!(reloaded.n_samples(), N_SAMPLES);
    assert_eq!(reloaded.n_columns(), 11);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_diabetes_get_data() {
    let download_dir = "./test_diabetes_get_data";

    let dataset = Diabetes::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached reference.
    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 11);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place and the change persists.
fn test_diabetes_get_data_mut() {
    let download_dir = "./test_diabetes_get_data_mut";

    let mut dataset = Diabetes::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // The mutation happens in place. It needs no clone and no reload.
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
