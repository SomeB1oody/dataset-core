#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::BreastCancer;
use dataset_ml::table::{ColumnData, Table};
use std::collections::HashMap;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/dataset/breast_cancer.rs`.
const BREAST_CANCER_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data";
const BREAST_CANCER_SHA256: &str =
    "d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a";

/// Number of samples.
const N_SAMPLES: usize = 569;

/// Number of feature columns.
const N_FEATURES: usize = 30;

/// The Breast Cancer class distribution: 212 malignant and 357 benign.
const N_MALIGNANT: usize = 212;
const N_BENIGN: usize = 357;

/// The 31 column names, in source order. The docs claim this exact layout.
const COLUMN_NAMES: [&str; 31] = [
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
];

/// Assert the column layout and types the docs claim.
fn assert_breast_cancer_schema(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    // The named constants agree with the source column order.
    assert_eq!(BreastCancer::TARGET, COLUMN_NAMES[0]);
    assert_eq!(BreastCancer::FEATURE_NAMES, COLUMN_NAMES[1..]);

    let diagnosis = table.column(BreastCancer::TARGET).unwrap();
    assert!(matches!(diagnosis.data(), ColumnData::String(_)));

    assert_eq!(BreastCancer::FEATURE_NAMES.len(), N_FEATURES);
    for name in BreastCancer::FEATURE_NAMES {
        assert!(
            matches!(table.column(name).unwrap().data(), ColumnData::Numeric(_)),
            "feature {name} should be numeric"
        );
    }
}

/// Assert the Breast Cancer invariants: the schema, the two diagnoses with their
/// exact distribution, and the non-negative measurement domain.
fn assert_breast_cancer_semantics(table: &Table) {
    assert_breast_cancer_schema(table);

    // Every label is one of the two known diagnoses, and both classes appear.
    let diagnosis = table
        .column(BreastCancer::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (i, name) in diagnosis.iter().enumerate() {
        assert!(
            name == "malignant" || name == "benign",
            "diagnosis[{}] = {:?} is not a known diagnosis",
            i,
            name
        );
        *counts.entry(name.as_str()).or_insert(0) += 1;
    }
    assert_eq!(counts.len(), 2);
    assert_eq!(
        counts["malignant"], N_MALIGNANT,
        "Breast Cancer should have {N_MALIGNANT} malignant rows"
    );
    assert_eq!(
        counts["benign"], N_BENIGN,
        "Breast Cancer should have {N_BENIGN} benign rows"
    );

    // Every measurement is a size or shape statistic, so none can be negative.
    let features = table.numeric_matrix(&BreastCancer::FEATURE_NAMES).unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let val = features[[row, col]];
            assert!(
                val.is_finite() && val >= 0.0,
                "feature[{}, {}] = {} is not a finite non-negative value",
                row,
                col,
                val
            );
        }
    }
}

#[test]
// Verifies that the Breast Cancer dataset loads with the correct column layout,
// types, and class distribution.
fn test_load_breast_cancer() {
    let download_dir = "./test_load_breast_cancer"; // the loader creates the directory if it does not exist

    let dataset = BreastCancer::new(download_dir);
    assert_breast_cancer_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name holds the same values as its position
// in the feature matrix.
fn test_breast_cancer_columns_agree_with_the_matrix() {
    let download_dir = "./test_breast_cancer_columns_agree_with_the_matrix";

    let dataset = BreastCancer::new(download_dir);
    let table = dataset.data().unwrap();
    let features = table.numeric_matrix(&BreastCancer::FEATURE_NAMES).unwrap();

    // The feature matrix skips `diagnosis`, so the 30 feature names start at
    // index 1 of the column list.
    for (col, name) in COLUMN_NAMES[1..].iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 284, N_SAMPLES - 1] {
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
// Verifies that Breast Cancer loading uses a pre-downloaded cached file without re-downloading.
fn test_breast_cancer_no_need_download() {
    let download_dir = "./test_breast_cancer_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download the Breast Cancer dataset in advance, under the filename the loader expects
    download_to(
        BREAST_CANCER_URL,
        download_dir_path,
        Some("breast_cancer.csv"),
    )
    .unwrap();

    let dataset = BreastCancer::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Breast Cancer data file and
// overwrites it with the real dataset.
fn test_breast_cancer_overwrite() {
    let download_dir = "./test_breast_cancer_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("breast_cancer.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = BreastCancer::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(
            &download_dir_path.join("breast_cancer.csv"),
            BREAST_CANCER_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_breast_cancer_into_data() {
    let download_dir = "./test_breast_cancer_into_data";

    let dataset = BreastCancer::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The table is now fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES);

    // Owned labels are correct: one of the two known diagnoses.
    let diagnosis = table
        .column(BreastCancer::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    for (i, name) in diagnosis.iter().enumerate() {
        assert!(
            name == "malignant" || name == "benign",
            "diagnosis[{}] = {:?} is not a known diagnosis",
            i,
            name
        );
    }

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("radius_mean").map(|c| c.data_mut())
    {
        values[0] = 15.0;
    }
    assert_eq!(
        table.column("radius_mean").unwrap().as_numeric().unwrap()[0],
        15.0
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the instance reusable.
fn test_breast_cancer_take_data() {
    let download_dir = "./test_breast_cancer_take_data";

    let mut dataset = BreastCancer::new(download_dir);
    let table = dataset.take_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());

    // take_data() resets the instance to unloaded, but it stays usable. The next
    // access reloads it from the cached file and yields the same table.
    let reloaded = dataset.data().unwrap();
    assert_eq!(reloaded.n_samples(), N_SAMPLES);
    assert_eq!(reloaded.n_columns(), COLUMN_NAMES.len());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_breast_cancer_get_data() {
    let download_dir = "./test_breast_cancer_get_data";

    let dataset = BreastCancer::new(download_dir);
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
fn test_breast_cancer_get_data_mut() {
    let download_dir = "./test_breast_cancer_get_data_mut";

    let mut dataset = BreastCancer::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Mutating the cached table needs no clone or reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) =
            table.column_mut("radius_mean").map(|c| c.data_mut())
    {
        values[0] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table.column("radius_mean").unwrap().as_numeric().unwrap()[0],
        99.0
    );

    remove_dir_all(download_dir).unwrap();
}
