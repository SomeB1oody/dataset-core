#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::WineRecognition;
use dataset_ml::table::{ColumnData, Table};
use std::collections::HashMap;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/dataset/wine_recognition.rs`.
const WINE_RECOGNITION_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data";
const WINE_RECOGNITION_SHA256: &str =
    "6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659";

/// Number of samples.
const N_SAMPLES: usize = 178;

/// Number of feature columns.
const N_FEATURES: usize = 13;

/// The Wine Recognition class distribution: 59, 71, and 48 samples.
const N_CLASS_1: usize = 59;
const N_CLASS_2: usize = 71;
const N_CLASS_3: usize = 48;

/// The 14 column names, in source order. The docs claim this exact layout.
const COLUMN_NAMES: [&str; 14] = [
    "class",
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
];

/// Assert the column layout and the constants the docs claim.
fn assert_wine_recognition_schema(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    assert_eq!(WineRecognition::TARGET, "class");
    let class = table.column(WineRecognition::TARGET).unwrap();
    assert!(matches!(class.data(), ColumnData::String(_)));

    assert_eq!(WineRecognition::FEATURE_NAMES.len(), N_FEATURES);
    for name in WineRecognition::FEATURE_NAMES {
        assert!(
            matches!(table.column(name).unwrap().data(), ColumnData::Numeric(_)),
            "feature {name} should be numeric"
        );
    }
}

/// Checks the Wine Recognition invariants: the schema, the three cultivars with
/// their exact distribution, and the positive measurement domain.
fn assert_wine_recognition_semantics(table: &Table) {
    assert_wine_recognition_schema(table);

    let class = table
        .column(WineRecognition::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (i, name) in class.iter().enumerate() {
        assert!(
            name == "class_1" || name == "class_2" || name == "class_3",
            "class[{}] = {:?} is not a known class",
            i,
            name
        );
        *counts.entry(name.as_str()).or_insert(0) += 1;
    }
    assert_eq!(counts.len(), 3);
    assert_eq!(counts["class_1"], N_CLASS_1);
    assert_eq!(counts["class_2"], N_CLASS_2);
    assert_eq!(counts["class_3"], N_CLASS_3);

    // Every constituent is a physical quantity, so none can be zero or negative.
    let features = table
        .numeric_matrix(&WineRecognition::FEATURE_NAMES)
        .unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let val = features[[row, col]];
            assert!(
                val.is_finite() && val > 0.0,
                "feature[{}, {}] = {} is not a finite positive value",
                row,
                col,
                val
            );
        }
    }
}

#[test]
fn test_load_wine_recognition() {
    let download_dir = "./test_load_wine_recognition"; // the loader creates the directory if it does not exist

    let dataset = WineRecognition::new(download_dir);
    assert_wine_recognition_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name holds the same values as its position
// in the feature matrix.
fn test_wine_recognition_columns_agree_with_the_matrix() {
    let download_dir = "./test_wine_recognition_columns_agree_with_the_matrix";

    let dataset = WineRecognition::new(download_dir);
    let table = dataset.data().unwrap();
    let features = table
        .numeric_matrix(&WineRecognition::FEATURE_NAMES)
        .unwrap();

    // The feature matrix skips `class`, so the 13 feature names start at index 1
    // of the column list.
    for (col, name) in COLUMN_NAMES[1..].iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 89, N_SAMPLES - 1] {
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
fn test_wine_recognition_no_need_download() {
    let download_dir = "./test_wine_recognition_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The loader expects the cached file to have this exact filename.
    download_to(
        WINE_RECOGNITION_URL,
        download_dir_path,
        Some("wine_recognition.csv"),
    )
    .unwrap();

    let dataset = WineRecognition::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_wine_recognition_overwrite() {
    let download_dir = "./test_wine_recognition_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("wine_recognition.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = WineRecognition::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(
            &download_dir_path.join("wine_recognition.csv"),
            WINE_RECOGNITION_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_wine_recognition_into_data() {
    let download_dir = "./test_wine_recognition_into_data";

    let dataset = WineRecognition::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumes `dataset`. The returned table is fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());

    let class = table
        .column(WineRecognition::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    for (i, name) in class.iter().enumerate() {
        assert!(
            name == "class_1" || name == "class_2" || name == "class_3",
            "class[{}] = {:?} is not a known class",
            i,
            name
        );
    }

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("alcohol").map(|c| c.data_mut()) {
        values[0] = 13.5;
    }
    assert_eq!(
        table.column("alcohol").unwrap().as_numeric().unwrap()[0],
        13.5
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_wine_recognition_take_data() {
    let download_dir = "./test_wine_recognition_take_data";

    let mut dataset = WineRecognition::new(download_dir);
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
fn test_wine_recognition_get_data() {
    let download_dir = "./test_wine_recognition_get_data";

    let dataset = WineRecognition::new(download_dir);
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
fn test_wine_recognition_get_data_mut() {
    let download_dir = "./test_wine_recognition_get_data_mut";

    let mut dataset = WineRecognition::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() mutates the cached table in place, with no clone or reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) = table.column_mut("alcohol").map(|c| c.data_mut())
    {
        values[0] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table.column("alcohol").unwrap().as_numeric().unwrap()[0],
        99.0
    );

    remove_dir_all(download_dir).unwrap();
}
