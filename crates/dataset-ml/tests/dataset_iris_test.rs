#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::table::{ColumnData, Table};
use dataset_ml::{Iris, MlDataset};
use std::collections::HashMap;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Iris dataset file.
const IRIS_SHA256: &str = "c52742e50315a99f956a383faedf7575552675f6409ef0f9a47076dd08479930";

/// Number of samples.
const N_SAMPLES: usize = 150;

/// The five column names, in source order.
const COLUMN_NAMES: [&str; 5] = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
];

/// Assert the Iris invariants: the column layout, the column types, the domains,
/// and the pinned first record.
fn assert_iris_semantics(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 5);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    // The named constants agree with the source column order.
    assert_eq!(Iris::FEATURE_NAMES, COLUMN_NAMES[..4]);
    assert_eq!(Iris::TARGET, COLUMN_NAMES[4]);

    // Four numeric measurements and one string label.
    for name in Iris::FEATURE_NAMES {
        assert!(
            matches!(table.column(name).unwrap().data(), ColumnData::Numeric(_)),
            "feature {name} should be numeric"
        );
    }
    assert!(matches!(
        table.column(Iris::TARGET).unwrap().data(),
        ColumnData::String(_)
    ));

    // Every measurement is finite and positive.
    let features = table.numeric_matrix(&Iris::FEATURE_NAMES).unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 4]);
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let value = features[[row, col]];
            assert!(
                value.is_finite() && value > 0.0,
                "feature[{}, {}] = {} is not a positive measurement",
                row,
                col,
                value
            );
        }
    }

    // Exactly three species, 50 samples each.
    let species = table.column(Iris::TARGET).unwrap().as_string().unwrap();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for name in species.iter() {
        *counts.entry(name.as_str()).or_insert(0) += 1;
    }
    assert_eq!(counts.len(), 3);
    for name in ["setosa", "versicolor", "virginica"] {
        assert_eq!(counts[name], 50, "{name} should hold 50 samples");
    }

    // The first record of the source, pinned value by value.
    assert_eq!(features.row(0).to_vec(), vec![5.1, 3.5, 1.4, 0.2]);
    assert_eq!(species[0], "setosa");
}

#[test]
// Verifies that the Iris dataset loads with the correct column layout, column
// types, and class balance.
fn test_load_iris() {
    let download_dir = "./test_load_iris"; // the loader creates this directory if it is missing

    let dataset = Iris::new(download_dir);
    assert_iris_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name holds the same values as its position
// in the feature matrix.
fn test_iris_columns_agree_with_the_matrix() {
    let download_dir = "./test_iris_columns_agree_with_the_matrix";

    let dataset = Iris::new(download_dir);
    let table = dataset.data().unwrap();
    let features = table.numeric_matrix(&Iris::FEATURE_NAMES).unwrap();

    for (col, name) in Iris::FEATURE_NAMES.iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 75, N_SAMPLES - 1] {
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
// Verifies that n_samples() reads the table without a per-loader accessor.
fn test_iris_n_samples() {
    let download_dir = "./test_iris_n_samples";

    let dataset = Iris::new(download_dir);
    assert_eq!(Iris::NAME, "iris");
    assert!(!dataset.is_loaded());
    assert_eq!(dataset.n_samples().unwrap(), N_SAMPLES);
    assert!(dataset.is_loaded());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Iris reuses a cached file instead of a new download.
fn test_iris_no_need_download() {
    let download_dir = "./test_iris_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    download_to(
        "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv",
        download_dir_path,
        None,
    )
    .unwrap();

    let dataset = Iris::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake data file and overwrites it
// with the real dataset.
fn test_iris_overwrite() {
    let download_dir = "./test_iris_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let mut fake = File::create(download_dir_path.join("iris.csv")).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = Iris::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(file_sha256_matches(&download_dir_path.join("iris.csv"), IRIS_SHA256).unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_iris_into_data() {
    let download_dir = "./test_iris_into_data";

    let dataset = Iris::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The table is now fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES);

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) =
        table.column_mut("sepal_length").map(|c| c.data_mut())
    {
        values[0] = 5.5;
    }
    assert_eq!(
        table.column("sepal_length").unwrap().as_numeric().unwrap()[0],
        5.5
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the instance reusable.
fn test_iris_take_data() {
    let download_dir = "./test_iris_take_data";

    let mut dataset = Iris::new(download_dir);
    let table = dataset.take_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file.
    assert!(!dataset.is_loaded());
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_iris_get_data() {
    let download_dir = "./test_iris_get_data";

    let dataset = Iris::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    dataset.data().unwrap();
    assert_eq!(dataset.get_data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place.
fn test_iris_get_data_mut() {
    let download_dir = "./test_iris_get_data_mut";

    let mut dataset = Iris::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) =
            table.column_mut("sepal_length").map(|c| c.data_mut())
    {
        values[0] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table.column("sepal_length").unwrap().as_numeric().unwrap()[0],
        99.0
    );
    assert_eq!(
        table.column("sepal_width").unwrap().as_numeric().unwrap()[0],
        3.5,
        "the other columns should stay untouched"
    );

    remove_dir_all(download_dir).unwrap();
}
