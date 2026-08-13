#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::dataset::linnerud::*;
use dataset_ml::table::{ColumnData, Table};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// The pinned source URLs, filenames, and SHA-256 hashes of the two Linnerud
/// files. This file mirrors them here, so the cache-reuse and overwrite tests
/// stay in sync with the loader.
const LINNERUD_EXERCISE_URL: &str = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/linnerud_exercise.csv";
const LINNERUD_PHYSIOLOGICAL_URL: &str = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/linnerud_physiological.csv";
const LINNERUD_EXERCISE_FILENAME: &str = "linnerud_exercise.csv";
const LINNERUD_PHYSIOLOGICAL_FILENAME: &str = "linnerud_physiological.csv";
const LINNERUD_EXERCISE_SHA256: &str =
    "cb8d8c24937643fa2459682efb86c5e667bcd6dd93109eef81964d9e9f11bf8c";
const LINNERUD_PHYSIOLOGICAL_SHA256: &str =
    "2bf7e05c1cd7d0adf0eca1e456941f624bed0a4fc96694d60d0ff7853ec5fcf7";

/// Number of samples.
const N_SAMPLES: usize = 20;

/// The six column names, in scikit-learn column order.
const COLUMN_NAMES: [&str; 6] = ["Chins", "Situps", "Jumps", "Weight", "Waist", "Pulse"];

/// Assert the column layout the documentation claims: six numeric columns, three
/// exercise features followed by three physiological targets.
fn assert_columns_match_the_docs(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 6);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    assert_eq!(Linnerud::FEATURE_NAMES, ["Chins", "Situps", "Jumps"]);

    // Three target columns make this a multi-output regression task.
    assert_eq!(Linnerud::TARGET_NAMES, ["Weight", "Waist", "Pulse"]);

    for column in table.columns() {
        assert!(
            matches!(column.data(), ColumnData::Numeric(_)),
            "column {} should be numeric",
            column.name()
        );
    }
}

/// Assert the defining properties of the Linnerud feature and target matrices:
/// shape `(20, 3)` and every value finite.
fn assert_matrix_ok(matrix: &ndarray::Array2<f64>) {
    assert_eq!(matrix.shape(), &[20, 3]);
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            let val = matrix[[row, col]];
            assert!(
                val.is_finite(),
                "value[{}, {}] = {} is not finite",
                row,
                col,
                val
            );
        }
    }
}

/// Assert both role matrices carry the shape and the finite values the
/// documentation claims.
fn assert_linnerud_matrices(table: &Table) {
    assert_matrix_ok(&table.numeric_matrix(&Linnerud::FEATURE_NAMES).unwrap());
    assert_matrix_ok(&table.numeric_matrix(&Linnerud::TARGET_NAMES).unwrap());
}

#[test]
// Verifies that the Linnerud dataset loads with the correct column layout and
// reference values.
fn test_load_linnerud() {
    // If the directory does not exist, the code creates it.
    let download_dir = "./test_load_linnerud";

    let dataset = Linnerud::new(download_dir);
    let table = dataset.data().unwrap();

    assert_columns_match_the_docs(table);
    assert_linnerud_matrices(table);

    // Check against scikit-learn's published reference. The first exercise row is
    // (Chins, Situps, Jumps) = (5, 162, 60). The first physiological row is
    // (Weight, Waist, Pulse) = (191, 36, 50).
    let value = |name: &str| table.column(name).unwrap().as_numeric().unwrap()[0];
    assert_eq!(value("Chins"), 5.0, "Chins[0] should be 5");
    assert_eq!(value("Situps"), 162.0, "Situps[0] should be 162");
    assert_eq!(value("Jumps"), 60.0, "Jumps[0] should be 60");
    assert_eq!(value("Weight"), 191.0, "Weight[0] should be 191");
    assert_eq!(value("Waist"), 36.0, "Waist[0] should be 36");
    assert_eq!(value("Pulse"), 50.0, "Pulse[0] should be 50");

    // Targets are continuous regression values across all three columns.
    for name in Linnerud::TARGET_NAMES {
        let column = table.column(name).unwrap();
        let mut distinct: Vec<f64> = column.as_numeric().unwrap().to_vec();
        distinct.sort_by(|a, b| a.total_cmp(b));
        distinct.dedup();
        assert!(
            distinct.len() > 2,
            "target column {} should be continuous (got {} distinct values)",
            column.name(),
            distinct.len()
        );
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name holds the same values as its position
// in the role matrix it belongs to.
fn test_linnerud_columns_agree_with_the_matrices() {
    let download_dir = "./test_linnerud_columns_agree_with_the_matrices";

    let dataset = Linnerud::new(download_dir);
    let table = dataset.data().unwrap();

    let name_groups: [(&[&str], usize); 2] = [
        (&Linnerud::FEATURE_NAMES, 0usize),
        (&Linnerud::TARGET_NAMES, 3usize),
    ];
    for (names, offset) in name_groups {
        let matrix = table.numeric_matrix(names).unwrap();
        for (col, name) in COLUMN_NAMES[offset..offset + 3].iter().enumerate() {
            let column = table.column(name).unwrap().as_numeric().unwrap();
            for row in [0usize, 1, 10, N_SAMPLES - 1] {
                assert_eq!(
                    column[row],
                    matrix[[row, col]],
                    "column {name} disagrees with matrix column {col} at row {row}"
                );
            }
        }
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Linnerud loading uses pre-downloaded cached files without re-downloading.
fn test_linnerud_no_need_download() {
    let download_dir = "./test_linnerud_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    {
        download_to(
            LINNERUD_EXERCISE_URL,
            download_dir_path,
            Some(LINNERUD_EXERCISE_FILENAME),
        )
        .unwrap();
        download_to(
            LINNERUD_PHYSIOLOGICAL_URL,
            download_dir_path,
            Some(LINNERUD_PHYSIOLOGICAL_FILENAME),
        )
        .unwrap();
    }

    let dataset = Linnerud::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects corrupt or fake Linnerud files and overwrites
// them with the real data.
fn test_linnerud_overwrite() {
    let download_dir = "./test_linnerud_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let mut fake_exercise =
            File::create(download_dir_path.join(LINNERUD_EXERCISE_FILENAME)).unwrap();
        fake_exercise.write_all(b"fake data").unwrap();
        let mut fake_physiological =
            File::create(download_dir_path.join(LINNERUD_PHYSIOLOGICAL_FILENAME)).unwrap();
        fake_physiological.write_all(b"fake data").unwrap();
    }

    let dataset = Linnerud::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(
            &download_dir_path.join(LINNERUD_EXERCISE_FILENAME),
            LINNERUD_EXERCISE_SHA256
        )
        .unwrap()
    );
    assert!(
        file_sha256_matches(
            &download_dir_path.join(LINNERUD_PHYSIOLOGICAL_FILENAME),
            LINNERUD_PHYSIOLOGICAL_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table, consuming the dataset.
fn test_linnerud_into_data() {
    let download_dir = "./test_linnerud_into_data";

    let dataset = Linnerud::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumes `dataset`. The returned table is fully owned.

    assert_columns_match_the_docs(&table);
    assert_linnerud_matrices(&table);

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("Chins").map(|c| c.data_mut()) {
        values[0] = 6.0;
    }
    assert_eq!(table.column("Chins").unwrap().as_numeric().unwrap()[0], 6.0);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the dataset reusable.
fn test_linnerud_take_data() {
    let download_dir = "./test_linnerud_take_data";

    let mut dataset = Linnerud::new(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 6);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it (from the cached files) and yields the same layout.
    let reloaded = dataset.data().unwrap();
    assert_eq!(reloaded.n_samples(), N_SAMPLES);
    assert_eq!(reloaded.n_columns(), 6);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_linnerud_get_data() {
    let download_dir = "./test_linnerud_get_data";

    let dataset = Linnerud::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached reference.
    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 6);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place and the change persists.
fn test_linnerud_get_data_mut() {
    let download_dir = "./test_linnerud_get_data_mut";

    let mut dataset = Linnerud::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // The mutation happens in place. It needs no clone and no reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) = table.column_mut("Chins").map(|c| c.data_mut())
    {
        values[0] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table.column("Chins").unwrap().as_numeric().unwrap()[0],
        99.0
    );

    remove_dir_all(download_dir).unwrap();
}
