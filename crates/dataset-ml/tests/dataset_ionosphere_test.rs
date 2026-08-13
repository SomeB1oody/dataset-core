#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::Ionosphere;
use dataset_ml::table::{ColumnData, Table};
use std::collections::HashMap;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/dataset/ionosphere.rs`.
const IONOSPHERE_URL: &str =
    "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data";
const IONOSPHERE_SHA256: &str = "46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd";

/// The Ionosphere dataset has this many samples.
const N_SAMPLES: usize = 351;

/// Number of feature columns.
const N_FEATURES: usize = 34;

/// The Ionosphere class distribution: 225 good and 126 bad.
const N_GOOD: usize = 225;
const N_BAD: usize = 126;

/// The 35 column names, in source order. The docs claim this exact layout.
const COLUMN_NAMES: [&str; 35] = [
    "attribute_1",
    "attribute_2",
    "attribute_3",
    "attribute_4",
    "attribute_5",
    "attribute_6",
    "attribute_7",
    "attribute_8",
    "attribute_9",
    "attribute_10",
    "attribute_11",
    "attribute_12",
    "attribute_13",
    "attribute_14",
    "attribute_15",
    "attribute_16",
    "attribute_17",
    "attribute_18",
    "attribute_19",
    "attribute_20",
    "attribute_21",
    "attribute_22",
    "attribute_23",
    "attribute_24",
    "attribute_25",
    "attribute_26",
    "attribute_27",
    "attribute_28",
    "attribute_29",
    "attribute_30",
    "attribute_31",
    "attribute_32",
    "attribute_33",
    "attribute_34",
    "class",
];

/// Assert the column layout the docs claim.
fn assert_ionosphere_schema(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    assert_eq!(Ionosphere::FEATURE_NAMES, COLUMN_NAMES[..N_FEATURES]);
    assert_eq!(Ionosphere::TARGET, COLUMN_NAMES[N_FEATURES]);

    let class = table.column(Ionosphere::TARGET).unwrap();
    assert!(matches!(class.data(), ColumnData::String(_)));

    assert_eq!(Ionosphere::FEATURE_NAMES.len(), N_FEATURES);
    for name in Ionosphere::FEATURE_NAMES {
        let column = table.column(name).unwrap();
        assert!(
            matches!(column.data(), ColumnData::Numeric(_)),
            "feature {name} should be numeric"
        );
    }
}

/// Checks the Ionosphere dataset invariants: the schema, the two `class`
/// classes with their exact distribution, and the normalized numeric feature
/// domain.
fn assert_ionosphere_semantics(table: &Table) {
    assert_ionosphere_schema(table);

    // Labels are one of the two mapped names, and both classes are present.
    let class = table
        .column(Ionosphere::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (i, name) in class.iter().enumerate() {
        assert!(
            name == "good" || name == "bad",
            "class[{}] = {:?} is not a known class",
            i,
            name
        );
        *counts.entry(name.as_str()).or_insert(0) += 1;
    }
    assert_eq!(counts.len(), 2);
    assert_eq!(
        counts["good"], N_GOOD,
        "Ionosphere should have {N_GOOD} good rows"
    );
    assert_eq!(
        counts["bad"], N_BAD,
        "Ionosphere should have {N_BAD} bad rows"
    );

    // The source normalizes every attribute to [-1, 1].
    let features = table.numeric_matrix(&Ionosphere::FEATURE_NAMES).unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let val = features[[row, col]];
            assert!(
                val.is_finite() && val.abs() <= 1.0,
                "feature[{}, {}] = {} is not a finite value in [-1, 1]",
                row,
                col,
                val
            );
        }
    }

    // The second attribute is constant 0 in this collection.
    let attribute_2 = table.column("attribute_2").unwrap().as_numeric().unwrap();
    for (row, value) in attribute_2.iter().enumerate() {
        assert_eq!(*value, 0.0, "attribute_2 should be constant 0 at row {row}");
    }
}

#[test]
// Verifies that the Ionosphere dataset loads with the correct column layout,
// class distribution, and normalized feature domain.
fn test_load_ionosphere() {
    let download_dir = "./test_load_ionosphere"; // if the directory does not exist, the code creates it

    let dataset = Ionosphere::new(download_dir);
    assert_ionosphere_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name holds the same values as its position
// in the feature matrix.
fn test_ionosphere_columns_agree_with_the_matrix() {
    let download_dir = "./test_ionosphere_columns_agree_with_the_matrix";

    let dataset = Ionosphere::new(download_dir);
    let table = dataset.data().unwrap();
    let features = table.numeric_matrix(&Ionosphere::FEATURE_NAMES).unwrap();

    for (col, name) in COLUMN_NAMES[..N_FEATURES].iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 175, N_SAMPLES - 1] {
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
// Verifies that Ionosphere loading reuses an existing cached file instead of downloading it again.
fn test_ionosphere_no_need_download() {
    let download_dir = "./test_ionosphere_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download the Ionosphere dataset in advance and save it under the filename the loader expects
    download_to(IONOSPHERE_URL, download_dir_path, Some("ionosphere.csv")).unwrap();

    let dataset = Ionosphere::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Ionosphere data file and overwrites it with the real dataset.
fn test_ionosphere_overwrite() {
    let download_dir = "./test_ionosphere_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("ionosphere.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = Ionosphere::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(&download_dir_path.join("ionosphere.csv"), IONOSPHERE_SHA256).unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_ionosphere_into_data() {
    let download_dir = "./test_ionosphere_into_data";

    let dataset = Ionosphere::new(download_dir);
    let mut table = dataset.into_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());

    let class = table
        .column(Ionosphere::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    for (i, name) in class.iter().enumerate() {
        assert!(
            name == "good" || name == "bad",
            "class[{}] = {:?} is not a known class",
            i,
            name
        );
    }

    // The owned table allows direct mutation and needs no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("attribute_1").map(|c| c.data_mut())
    {
        values[0] = 0.5;
    }
    assert_eq!(
        table.column("attribute_1").unwrap().as_numeric().unwrap()[0],
        0.5
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the dataset reusable.
fn test_ionosphere_take_data() {
    let download_dir = "./test_ionosphere_take_data";

    let mut dataset = Ionosphere::new(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file and yields the same table.
    let reloaded = dataset.data().unwrap();
    assert_eq!(reloaded.n_samples(), N_SAMPLES);
    assert_eq!(reloaded.n_columns(), COLUMN_NAMES.len());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_ionosphere_get_data() {
    let download_dir = "./test_ionosphere_get_data";

    let dataset = Ionosphere::new(download_dir);
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
fn test_ionosphere_get_data_mut() {
    let download_dir = "./test_ionosphere_get_data_mut";

    let mut dataset = Ionosphere::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() mutates the cached table in place. It needs no clone or reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) =
            table.column_mut("attribute_1").map(|c| c.data_mut())
    {
        values[0] = 0.25;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table.column("attribute_1").unwrap().as_numeric().unwrap()[0],
        0.25
    );

    remove_dir_all(download_dir).unwrap();
}
