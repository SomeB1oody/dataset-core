#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::table::{ColumnData, Table};
use dataset_ml::{MlDataset, Titanic};
use ndarray::Array1;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Titanic dataset file.
const TITANIC_SHA256: &str = "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7";

/// The URL the loader downloads from.
const TITANIC_URL: &str =
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv";

/// Number of samples.
const N_SAMPLES: usize = 891;

/// The twelve column names, in source order.
const COLUMN_NAMES: [&str; 12] = [
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
];

/// The numeric columns, in source order.
const NUMERIC_NAMES: [&str; 7] = [
    "PassengerId",
    "Survived",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
];

/// The text columns, in source order.
const STRING_NAMES: [&str; 5] = ["Name", "Sex", "Ticket", "Cabin", "Embarked"];

/// Read one numeric column by name.
fn numeric<'a>(table: &'a Table, name: &str) -> &'a Array1<f64> {
    table.column(name).unwrap().as_numeric().unwrap()
}

/// Read one text column by name.
fn text<'a>(table: &'a Table, name: &str) -> &'a Array1<String> {
    table.column(name).unwrap().as_string().unwrap()
}

/// Assert the Titanic invariants: the column layout, the types, the domains,
/// the missing-value counts, and the pinned first and last records.
fn assert_titanic_semantics(table: &Table) {
    // Every name in the loader's constants reaches a real column.
    for name in Titanic::FEATURE_NAMES {
        assert!(
            table.column(name).is_some(),
            "FEATURE_NAMES entry `{name}` names no column"
        );
    }
    assert!(
        table.column(Titanic::TARGET).is_some(),
        "TARGET `{}` names no column",
        Titanic::TARGET
    );
    assert!(
        !Titanic::FEATURE_NAMES.contains(&Titanic::TARGET),
        "the target must not also be a feature"
    );
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 12);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    // The names and the types match the documented layout.
    for (index, column) in table.columns().iter().enumerate() {
        assert_eq!(column.name(), COLUMN_NAMES[index]);
        if NUMERIC_NAMES.contains(&column.name()) {
            assert!(
                matches!(column.data(), ColumnData::Numeric(_)),
                "column {} should be numeric",
                column.name()
            );
        } else {
            assert!(
                STRING_NAMES.contains(&column.name()),
                "column {} is neither numeric nor text",
                column.name()
            );
            assert!(
                matches!(column.data(), ColumnData::String(_)),
                "column {} should be text",
                column.name()
            );
        }
    }

    // One identifier, one target, and ten features.
    assert_eq!(Titanic::FEATURE_NAMES.len(), 10);
    let non_feature_names: Vec<&str> = COLUMN_NAMES
        .iter()
        .copied()
        .filter(|name| !Titanic::FEATURE_NAMES.contains(name))
        .collect();
    assert_eq!(non_feature_names, vec!["PassengerId", Titanic::TARGET]);

    // Five of the ten features are text.
    assert_eq!(
        STRING_NAMES.len(),
        5,
        "five of the ten features should be text"
    );
    for name in STRING_NAMES {
        assert_eq!(text(table, name).len(), N_SAMPLES);
    }

    // Every numeric value is finite or NaN. NaN marks a missing value.
    for name in NUMERIC_NAMES {
        let values = numeric(table, name);
        for (row, &value) in values.iter().enumerate() {
            assert!(
                value.is_finite() || value.is_nan(),
                "{}[{}] = {} is not finite or NaN",
                name,
                row,
                value
            );
        }
    }

    // The target is binary. NaN marks a missing value.
    let survived = numeric(table, "Survived");
    let mut died = 0usize;
    let mut lived = 0usize;
    for (row, &value) in survived.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        assert!(
            value == 0.0 || value == 1.0,
            "Survived[{}] = {} is not a binary value (expected 0.0 or 1.0, or NaN)",
            row,
            value
        );
        if value == 0.0 {
            died += 1;
        } else {
            lived += 1;
        }
    }
    assert_eq!(died, 549, "549 passengers should carry Survived = 0.0");
    assert_eq!(lived, 342, "342 passengers should carry Survived = 1.0");

    // The Titanic dataset is known to have missing values.
    let nan_count: usize = Titanic::FEATURE_NAMES
        .iter()
        .filter_map(|name| table.column(name).unwrap().as_numeric())
        .map(|values| values.iter().filter(|value| value.is_nan()).count())
        .sum();
    assert!(
        nan_count > 0,
        "the numeric features should contain at least one NaN (missing Age values)"
    );

    let empty_string_count: usize = STRING_NAMES
        .iter()
        .map(|name| text(table, name).iter().filter(|v| v.is_empty()).count())
        .sum();
    assert!(
        empty_string_count > 0,
        "the text features should contain at least one empty string (missing Cabin and Embarked values)"
    );

    // The exact missing-value counts of the source.
    assert_eq!(
        numeric(table, "Age").iter().filter(|v| v.is_nan()).count(),
        177,
        "Age should have 177 missing values"
    );
    assert_eq!(nan_count, 177, "Age holds the only missing numeric feature");
    assert_eq!(
        text(table, "Cabin").iter().filter(|v| v.is_empty()).count(),
        687,
        "Cabin should have 687 missing values"
    );
    assert_eq!(
        text(table, "Embarked")
            .iter()
            .filter(|v| v.is_empty())
            .count(),
        2,
        "Embarked should have 2 missing values"
    );
    assert_eq!(empty_string_count, 687 + 2);

    // The identifier runs from 1 to 891.
    let passenger_id = numeric(table, "PassengerId");
    assert_eq!(passenger_id[0], 1.0);
    assert_eq!(passenger_id[N_SAMPLES - 1], 891.0);

    // The first record of the source, pinned value by value.
    assert_eq!(numeric(table, "Pclass")[0], 3.0);
    assert_eq!(text(table, "Name")[0], "Braund, Mr. Owen Harris");
    assert_eq!(text(table, "Sex")[0], "male");
    assert_eq!(numeric(table, "Age")[0], 22.0);
    assert_eq!(numeric(table, "SibSp")[0], 1.0);
    assert_eq!(numeric(table, "Parch")[0], 0.0);
    assert_eq!(text(table, "Ticket")[0], "A/5 21171");
    assert_eq!(numeric(table, "Fare")[0], 7.25);
    assert_eq!(text(table, "Cabin")[0], "");
    assert_eq!(text(table, "Embarked")[0], "S");
    assert_eq!(survived[0], 0.0);

    // The last record of the source, pinned value by value.
    let last = N_SAMPLES - 1;
    assert_eq!(numeric(table, "Pclass")[last], 3.0);
    assert_eq!(text(table, "Name")[last], "Dooley, Mr. Patrick");
    assert_eq!(text(table, "Sex")[last], "male");
    assert_eq!(numeric(table, "Age")[last], 32.0);
    assert_eq!(numeric(table, "SibSp")[last], 0.0);
    assert_eq!(numeric(table, "Parch")[last], 0.0);
    assert_eq!(text(table, "Ticket")[last], "370376");
    assert_eq!(numeric(table, "Fare")[last], 7.75);
    assert_eq!(text(table, "Cabin")[last], "");
    assert_eq!(text(table, "Embarked")[last], "Q");
    assert_eq!(survived[last], 0.0);
}

#[test]
// Verifies that the Titanic dataset loads with the correct column layout,
// types, and missing-value counts.
fn test_load_titanic() {
    let download_dir = "./test_load_titanic"; // the code creates the directory if it does not exist

    let dataset = Titanic::new(download_dir);
    assert_titanic_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that n_samples() reads the table without a per-loader accessor.
fn test_titanic_n_samples() {
    let download_dir = "./test_titanic_n_samples";

    let dataset = Titanic::new(download_dir);
    assert_eq!(Titanic::NAME, "titanic");
    assert!(!dataset.is_loaded());
    assert_eq!(dataset.n_samples().unwrap(), N_SAMPLES);
    assert!(dataset.is_loaded());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake data file and overwrites it
// with the real dataset.
fn test_titanic_overwrite() {
    let download_dir = "./test_titanic_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let titanic_path = download_dir_path.join("titanic.csv");
        let mut fake_titanic = File::create(titanic_path).unwrap();
        fake_titanic.write_all(b"fake data").unwrap();
    }

    let dataset = Titanic::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(file_sha256_matches(&download_dir_path.join("titanic.csv"), TITANIC_SHA256).unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Titanic reuses a cached file instead of a new download.
fn test_titanic_no_need_download() {
    let download_dir = "./test_titanic_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    download_to(TITANIC_URL, download_dir_path, None).unwrap();

    let dataset = Titanic::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_titanic_into_data() {
    let download_dir = "./test_titanic_into_data";

    let dataset = Titanic::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The table is now fully owned.

    assert_titanic_semantics(&table);

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("Age").map(|c| c.data_mut()) {
        values[0] = 999.0;
    }
    assert_eq!(table.column("Age").unwrap().as_numeric().unwrap()[0], 999.0);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the instance reusable.
fn test_titanic_take_data() {
    let download_dir = "./test_titanic_take_data";

    let mut dataset = Titanic::new(download_dir);
    let table = dataset.take_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 12);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file.
    assert!(!dataset.is_loaded());
    assert_titanic_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_titanic_get_data() {
    let download_dir = "./test_titanic_get_data";

    let dataset = Titanic::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 12);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place.
fn test_titanic_get_data_mut() {
    let download_dir = "./test_titanic_get_data_mut";

    let mut dataset = Titanic::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) = table.column_mut("Age").map(|c| c.data_mut())
    {
        values[0] = 999.0;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(table.column("Age").unwrap().as_numeric().unwrap()[0], 999.0);
    assert_eq!(
        table.column("Fare").unwrap().as_numeric().unwrap()[0],
        7.25,
        "the other columns should stay untouched"
    );

    remove_dir_all(download_dir).unwrap();
}
