#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::dataset::california_housing::*;
use dataset_ml::table::{ColumnData, Table};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/california_housing.rs`.
const CALIFORNIA_HOUSING_URL: &str =
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv";
const CALIFORNIA_HOUSING_SHA256: &str =
    "8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e";

/// Number of samples.
const N_SAMPLES: usize = 20640;

/// The nine column names, in source order.
const COLUMN_NAMES: [&str; 9] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "MedHouseVal",
];

/// The name of the only column that can be `NaN`. The source omits
/// `total_bedrooms` from 207 rows.
const AVE_BEDRMS: &str = "AveBedrms";

/// Assert the column layout the documentation claims: nine numeric columns in
/// sklearn order, eight features and one target named `MedHouseVal`.
fn assert_columns_match_the_docs(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 9);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    // The named constants agree with the source column order.
    assert_eq!(CaliforniaHousing::FEATURE_NAMES.len(), 8);
    assert_eq!(CaliforniaHousing::FEATURE_NAMES, COLUMN_NAMES[..8]);
    assert_eq!(CaliforniaHousing::TARGET, "MedHouseVal");
    assert_eq!(CaliforniaHousing::TARGET, COLUMN_NAMES[8]);

    for column in table.columns() {
        assert!(
            matches!(column.data(), ColumnData::Numeric(_)),
            "column {} should be numeric",
            column.name()
        );
    }
}

/// Assert the target column holds finite median house values in units of
/// $100,000. Because of this unit, targets fall in about the [0.15, 5.0] range.
fn assert_targets_in_range(table: &Table) {
    let targets = table
        .column(CaliforniaHousing::TARGET)
        .unwrap()
        .as_numeric()
        .unwrap();
    assert_eq!(targets.len(), N_SAMPLES);
    for i in 0..targets.len() {
        let val = targets[i];
        assert!(
            val.is_finite() && (0.1..=5.1).contains(&val),
            "target[{}] = {} is out of the expected $100k range",
            i,
            val
        );
    }
}

#[test]
fn test_load_california_housing() {
    let download_dir = "./test_load_california_housing"; // the loader creates the directory if it does not exist

    let dataset = CaliforniaHousing::new(download_dir);
    let table = dataset.data().unwrap();

    assert_columns_match_the_docs(table);
    assert_targets_in_range(table);

    // Semantic assertions: every feature is finite, except `AveBedrms`. `AveBedrms`
    // is NaN for the 207 rows with a missing `total_bedrooms` value. No other
    // column may contain NaN. At least one NaN must appear in `AveBedrms`. This
    // confirms the missing-value handling works.
    let mut saw_ave_bedrms_nan = false;
    for name in CaliforniaHousing::FEATURE_NAMES {
        let column = table.column(name).unwrap();
        let values = column.as_numeric().unwrap();
        for row in 0..values.len() {
            let val = values[row];
            if name == AVE_BEDRMS && val.is_nan() {
                saw_ave_bedrms_nan = true;
                continue;
            }
            assert!(
                val.is_finite(),
                "feature[{}, {}] = {} is not finite (only AveBedrms may be NaN)",
                row,
                name,
                val
            );
        }
    }
    assert!(
        saw_ave_bedrms_nan,
        "expected at least one NaN in AveBedrms from the missing total_bedrooms values"
    );

    // Check the geographic columns: Latitude and Longitude must fall within
    // California's bounding box.
    let latitude = table.column("Latitude").unwrap().as_numeric().unwrap();
    let longitude = table.column("Longitude").unwrap().as_numeric().unwrap();
    for row in 0..N_SAMPLES {
        assert!(
            (32.0..=43.0).contains(&latitude[row]),
            "Latitude[{}] = {} is outside California",
            row,
            latitude[row]
        );
        assert!(
            (-125.0..=-113.0).contains(&longitude[row]),
            "Longitude[{}] = {} is outside California",
            row,
            longitude[row]
        );
    }

    // The feature matrix keeps source order and holds the eight feature columns.
    let features = table
        .numeric_matrix(&CaliforniaHousing::FEATURE_NAMES)
        .unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 8]);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name holds the same values as its position
// in the feature matrix.
fn test_california_housing_columns_agree_with_the_matrix() {
    let download_dir = "./test_california_housing_columns_agree_with_the_matrix";

    let dataset = CaliforniaHousing::new(download_dir);
    let table = dataset.data().unwrap();
    let features = table
        .numeric_matrix(&CaliforniaHousing::FEATURE_NAMES)
        .unwrap();

    for (col, name) in COLUMN_NAMES[..8].iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 10_000, N_SAMPLES - 1] {
            let expected = features[[row, col]];
            if expected.is_nan() {
                assert!(
                    column[row].is_nan(),
                    "column {name} disagrees with matrix column {col} at row {row}"
                );
            } else {
                assert_eq!(
                    column[row], expected,
                    "column {name} disagrees with matrix column {col} at row {row}"
                );
            }
        }
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_california_housing_no_need_download() {
    let download_dir = "./test_california_housing_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download the dataset before the test, using the file name the loader expects
    download_to(
        CALIFORNIA_HOUSING_URL,
        download_dir_path,
        Some("california_housing.csv"),
    )
    .unwrap();

    // this call uses the cached dataset instead of downloading it again
    let dataset = CaliforniaHousing::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_california_housing_overwrite() {
    let download_dir = "./test_california_housing_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("california_housing.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // this call replaces the fake file with the real dataset
    let dataset = CaliforniaHousing::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(
            &download_dir_path.join("california_housing.csv"),
            CALIFORNIA_HOUSING_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_california_housing_into_data() {
    let download_dir = "./test_california_housing_into_data";

    let dataset = CaliforniaHousing::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // `into_data` consumes `dataset`. The table is now fully owned.

    assert_columns_match_the_docs(&table);
    assert_targets_in_range(&table);

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("MedInc").map(|c| c.data_mut()) {
        values[0] = 5.0;
    }
    assert_eq!(
        table.column("MedInc").unwrap().as_numeric().unwrap()[0],
        5.0
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_california_housing_take_data() {
    let download_dir = "./test_california_housing_take_data";

    let mut dataset = CaliforniaHousing::new(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 9);

    // After `take_data`, the instance resets to unloaded, but remains usable. The
    // next access reloads the data from the cached file and returns the same layout.
    let reloaded = dataset.data().unwrap();
    assert_eq!(reloaded.n_samples(), N_SAMPLES);
    assert_eq!(reloaded.n_columns(), 9);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_california_housing_get_data() {
    let download_dir = "./test_california_housing_get_data";

    let dataset = CaliforniaHousing::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, `get_data` returns the cached reference.
    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 9);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_california_housing_get_data_mut() {
    let download_dir = "./test_california_housing_get_data_mut";

    let mut dataset = CaliforniaHousing::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // This loads the dataset, then mutates the cached column in place. No clone
    // or reload occurs.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) = table.column_mut("MedInc").map(|c| c.data_mut())
    {
        values[0] = 99.0;
    }

    // The change persists in the cache. A later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table.column("MedInc").unwrap().as_numeric().unwrap()[0],
        99.0
    );

    remove_dir_all(download_dir).unwrap();
}
