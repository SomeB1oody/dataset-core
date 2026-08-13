#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::covtype::Covtype;
use dataset_ml::table::{ColumnData, Table};
use ndarray::Array1;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached (decompressed) Cover Type dataset file (`covtype.csv`).
const COVTYPE_SHA256: &str = "0a9371cef7c964b5475d6053cc3e0894a5aa6f65ad1ed3ecb01c45aa96217945";

/// Check every Nth row to keep the 581k×54 semantic assertions fast.
const ROW_STRIDE: usize = 5000;

/// Number of samples.
const N_SAMPLES: usize = 581_012;

/// Number of feature columns.
const N_FEATURES: usize = 54;

/// Number of columns, which is the 54 features and the target.
const N_COLUMNS: usize = N_FEATURES + 1;

/// Number of one-hot `Wilderness_Area` columns.
const N_WILDERNESS: usize = 4;

/// Number of one-hot `Soil_Type` columns.
const N_SOIL: usize = 40;

/// The 10 quantitative feature names, in source order.
const QUANTITATIVE_NAMES: [&str; 10] = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
];

/// Build the 55 column names the documentation claims, in source order.
fn expected_column_names() -> Vec<String> {
    let mut names: Vec<String> = QUANTITATIVE_NAMES.iter().map(|n| n.to_string()).collect();
    names.extend((0..N_WILDERNESS).map(|i| format!("Wilderness_Area_{i}")));
    names.extend((0..N_SOIL).map(|i| format!("Soil_Type_{i}")));
    names.push("Cover_Type".to_string());
    names
}

/// Borrow the `Cover_Type` column of the table.
fn cover_type_of(table: &Table) -> &Array1<i64> {
    table.column(Covtype::TARGET).unwrap().as_integer().unwrap()
}

/// Assert the column layout the documentation claims: the names and the column
/// types.
fn assert_covtype_columns(table: &Table) {
    let names = expected_column_names();

    assert_eq!(table.n_columns(), N_COLUMNS);
    assert_eq!(table.names().map(str::to_string).collect::<Vec<_>>(), names);

    // The named constants agree with the source column order.
    assert_eq!(Covtype::FEATURE_NAMES.len(), N_FEATURES);
    assert_eq!(Covtype::FEATURE_NAMES, names[..N_FEATURES]);
    assert_eq!(Covtype::TARGET, names[N_FEATURES]);

    for name in Covtype::FEATURE_NAMES {
        assert!(
            matches!(table.column(name).unwrap().data(), ColumnData::Numeric(_)),
            "feature {} should be numeric",
            name
        );
    }

    let cover_type = table.column(Covtype::TARGET).unwrap();
    assert!(
        matches!(cover_type.data(), ColumnData::Integer(_)),
        "Cover_Type should be an integer column"
    );
}

#[test]
// Verifies that the Cover Type dataset loads with the correct column layout,
// label set, and one-hot structure of the wilderness/soil feature blocks.
fn test_load_covtype() {
    let download_dir = "./test_load_covtype"; // the loader creates this directory if it is missing

    let dataset = Covtype::new(download_dir);
    let table = dataset.data().unwrap();

    assert_covtype_columns(table);
    assert_eq!(table.n_samples(), N_SAMPLES);

    let cover_type = cover_type_of(table);
    assert_eq!(cover_type.len(), N_SAMPLES);

    let unique_labels: HashSet<_> = cover_type.iter().copied().collect();
    assert_eq!(
        unique_labels,
        (1i64..=7).collect::<HashSet<_>>(),
        "Covtype should have exactly the seven cover-type classes 1..=7"
    );

    // The codes start at 1, so a code indexes CLASS_NAMES after you subtract 1.
    assert_eq!(Covtype::CLASS_NAMES.len(), unique_labels.len());
    for &code in &unique_labels {
        assert!(
            Covtype::CLASS_NAMES.get(code as usize - 1).is_some(),
            "code {code} names no class"
        );
    }

    // The feature columns, in source order.
    let features: Vec<&Array1<f64>> = Covtype::FEATURE_NAMES
        .iter()
        .map(|name| table.column(name).unwrap().as_numeric().unwrap())
        .collect();
    assert_eq!(features.len(), N_FEATURES);

    let elevation = table.column("Elevation").unwrap().as_numeric().unwrap();

    for row in (0..N_SAMPLES).step_by(ROW_STRIDE) {
        for (col, values) in features.iter().enumerate() {
            assert!(
                values[row].is_finite(),
                "feature[{}, {}] = {} is not finite",
                row,
                col,
                values[row]
            );
        }

        // The 4 Wilderness_Area columns are a one-hot block.
        let wilderness: f64 = (0..N_WILDERNESS)
            .map(|i| {
                table
                    .column(&format!("Wilderness_Area_{i}"))
                    .unwrap()
                    .as_numeric()
                    .unwrap()[row]
            })
            .sum();
        assert_eq!(
            wilderness, 1.0,
            "row {} Wilderness_Area block must be one-hot (sum == 1)",
            row
        );

        // The 40 Soil_Type columns are a one-hot block.
        let soil: f64 = (0..N_SOIL)
            .map(|i| {
                table
                    .column(&format!("Soil_Type_{i}"))
                    .unwrap()
                    .as_numeric()
                    .unwrap()[row]
            })
            .sum();
        assert_eq!(
            soil, 1.0,
            "row {} Soil_Type block must be one-hot (sum == 1)",
            row
        );

        for (col, values) in features.iter().enumerate().skip(10) {
            let v = values[row];
            assert!(
                v == 0.0 || v == 1.0,
                "feature[{}, {}] = {} is not binary",
                row,
                col,
                v
            );
        }

        // Elevation is a plausible positive altitude in meters.
        assert!(
            (1000.0..=5000.0).contains(&elevation[row]),
            "row {} elevation {} out of plausible range",
            row,
            elevation[row]
        );
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Covtype reuses a cached file instead of a new download.
fn test_covtype_no_need_download() {
    let download_dir = "./test_load_covtype_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    Covtype::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(&download_dir_path.join("covtype.csv"), COVTYPE_SHA256).unwrap(),
        "cached covtype.csv should match the expected SHA256"
    );

    let dataset = Covtype::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake Covtype data file and
// overwrites it with the real dataset.
fn test_covtype_overwrite() {
    let download_dir = "./test_load_covtype_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let covtype_path = download_dir_path.join("covtype.csv");
        let mut fake_covtype = File::create(covtype_path).unwrap();
        fake_covtype.write_all(b"fake data").unwrap();
    }

    let dataset = Covtype::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(file_sha256_matches(&download_dir_path.join("covtype.csv"), COVTYPE_SHA256).unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_covtype_into_data() {
    let download_dir = "./test_covtype_into_data";

    let dataset = Covtype::new(download_dir);
    let mut table = dataset.into_data().unwrap();

    assert_covtype_columns(&table);
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(cover_type_of(&table).len(), N_SAMPLES);

    // Owned labels are correct: exactly the seven cover-type classes.
    let unique_labels: HashSet<_> = cover_type_of(&table).iter().copied().collect();
    assert_eq!(unique_labels, (1i64..=7).collect::<HashSet<_>>());

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("Elevation").map(|c| c.data_mut()) {
        values[0] = 1234.0;
    }
    assert_eq!(
        table.column("Elevation").unwrap().as_numeric().unwrap()[0],
        1234.0
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_covtype_get_data() {
    let download_dir = "./test_covtype_get_data";

    let dataset = Covtype::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached reference.
    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), N_COLUMNS);
    assert_eq!(cover_type_of(table).len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that CLASS_NAMES names the seven cover types in the source's order.
// This needs no download.
fn test_covtype_class_names() {
    assert_eq!(Covtype::CLASS_NAMES.len(), 7);
    assert_eq!(
        Covtype::CLASS_NAMES,
        [
            "Spruce/Fir",
            "Lodgepole Pine",
            "Ponderosa Pine",
            "Cottonwood/Willow",
            "Aspen",
            "Douglas-fir",
            "Krummholz",
        ]
    );

    // Code 1 is the first name, and code 7 is the last.
    assert_eq!(Covtype::CLASS_NAMES[1 - 1], "Spruce/Fir");
    assert_eq!(Covtype::CLASS_NAMES[7 - 1], "Krummholz");
}
