#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::table::{ColumnData, Table};
use dataset_ml::{MlDataset, PalmerPenguins};
use ndarray::Array1;
use std::collections::HashMap;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/dataset/palmer_penguins.rs`.
const PENGUINS_URL: &str =
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv";
const PENGUINS_SHA256: &str = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93";

/// Number of samples.
const N_SAMPLES: usize = 344;

/// The eight column names, in source order.
const COLUMN_NAMES: [&str; 8] = [
    "species",
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
    "year",
];

/// The numeric columns, in source order.
const NUMERIC_NAMES: [&str; 5] = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "year",
];

/// The text columns, in source order.
const STRING_NAMES: [&str; 3] = ["species", "island", "sex"];

/// The four measurement columns. Each one misses the same two samples.
const MEASUREMENT_NAMES: [&str; 4] = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
];

/// Read one numeric column by name.
fn numeric<'a>(table: &'a Table, name: &str) -> &'a Array1<f64> {
    table.column(name).unwrap().as_numeric().unwrap()
}

/// Read one text column by name.
fn text<'a>(table: &'a Table, name: &str) -> &'a Array1<String> {
    table.column(name).unwrap().as_string().unwrap()
}

/// Count how often each value appears in a text column.
fn counts(values: &Array1<String>) -> HashMap<&str, usize> {
    let mut result: HashMap<&str, usize> = HashMap::new();
    for value in values.iter() {
        *result.entry(value.as_str()).or_insert(0) += 1;
    }
    result
}

/// Assert the Palmer Penguins invariants: the column layout, the types, the
/// domains, the class balance, the missing values, and the pinned
/// first and last records.
fn assert_penguins_semantics(table: &Table) {
    // Every name in the loader's constants reaches a real column.
    for name in PalmerPenguins::FEATURE_NAMES {
        assert!(
            table.column(name).is_some(),
            "FEATURE_NAMES entry `{name}` names no column"
        );
    }
    assert!(
        table.column(PalmerPenguins::TARGET).is_some(),
        "TARGET `{}` names no column",
        PalmerPenguins::TARGET
    );
    assert!(
        !PalmerPenguins::FEATURE_NAMES.contains(&PalmerPenguins::TARGET),
        "the target must not also be a feature"
    );
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 8);
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

    // One target, six features, and one time column.
    assert_eq!(PalmerPenguins::FEATURE_NAMES.len(), 6);
    let non_feature_names: Vec<&str> = COLUMN_NAMES
        .iter()
        .copied()
        .filter(|name| !PalmerPenguins::FEATURE_NAMES.contains(name))
        .collect();
    assert_eq!(non_feature_names, vec![PalmerPenguins::TARGET, "year"]);

    // Two of the six features are text.
    let text_feature_names: Vec<&str> = PalmerPenguins::FEATURE_NAMES
        .into_iter()
        .filter(|name| STRING_NAMES.contains(name))
        .collect();
    assert_eq!(text_feature_names, vec!["island", "sex"]);
    for name in &text_feature_names {
        assert_eq!(text(table, name).len(), N_SAMPLES);
    }

    // The target names one of the three known species, and all three appear.
    let species = text(table, PalmerPenguins::TARGET);
    for (row, value) in species.iter().enumerate() {
        assert!(
            value == "Adelie" || value == "Chinstrap" || value == "Gentoo",
            "species[{}] = {:?} is not a known species",
            row,
            value
        );
    }
    let species_counts = counts(species);
    assert_eq!(species_counts.len(), 3);
    assert_eq!(species_counts["Adelie"], 152);
    assert_eq!(species_counts["Chinstrap"], 68);
    assert_eq!(species_counts["Gentoo"], 124);

    // `island` names one of the three known islands, or it is empty. An empty
    // string marks a missing value.
    let island = text(table, "island");
    for (row, value) in island.iter().enumerate() {
        assert!(
            value.is_empty() || value == "Biscoe" || value == "Dream" || value == "Torgersen",
            "island[{}] = {:?} is not a known island",
            row,
            value
        );
    }
    let island_counts = counts(island);
    assert_eq!(island_counts["Biscoe"], 168);
    assert_eq!(island_counts["Dream"], 124);
    assert_eq!(island_counts["Torgersen"], 52);

    // `sex` is male, female, or empty. An empty string marks a missing value.
    let sex = text(table, "sex");
    for (row, value) in sex.iter().enumerate() {
        assert!(
            value.is_empty() || value == "male" || value == "female",
            "sex[{}] = {:?} is not a known value",
            row,
            value
        );
    }
    let sex_counts = counts(sex);
    assert_eq!(sex_counts["male"], 168);
    assert_eq!(sex_counts["female"], 165);
    assert_eq!(sex_counts[""], 11, "sex should have 11 missing values");

    // Every numeric value is either positive or NaN, the source's
    // missing-value marker.
    let mut saw_nan = false;
    for name in NUMERIC_NAMES {
        let values = numeric(table, name);
        for (row, &value) in values.iter().enumerate() {
            if value.is_nan() {
                saw_nan = true;
            } else {
                assert!(
                    value > 0.0,
                    "{}[{}] = {} is neither NaN nor a positive value",
                    name,
                    row,
                    value
                );
            }
        }
    }
    assert!(
        saw_nan,
        "expected at least one NaN from the dataset's missing values"
    );

    // The same two samples miss all four measurements.
    for name in MEASUREMENT_NAMES {
        let values = numeric(table, name);
        let missing: Vec<usize> = values
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_nan())
            .map(|(row, _)| row)
            .collect();
        assert_eq!(missing, vec![3, 271], "{name} misses samples 3 and 271");
    }

    // The study covers three years.
    let year = numeric(table, "year");
    for (row, &value) in year.iter().enumerate() {
        assert!(
            value == 2007.0 || value == 2008.0 || value == 2009.0,
            "year[{}] = {} is not a study year",
            row,
            value
        );
    }
    assert_eq!(year.iter().filter(|&&v| v == 2007.0).count(), 110);
    assert_eq!(year.iter().filter(|&&v| v == 2008.0).count(), 114);
    assert_eq!(year.iter().filter(|&&v| v == 2009.0).count(), 120);

    // The first record of the source, pinned value by value.
    assert_eq!(species[0], "Adelie");
    assert_eq!(island[0], "Torgersen");
    assert_eq!(numeric(table, "bill_length_mm")[0], 39.1);
    assert_eq!(numeric(table, "bill_depth_mm")[0], 18.7);
    assert_eq!(numeric(table, "flipper_length_mm")[0], 181.0);
    assert_eq!(numeric(table, "body_mass_g")[0], 3750.0);
    assert_eq!(sex[0], "male");
    assert_eq!(year[0], 2007.0);

    // The last record of the source, pinned value by value.
    let last = N_SAMPLES - 1;
    assert_eq!(species[last], "Chinstrap");
    assert_eq!(island[last], "Dream");
    assert_eq!(numeric(table, "bill_length_mm")[last], 50.2);
    assert_eq!(numeric(table, "bill_depth_mm")[last], 18.7);
    assert_eq!(numeric(table, "flipper_length_mm")[last], 198.0);
    assert_eq!(numeric(table, "body_mass_g")[last], 3775.0);
    assert_eq!(sex[last], "female");
    assert_eq!(year[last], 2009.0);
}

#[test]
// Verifies that the Palmer Penguins dataset loads with the correct column
// layout, class balance, and missing values.
fn test_load_palmer_penguins() {
    let download_dir = "./test_load_palmer_penguins"; // the code creates the directory if it does not exist

    let dataset = PalmerPenguins::new(download_dir);
    assert_penguins_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that n_samples() reads the table without a per-loader accessor.
fn test_palmer_penguins_n_samples() {
    let download_dir = "./test_palmer_penguins_n_samples";

    let dataset = PalmerPenguins::new(download_dir);
    assert_eq!(PalmerPenguins::NAME, "palmer_penguins");
    assert!(!dataset.is_loaded());
    assert_eq!(dataset.n_samples().unwrap(), N_SAMPLES);
    assert!(dataset.is_loaded());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that PalmerPenguins reuses a cached file instead of a new download.
fn test_palmer_penguins_no_need_download() {
    let download_dir = "./test_palmer_penguins_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    download_to(PENGUINS_URL, download_dir_path, None).unwrap();

    // this call uses the cached dataset instead of downloading it again
    let dataset = PalmerPenguins::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake data file and overwrites it
// with the real dataset.
fn test_palmer_penguins_overwrite() {
    let download_dir = "./test_palmer_penguins_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("penguins.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // this call replaces the fake file with the real dataset
    let dataset = PalmerPenguins::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(file_sha256_matches(&download_dir_path.join("penguins.csv"), PENGUINS_SHA256).unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_palmer_penguins_into_data() {
    let download_dir = "./test_palmer_penguins_into_data";

    let dataset = PalmerPenguins::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The table is now fully owned.

    assert_penguins_semantics(&table);

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) =
        table.column_mut("bill_length_mm").map(|c| c.data_mut())
    {
        values[0] = 40.0;
    }
    assert_eq!(
        table
            .column("bill_length_mm")
            .unwrap()
            .as_numeric()
            .unwrap()[0],
        40.0
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the instance reusable.
fn test_palmer_penguins_take_data() {
    let download_dir = "./test_palmer_penguins_take_data";

    let mut dataset = PalmerPenguins::new(download_dir);
    let table = dataset.take_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 8);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file.
    assert!(!dataset.is_loaded());
    assert_penguins_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_palmer_penguins_get_data() {
    let download_dir = "./test_palmer_penguins_get_data";

    let dataset = PalmerPenguins::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached reference.
    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 8);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place.
fn test_palmer_penguins_get_data_mut() {
    let download_dir = "./test_palmer_penguins_get_data_mut";

    let mut dataset = PalmerPenguins::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // The mutation happens in place. It needs no clone and no reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) =
            table.column_mut("bill_length_mm").map(|c| c.data_mut())
    {
        values[0] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table
            .column("bill_length_mm")
            .unwrap()
            .as_numeric()
            .unwrap()[0],
        99.0
    );
    assert_eq!(
        table.column("bill_depth_mm").unwrap().as_numeric().unwrap()[0],
        18.7,
        "the other columns should stay untouched"
    );

    remove_dir_all(download_dir).unwrap();
}
