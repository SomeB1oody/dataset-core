#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::dataset::wine_quality::white_wine_quality::WhiteWineQuality;
use dataset_ml::table::{ColumnData, Table};
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// Number of samples.
const N_SAMPLES: usize = 4898;

/// The 12 column names, in source order.
const COLUMN_NAMES: [&str; 12] = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
];

/// Assert the column layout the documentation claims: 12 numeric columns in
/// source order, 11 features and one target named `quality`.
fn assert_columns_match_the_docs(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 12);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    assert_eq!(WhiteWineQuality::FEATURE_NAMES.len(), 11);
    assert_eq!(WhiteWineQuality::TARGET, "quality");
    assert!(
        table.column(WhiteWineQuality::TARGET).is_some(),
        "TARGET `{}` names no column",
        WhiteWineQuality::TARGET
    );
    assert!(
        !WhiteWineQuality::FEATURE_NAMES.contains(&WhiteWineQuality::TARGET),
        "the target must not also be a feature"
    );

    for column in table.columns() {
        assert!(
            matches!(column.data(), ColumnData::Numeric(_)),
            "column {} should be numeric",
            column.name()
        );
    }
}

/// Assert the quality target holds whole numbers in the valid score range, and
/// that the subset holds more than one score.
fn assert_quality_scores(table: &Table) {
    let targets = table.column("quality").unwrap().as_numeric().unwrap();
    assert_eq!(targets.len(), N_SAMPLES);

    let mut unique_qualities = HashSet::new();
    for i in 0..targets.len() {
        let val = targets[i];
        assert!(val.is_finite(), "target[{}] = {} is not finite", i, val);
        assert!(
            val.fract() == 0.0,
            "target[{}] = {} is not an integer value",
            i,
            val
        );
        assert!(
            (0.0..=10.0).contains(&val),
            "target[{}] = {} is outside the valid quality range [0, 10]",
            i,
            val
        );
        unique_qualities.insert(val as i32);
    }
    assert!(
        unique_qualities.len() > 1,
        "targets should contain more than one unique quality score"
    );
}

#[test]
fn test_load_white_wine_quality() {
    let download_dir = "./test_load_white_wine_quality"; // if this directory does not exist, the code creates it

    let dataset = WhiteWineQuality::new(download_dir);
    let table = dataset.data().unwrap();

    assert_columns_match_the_docs(table);
    assert_quality_scores(table);

    // Every physicochemical measurement is finite.
    let features = table
        .numeric_matrix(&WhiteWineQuality::FEATURE_NAMES)
        .unwrap();
    assert_eq!(features.shape(), &[N_SAMPLES, 11]);
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

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name holds the same values as its position
// in the feature matrix.
fn test_white_wine_quality_columns_agree_with_the_matrix() {
    let download_dir = "./test_white_wine_quality_columns_agree_with_the_matrix";

    let dataset = WhiteWineQuality::new(download_dir);
    let table = dataset.data().unwrap();
    let features = table
        .numeric_matrix(&WhiteWineQuality::FEATURE_NAMES)
        .unwrap();

    for (col, name) in COLUMN_NAMES[..11].iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 2449, N_SAMPLES - 1] {
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
fn test_white_wine_quality_no_need_download() {
    let download_dir = "./test_white_wine_quality_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    download_to(
        "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-white.csv",
        download_dir_path,
        None,
    )
    .unwrap();

    let dataset = WhiteWineQuality::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_white_wine_quality_overwrite() {
    let download_dir = "./test_white_wine_quality_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    {
        let fake_white_wine_dataset_path = download_dir_path.join("winequality-white.csv");
        let mut fake_white_wine = File::create(fake_white_wine_dataset_path).unwrap();
        fake_white_wine.write_all(b"fake data").unwrap();
    }

    let dataset = WhiteWineQuality::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(
            &download_dir_path.join("winequality-white.csv"),
            "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836"
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_white_wine_quality_into_data() {
    let download_dir = "./test_white_wine_quality_into_data";

    let dataset = WhiteWineQuality::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The table is now fully owned.

    assert_columns_match_the_docs(&table);
    assert_quality_scores(&table);

    // Owned data allows direct mutation and needs no clone.
    if let Some(ColumnData::Numeric(values)) =
        table.column_mut("fixed acidity").map(|c| c.data_mut())
    {
        values[0] = 10.0;
    }
    assert_eq!(
        table.column("fixed acidity").unwrap().as_numeric().unwrap()[0],
        10.0
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_white_wine_quality_take_data() {
    let download_dir = "./test_white_wine_quality_take_data";

    let mut dataset = WhiteWineQuality::new(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 12);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file and yields the same layout.
    let reloaded = dataset.data().unwrap();
    assert_eq!(reloaded.n_samples(), N_SAMPLES);
    assert_eq!(reloaded.n_columns(), 12);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_white_wine_quality_get_data() {
    let download_dir = "./test_white_wine_quality_get_data";

    let dataset = WhiteWineQuality::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached reference.
    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 12);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_white_wine_quality_get_data_mut() {
    let download_dir = "./test_white_wine_quality_get_data_mut";

    let mut dataset = WhiteWineQuality::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() mutates the cached column in place, with no clone or reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) =
            table.column_mut("fixed acidity").map(|c| c.data_mut())
    {
        values[0] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table.column("fixed acidity").unwrap().as_numeric().unwrap()[0],
        99.0
    );

    remove_dir_all(download_dir).unwrap();
}
