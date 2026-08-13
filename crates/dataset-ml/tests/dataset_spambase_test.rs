#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::Spambase;
use dataset_ml::table::{ColumnData, Table};
use std::collections::HashMap;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA-256 mirrors the constant in `src/dataset/spambase.rs` (`spambase.data`'s bytes).
const SPAMBASE_SHA256: &str = "b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12";

/// The Spambase dataset has this many samples.
const N_SAMPLES: usize = 4601;

/// Number of feature columns.
const N_FEATURES: usize = 57;

/// The Spambase class distribution: 1,813 spam and 2,788 ham.
const N_SPAM: usize = 1813;
const N_HAM: usize = 2788;

/// The 58 column names, in source order. The docs claim this exact layout.
const COLUMN_NAMES: [&str; 58] = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
    "class",
];

/// Assert the column layout and types the docs claim.
fn assert_spambase_schema(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    // The named constants agree with the source column order.
    assert_eq!(Spambase::FEATURE_NAMES, COLUMN_NAMES[..N_FEATURES]);
    assert_eq!(Spambase::TARGET, COLUMN_NAMES[N_FEATURES]);

    let class = table.column(Spambase::TARGET).unwrap();
    assert!(matches!(class.data(), ColumnData::String(_)));

    assert_eq!(Spambase::FEATURE_NAMES.len(), N_FEATURES);
    for name in Spambase::FEATURE_NAMES {
        assert!(
            matches!(table.column(name).unwrap().data(), ColumnData::Numeric(_)),
            "feature {name} should be numeric"
        );
    }
}

/// Checks the Spambase dataset invariants: the schema, the two `class` classes
/// with their exact distribution, and the non-negative numeric feature domain.
fn assert_spambase_semantics(table: &Table) {
    assert_spambase_schema(table);

    let class = table.column(Spambase::TARGET).unwrap().as_string().unwrap();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (i, name) in class.iter().enumerate() {
        assert!(
            name == "ham" || name == "spam",
            "class[{}] = {:?} is not a known class",
            i,
            name
        );
        *counts.entry(name.as_str()).or_insert(0) += 1;
    }
    assert_eq!(counts.len(), 2);
    assert_eq!(
        counts["spam"], N_SPAM,
        "Spambase should have {N_SPAM} spam rows"
    );
    assert_eq!(
        counts["ham"], N_HAM,
        "Spambase should have {N_HAM} ham rows"
    );

    // Every feature is a finite, non-negative frequency or run-length statistic.
    let features = table.numeric_matrix(&Spambase::FEATURE_NAMES).unwrap();
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
fn test_load_spambase() {
    let download_dir = "./test_load_spambase"; // if the directory does not exist, the code creates it

    let dataset = Spambase::new(download_dir);
    assert_spambase_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a column reached by name holds the same values as its position
// in the feature matrix.
fn test_spambase_columns_agree_with_the_matrix() {
    let download_dir = "./test_spambase_columns_agree_with_the_matrix";

    let dataset = Spambase::new(download_dir);
    let table = dataset.data().unwrap();
    let features = table.numeric_matrix(&Spambase::FEATURE_NAMES).unwrap();

    for (col, name) in COLUMN_NAMES[..N_FEATURES].iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 2300, N_SAMPLES - 1] {
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
fn test_spambase_no_need_download() {
    let download_dir = "./test_spambase_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load downloads and extracts the ZIP archive. This primes the cache.
    // The second instance then reuses the extracted file.
    Spambase::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(&download_dir_path.join("spambase.csv"), SPAMBASE_SHA256).unwrap(),
        "cached spambase.csv should match the expected SHA256"
    );

    let dataset = Spambase::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_spambase_overwrite() {
    let download_dir = "./test_spambase_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("spambase.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = Spambase::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(file_sha256_matches(&download_dir_path.join("spambase.csv"), SPAMBASE_SHA256).unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_spambase_into_data() {
    let download_dir = "./test_spambase_into_data";

    let dataset = Spambase::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The table is now fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), COLUMN_NAMES.len());

    let class = table.column(Spambase::TARGET).unwrap().as_string().unwrap();
    for (i, name) in class.iter().enumerate() {
        assert!(
            name == "ham" || name == "spam",
            "class[{}] = {:?} is not a known class",
            i,
            name
        );
    }

    // The owned table allows direct mutation and needs no clone.
    if let Some(ColumnData::Numeric(values)) =
        table.column_mut("word_freq_make").map(|c| c.data_mut())
    {
        values[0] = 0.5;
    }
    assert_eq!(
        table
            .column("word_freq_make")
            .unwrap()
            .as_numeric()
            .unwrap()[0],
        0.5
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_spambase_take_data() {
    let download_dir = "./test_spambase_take_data";

    let mut dataset = Spambase::new(download_dir);
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
fn test_spambase_get_data() {
    let download_dir = "./test_spambase_get_data";

    let dataset = Spambase::new(download_dir);
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
fn test_spambase_get_data_mut() {
    let download_dir = "./test_spambase_get_data_mut";

    let mut dataset = Spambase::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() mutates the cached table in place, with no clone or reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) =
            table.column_mut("word_freq_make").map(|c| c.data_mut())
    {
        values[0] = 0.25;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table
            .column("word_freq_make")
            .unwrap()
            .as_numeric()
            .unwrap()[0],
        0.25
    );

    remove_dir_all(download_dir).unwrap();
}
