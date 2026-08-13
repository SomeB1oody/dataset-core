#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::kddcup99::*;
use dataset_ml::table::{ColumnData, Table};
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached (decompressed) 10% subset file (`kddcup99_10_percent.csv`).
const KDDCUP99_10_PERCENT_SHA256: &str =
    "f8c8267ebcd9c0ed1fd7d6277fe5bfff8732e9b7db8e61b873542b2a534b6f9a";

/// The default 10% subset has this many samples.
const N_SAMPLES_10_PERCENT: usize = 494_021;

/// The full set has this many samples.
const N_SAMPLES_FULL: usize = 4_898_431;

/// Check every Nth row to keep the large semantic assertions fast.
const ROW_STRIDE: usize = 5_000;

/// The 42 columns the documentation lists, as `(name, type)`, in source order.
const COLUMN_SCHEMA: [(&str, &str); 42] = [
    ("duration", "numeric"),
    ("protocol_type", "string"),
    ("service", "string"),
    ("flag", "string"),
    ("src_bytes", "numeric"),
    ("dst_bytes", "numeric"),
    ("land", "numeric"),
    ("wrong_fragment", "numeric"),
    ("urgent", "numeric"),
    ("hot", "numeric"),
    ("num_failed_logins", "numeric"),
    ("logged_in", "numeric"),
    ("num_compromised", "numeric"),
    ("root_shell", "numeric"),
    ("su_attempted", "numeric"),
    ("num_root", "numeric"),
    ("num_file_creations", "numeric"),
    ("num_shells", "numeric"),
    ("num_access_files", "numeric"),
    ("num_outbound_cmds", "numeric"),
    ("is_host_login", "numeric"),
    ("is_guest_login", "numeric"),
    ("count", "numeric"),
    ("srv_count", "numeric"),
    ("serror_rate", "numeric"),
    ("srv_serror_rate", "numeric"),
    ("rerror_rate", "numeric"),
    ("srv_rerror_rate", "numeric"),
    ("same_srv_rate", "numeric"),
    ("diff_srv_rate", "numeric"),
    ("srv_diff_host_rate", "numeric"),
    ("dst_host_count", "numeric"),
    ("dst_host_srv_count", "numeric"),
    ("dst_host_same_srv_rate", "numeric"),
    ("dst_host_diff_srv_rate", "numeric"),
    ("dst_host_same_src_port_rate", "numeric"),
    ("dst_host_srv_diff_host_rate", "numeric"),
    ("dst_host_serror_rate", "numeric"),
    ("dst_host_srv_serror_rate", "numeric"),
    ("dst_host_rerror_rate", "numeric"),
    ("dst_host_srv_rerror_rate", "numeric"),
    ("label", "string"),
];

/// Assert that the column names and types match the documented table, and that
/// the named constants agree with the source column order.
fn assert_kddcup99_schema(table: &Table) {
    assert_eq!(table.n_columns(), COLUMN_SCHEMA.len());
    for (column, &(name, kind)) in table.columns().iter().zip(COLUMN_SCHEMA.iter()) {
        assert_eq!(column.name(), name, "column order differs from the source");
        assert_eq!(
            column.data().kind(),
            kind,
            "column {name} has an unexpected type"
        );
    }

    // The named constants agree with the source column order: the 41 features,
    // then the label, account for exactly the 42 documented columns.
    let column_names: Vec<&str> = COLUMN_SCHEMA.iter().map(|&(name, _)| name).collect();
    assert_eq!(Kddcup99::FEATURE_NAMES.len(), 41);
    assert_eq!(&Kddcup99::FEATURE_NAMES[..], &column_names[..41]);
    assert_eq!(Kddcup99::TARGET, column_names[41]);
    assert_eq!(Kddcup99::FEATURE_NAMES.len() + 1, table.n_columns());
    for name in Kddcup99::FEATURE_NAMES {
        assert!(
            table.column(name).is_some(),
            "feature {name} should be a column"
        );
    }

    // Three string features and 38 numeric features (`label` is the fourth
    // string column).
    let numeric_count = table
        .columns()
        .iter()
        .filter(|column| column.as_numeric().is_some())
        .count();
    assert_eq!(numeric_count, 38);
    let string_count = table
        .columns()
        .iter()
        .filter(|column| column.as_string().is_some())
        .count();
    assert_eq!(string_count, 4);
}

/// Assert the shared cross-partition invariants on a loaded KDD Cup 1999 dataset:
/// the schema, the 23 trailing-period classes, and per-row column domains.
fn assert_kddcup99_semantics(table: &Table, n_samples: usize) {
    assert_eq!(table.n_samples(), n_samples);
    assert_kddcup99_schema(table);

    // The loader keeps labels verbatim, including the trailing period. Both
    // partitions contain exactly the same 23 classes (normal + 22 attack types).
    let labels = table.column(Kddcup99::TARGET).unwrap().as_string().unwrap();
    assert_eq!(labels.len(), n_samples);
    let unique_labels: HashSet<&str> = labels.iter().map(|s| s.as_str()).collect();
    assert!(
        unique_labels.contains("normal."),
        "the `normal.` class (with trailing period) must be present"
    );
    assert!(
        unique_labels.iter().all(|l| l.ends_with('.')),
        "every label must keep its trailing period"
    );
    assert_eq!(
        unique_labels.len(),
        23,
        "KDD Cup 1999 should have exactly 23 distinct classes"
    );

    let valid_protocols: HashSet<&str> = ["tcp", "udp", "icmp"].into_iter().collect();
    let protocol_type = table.column("protocol_type").unwrap().as_string().unwrap();
    let service = table.column("service").unwrap().as_string().unwrap();
    let flag = table.column("flag").unwrap().as_string().unwrap();
    let duration = table.column("duration").unwrap().as_numeric().unwrap();
    let src_bytes = table.column("src_bytes").unwrap().as_numeric().unwrap();
    let dst_bytes = table.column("dst_bytes").unwrap().as_numeric().unwrap();

    for row in (0..n_samples).step_by(ROW_STRIDE) {
        // protocol_type is one of the three known protocols.
        assert!(
            valid_protocols.contains(protocol_type[row].as_str()),
            "row {} protocol_type {:?} is not a known protocol",
            row,
            protocol_type[row]
        );
        // service and flag are non-empty.
        assert!(!service[row].is_empty(), "row {} service is empty", row);
        assert!(!flag[row].is_empty(), "row {} flag is empty", row);

        // Every numeric column is finite.
        for column in table.columns() {
            if let Some(values) = column.as_numeric() {
                assert!(
                    values[row].is_finite(),
                    "column {} row {} = {} is not finite",
                    column.name(),
                    row,
                    values[row]
                );
            }
        }

        // `duration`, `src_bytes`, and `dst_bytes` are counts.
        assert!(
            duration[row] >= 0.0 && src_bytes[row] >= 0.0 && dst_bytes[row] >= 0.0,
            "row {} duration/src_bytes/dst_bytes should be non-negative",
            row
        );
    }
}

#[test]
// Verifies that the default (10% subset) dataset loads with the documented
// columns, label values, and feature-domain invariants.
fn test_load_kddcup99() {
    // If the directory does not exist, the code creates it.
    let download_dir = "./test_load_kddcup99";

    let dataset = Kddcup99::new(download_dir);
    let table = dataset.data().unwrap();

    assert_kddcup99_semantics(table, N_SAMPLES_10_PERCENT);

    // Read the three string features individually, by name.
    for name in ["protocol_type", "service", "flag"] {
        let column = table.column(name).unwrap().as_string().unwrap();
        assert_eq!(column.len(), N_SAMPLES_10_PERCENT);
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the full set (new_full) loads with the documented columns, label
// values, and feature-domain invariants. This is large (~743 MB decompressed,
// several GB parsed) and slow, so it is the only test that exercises the full set.
fn test_load_kddcup99_full() {
    let download_dir = "./test_load_kddcup99_full";

    let dataset = Kddcup99::new_full(download_dir);
    assert_kddcup99_semantics(dataset.data().unwrap(), N_SAMPLES_FULL);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the two partitions cache to distinct filenames, so a 10% subset
// and a full set can coexist in the same storage directory.
fn test_kddcup99_partitions_distinct_files() {
    let download_dir = "./test_kddcup99_partitions";
    let download_dir_path = Path::new(download_dir);

    // Load the default 10% subset. It caches under `kddcup99_10_percent.csv`.
    Kddcup99::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join("kddcup99_10_percent.csv"),
            KDDCUP99_10_PERCENT_SHA256
        )
        .unwrap(),
        "cached 10% subset should match the expected SHA256"
    );
    assert!(
        !download_dir_path.join("kddcup99.csv").exists(),
        "loading the 10% subset must not create the full-set file"
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that KDD Cup 1999 loading uses a pre-existing cached file without re-downloading.
fn test_kddcup99_no_need_download() {
    let download_dir = "./test_load_kddcup99_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    Kddcup99::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join("kddcup99_10_percent.csv"),
            KDDCUP99_10_PERCENT_SHA256
        )
        .unwrap(),
        "cached kddcup99_10_percent.csv should match the expected SHA256"
    );

    let dataset = Kddcup99::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES_10_PERCENT);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake KDD Cup 1999 data file and
// overwrites it with the real dataset.
fn test_kddcup99_overwrite() {
    let download_dir = "./test_load_kddcup99_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let kddcup99_path = download_dir_path.join("kddcup99_10_percent.csv");
        let mut fake_kddcup99 = File::create(kddcup99_path).unwrap();
        fake_kddcup99.write_all(b"fake data").unwrap();
    }

    let dataset = Kddcup99::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES_10_PERCENT);

    assert!(
        file_sha256_matches(
            &download_dir_path.join("kddcup99_10_percent.csv"),
            KDDCUP99_10_PERCENT_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table, consuming the dataset.
fn test_kddcup99_into_data() {
    let download_dir = "./test_kddcup99_into_data";

    let dataset = Kddcup99::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumes `dataset`. The table is fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES_10_PERCENT);
    assert_eq!(table.n_columns(), 42);

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("duration").map(|c| c.data_mut()) {
        values[0] = 1234.0;
    }
    assert_eq!(
        table.column("duration").unwrap().as_numeric().unwrap()[0],
        1234.0
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_kddcup99_get_data() {
    let download_dir = "./test_kddcup99_get_data";

    let dataset = Kddcup99::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached reference.
    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES_10_PERCENT);
    assert_eq!(table.n_columns(), 42);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place.
fn test_kddcup99_get_data_mut() {
    let download_dir = "./test_kddcup99_get_data_mut";

    let mut dataset = Kddcup99::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) =
            table.column_mut("duration").map(|c| c.data_mut())
    {
        values[0] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table.column("duration").unwrap().as_numeric().unwrap()[0],
        99.0
    );

    remove_dir_all(download_dir).unwrap();
}
