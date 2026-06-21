mod common;

use common::file_sha256_matches;
use dataset_ml::kddcup99::*;
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

/// Assert the shared cross-partition invariants on a loaded KDD Cup 1999 dataset:
/// the schema shapes, the 23 trailing-period classes, and per-row feature domains.
fn assert_kddcup99_semantics(
    strings: &ndarray::Array2<String>,
    numerics: &ndarray::Array2<f64>,
    labels: &ndarray::Array1<String>,
    n_samples: usize,
) {
    assert_eq!(strings.shape(), &[n_samples, 3]);
    assert_eq!(numerics.shape(), &[n_samples, 38]);
    assert_eq!(labels.len(), n_samples);

    // Labels are kept verbatim, including the trailing period. Both partitions
    // contain exactly the same 23 classes (normal + 22 attack types).
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

    for row in (0..numerics.nrows()).step_by(ROW_STRIDE) {
        // protocol_type (string column 0) is one of the three known protocols.
        assert!(
            valid_protocols.contains(strings[[row, 0]].as_str()),
            "row {} protocol_type {:?} is not a known protocol",
            row,
            strings[[row, 0]]
        );
        // service and flag (string columns 1, 2) are non-empty.
        assert!(
            !strings[[row, 1]].is_empty(),
            "row {} service is empty",
            row
        );
        assert!(!strings[[row, 2]].is_empty(), "row {} flag is empty", row);

        // Every numeric feature is finite, and the byte counters are non-negative.
        for col in 0..numerics.ncols() {
            assert!(
                numerics[[row, col]].is_finite(),
                "numeric[{}, {}] = {} is not finite",
                row,
                col,
                numerics[[row, col]]
            );
        }
        // `duration` (col 0), `src_bytes` (col 1), `dst_bytes` (col 2) are counts.
        for col in 0..3 {
            assert!(
                numerics[[row, col]] >= 0.0,
                "row {} numeric col {} = {} should be non-negative",
                row,
                col,
                numerics[[row, col]]
            );
        }
    }
}

#[test]
// Verifies that the default (10% subset) dataset loads with the correct shapes,
// label values, and feature-domain invariants.
fn test_load_kddcup99() {
    let download_dir = "./test_load_kddcup99"; // the code will create the directory if it doesn't exist

    let dataset = Kddcup99::new(download_dir);
    let (strings, numerics, labels) = dataset.data().unwrap();

    assert_kddcup99_semantics(strings, numerics, labels, N_SAMPLES_10_PERCENT);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the full set (new_full) loads with the correct shapes, label
// values, and feature-domain invariants. This is large (~743 MB decompressed,
// several GB parsed) and slow, so it is the only test exercising the full set.
fn test_load_kddcup99_full() {
    let download_dir = "./test_load_kddcup99_full";

    let dataset = Kddcup99::new_full(download_dir);
    let (strings, numerics, labels) = dataset.data().unwrap();

    assert_kddcup99_semantics(strings, numerics, labels, N_SAMPLES_FULL);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the two partitions cache to distinct filenames, so a 10% subset
// and a full set can coexist in the same storage directory.
fn test_kddcup99_partitions_distinct_files() {
    let download_dir = "./test_kddcup99_partitions";
    let download_dir_path = Path::new(download_dir);

    // Load the default 10% subset; it caches under `kddcup99_10_percent.csv`.
    Kddcup99::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join("kddcup99_10_percent.csv"),
            KDDCUP99_10_PERCENT_SHA256
        )
        .unwrap(),
        "cached 10% subset should match the expected SHA256"
    );
    // The full-set filename must not have been created by the subset load.
    assert!(
        !download_dir_path.join("kddcup99.csv").exists(),
        "loading the 10% subset must not create the full-set file"
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that KDD Cup 1999 loading uses a pre-existing cached file without re-downloading.
fn test_kddcup99_no_need_download() {
    let download_dir = "./test_load_kddcup99_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Prime the cache by loading once, then confirm a second instance reuses it.
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
    let (_strings, _numerics, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake KDD Cup 1999 data file is detected and overwritten with the real dataset.
fn test_kddcup99_overwrite() {
    let download_dir = "./test_load_kddcup99_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake KDD Cup 1999 dataset in advance
    {
        let kddcup99_path = download_dir_path.join("kddcup99_10_percent.csv");
        let mut fake_kddcup99 = File::create(kddcup99_path).unwrap();
        fake_kddcup99.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake KDD Cup 1999 dataset
    let dataset = Kddcup99::new(download_dir);
    let (_strings, _numerics, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("kddcup99_10_percent.csv"),
            KDDCUP99_10_PERCENT_SHA256
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays, consuming the dataset.
fn test_kddcup99_into_data() {
    let download_dir = "./test_kddcup99_into_data";

    let dataset = Kddcup99::new(download_dir);
    let (strings, mut numerics, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; the arrays are fully owned.

    assert_eq!(strings.shape(), &[N_SAMPLES_10_PERCENT, 3]);
    assert_eq!(numerics.shape(), &[N_SAMPLES_10_PERCENT, 38]);
    assert_eq!(labels.len(), N_SAMPLES_10_PERCENT);

    // Owned data can be mutated directly, with no `to_owned()` clone.
    numerics[[0, 0]] = 1234.0;
    assert_eq!(numerics[[0, 0]], 1234.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_kddcup99_get_data() {
    let download_dir = "./test_kddcup99_get_data";

    let dataset = Kddcup99::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (strings, numerics, labels) = dataset.get_data().unwrap();
    assert_eq!(strings.shape(), &[N_SAMPLES_10_PERCENT, 3]);
    assert_eq!(numerics.shape(), &[N_SAMPLES_10_PERCENT, 38]);
    assert_eq!(labels.len(), N_SAMPLES_10_PERCENT);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
