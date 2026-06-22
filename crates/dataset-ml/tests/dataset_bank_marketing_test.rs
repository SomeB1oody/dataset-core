mod common;

use common::file_sha256_matches;
use dataset_ml::bank_marketing::*;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Bank Marketing dataset file (`bank_marketing.csv`).
const BANK_SHA256: &str = "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d";

/// The `bank-full.csv` partition has this many samples.
const N_SAMPLES: usize = 45_211;

/// Assert the Bank Marketing dataset invariants: the schema shapes, the two
/// `y` classes, and the per-feature domains.
fn assert_bank_semantics(
    strings: &ndarray::Array2<String>,
    numerics: &ndarray::Array2<f64>,
    labels: &ndarray::Array1<String>,
) {
    assert_eq!(strings.shape(), &[N_SAMPLES, 9]);
    assert_eq!(numerics.shape(), &[N_SAMPLES, 7]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Exactly two target classes, kept verbatim.
    let unique_labels: HashSet<&str> = labels.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        unique_labels,
        HashSet::from(["yes", "no"]),
        "Bank Marketing should have exactly the two `y` classes `yes` and `no`"
    );

    // `marital` (string column 1) is one of the recorded values; `default`,
    // `housing`, `loan` (cols 3, 4, 5) are binary yes/no.
    let valid_marital: HashSet<&str> = ["married", "single", "divorced"].into_iter().collect();
    let yes_no: HashSet<&str> = ["yes", "no"].into_iter().collect();
    for row in 0..strings.nrows() {
        assert!(
            valid_marital.contains(strings[[row, 1]].as_str()),
            "row {} marital {:?} is unexpected",
            row,
            strings[[row, 1]]
        );
        for &col in &[3usize, 4, 5] {
            assert!(
                yes_no.contains(strings[[row, col]].as_str()),
                "row {} string col {} = {:?} should be yes/no",
                row,
                col,
                strings[[row, col]]
            );
        }
    }

    // Every numeric feature is finite; `age` (col 0) is positive, while
    // `duration` (col 3), `campaign` (col 4), and `previous` (col 6) are
    // non-negative (`duration` can be 0 for a 0-second call).
    for row in 0..numerics.nrows() {
        for col in 0..numerics.ncols() {
            assert!(
                numerics[[row, col]].is_finite(),
                "numeric[{}, {}] = {} is not finite",
                row,
                col,
                numerics[[row, col]]
            );
        }
        assert!(numerics[[row, 0]] > 0.0, "row {} age must be positive", row);
        assert!(
            numerics[[row, 3]] >= 0.0 && numerics[[row, 4]] >= 0.0 && numerics[[row, 6]] >= 0.0,
            "row {} duration/campaign/previous must be non-negative",
            row
        );
    }

    // `unknown` is kept verbatim as a category value (not mapped to empty), and it
    // does appear in the source (e.g. in `poutcome`/`contact`).
    assert!(
        strings.iter().any(|s| s == "unknown"),
        "the `unknown` category label should be preserved verbatim"
    );
    assert!(
        !strings.iter().any(|s| s.is_empty()),
        "no string feature should be empty (missing is encoded as `unknown`)"
    );
}

#[test]
// Verifies that the Bank Marketing dataset loads with the correct shapes, label
// values, and feature-domain invariants.
fn test_load_bank_marketing() {
    let download_dir = "./test_load_bank_marketing"; // the code will create the directory if it doesn't exist

    let dataset = BankMarketing::new(download_dir);
    let (strings, numerics, labels) = dataset.data().unwrap();

    assert_bank_semantics(strings, numerics, labels);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Bank Marketing loading uses a pre-existing cached file without re-downloading.
fn test_bank_marketing_no_need_download() {
    let download_dir = "./test_load_bank_marketing_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Prime the cache by loading once, then confirm a second instance reuses it.
    BankMarketing::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(&download_dir_path.join("bank_marketing.csv"), BANK_SHA256).unwrap(),
        "cached bank_marketing.csv should match the expected SHA256"
    );

    let dataset = BankMarketing::new(download_dir);
    let (_strings, _numerics, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Bank Marketing data file is detected and overwritten with the real dataset.
fn test_bank_marketing_overwrite() {
    let download_dir = "./test_load_bank_marketing_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Bank Marketing dataset in advance
    {
        let bank_path = download_dir_path.join("bank_marketing.csv");
        let mut fake_bank = File::create(bank_path).unwrap();
        fake_bank.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Bank Marketing dataset
    let dataset = BankMarketing::new(download_dir);
    let (_strings, _numerics, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(&download_dir_path.join("bank_marketing.csv"), BANK_SHA256).unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays, consuming the dataset.
fn test_bank_marketing_into_data() {
    let download_dir = "./test_bank_marketing_into_data";

    let dataset = BankMarketing::new(download_dir);
    let (strings, mut numerics, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; the arrays are fully owned.

    assert_eq!(strings.shape(), &[N_SAMPLES, 9]);
    assert_eq!(numerics.shape(), &[N_SAMPLES, 7]);
    assert_eq!(labels.len(), N_SAMPLES);

    // Owned data can be mutated directly, with no `to_owned()` clone.
    numerics[[0, 0]] = 1234.0;
    assert_eq!(numerics[[0, 0]], 1234.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_bank_marketing_get_data() {
    let download_dir = "./test_bank_marketing_get_data";

    let dataset = BankMarketing::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (strings, numerics, labels) = dataset.get_data().unwrap();
    assert_eq!(strings.shape(), &[N_SAMPLES, 9]);
    assert_eq!(numerics.shape(), &[N_SAMPLES, 7]);
    assert_eq!(labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
