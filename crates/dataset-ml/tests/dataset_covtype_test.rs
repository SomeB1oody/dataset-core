mod common;

use common::file_sha256_matches;
use dataset_ml::covtype::*;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached (decompressed) Cover Type dataset file (`covtype.csv`).
const COVTYPE_SHA256: &str = "0a9371cef7c964b5475d6053cc3e0894a5aa6f65ad1ed3ecb01c45aa96217945";

/// Check every Nth row to keep the 581k×54 semantic assertions fast.
const ROW_STRIDE: usize = 5000;

#[test]
// Verifies that the Cover Type dataset loads with the correct shape, label set,
// and one-hot structure of the wilderness/soil feature blocks.
fn test_load_covtype() {
    let download_dir = "./test_load_covtype"; // the code will create the directory if it doesn't exist

    let dataset = Covtype::new(download_dir);
    let (features, labels) = dataset.data().unwrap();

    assert_eq!(features.shape(), &[581012, 54]);
    assert_eq!(labels.len(), 581012);

    // Semantic assertions: labels are exactly the seven cover-type classes 1..=7.
    let unique_labels: HashSet<_> = labels.iter().copied().collect();
    assert_eq!(
        unique_labels,
        (1u8..=7).collect::<HashSet<_>>(),
        "Covtype should have exactly the seven cover-type classes 1..=7"
    );

    // Semantic assertions on a strided sample of rows: every feature is finite, the
    // 4 Wilderness_Area and 40 Soil_Type columns are binary and one-hot.
    for row in (0..features.nrows()).step_by(ROW_STRIDE) {
        for col in 0..features.ncols() {
            assert!(
                features[[row, col]].is_finite(),
                "feature[{}, {}] = {} is not finite",
                row,
                col,
                features[[row, col]]
            );
        }

        // Columns 10..14 are the one-hot Wilderness_Area block.
        let wilderness: f64 = (10..14).map(|c| features[[row, c]]).sum();
        assert_eq!(
            wilderness, 1.0,
            "row {} Wilderness_Area block must be one-hot (sum == 1)",
            row
        );
        // Columns 14..54 are the one-hot Soil_Type block.
        let soil: f64 = (14..54).map(|c| features[[row, c]]).sum();
        assert_eq!(
            soil, 1.0,
            "row {} Soil_Type block must be one-hot (sum == 1)",
            row
        );
        // Every one-hot column value is exactly 0 or 1.
        for col in 10..54 {
            let v = features[[row, col]];
            assert!(
                v == 0.0 || v == 1.0,
                "feature[{}, {}] = {} is not binary",
                row,
                col,
                v
            );
        }
        // Elevation (column 0) is a plausible positive altitude in metres.
        assert!(
            (1000.0..=5000.0).contains(&features[[row, 0]]),
            "row {} elevation {} out of plausible range",
            row,
            features[[row, 0]]
        );
    }

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Covtype loading uses a pre-existing cached file without re-downloading.
fn test_covtype_no_need_download() {
    let download_dir = "./test_load_covtype_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Prime the cache by loading once, then confirm a second instance reuses it.
    Covtype::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(&download_dir_path.join("covtype.csv"), COVTYPE_SHA256).unwrap(),
        "cached covtype.csv should match the expected SHA256"
    );

    let dataset = Covtype::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Covtype data file is detected and overwritten with the real dataset.
fn test_covtype_overwrite() {
    let download_dir = "./test_load_covtype_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Covtype dataset in advance
    {
        let covtype_path = download_dir_path.join("covtype.csv");
        let mut fake_covtype = File::create(covtype_path).unwrap();
        fake_covtype.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Covtype dataset
    let dataset = Covtype::new(download_dir);
    let (_features, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(file_sha256_matches(&download_dir_path.join("covtype.csv"), COVTYPE_SHA256).unwrap());

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and labels, consuming the dataset.
fn test_covtype_into_data() {
    let download_dir = "./test_covtype_into_data";

    let dataset = Covtype::new(download_dir);
    let (mut features, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; `features`/`labels` are fully owned.

    assert_eq!(features.shape(), &[581012, 54]);
    assert_eq!(labels.len(), 581012);

    // Owned labels are correct: exactly the seven cover-type classes.
    let unique_labels: HashSet<_> = labels.iter().copied().collect();
    assert_eq!(unique_labels, (1u8..=7).collect::<HashSet<_>>());

    // Owned data can be mutated directly, with no `to_owned()` clone.
    features[[0, 0]] = 1234.0;
    assert_eq!(features[[0, 0]], 1234.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_covtype_get_data() {
    let download_dir = "./test_covtype_get_data";

    let dataset = Covtype::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (features, labels) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[581012, 54]);
    assert_eq!(labels.len(), 581012);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
