#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::wholesale_customers::WholesaleCustomers;
use dataset_ml::traits::MlDataset;
use ndarray::Array2;
use std::collections::HashMap;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Wholesale Customers dataset file.
const WHOLESALE_SHA256: &str = "c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef";

/// The name of the cached dataset file.
const WHOLESALE_FILENAME: &str = "wholesale_customers.csv";

/// Number of clients.
const N_SAMPLES: usize = 440;

/// Number of feature columns.
const N_FEATURES: usize = 8;

/// Sum of each spending column, for columns `2` to `7`.
const SPENDING_SUMS: [f64; 6] = [
    5_280_131.0,
    2_550_357.0,
    3_498_562.0,
    1_351_650.0,
    1_267_857.0,
    670_943.0,
];

/// Count how many rows hold each value of one column.
fn value_counts(features: &Array2<f64>, col: usize) -> HashMap<i64, usize> {
    let mut counts = HashMap::new();
    for row in 0..features.nrows() {
        *counts.entry(features[[row, col]] as i64).or_insert(0) += 1;
    }
    counts
}

/// Assert the Wholesale Customers invariants: the shape, the two code domains,
/// the spending domain, and the pinned records.
fn assert_wholesale_semantics(features: &Array2<f64>) {
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);

    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let value = features[[row, col]];
            assert!(
                value.is_finite(),
                "feature[{}, {}] = {} is not finite",
                row,
                col,
                value
            );
            // Every source value is a whole number, and no client spends a
            // negative amount.
            assert!(
                value.fract() == 0.0 && value > 0.0,
                "feature[{}, {}] = {} is not a positive whole number",
                row,
                col,
                value
            );
        }

        let channel = features[[row, 0]];
        let region = features[[row, 1]];
        assert!(
            channel == 1.0 || channel == 2.0,
            "row {} Channel {} is outside 1..=2",
            row,
            channel
        );
        assert!(
            (1.0..=3.0).contains(&region),
            "row {} Region {} is outside 1..=3",
            row,
            region
        );
    }

    // The two codes keep the published balance.
    let channels = value_counts(features, 0);
    assert_eq!(
        channels[&1], 298,
        "298 clients should use the Horeca channel"
    );
    assert_eq!(
        channels[&2], 142,
        "142 clients should use the Retail channel"
    );

    let regions = value_counts(features, 1);
    assert_eq!(regions[&1], 77, "77 clients should sit in Lisbon");
    assert_eq!(regions[&2], 47, "47 clients should sit in Oporto");
    assert_eq!(regions[&3], 316, "316 clients should sit in another region");

    // Each spending column sums to its published total.
    for (offset, expected) in SPENDING_SUMS.iter().enumerate() {
        let col = offset + 2;
        let total: f64 = features.column(col).iter().sum();
        assert_eq!(
            total,
            *expected,
            "column {} ({}) should sum to {}",
            col,
            WholesaleCustomers::COLUMN_NAMES[col],
            expected
        );
    }

    // The first and the last record of the source, pinned value by value.
    assert_eq!(
        features.row(0).to_vec(),
        vec![2.0, 3.0, 12669.0, 9656.0, 7561.0, 214.0, 2674.0, 1338.0]
    );
    assert_eq!(
        features.row(N_SAMPLES - 1).to_vec(),
        vec![1.0, 3.0, 2787.0, 1698.0, 2510.0, 65.0, 477.0, 52.0]
    );
}

#[test]
// Verifies that COLUMN_NAMES names the 8 columns in the source's order.
fn test_wholesale_customers_column_names() {
    assert_eq!(WholesaleCustomers::COLUMN_NAMES.len(), N_FEATURES);
    assert_eq!(
        WholesaleCustomers::COLUMN_NAMES,
        [
            "Channel",
            "Region",
            "Fresh",
            "Milk",
            "Grocery",
            "Frozen",
            "Detergents_Paper",
            "Delicassen",
        ]
    );
}

#[test]
// Verifies that the Wholesale Customers dataset loads with the correct shape,
// code domains, and pinned totals.
fn test_load_wholesale_customers() {
    let download_dir = "./test_load_wholesale_customers"; // the loader creates this directory if it is missing

    let dataset = WholesaleCustomers::new(download_dir);
    let features = dataset.features().unwrap();

    assert_wholesale_semantics(features);

    // This dataset has no target, so data() returns the same matrix.
    let data = dataset.data().unwrap();
    assert_eq!(data, features);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that n_samples() counts the leading axis of a single-array dataset.
fn test_wholesale_customers_n_samples() {
    let download_dir = "./test_wholesale_customers_n_samples";

    let dataset = WholesaleCustomers::new(download_dir);
    assert_eq!(WholesaleCustomers::NAME, "wholesale_customers");
    assert!(!dataset.is_loaded());
    assert_eq!(dataset.n_samples().unwrap(), N_SAMPLES);
    assert!(dataset.is_loaded());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Wholesale Customers reuses a cached file instead of a new download.
fn test_wholesale_customers_no_need_download() {
    let download_dir = "./test_wholesale_customers_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load primes the cache. The second instance then reuses it.
    WholesaleCustomers::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join(WHOLESALE_FILENAME),
            WHOLESALE_SHA256
        )
        .unwrap(),
        "the cached file should match the expected SHA256"
    );

    let dataset = WholesaleCustomers::new(download_dir);
    assert_eq!(dataset.features().unwrap().nrows(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake data file and overwrites it
// with the real dataset.
fn test_wholesale_customers_overwrite() {
    let download_dir = "./test_wholesale_customers_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let mut fake = File::create(download_dir_path.join(WHOLESALE_FILENAME)).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = WholesaleCustomers::new(download_dir);
    assert_eq!(dataset.features().unwrap().nrows(), N_SAMPLES);

    assert!(
        file_sha256_matches(
            &download_dir_path.join(WHOLESALE_FILENAME),
            WHOLESALE_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned matrix and consumes the dataset.
fn test_wholesale_customers_into_data() {
    let download_dir = "./test_wholesale_customers_into_data";

    let dataset = WholesaleCustomers::new(download_dir);
    let mut features = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The matrix is now fully owned.

    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);

    // The caller can mutate the owned data directly, with no `to_owned()` clone.
    features[[0, 2]] = 1.0;
    assert_eq!(features[[0, 2]], 1.0);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned matrix and leaves the instance reusable.
fn test_wholesale_customers_take_data() {
    let download_dir = "./test_wholesale_customers_take_data";

    let mut dataset = WholesaleCustomers::new(download_dir);
    let features = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file and yields the same shape.
    assert!(!dataset.is_loaded());
    assert_eq!(dataset.data().unwrap().shape(), &[N_SAMPLES, N_FEATURES]);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_wholesale_customers_get_data() {
    let download_dir = "./test_wholesale_customers_get_data";

    let dataset = WholesaleCustomers::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    dataset.data().unwrap();
    assert_eq!(
        dataset.get_data().unwrap().shape(),
        &[N_SAMPLES, N_FEATURES]
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached matrix in place.
fn test_wholesale_customers_get_data_mut() {
    let download_dir = "./test_wholesale_customers_get_data_mut";

    let mut dataset = WholesaleCustomers::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() takes the logarithm of the first client's `Fresh` spending
    // in place, with no clone and no reload.
    dataset.data().unwrap();
    if let Some(features) = dataset.get_data_mut() {
        features[[0, 2]] = features[[0, 2]].ln();
    }

    // The change persisted in the cache: a later access observes it.
    let features = dataset.data().unwrap();
    assert_eq!(features[[0, 2]], 12669f64.ln());
    assert_eq!(
        features[[1, 2]],
        7057.0,
        "the second row should stay untouched"
    );

    remove_dir_all(download_dir).unwrap();
}
