mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::diabetes::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// The pinned source URL and SHA-256 of the Diabetes dataset, mirrored here so the
/// cache-reuse and overwrite tests stay in sync with the loader.
const DIABETES_URL: &str = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt";
const DIABETES_FILENAME: &str = "diabetes.tab";
const DIABETES_SHA256: &str = "4733febee697862c22139cdac87478a300ce0d101593deb07ed6c0f3328a99cd";

/// Assert the defining properties of scikit-learn's standardized diabetes
/// features: every value is finite, each column has mean ~0, and each column's
/// sum of squares is ~1.
fn assert_features_standardized(features: &ndarray::Array2<f64>) {
    assert_eq!(features.shape(), &[442, 10]);

    // All feature values must be finite (no NaN or Inf).
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

    // Defining property of sklearn's scaling: per column, mean ~ 0 and the sum of
    // squares totals 1.
    let n = features.nrows() as f64;
    for j in 0..features.ncols() {
        let col = features.column(j);
        let mean = col.sum() / n;
        let sum_sq: f64 = col.iter().map(|&x| x * x).sum();
        assert!(
            mean.abs() < 1e-6,
            "column {} mean = {} is not ~0 (features must be mean-centered)",
            j,
            mean
        );
        assert!(
            (sum_sq - 1.0).abs() < 1e-6,
            "column {} sum-of-squares = {} is not ~1 (features must be L2-normalized)",
            j,
            sum_sq
        );
    }
}

/// Assert the regression targets are the unscaled disease-progression values:
/// finite and within the known range 25..=346.
fn assert_targets_in_range(targets: &ndarray::Array1<f64>) {
    assert_eq!(targets.len(), 442);
    for i in 0..targets.len() {
        let val = targets[i];
        assert!(val.is_finite(), "targets[{}] = {} is not finite", i, val);
        assert!(
            (25.0..=346.0).contains(&val),
            "targets[{}] = {} is outside the expected range [25, 346]",
            i,
            val
        );
    }
}

#[test]
// Verifies that the Diabetes dataset loads with the correct shape, standardized
// features, and unscaled regression targets.
fn test_load_diabetes() {
    let download_dir = "./test_load_diabetes"; // the code will create the directory if it doesn't exist

    let dataset = Diabetes::new(download_dir);
    let features = dataset.features().unwrap();
    let targets = dataset.targets().unwrap();

    // Accessor consistency: data() returns the same arrays as features() and targets()
    assert_eq!(features.shape(), &[442, 10]);
    assert_eq!(targets.len(), 442);

    let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets

    // Standardized features (sklearn load_diabetes default) and unscaled targets.
    assert_features_standardized(features);
    assert_targets_in_range(targets);

    // Pin against scikit-learn's published reference: the first standardized `age`
    // value and the first (unscaled) target.
    assert!(
        (features[[0, 0]] - 0.0380759).abs() < 1e-4,
        "features[0, 0] = {} does not match sklearn's reference 0.0380759",
        features[[0, 0]]
    );
    assert_eq!(
        targets[0], 151.0,
        "targets[0] should be the unscaled Y = 151"
    );

    // Targets are continuous regression values, not a binary label: there must be
    // more than two distinct values.
    let mut distinct: Vec<f64> = targets.to_vec();
    distinct.sort_by(|a, b| a.total_cmp(b));
    distinct.dedup();
    assert!(
        distinct.len() > 2,
        "targets should be continuous (got {} distinct values)",
        distinct.len()
    );

    // You can use `.to_owned()` to get an owned, mutable copy of the data.
    let mut features_owned = features.to_owned();
    let mut targets_owned = targets.to_owned();
    features_owned[[0, 0]] = 0.05;
    targets_owned[0] = 200.0;

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Diabetes loading uses a pre-downloaded cached file without re-downloading.
fn test_diabetes_no_need_download() {
    let download_dir = "./test_diabetes_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Diabetes dataset in advance, under the loader's expected filename
    {
        download_to(DIABETES_URL, download_dir_path, Some(DIABETES_FILENAME)).unwrap();
    }

    // should use cached Diabetes dataset
    let dataset = Diabetes::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Diabetes data file is detected and overwritten with the real dataset.
fn test_diabetes_overwrite() {
    let download_dir = "./test_diabetes_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Diabetes dataset in advance
    {
        let diabetes_path = download_dir_path.join(DIABETES_FILENAME);
        let mut fake_diabetes = File::create(diabetes_path).unwrap();
        fake_diabetes.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Diabetes dataset
    let dataset = Diabetes::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(&download_dir_path.join(DIABETES_FILENAME), DIABETES_SHA256).unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and targets, consuming the dataset.
fn test_diabetes_into_data() {
    let download_dir = "./test_diabetes_into_data";

    let dataset = Diabetes::new(download_dir);
    let (mut features, targets) = dataset.into_data().unwrap();
    // `dataset` has been consumed; `features`/`targets` are fully owned.

    assert_features_standardized(&features);
    assert_targets_in_range(&targets);

    // Owned data can be mutated directly, with no `to_owned()` clone.
    features[[0, 0]] = 0.05;
    assert_eq!(features[[0, 0]], 0.05);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_diabetes_take_data() {
    let download_dir = "./test_diabetes_take_data";

    let mut dataset = Diabetes::new(download_dir);
    let (features, targets) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[442, 10]);
    assert_eq!(targets.len(), 442);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_features, reloaded_targets) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[442, 10]);
    assert_eq!(reloaded_targets.len(), 442);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_diabetes_get_data() {
    let download_dir = "./test_diabetes_get_data";

    let dataset = Diabetes::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (features, targets) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[442, 10]);
    assert_eq!(targets.len(), 442);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_diabetes_get_data_mut() {
    let download_dir = "./test_diabetes_get_data_mut";

    let mut dataset = Diabetes::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load, then mutate the cached features in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((features, _targets)) = dataset.get_data_mut() {
        features[[0, 0]] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (features, _targets) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 99.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
