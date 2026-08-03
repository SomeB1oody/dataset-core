mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::dataset::california_housing::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/california_housing.rs`.
const CALIFORNIA_HOUSING_URL: &str =
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv";
const CALIFORNIA_HOUSING_SHA256: &str =
    "8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e";

/// Column index of the derived `AveBedrms` feature, the only one that can be
/// `NaN` (the source has 207 missing `total_bedrooms` values).
const AVE_BEDRMS_COL: usize = 3;

#[test]
fn test_load_california_housing() {
    let download_dir = "./test_load_california_housing"; // the code creates the directory if it does not exist

    let dataset = CaliforniaHousing::new(download_dir);
    let features = dataset.features().unwrap();
    let targets = dataset.targets().unwrap();

    assert_eq!(features.shape(), &[20640, 8]);
    assert_eq!(targets.len(), 20640);

    let (features, targets) = dataset.data().unwrap(); // data() is another way to get features and targets

    // Semantic assertions: targets are median house values in units of $100,000.
    // Because of this unit, targets fall in about the [0.15, 5.0] range. All
    // targets must be finite.
    for i in 0..targets.len() {
        let val = targets[i];
        assert!(
            val.is_finite() && (0.1..=5.1).contains(&val),
            "target[{}] = {} is out of the expected $100k range",
            i,
            val
        );
    }

    // Semantic assertions: every feature is finite, except `AveBedrms`. `AveBedrms`
    // is NaN for the 207 rows with a missing `total_bedrooms` value. No other
    // column may contain NaN. At least one NaN must appear in `AveBedrms`. This
    // confirms the missing-value handling works.
    let mut saw_ave_bedrms_nan = false;
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            let val = features[[row, col]];
            if col == AVE_BEDRMS_COL && val.is_nan() {
                saw_ave_bedrms_nan = true;
                continue;
            }
            assert!(
                val.is_finite(),
                "feature[{}, {}] = {} is not finite (only AveBedrms may be NaN)",
                row,
                col,
                val
            );
        }
    }
    assert!(
        saw_ave_bedrms_nan,
        "expected at least one NaN in AveBedrms from the missing total_bedrooms values"
    );

    // Check the geographic columns: Latitude (column 6) and Longitude (column 7)
    // must fall within California's bounding box.
    for row in 0..features.nrows() {
        let latitude = features[[row, 6]];
        let longitude = features[[row, 7]];
        assert!(
            (32.0..=43.0).contains(&latitude),
            "Latitude[{}] = {} is outside California",
            row,
            latitude
        );
        assert!(
            (-125.0..=-113.0).contains(&longitude),
            "Longitude[{}] = {} is outside California",
            row,
            longitude
        );
    }

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_california_housing_no_need_download() {
    let download_dir = "./test_california_housing_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download the dataset before the test, using the file name the loader expects
    download_to(
        CALIFORNIA_HOUSING_URL,
        download_dir_path,
        Some("california_housing.csv"),
    )
    .unwrap();

    // this call uses the cached dataset instead of downloading it again
    let dataset = CaliforniaHousing::new(download_dir);
    let _ = dataset.data().unwrap();

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_california_housing_overwrite() {
    let download_dir = "./test_california_housing_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("california_housing.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // this call replaces the fake file with the real dataset
    let dataset = CaliforniaHousing::new(download_dir);
    let _ = dataset.data().unwrap();

    assert!(
        file_sha256_matches(
            &download_dir_path.join("california_housing.csv"),
            CALIFORNIA_HOUSING_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_california_housing_into_data() {
    let download_dir = "./test_california_housing_into_data";

    let dataset = CaliforniaHousing::new(download_dir);
    let (mut features, targets) = dataset.into_data().unwrap();
    // `into_data` consumes `dataset`. `features` and `targets` are now fully owned.

    assert_eq!(features.shape(), &[20640, 8]);
    assert_eq!(targets.len(), 20640);

    // Owned targets are correct: finite values in the $100k range.
    for i in 0..targets.len() {
        let val = targets[i];
        assert!(
            val.is_finite() && (0.1..=5.1).contains(&val),
            "target[{}] = {} is out of the expected $100k range",
            i,
            val
        );
    }

    // Owned data can be mutated directly, with no `to_owned()` clone.
    features[[0, 0]] = 5.0;
    assert_eq!(features[[0, 0]], 5.0);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_california_housing_take_data() {
    let download_dir = "./test_california_housing_take_data";

    let mut dataset = CaliforniaHousing::new(download_dir);
    let (features, targets) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[20640, 8]);
    assert_eq!(targets.len(), 20640);

    // After `take_data`, the instance resets to unloaded, but remains usable. The
    // next access reloads the data from the cached file and returns the same shapes.
    let (reloaded_features, reloaded_targets) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[20640, 8]);
    assert_eq!(reloaded_targets.len(), 20640);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_california_housing_get_data() {
    let download_dir = "./test_california_housing_get_data";

    let dataset = CaliforniaHousing::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, `get_data` returns the cached references.
    dataset.data().unwrap();
    let (features, targets) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[20640, 8]);
    assert_eq!(targets.len(), 20640);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_california_housing_get_data_mut() {
    let download_dir = "./test_california_housing_get_data_mut";

    let mut dataset = CaliforniaHousing::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // This loads the dataset, then mutates the cached features in place. No clone
    // or reload occurs.
    dataset.data().unwrap();
    if let Some((features, _targets)) = dataset.get_data_mut() {
        features[[0, 0]] = 99.0;
    }

    // The change persists in the cache. A later access observes it.
    let (features, _targets) = dataset.data().unwrap();
    assert_eq!(features[[0, 0]], 99.0);

    remove_dir_all(download_dir).unwrap();
}
