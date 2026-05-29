use dataset_core::utils::{download_to, file_sha256_matches};
use dataset_ml::wine_quality::white_wine_quality::WhiteWineQuality;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

#[test]
// Verifies that the White Wine Quality dataset loads with the correct feature shape and target count.
fn test_load_white_wine_quality() {
    let download_dir = "./test_load_white_wine_quality"; // the code will create this directory if it doesn't exist

    let dataset = WhiteWineQuality::new(download_dir);
    let features = dataset.features().unwrap();
    let targets = dataset.targets().unwrap();

    // Accessor consistency: data() returns the same arrays as features() and targets()
    assert_eq!(features.shape(), &[4898, 11]);
    assert_eq!(targets.len(), 4898);

    let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
    // you can use `.to_owned()` to get owned copies of the data
    let mut features_owned = features.to_owned();
    let mut targets_owned = targets.to_owned();

    // Semantic assertions: all feature and target values must be finite (no NaN or Inf)
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
    for i in 0..targets.len() {
        let val = targets[i];
        assert!(val.is_finite(), "target[{}] = {} is not finite", i, val);
    }

    // Semantic assertions: quality scores must be integer-valued and within the valid range [0, 10]
    let mut unique_qualities = HashSet::new();
    for i in 0..targets.len() {
        let val = targets[i];
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
    // The actual dataset contains multiple quality scores; verify we have more than one
    assert!(
        unique_qualities.len() > 1,
        "targets should contain more than one unique quality score"
    );

    // Example: Modify feature values
    features_owned[[0, 0]] = 10.0;
    targets_owned[0] = 7.0;

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that White Wine Quality loading uses a pre-downloaded cached file without re-downloading.
fn test_white_wine_quality_no_need_download() {
    let download_dir = "./test_white_wine_quality_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download dataset in advance
    download_to(
        "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-white.csv",
        download_dir_path,
        None,
    )
    .unwrap();

    // should use cached dataset
    let dataset = WhiteWineQuality::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake White Wine Quality data file is detected and overwritten with the real dataset.
fn test_white_wine_quality_overwrite() {
    let download_dir = "./test_white_wine_quality_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // create fake dataset
    {
        let fake_white_wine_dataset_path = download_dir_path.join("winequality-white.csv");
        let mut fake_white_wine = File::create(fake_white_wine_dataset_path).unwrap();
        fake_white_wine.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake dataset
    let dataset = WhiteWineQuality::new(download_dir);
    let (_features, _targets) = dataset.data().unwrap();

    // check that the downloaded file is correct
    assert!(
        file_sha256_matches(
            &download_dir_path.join("winequality-white.csv"),
            "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836"
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and targets, consuming the dataset.
fn test_white_wine_quality_into_data() {
    let download_dir = "./test_white_wine_quality_into_data";

    let dataset = WhiteWineQuality::new(download_dir);
    let (mut features, targets) = dataset.into_data().unwrap();
    // `dataset` has been consumed; `features`/`targets` are fully owned.

    assert_eq!(features.shape(), &[4898, 11]);
    assert_eq!(targets.len(), 4898);

    // Owned targets are correct: quality scores within [0, 10].
    for i in 0..targets.len() {
        let val = targets[i];
        assert!(
            (0.0..=10.0).contains(&val),
            "target[{}] = {} is outside the valid quality range [0, 10]",
            i,
            val
        );
    }

    // Owned data can be mutated directly, with no `to_owned()` clone.
    features[[0, 0]] = 10.0;
    assert_eq!(features[[0, 0]], 10.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_white_wine_quality_take_data() {
    let download_dir = "./test_white_wine_quality_take_data";

    let mut dataset = WhiteWineQuality::new(download_dir);
    let (features, targets) = dataset.take_data().unwrap();

    assert_eq!(features.shape(), &[4898, 11]);
    assert_eq!(targets.len(), 4898);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_features, reloaded_targets) = dataset.data().unwrap();
    assert_eq!(reloaded_features.shape(), &[4898, 11]);
    assert_eq!(reloaded_targets.len(), 4898);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_white_wine_quality_get_data() {
    let download_dir = "./test_white_wine_quality_get_data";

    let dataset = WhiteWineQuality::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (features, targets) = dataset.get_data().unwrap();
    assert_eq!(features.shape(), &[4898, 11]);
    assert_eq!(targets.len(), 4898);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_white_wine_quality_get_data_mut() {
    let download_dir = "./test_white_wine_quality_get_data_mut";

    let mut dataset = WhiteWineQuality::new(download_dir);
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
