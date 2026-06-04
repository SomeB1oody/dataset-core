mod common;

use common::file_sha256_matches;
use dataset_core::utils::download_to;
use dataset_ml::palmer_penguins::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// URL and SHA-256 mirror the constants in `src/palmer_penguins.rs`.
const PENGUINS_URL: &str =
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv";
const PENGUINS_SHA256: &str = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93";

#[test]
// Verifies that the Palmer Penguins dataset loads with the correct feature shapes and label count.
fn test_load_palmer_penguins() {
    let download_dir = "./test_load_palmer_penguins"; // the code will create the directory if it doesn't exist

    let dataset = PalmerPenguins::new(download_dir);
    let (string_features, numeric_features) = dataset.features().unwrap();
    let labels = dataset.labels().unwrap();

    assert_eq!(string_features.shape(), &[344, 2]);
    assert_eq!(numeric_features.shape(), &[344, 5]);
    assert_eq!(labels.len(), 344);

    let (string_features, numeric_features, labels) = dataset.data().unwrap();

    // Semantic assertions: labels must be one of the three known species, and
    // all three species must be present.
    let mut has_adelie = false;
    let mut has_chinstrap = false;
    let mut has_gentoo = false;
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label == "Adelie" || label == "Chinstrap" || label == "Gentoo",
            "labels[{}] = {:?} is not a known species",
            i,
            label
        );
        match label {
            "Adelie" => has_adelie = true,
            "Chinstrap" => has_chinstrap = true,
            "Gentoo" => has_gentoo = true,
            _ => {}
        }
    }
    assert!(has_adelie, "labels must contain at least one Adelie");
    assert!(has_chinstrap, "labels must contain at least one Chinstrap");
    assert!(has_gentoo, "labels must contain at least one Gentoo");

    // Semantic assertions: string features are either a known category or empty
    // (empty marks a missing value). `island` is the first string column.
    for row in 0..string_features.nrows() {
        let island = &string_features[[row, 0]];
        assert!(
            island.is_empty() || island == "Biscoe" || island == "Dream" || island == "Torgersen",
            "island[{}] = {:?} is not a known island",
            row,
            island
        );
        let sex = &string_features[[row, 1]];
        assert!(
            sex.is_empty() || sex == "male" || sex == "female",
            "sex[{}] = {:?} is not a known value",
            row,
            sex
        );
    }

    // Semantic assertions: numeric features are either finite-and-positive or
    // NaN (the source's missing-value marker). The dataset is known to contain a
    // few missing measurements, so at least one NaN is expected.
    let mut saw_nan = false;
    for row in 0..numeric_features.nrows() {
        for col in 0..numeric_features.ncols() {
            let val = numeric_features[[row, col]];
            if val.is_nan() {
                saw_nan = true;
            } else {
                assert!(
                    val > 0.0,
                    "numeric[{}, {}] = {} is neither NaN nor a positive value",
                    row,
                    col,
                    val
                );
            }
        }
    }
    assert!(
        saw_nan,
        "expected at least one NaN from the dataset's missing values"
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that Palmer Penguins loading uses a pre-downloaded cached file without re-downloading.
fn test_palmer_penguins_no_need_download() {
    let download_dir = "./test_palmer_penguins_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // download Palmer Penguins dataset in advance
    download_to(PENGUINS_URL, download_dir_path, None).unwrap();

    // should use cached Palmer Penguins dataset
    let dataset = PalmerPenguins::new(download_dir);
    let _ = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake Palmer Penguins data file is detected and overwritten with the real dataset.
fn test_palmer_penguins_overwrite() {
    let download_dir = "./test_palmer_penguins_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake Palmer Penguins dataset in advance
    {
        let path = download_dir_path.join("penguins.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake Palmer Penguins dataset
    let dataset = PalmerPenguins::new(download_dir);
    let _ = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(file_sha256_matches(&download_dir_path.join("penguins.csv"), PENGUINS_SHA256).unwrap());

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned features and labels, consuming the dataset.
fn test_palmer_penguins_into_data() {
    let download_dir = "./test_palmer_penguins_into_data";

    let dataset = PalmerPenguins::new(download_dir);
    let (strings, mut numerics, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; the arrays are fully owned.

    assert_eq!(strings.shape(), &[344, 2]);
    assert_eq!(numerics.shape(), &[344, 5]);
    assert_eq!(labels.len(), 344);

    // Owned labels are correct: one of the three known species.
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label == "Adelie" || label == "Chinstrap" || label == "Gentoo",
            "labels[{}] = {:?} is not a known species",
            i,
            label
        );
    }

    // Owned data can be mutated directly, with no `to_owned()` clone.
    numerics[[0, 0]] = 40.0;
    assert_eq!(numerics[[0, 0]], 40.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_palmer_penguins_take_data() {
    let download_dir = "./test_palmer_penguins_take_data";

    let mut dataset = PalmerPenguins::new(download_dir);
    let (strings, numerics, labels) = dataset.take_data().unwrap();

    assert_eq!(strings.shape(), &[344, 2]);
    assert_eq!(numerics.shape(), &[344, 5]);
    assert_eq!(labels.len(), 344);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (r_strings, r_numerics, r_labels) = dataset.data().unwrap();
    assert_eq!(r_strings.shape(), &[344, 2]);
    assert_eq!(r_numerics.shape(), &[344, 5]);
    assert_eq!(r_labels.len(), 344);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_palmer_penguins_get_data() {
    let download_dir = "./test_palmer_penguins_get_data";

    let dataset = PalmerPenguins::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (strings, numerics, labels) = dataset.get_data().unwrap();
    assert_eq!(strings.shape(), &[344, 2]);
    assert_eq!(numerics.shape(), &[344, 5]);
    assert_eq!(labels.len(), 344);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_palmer_penguins_get_data_mut() {
    let download_dir = "./test_palmer_penguins_get_data_mut";

    let mut dataset = PalmerPenguins::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load, then mutate the cached numeric features in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((_strings, numerics, _labels)) = dataset.get_data_mut() {
        numerics[[0, 0]] = 99.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (_strings, numerics, _labels) = dataset.data().unwrap();
    assert_eq!(numerics[[0, 0]], 99.0);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
