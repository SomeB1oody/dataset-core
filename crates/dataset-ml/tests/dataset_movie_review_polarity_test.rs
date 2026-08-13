#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::movie_review_polarity::*;
use dataset_ml::table::{ColumnData, Table};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Movie Review Polarity archive (`review_polarity.tar.gz`).
const MOVIE_REVIEW_POLARITY_SHA256: &str =
    "fc0dccc2671af5db3c5d8f81f77a1ebfec953ecdd422334062df61ede36b2179";

/// The Movie Review Polarity dataset has this many samples.
const N_SAMPLES: usize = 2_000;

/// The two column names, in source order.
const COLUMN_NAMES: [&str; 2] = ["text", "label"];

/// Assert the column layout the documentation states: two columns, one text
/// feature and one string target.
fn assert_movie_review_polarity_schema(table: &Table) {
    // Every name in the loader's constants reaches a real column.
    for name in MovieReviewPolarity::FEATURE_NAMES {
        assert!(
            table.column(name).is_some(),
            "FEATURE_NAMES entry `{name}` names no column"
        );
    }
    assert!(
        table.column(MovieReviewPolarity::TARGET).is_some(),
        "TARGET `{}` names no column",
        MovieReviewPolarity::TARGET
    );
    assert!(
        !MovieReviewPolarity::FEATURE_NAMES.contains(&MovieReviewPolarity::TARGET),
        "the target must not also be a feature"
    );
    assert_eq!(table.n_columns(), 2);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    assert_eq!(MovieReviewPolarity::FEATURE_NAMES, ["text"]);
    assert_eq!(MovieReviewPolarity::TARGET, "label");

    let text = table.column(MovieReviewPolarity::FEATURE_NAMES[0]).unwrap();
    assert!(matches!(text.data(), ColumnData::String(_)));

    let label = table.column(MovieReviewPolarity::TARGET).unwrap();
    assert!(matches!(label.data(), ColumnData::String(_)));
}

/// Checks the Movie Review Polarity invariants: the column layout, the sample
/// count, the two label classes with their exact balanced counts, and non-empty
/// reviews. It also checks the pinned first document, which the deterministic
/// neg-then-pos lexicographic walk fixes.
fn assert_movie_review_polarity_semantics(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_movie_review_polarity_schema(table);

    let texts = table
        .column(MovieReviewPolarity::FEATURE_NAMES[0])
        .unwrap()
        .as_string()
        .unwrap();
    let labels = table
        .column(MovieReviewPolarity::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    let mut positive = 0usize;
    let mut negative = 0usize;
    for (i, label) in labels.iter().enumerate() {
        match label.as_str() {
            "positive" => positive += 1,
            "negative" => negative += 1,
            other => panic!("labels[{i}] = {other:?} is not `positive` or `negative`"),
        }
    }
    assert_eq!(positive, 1000, "expected 1,000 positive reviews");
    assert_eq!(negative, 1000, "expected 1,000 negative reviews");

    for (i, text) in texts.iter().enumerate() {
        assert!(!text.is_empty(), "texts[{i}] should not be empty");
    }

    // The deterministic walk (neg folder first, then files in lexicographic order)
    // fixes the first record: neg / file "cv000_29416.txt".
    assert_eq!(labels[0], "negative");
    assert!(
        texts[0].starts_with("plot : two teen couples go to a church party"),
        "texts[0] = {:?} does not match the known first review",
        &texts[0][..texts[0].len().min(60)]
    );
}

#[test]
// Verifies that the Movie Review Polarity dataset loads with the correct column
// layout, sample count, label classes, and non-empty reviews.
fn test_load_movie_review_polarity() {
    let download_dir = "./test_load_movie_review_polarity"; // the loader creates the directory if it does not exist

    let dataset = MovieReviewPolarity::new(download_dir);
    assert_movie_review_polarity_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that loading uses a pre-existing cached archive without re-downloading.
fn test_movie_review_polarity_no_need_download() {
    let download_dir = "./test_movie_review_polarity_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    MovieReviewPolarity::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join("review_polarity.tar.gz"),
            MOVIE_REVIEW_POLARITY_SHA256
        )
        .unwrap(),
        "cached review_polarity.tar.gz should match the expected SHA256"
    );

    let dataset = MovieReviewPolarity::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake archive and overwrites it with the real one.
fn test_movie_review_polarity_overwrite() {
    let download_dir = "./test_movie_review_polarity_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("review_polarity.tar.gz");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = MovieReviewPolarity::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(
            &download_dir_path.join("review_polarity.tar.gz"),
            MOVIE_REVIEW_POLARITY_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_movie_review_polarity_into_data() {
    let download_dir = "./test_movie_review_polarity_into_data";

    let dataset = MovieReviewPolarity::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumes `dataset`. The returned table is fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_movie_review_polarity_schema(&table);

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::String(values)) = table.column_mut("text").map(|c| c.data_mut()) {
        values[0] = "cleaned text".to_string();
    }
    assert_eq!(
        table.column("text").unwrap().as_string().unwrap()[0],
        "cleaned text"
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the dataset reusable.
fn test_movie_review_polarity_take_data() {
    let download_dir = "./test_movie_review_polarity_take_data";

    let mut dataset = MovieReviewPolarity::new(download_dir);
    let table = dataset.take_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);

    // take_data() resets the instance to unloaded, but it stays usable. The next
    // access reloads it (from the cached archive) and yields the same shape.
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_movie_review_polarity_get_data() {
    let download_dir = "./test_movie_review_polarity_get_data";

    let dataset = MovieReviewPolarity::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached reference.
    dataset.data().unwrap();
    assert_eq!(dataset.get_data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place and the change persists.
fn test_movie_review_polarity_get_data_mut() {
    let download_dir = "./test_movie_review_polarity_get_data_mut";

    let mut dataset = MovieReviewPolarity::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // The mutation happens in place. It needs no clone and no reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::String(values)) = table.column_mut("text").map(|c| c.data_mut())
    {
        values[0] = "normalized".to_string();
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table.column("text").unwrap().as_string().unwrap()[0],
        "normalized"
    );
    assert_eq!(
        table.column("label").unwrap().as_string().unwrap()[0],
        "negative",
        "the other columns should stay untouched"
    );

    remove_dir_all(download_dir).unwrap();
}
