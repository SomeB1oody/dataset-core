mod common;

use common::file_sha256_matches;
use dataset_ml::movie_review_polarity::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Movie Review Polarity archive (`review_polarity.tar.gz`).
const MOVIE_REVIEW_POLARITY_SHA256: &str =
    "fc0dccc2671af5db3c5d8f81f77a1ebfec953ecdd422334062df61ede36b2179";

/// The Movie Review Polarity dataset has this many samples.
const N_SAMPLES: usize = 2_000;

/// Assert the Movie Review Polarity invariants: the sample count, the two label
/// classes with their exact (balanced) counts, non-empty reviews, and the pinned
/// first document (fixed by the deterministic neg-then-pos lexicographic walk).
fn assert_movie_review_polarity_semantics(
    texts: &ndarray::Array1<String>,
    labels: &ndarray::Array1<&'static str>,
) {
    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // Labels are one of the two classes, perfectly balanced (1,000 each).
    let mut positive = 0usize;
    let mut negative = 0usize;
    for (i, &label) in labels.iter().enumerate() {
        match label {
            "positive" => positive += 1,
            "negative" => negative += 1,
            other => panic!("labels[{i}] = {other:?} is not `positive` or `negative`"),
        }
    }
    assert_eq!(positive, 1000, "expected 1,000 positive reviews");
    assert_eq!(negative, 1000, "expected 1,000 negative reviews");

    // Every review is non-empty.
    for (i, text) in texts.iter().enumerate() {
        assert!(!text.is_empty(), "texts[{i}] should not be empty");
    }

    // The first record is fixed by the deterministic walk (neg folder first, then
    // files in lexicographic order): neg / file "cv000_29416.txt".
    assert_eq!(labels[0], "negative");
    assert!(
        texts[0].starts_with("plot : two teen couples go to a church party"),
        "texts[0] = {:?} does not match the known first review",
        &texts[0][..texts[0].len().min(60)]
    );
}

#[test]
// Verifies that the Movie Review Polarity dataset loads with the correct sample
// count, label classes, and non-empty reviews.
fn test_load_movie_review_polarity() {
    let download_dir = "./test_load_movie_review_polarity"; // the code will create the directory if it doesn't exist

    let dataset = MovieReviewPolarity::new(download_dir);
    let (texts, labels) = dataset.data().unwrap();

    assert_movie_review_polarity_semantics(texts, labels);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that loading uses a pre-existing cached archive without re-downloading.
fn test_movie_review_polarity_no_need_download() {
    let download_dir = "./test_movie_review_polarity_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Prime the cache by loading once (downloads the archive), then confirm a
    // second instance reuses it.
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
    let (_texts, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake archive is detected and overwritten with the real one.
fn test_movie_review_polarity_overwrite() {
    let download_dir = "./test_movie_review_polarity_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake archive in advance
    {
        let path = download_dir_path.join("review_polarity.tar.gz");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake archive
    let dataset = MovieReviewPolarity::new(download_dir);
    let (_texts, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("review_polarity.tar.gz"),
            MOVIE_REVIEW_POLARITY_SHA256
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays, consuming the dataset.
fn test_movie_review_polarity_into_data() {
    let download_dir = "./test_movie_review_polarity_into_data";

    let dataset = MovieReviewPolarity::new(download_dir);
    let (mut texts, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; the arrays are fully owned.

    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // Owned data can be mutated directly, with no `to_owned()` clone.
    texts[0] = "cleaned text".to_string();
    assert_eq!(texts[0], "cleaned text");

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_movie_review_polarity_take_data() {
    let download_dir = "./test_movie_review_polarity_take_data";

    let mut dataset = MovieReviewPolarity::new(download_dir);
    let (texts, labels) = dataset.take_data().unwrap();

    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached archive) and yields the same shapes.
    let (reloaded_texts, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_texts.len(), N_SAMPLES);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_movie_review_polarity_get_data() {
    let download_dir = "./test_movie_review_polarity_get_data";

    let dataset = MovieReviewPolarity::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (texts, labels) = dataset.get_data().unwrap();
    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_movie_review_polarity_get_data_mut() {
    let download_dir = "./test_movie_review_polarity_get_data_mut";

    let mut dataset = MovieReviewPolarity::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load, then mutate the cached texts in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((texts, _labels)) = dataset.get_data_mut() {
        texts[0] = "normalized".to_string();
    }

    // The change persisted in the cache: a later access observes it.
    let (texts, _labels) = dataset.data().unwrap();
    assert_eq!(texts[0], "normalized");

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
