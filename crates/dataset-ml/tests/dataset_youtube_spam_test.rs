mod common;

use common::file_sha256_matches;
use dataset_ml::youtube_spam::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached YouTube Spam dataset file (`youtube_spam.csv`).
const YOUTUBE_SPAM_SHA256: &str =
    "f172e32ca7b4ecadb926df0c836dbe6c6485c519a47a5e7d7f719f2b3553906b";

/// The YouTube Spam dataset has this many samples.
const N_SAMPLES: usize = 1_956;

/// Assert the YouTube Spam dataset invariants: the sample count, the two label
/// classes with their exact counts, and non-empty comment texts.
fn assert_youtube_spam_semantics(
    texts: &ndarray::Array1<String>,
    labels: &ndarray::Array1<&'static str>,
) {
    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // Labels are one of the two classes, with the documented per-class counts.
    let mut ham = 0usize;
    let mut spam = 0usize;
    for (i, &label) in labels.iter().enumerate() {
        match label {
            "ham" => ham += 1,
            "spam" => spam += 1,
            other => panic!("labels[{i}] = {other:?} is not `ham` or `spam`"),
        }
    }
    assert_eq!(ham, 951, "expected 951 ham comments");
    assert_eq!(spam, 1005, "expected 1,005 spam comments");

    // Every comment body is non-empty.
    for (i, text) in texts.iter().enumerate() {
        assert!(!text.is_empty(), "texts[{i}] should not be empty");
    }

    // The first record is a known spam comment (the dataset ordering is fixed by
    // the pinned SHA-256: the five per-video CSVs concatenated Psy-first).
    assert_eq!(labels[0], "spam");
    assert!(
        texts[0].starts_with("Huh, anyway check out this"),
        "texts[0] = {:?} does not match the known first comment",
        texts[0]
    );
}

#[test]
// Verifies that the YouTube Spam dataset loads with the correct sample count,
// label classes, and non-empty comment texts.
fn test_load_youtube_spam() {
    let download_dir = "./test_load_youtube_spam"; // the code will create the directory if it doesn't exist

    let dataset = YoutubeSpam::new(download_dir);
    let (texts, labels) = dataset.data().unwrap();

    assert_youtube_spam_semantics(texts, labels);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that YouTube Spam loading uses a pre-existing cached file without re-downloading.
fn test_youtube_spam_no_need_download() {
    let download_dir = "./test_youtube_spam_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Prime the cache by loading once (downloads, extracts, and concatenates the
    // five CSVs), then confirm a second instance reuses the combined file.
    YoutubeSpam::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join("youtube_spam.csv"),
            YOUTUBE_SPAM_SHA256
        )
        .unwrap(),
        "cached youtube_spam.csv should match the expected SHA256"
    );

    let dataset = YoutubeSpam::new(download_dir);
    let (_texts, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake YouTube Spam data file is detected and overwritten with the real dataset.
fn test_youtube_spam_overwrite() {
    let download_dir = "./test_youtube_spam_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake YouTube Spam dataset in advance
    {
        let path = download_dir_path.join("youtube_spam.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake YouTube Spam dataset
    let dataset = YoutubeSpam::new(download_dir);
    let (_texts, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("youtube_spam.csv"),
            YOUTUBE_SPAM_SHA256
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays, consuming the dataset.
fn test_youtube_spam_into_data() {
    let download_dir = "./test_youtube_spam_into_data";

    let dataset = YoutubeSpam::new(download_dir);
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
fn test_youtube_spam_take_data() {
    let download_dir = "./test_youtube_spam_take_data";

    let mut dataset = YoutubeSpam::new(download_dir);
    let (texts, labels) = dataset.take_data().unwrap();

    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_texts, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_texts.len(), N_SAMPLES);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_youtube_spam_get_data() {
    let download_dir = "./test_youtube_spam_get_data";

    let dataset = YoutubeSpam::new(download_dir);
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
fn test_youtube_spam_get_data_mut() {
    let download_dir = "./test_youtube_spam_get_data_mut";

    let mut dataset = YoutubeSpam::new(download_dir);
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
