#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::youtube_spam::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached YouTube Spam dataset file (`youtube_spam.csv`).
const YOUTUBE_SPAM_SHA256: &str =
    "f172e32ca7b4ecadb926df0c836dbe6c6485c519a47a5e7d7f719f2b3553906b";

/// The YouTube Spam dataset has this many samples.
const N_SAMPLES: usize = 1_956;

/// Checks the YouTube Spam dataset invariants: the sample count, the two label
/// classes with their exact counts, and non-empty comment texts.
fn assert_youtube_spam_semantics(
    texts: &ndarray::Array1<String>,
    labels: &ndarray::Array1<&'static str>,
) {
    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

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

    for (i, text) in texts.iter().enumerate() {
        assert!(!text.is_empty(), "texts[{i}] should not be empty");
    }

    // The first record is a known spam comment. The pinned SHA-256 fixes the
    // dataset order: the loader concatenates the five per-video CSVs Psy-first.
    assert_eq!(labels[0], "spam");
    assert!(
        texts[0].starts_with("Huh, anyway check out this"),
        "texts[0] = {:?} does not match the known first comment",
        texts[0]
    );
}

#[test]
fn test_load_youtube_spam() {
    let download_dir = "./test_load_youtube_spam"; // the loader creates the directory if it does not exist

    let dataset = YoutubeSpam::new(download_dir);
    let (texts, labels) = dataset.data().unwrap();

    assert_youtube_spam_semantics(texts, labels);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_youtube_spam_no_need_download() {
    let download_dir = "./test_youtube_spam_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load downloads, extracts, and concatenates the five per-video
    // CSVs into the cache. The next instance then reuses the combined file.
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

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_youtube_spam_overwrite() {
    let download_dir = "./test_youtube_spam_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("youtube_spam.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = YoutubeSpam::new(download_dir);
    let (_texts, _labels) = dataset.data().unwrap();

    assert!(
        file_sha256_matches(
            &download_dir_path.join("youtube_spam.csv"),
            YOUTUBE_SPAM_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_youtube_spam_into_data() {
    let download_dir = "./test_youtube_spam_into_data";

    let dataset = YoutubeSpam::new(download_dir);
    let (mut texts, labels) = dataset.into_data().unwrap();
    // into_data() consumes `dataset`. The returned arrays are fully owned.

    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // The caller can mutate owned data directly, with no `to_owned()` clone.
    texts[0] = "cleaned text".to_string();
    assert_eq!(texts[0], "cleaned text");

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_youtube_spam_take_data() {
    let download_dir = "./test_youtube_spam_take_data";

    let mut dataset = YoutubeSpam::new(download_dir);
    let (texts, labels) = dataset.take_data().unwrap();

    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // take_data() resets the instance to unloaded, but it stays usable. The next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_texts, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_texts.len(), N_SAMPLES);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_youtube_spam_get_data() {
    let download_dir = "./test_youtube_spam_get_data";

    let dataset = YoutubeSpam::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached references.
    dataset.data().unwrap();
    let (texts, labels) = dataset.get_data().unwrap();
    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_youtube_spam_get_data_mut() {
    let download_dir = "./test_youtube_spam_get_data_mut";

    let mut dataset = YoutubeSpam::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() mutates the cached texts in place, with no clone or reload.
    dataset.data().unwrap();
    if let Some((texts, _labels)) = dataset.get_data_mut() {
        texts[0] = "normalized".to_string();
    }

    // The change persisted in the cache: a later access observes it.
    let (texts, _labels) = dataset.data().unwrap();
    assert_eq!(texts[0], "normalized");

    remove_dir_all(download_dir).unwrap();
}
