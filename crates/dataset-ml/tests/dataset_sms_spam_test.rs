mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::sms_spam::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached SMS Spam dataset file (`sms_spam.csv`).
const SMS_SPAM_SHA256: &str = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d";

/// The SMS Spam dataset has this many samples.
const N_SAMPLES: usize = 5_574;

/// Assert the SMS Spam dataset invariants: the sample count, the two label
/// classes with their exact counts, and non-empty message texts.
fn assert_sms_spam_semantics(
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
    assert_eq!(ham, 4827, "expected 4,827 ham messages");
    assert_eq!(spam, 747, "expected 747 spam messages");

    // Every message body is non-empty.
    for (i, text) in texts.iter().enumerate() {
        assert!(!text.is_empty(), "texts[{i}] should not be empty");
    }

    // The first record is a known ham message (the dataset ordering is fixed by
    // the pinned SHA-256).
    assert_eq!(labels[0], "ham");
    assert!(
        texts[0].starts_with("Go until jurong point"),
        "texts[0] = {:?} does not match the known first message",
        texts[0]
    );
}

#[test]
// Verifies that the SMS Spam dataset loads with the correct sample count, label
// classes, and non-empty message texts.
fn test_load_sms_spam() {
    let download_dir = "./test_load_sms_spam"; // the loader creates this directory if it is missing

    let dataset = SmsSpam::new(download_dir);
    let (texts, labels) = dataset.data().unwrap();

    assert_sms_spam_semantics(texts, labels);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that SMS Spam reuses a cached file instead of a new download.
fn test_sms_spam_no_need_download() {
    let download_dir = "./test_sms_spam_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load downloads and extracts the ZIP file. This primes the cache.
    // The second instance then reuses the extracted file.
    SmsSpam::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(&download_dir_path.join("sms_spam.csv"), SMS_SPAM_SHA256).unwrap(),
        "cached sms_spam.csv should match the expected SHA256"
    );

    let dataset = SmsSpam::new(download_dir);
    let (_texts, _labels) = dataset.data().unwrap();

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake SMS Spam data file and
// overwrites it with the real dataset.
fn test_sms_spam_overwrite() {
    let download_dir = "./test_sms_spam_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("sms_spam.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // The loader overwrites the fake file with the real dataset.
    let dataset = SmsSpam::new(download_dir);
    let (_texts, _labels) = dataset.data().unwrap();

    assert!(file_sha256_matches(&download_dir_path.join("sms_spam.csv"), SMS_SPAM_SHA256).unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays and consumes the dataset.
fn test_sms_spam_into_data() {
    let download_dir = "./test_sms_spam_into_data";

    let dataset = SmsSpam::new(download_dir);
    let (mut texts, labels) = dataset.into_data().unwrap();
    // `into_data()` consumes `dataset`. The arrays are fully owned.

    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // The caller can mutate the owned data directly, with no `to_owned()` clone.
    texts[0] = "cleaned text".to_string();
    assert_eq!(texts[0], "cleaned text");

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_sms_spam_take_data() {
    let download_dir = "./test_sms_spam_take_data";

    let mut dataset = SmsSpam::new(download_dir);
    let (texts, labels) = dataset.take_data().unwrap();

    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // take_data() resets the instance to unloaded, but it stays usable. The next
    // access reloads it from the cached file and yields the same shapes.
    let (reloaded_texts, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_texts.len(), N_SAMPLES);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_sms_spam_get_data() {
    let download_dir = "./test_sms_spam_get_data";

    let dataset = SmsSpam::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // This loads the dataset. get_data() then returns the cached references.
    dataset.data().unwrap();
    let (texts, labels) = dataset.get_data().unwrap();
    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_sms_spam_get_data_mut() {
    let download_dir = "./test_sms_spam_get_data_mut";

    let mut dataset = SmsSpam::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // This loads the dataset. It then mutates the cached texts in place, with no
    // clone and no reload.
    dataset.data().unwrap();
    if let Some((texts, _labels)) = dataset.get_data_mut() {
        texts[0] = "normalized".to_string();
    }

    // The change persisted in the cache: a later access observes it.
    let (texts, _labels) = dataset.data().unwrap();
    assert_eq!(texts[0], "normalized");

    remove_dir_all(download_dir).unwrap();
}
