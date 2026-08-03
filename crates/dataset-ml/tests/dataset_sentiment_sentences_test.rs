mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::sentiment_sentences::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Sentiment Labelled Sentences file (`sentiment_sentences.csv`).
const SENTIMENT_SENTENCES_SHA256: &str =
    "3a6aac64fa37c8075d49678cd73140eaa70a95c984d540ddf93ec7b021e05725";

/// The Sentiment Labelled Sentences dataset has this many samples.
const N_SAMPLES: usize = 3_000;

/// Checks the Sentiment Labelled Sentences invariants. It checks the sample count, the two
/// sentiment classes, the three sources with their exact balanced counts, and the
/// non-empty sentence texts.
fn assert_sentiment_sentences_semantics(
    texts: &ndarray::Array1<String>,
    sources: &ndarray::Array1<&'static str>,
    labels: &ndarray::Array1<&'static str>,
) {
    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(sources.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // Labels are one of the two classes. Each class has exactly 1,500 sentences.
    let mut positive = 0usize;
    let mut negative = 0usize;
    for (i, &label) in labels.iter().enumerate() {
        match label {
            "positive" => positive += 1,
            "negative" => negative += 1,
            other => panic!("labels[{i}] = {other:?} is not `positive` or `negative`"),
        }
    }
    assert_eq!(positive, 1500, "expected 1,500 positive sentences");
    assert_eq!(negative, 1500, "expected 1,500 negative sentences");

    // Sources are one of the three sites. Each site has 1,000 sentences.
    let mut amazon = 0usize;
    let mut imdb = 0usize;
    let mut yelp = 0usize;
    for (i, &source) in sources.iter().enumerate() {
        match source {
            "amazon" => amazon += 1,
            "imdb" => imdb += 1,
            "yelp" => yelp += 1,
            other => panic!("sources[{i}] = {other:?} is not `amazon`, `imdb`, or `yelp`"),
        }
    }
    assert_eq!(amazon, 1000, "expected 1,000 amazon sentences");
    assert_eq!(imdb, 1000, "expected 1,000 imdb sentences");
    assert_eq!(yelp, 1000, "expected 1,000 yelp sentences");

    // Every sentence is non-empty.
    for (i, text) in texts.iter().enumerate() {
        assert!(!text.is_empty(), "texts[{i}] should not be empty");
    }

    // The first record is a known negative Amazon sentence. The loader combines the
    // three per-site files with Amazon first. This order matches the pinned SHA-256.
    assert_eq!(sources[0], "amazon");
    assert_eq!(labels[0], "negative");
    assert!(
        texts[0].starts_with("So there is no way for me to plug it in"),
        "texts[0] = {:?} does not match the known first sentence",
        texts[0]
    );
}

#[test]
// Verifies that the Sentiment Labelled Sentences dataset loads with the correct
// sample count, sentiment classes, sources, and non-empty texts.
fn test_load_sentiment_sentences() {
    let download_dir = "./test_load_sentiment_sentences"; // if the directory does not exist, the code creates it

    let dataset = SentimentSentences::new(download_dir);
    let (texts, sources, labels) = dataset.data().unwrap();

    assert_sentiment_sentences_semantics(texts, sources, labels);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that loading reuses an existing cached file instead of downloading it again.
fn test_sentiment_sentences_no_need_download() {
    let download_dir = "./test_sentiment_sentences_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Load once to prime the cache. This downloads, extracts, and combines the three
    // per-site files. Then confirm that a second instance reuses the combined file.
    SentimentSentences::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join("sentiment_sentences.csv"),
            SENTIMENT_SENTENCES_SHA256
        )
        .unwrap(),
        "cached sentiment_sentences.csv should match the expected SHA256"
    );

    let dataset = SentimentSentences::new(download_dir);
    let (_texts, _sources, _labels) = dataset.data().unwrap();

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake data file and overwrites it with the real dataset.
fn test_sentiment_sentences_overwrite() {
    let download_dir = "./test_sentiment_sentences_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("sentiment_sentences.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // this call overwrites the fake dataset with the real data
    let dataset = SentimentSentences::new(download_dir);
    let (_texts, _sources, _labels) = dataset.data().unwrap();

    assert!(
        file_sha256_matches(
            &download_dir_path.join("sentiment_sentences.csv"),
            SENTIMENT_SENTENCES_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays and consumes the dataset.
fn test_sentiment_sentences_into_data() {
    let download_dir = "./test_sentiment_sentences_into_data";

    let dataset = SentimentSentences::new(download_dir);
    let (mut texts, sources, labels) = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The arrays are now fully owned.

    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(sources.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // Owned data allows direct mutation and needs no `to_owned()` clone.
    texts[0] = "cleaned text".to_string();
    assert_eq!(texts[0], "cleaned text");

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_sentiment_sentences_take_data() {
    let download_dir = "./test_sentiment_sentences_take_data";

    let mut dataset = SentimentSentences::new(download_dir);
    let (texts, sources, labels) = dataset.take_data().unwrap();

    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(sources.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file and yields the same shapes.
    let (reloaded_texts, reloaded_sources, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_texts.len(), N_SAMPLES);
    assert_eq!(reloaded_sources.len(), N_SAMPLES);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_sentiment_sentences_get_data() {
    let download_dir = "./test_sentiment_sentences_get_data";

    let dataset = SentimentSentences::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading. get_data() then returns the cached references.
    dataset.data().unwrap();
    let (texts, sources, labels) = dataset.get_data().unwrap();
    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(sources.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_sentiment_sentences_get_data_mut() {
    let download_dir = "./test_sentiment_sentences_get_data_mut";

    let mut dataset = SentimentSentences::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load the data. Then mutate the cached texts in place, with no clone or reload.
    dataset.data().unwrap();
    if let Some((texts, _sources, _labels)) = dataset.get_data_mut() {
        texts[0] = "normalized".to_string();
    }

    // The change persisted in the cache: a later access observes it.
    let (texts, _sources, _labels) = dataset.data().unwrap();
    assert_eq!(texts[0], "normalized");

    remove_dir_all(download_dir).unwrap();
}
