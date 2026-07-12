mod common;

use common::file_sha256_matches;
use dataset_ml::sentiment_sentences::*;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Sentiment Labelled Sentences file (`sentiment_sentences.csv`).
const SENTIMENT_SENTENCES_SHA256: &str =
    "3a6aac64fa37c8075d49678cd73140eaa70a95c984d540ddf93ec7b021e05725";

/// The Sentiment Labelled Sentences dataset has this many samples.
const N_SAMPLES: usize = 3_000;

/// Assert the Sentiment Labelled Sentences invariants: the sample count, the two
/// sentiment classes and three sources with their exact (balanced) counts, and
/// non-empty sentence texts.
fn assert_sentiment_sentences_semantics(
    texts: &ndarray::Array1<String>,
    sources: &ndarray::Array1<&'static str>,
    labels: &ndarray::Array1<&'static str>,
) {
    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(sources.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // Labels are one of the two classes, perfectly balanced (1,500 each).
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

    // Sources are one of the three sites, 1,000 sentences each.
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

    // The first record is a known negative Amazon sentence (the ordering is fixed
    // by the pinned SHA-256: the three per-site files combined amazon-first).
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
    let download_dir = "./test_load_sentiment_sentences"; // the code will create the directory if it doesn't exist

    let dataset = SentimentSentences::new(download_dir);
    let (texts, sources, labels) = dataset.data().unwrap();

    assert_sentiment_sentences_semantics(texts, sources, labels);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that loading uses a pre-existing cached file without re-downloading.
fn test_sentiment_sentences_no_need_download() {
    let download_dir = "./test_sentiment_sentences_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Prime the cache by loading once (downloads, extracts, and combines the three
    // per-site files), then confirm a second instance reuses the combined file.
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

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake data file is detected and overwritten with the real dataset.
fn test_sentiment_sentences_overwrite() {
    let download_dir = "./test_sentiment_sentences_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake dataset in advance
    {
        let path = download_dir_path.join("sentiment_sentences.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake dataset
    let dataset = SentimentSentences::new(download_dir);
    let (_texts, _sources, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("sentiment_sentences.csv"),
            SENTIMENT_SENTENCES_SHA256
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays, consuming the dataset.
fn test_sentiment_sentences_into_data() {
    let download_dir = "./test_sentiment_sentences_into_data";

    let dataset = SentimentSentences::new(download_dir);
    let (mut texts, sources, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; the arrays are fully owned.

    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(sources.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // Owned data can be mutated directly, with no `to_owned()` clone.
    texts[0] = "cleaned text".to_string();
    assert_eq!(texts[0], "cleaned text");

    // clean up: remove the downloaded files
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

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached file) and yields the same shapes.
    let (reloaded_texts, reloaded_sources, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_texts.len(), N_SAMPLES);
    assert_eq!(reloaded_sources.len(), N_SAMPLES);
    assert_eq!(reloaded_labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_sentiment_sentences_get_data() {
    let download_dir = "./test_sentiment_sentences_get_data";

    let dataset = SentimentSentences::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (texts, sources, labels) = dataset.get_data().unwrap();
    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(sources.len(), N_SAMPLES);
    assert_eq!(labels.len(), N_SAMPLES);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_sentiment_sentences_get_data_mut() {
    let download_dir = "./test_sentiment_sentences_get_data_mut";

    let mut dataset = SentimentSentences::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // Load, then mutate the cached texts in place (no clone, no reload).
    dataset.data().unwrap();
    if let Some((texts, _sources, _labels)) = dataset.get_data_mut() {
        texts[0] = "normalized".to_string();
    }

    // The change persisted in the cache: a later access observes it.
    let (texts, _sources, _labels) = dataset.data().unwrap();
    assert_eq!(texts[0], "normalized");

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
