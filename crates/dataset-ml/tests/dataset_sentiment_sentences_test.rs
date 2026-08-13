#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::sentiment_sentences::*;
use dataset_ml::table::{ColumnData, Table};
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached Sentiment Labelled Sentences file (`sentiment_sentences.csv`).
const SENTIMENT_SENTENCES_SHA256: &str =
    "3a6aac64fa37c8075d49678cd73140eaa70a95c984d540ddf93ec7b021e05725";

/// The Sentiment Labelled Sentences dataset has this many samples.
const N_SAMPLES: usize = 3_000;

/// The three column names, in source order.
const COLUMN_NAMES: [&str; 3] = ["text", "source", "label"];

/// Assert the column layout the documentation states: one text feature, one
/// string metadata column, and one string target.
fn assert_sentiment_sentences_schema(table: &Table) {
    // Every name in the loader's constants reaches a real column.
    for name in SentimentSentences::FEATURE_NAMES {
        assert!(
            table.column(name).is_some(),
            "FEATURE_NAMES entry `{name}` names no column"
        );
    }
    assert!(
        table.column(SentimentSentences::TARGET).is_some(),
        "TARGET `{}` names no column",
        SentimentSentences::TARGET
    );
    assert!(
        !SentimentSentences::FEATURE_NAMES.contains(&SentimentSentences::TARGET),
        "the target must not also be a feature"
    );
    assert_eq!(table.n_columns(), 3);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    assert_eq!(SentimentSentences::FEATURE_NAMES, ["text"]);
    assert_eq!(SentimentSentences::TARGET, "label");

    let text = table.column(SentimentSentences::FEATURE_NAMES[0]).unwrap();
    assert!(matches!(text.data(), ColumnData::String(_)));

    let source = table.column("source").unwrap();
    assert!(matches!(source.data(), ColumnData::String(_)));

    let label = table.column(SentimentSentences::TARGET).unwrap();
    assert!(matches!(label.data(), ColumnData::String(_)));
}

/// Checks the Sentiment Labelled Sentences invariants: the column layout, the
/// sample count, and the two sentiment classes. It also checks the three sources
/// with their exact balanced counts and the non-empty sentence texts.
fn assert_sentiment_sentences_semantics(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_sentiment_sentences_schema(table);

    let texts = table
        .column(SentimentSentences::FEATURE_NAMES[0])
        .unwrap()
        .as_string()
        .unwrap();
    let sources = table.column("source").unwrap().as_string().unwrap();
    let labels = table
        .column(SentimentSentences::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    assert_eq!(texts.len(), N_SAMPLES);
    assert_eq!(sources.len(), N_SAMPLES);
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
    assert_eq!(positive, 1500, "expected 1,500 positive sentences");
    assert_eq!(negative, 1500, "expected 1,500 negative sentences");

    let mut amazon = 0usize;
    let mut imdb = 0usize;
    let mut yelp = 0usize;
    for (i, source) in sources.iter().enumerate() {
        match source.as_str() {
            "amazon" => amazon += 1,
            "imdb" => imdb += 1,
            "yelp" => yelp += 1,
            other => panic!("sources[{i}] = {other:?} is not `amazon`, `imdb`, or `yelp`"),
        }
    }
    assert_eq!(amazon, 1000, "expected 1,000 amazon sentences");
    assert_eq!(imdb, 1000, "expected 1,000 imdb sentences");
    assert_eq!(yelp, 1000, "expected 1,000 yelp sentences");

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
fn test_load_sentiment_sentences() {
    let download_dir = "./test_load_sentiment_sentences"; // if the directory does not exist, the code creates it

    let dataset = SentimentSentences::new(download_dir);
    assert_sentiment_sentences_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_sentiment_sentences_no_need_download() {
    let download_dir = "./test_sentiment_sentences_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load downloads, extracts, and combines the three per-site files
    // into the cache. The next instance then reuses the combined file.
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
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_sentiment_sentences_overwrite() {
    let download_dir = "./test_sentiment_sentences_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("sentiment_sentences.csv");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = SentimentSentences::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

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
fn test_sentiment_sentences_into_data() {
    let download_dir = "./test_sentiment_sentences_into_data";

    let dataset = SentimentSentences::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The table is now fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_sentiment_sentences_schema(&table);

    // The owned table allows direct mutation and needs no clone.
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
fn test_sentiment_sentences_take_data() {
    let download_dir = "./test_sentiment_sentences_take_data";

    let mut dataset = SentimentSentences::new(download_dir);
    let table = dataset.take_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file and yields the same shape.
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_sentiment_sentences_get_data() {
    let download_dir = "./test_sentiment_sentences_get_data";

    let dataset = SentimentSentences::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached reference.
    dataset.data().unwrap();
    assert_eq!(dataset.get_data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_sentiment_sentences_get_data_mut() {
    let download_dir = "./test_sentiment_sentences_get_data_mut";

    let mut dataset = SentimentSentences::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() mutates the cached text column in place, with no clone or
    // reload.
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
        table.column("source").unwrap().as_string().unwrap()[0],
        "amazon",
        "the other columns should stay untouched"
    );

    remove_dir_all(download_dir).unwrap();
}
