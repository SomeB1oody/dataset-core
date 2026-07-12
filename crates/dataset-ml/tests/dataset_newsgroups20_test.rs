mod common;

use common::file_sha256_matches;
use dataset_ml::newsgroups20::*;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached 20 Newsgroups archive (`20news-bydate.tar.gz`).
const NEWSGROUPS20_SHA256: &str =
    "8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610";

/// Sample counts for the three subsets.
const N_TRAIN: usize = 11_314;
const N_TEST: usize = 7_532;
const N_ALL: usize = 18_846;

/// The 20 newsgroup category names.
const CATEGORIES: [&str; 20] = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
];

/// Assert the train-partition invariants: the sample count, all 20 categories
/// present with a known per-class count, non-empty posts, and the pinned first
/// document (fixed by the deterministic lexicographic walk).
fn assert_newsgroups20_train_semantics(
    texts: &ndarray::Array1<String>,
    labels: &ndarray::Array1<&'static str>,
) {
    assert_eq!(texts.len(), N_TRAIN);
    assert_eq!(labels.len(), N_TRAIN);

    let known: HashSet<&str> = CATEGORIES.into_iter().collect();

    // Every label is one of the 20 newsgroups, and all 20 appear.
    let mut seen: HashSet<&str> = HashSet::new();
    let mut alt_atheism = 0usize;
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            known.contains(label),
            "labels[{i}] = {label:?} is not a known newsgroup"
        );
        seen.insert(label);
        if label == "alt.atheism" {
            alt_atheism += 1;
        }
    }
    assert_eq!(seen.len(), 20, "expected all 20 newsgroups to be present");
    assert_eq!(alt_atheism, 480, "expected 480 alt.atheism training posts");

    // Every post is non-empty.
    for (i, text) in texts.iter().enumerate() {
        assert!(!text.is_empty(), "texts[{i}] should not be empty");
    }

    // The first record is fixed by the deterministic walk (categories then files
    // in lexicographic order): alt.atheism / file "49960".
    assert_eq!(labels[0], "alt.atheism");
    assert!(
        texts[0].starts_with("From: mathew <mathew@mantis.co.uk>"),
        "texts[0] = {:?} does not match the known first post",
        &texts[0][..texts[0].len().min(60)]
    );
}

#[test]
// Verifies that the 20 Newsgroups train partition loads with the correct sample
// count, categories, per-class counts, and non-empty posts.
fn test_load_newsgroups20() {
    let download_dir = "./test_load_newsgroups20"; // the code will create the directory if it doesn't exist

    let dataset = Newsgroups20::new(download_dir);
    let (texts, labels) = dataset.data().unwrap();

    assert_newsgroups20_train_semantics(texts, labels);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies the test and all subsets load with the correct sample counts and that
// every label is a known newsgroup.
fn test_newsgroups20_subsets() {
    let download_dir = "./test_newsgroups20_subsets";
    let known: HashSet<&str> = CATEGORIES.into_iter().collect();

    // `new_test` and `new_all` share the same cached archive in this directory.
    let test_set = Newsgroups20::new_test(download_dir);
    let (test_texts, test_labels) = test_set.data().unwrap();
    assert_eq!(test_texts.len(), N_TEST);
    assert_eq!(test_labels.len(), N_TEST);
    assert!(test_labels.iter().all(|l| known.contains(l)));

    let all_set = Newsgroups20::new_all(download_dir);
    let (all_texts, all_labels) = all_set.data().unwrap();
    assert_eq!(all_texts.len(), N_ALL);
    assert_eq!(all_labels.len(), N_ALL);
    assert_eq!(N_TRAIN + N_TEST, N_ALL);
    assert!(all_labels.iter().all(|l| known.contains(l)));

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that loading uses a pre-existing cached archive without re-downloading.
fn test_newsgroups20_no_need_download() {
    let download_dir = "./test_newsgroups20_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // Prime the cache by loading once (downloads the archive), then confirm a
    // second instance reuses it.
    Newsgroups20::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join("20news-bydate.tar.gz"),
            NEWSGROUPS20_SHA256
        )
        .unwrap(),
        "cached 20news-bydate.tar.gz should match the expected SHA256"
    );

    let dataset = Newsgroups20::new(download_dir);
    let (_texts, _labels) = dataset.data().unwrap();

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that a corrupt or fake archive is detected and overwritten with the real one.
fn test_newsgroups20_overwrite() {
    let download_dir = "./test_newsgroups20_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    // create a fake archive in advance
    {
        let path = download_dir_path.join("20news-bydate.tar.gz");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    // should overwrite the fake archive
    let dataset = Newsgroups20::new(download_dir);
    let (_texts, _labels) = dataset.data().unwrap();

    // check the fake file is overwritten
    assert!(
        file_sha256_matches(
            &download_dir_path.join("20news-bydate.tar.gz"),
            NEWSGROUPS20_SHA256
        )
        .unwrap()
    );

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays, consuming the dataset.
fn test_newsgroups20_into_data() {
    let download_dir = "./test_newsgroups20_into_data";

    let dataset = Newsgroups20::new(download_dir);
    let (mut texts, labels) = dataset.into_data().unwrap();
    // `dataset` has been consumed; the arrays are fully owned.

    assert_eq!(texts.len(), N_TRAIN);
    assert_eq!(labels.len(), N_TRAIN);

    // Owned data can be mutated directly, with no `to_owned()` clone.
    texts[0] = "cleaned text".to_string();
    assert_eq!(texts[0], "cleaned text");

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned data and leaves the dataset reusable.
fn test_newsgroups20_take_data() {
    let download_dir = "./test_newsgroups20_take_data";

    let mut dataset = Newsgroups20::new(download_dir);
    let (texts, labels) = dataset.take_data().unwrap();

    assert_eq!(texts.len(), N_TRAIN);
    assert_eq!(labels.len(), N_TRAIN);

    // After take_data the instance is reset to unloaded but still usable: the next
    // access reloads it (from the cached archive) and yields the same shapes.
    let (reloaded_texts, reloaded_labels) = dataset.data().unwrap();
    assert_eq!(reloaded_texts.len(), N_TRAIN);
    assert_eq!(reloaded_labels.len(), N_TRAIN);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_newsgroups20_get_data() {
    let download_dir = "./test_newsgroups20_get_data";

    let dataset = Newsgroups20::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // Trigger loading, then get_data() hands back the cached references.
    dataset.data().unwrap();
    let (texts, labels) = dataset.get_data().unwrap();
    assert_eq!(texts.len(), N_TRAIN);
    assert_eq!(labels.len(), N_TRAIN);

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached data in place and the change persists.
fn test_newsgroups20_get_data_mut() {
    let download_dir = "./test_newsgroups20_get_data_mut";

    let mut dataset = Newsgroups20::new(download_dir);
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
