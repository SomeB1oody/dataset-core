#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::newsgroups20::*;
use dataset_ml::table::{ColumnData, Table};
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

/// The two column names, in source order.
const COLUMN_NAMES: [&str; 2] = ["text", "label"];

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

/// Assert the column layout the documentation states: two columns, one text
/// feature and one string target.
fn assert_newsgroups20_schema(table: &Table) {
    // Every name in the loader's constants reaches a real column.
    for name in Newsgroups20::FEATURE_NAMES {
        assert!(
            table.column(name).is_some(),
            "FEATURE_NAMES entry `{name}` names no column"
        );
    }
    assert!(
        table.column(Newsgroups20::TARGET).is_some(),
        "TARGET `{}` names no column",
        Newsgroups20::TARGET
    );
    assert!(
        !Newsgroups20::FEATURE_NAMES.contains(&Newsgroups20::TARGET),
        "the target must not also be a feature"
    );
    assert_eq!(table.n_columns(), 2);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    assert_eq!(Newsgroups20::FEATURE_NAMES, ["text"]);
    assert_eq!(Newsgroups20::TARGET, "label");

    let text = table.column(Newsgroups20::FEATURE_NAMES[0]).unwrap();
    assert!(matches!(text.data(), ColumnData::String(_)));

    let label = table.column(Newsgroups20::TARGET).unwrap();
    assert!(matches!(label.data(), ColumnData::String(_)));
}

/// Assert the train-partition invariants: the column layout, the sample count,
/// all 20 categories with a known per-class count, non-empty posts, and the first
/// document. The deterministic lexicographic walk fixes this first document.
fn assert_newsgroups20_train_semantics(table: &Table) {
    assert_eq!(table.n_samples(), N_TRAIN);
    assert_newsgroups20_schema(table);

    let texts = table
        .column(Newsgroups20::FEATURE_NAMES[0])
        .unwrap()
        .as_string()
        .unwrap();
    let labels = table
        .column(Newsgroups20::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    assert_eq!(texts.len(), N_TRAIN);
    assert_eq!(labels.len(), N_TRAIN);

    let known: HashSet<&str> = CATEGORIES.into_iter().collect();

    let mut seen: HashSet<&str> = HashSet::new();
    let mut alt_atheism = 0usize;
    for (i, label) in labels.iter().enumerate() {
        assert!(
            known.contains(label.as_str()),
            "labels[{i}] = {label:?} is not a known newsgroup"
        );
        seen.insert(label.as_str());
        if label == "alt.atheism" {
            alt_atheism += 1;
        }
    }
    assert_eq!(seen.len(), 20, "expected all 20 newsgroups to be present");
    assert_eq!(alt_atheism, 480, "expected 480 alt.atheism training posts");

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
// Verifies that the 20 Newsgroups train partition loads with the correct column
// layout, sample count, categories, per-class counts, and non-empty posts.
fn test_load_newsgroups20() {
    // If the directory does not exist, the code creates it.
    let download_dir = "./test_load_newsgroups20";

    let dataset = Newsgroups20::new(download_dir);
    assert_newsgroups20_train_semantics(dataset.data().unwrap());

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
    let test_table = test_set.data().unwrap();
    assert_eq!(test_table.n_samples(), N_TEST);
    assert_newsgroups20_schema(test_table);
    let test_labels = test_table
        .column(Newsgroups20::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    assert_eq!(test_labels.len(), N_TEST);
    assert!(test_labels.iter().all(|l| known.contains(l.as_str())));

    let all_set = Newsgroups20::new_all(download_dir);
    let all_table = all_set.data().unwrap();
    assert_eq!(all_table.n_samples(), N_ALL);
    assert_newsgroups20_schema(all_table);
    assert_eq!(N_TRAIN + N_TEST, N_ALL);
    let all_labels = all_table
        .column(Newsgroups20::TARGET)
        .unwrap()
        .as_string()
        .unwrap();
    assert_eq!(all_labels.len(), N_ALL);
    assert!(all_labels.iter().all(|l| known.contains(l.as_str())));

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that loading uses a pre-existing cached archive without re-downloading.
fn test_newsgroups20_no_need_download() {
    let download_dir = "./test_newsgroups20_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

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
    assert_eq!(dataset.data().unwrap().n_samples(), N_TRAIN);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake archive and overwrites it
// with the real one.
fn test_newsgroups20_overwrite() {
    let download_dir = "./test_newsgroups20_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let path = download_dir_path.join("20news-bydate.tar.gz");
        let mut fake = File::create(path).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = Newsgroups20::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_TRAIN);

    assert!(
        file_sha256_matches(
            &download_dir_path.join("20news-bydate.tar.gz"),
            NEWSGROUPS20_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table, consuming the dataset.
fn test_newsgroups20_into_data() {
    let download_dir = "./test_newsgroups20_into_data";

    let dataset = Newsgroups20::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumes `dataset`. The returned table is fully owned.

    assert_eq!(table.n_samples(), N_TRAIN);
    assert_newsgroups20_schema(&table);

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
fn test_newsgroups20_take_data() {
    let download_dir = "./test_newsgroups20_take_data";

    let mut dataset = Newsgroups20::new(download_dir);
    let table = dataset.take_data().unwrap();
    assert_eq!(table.n_samples(), N_TRAIN);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads the data (from the cached archive) and yields the same shape.
    assert_eq!(dataset.data().unwrap().n_samples(), N_TRAIN);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_newsgroups20_get_data() {
    let download_dir = "./test_newsgroups20_get_data";

    let dataset = Newsgroups20::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    // After loading, get_data() returns the cached reference.
    dataset.data().unwrap();
    assert_eq!(dataset.get_data().unwrap().n_samples(), N_TRAIN);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place and the change persists.
fn test_newsgroups20_get_data_mut() {
    let download_dir = "./test_newsgroups20_get_data_mut";

    let mut dataset = Newsgroups20::new(download_dir);
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
        "alt.atheism",
        "the other columns should stay untouched"
    );

    remove_dir_all(download_dir).unwrap();
}
