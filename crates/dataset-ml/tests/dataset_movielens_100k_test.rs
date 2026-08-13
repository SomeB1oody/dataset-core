#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::movielens_100k::MovieLens100k;
use dataset_ml::table::{ColumnData, Table};
use dataset_ml::traits::MlDataset;
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached rating log.
const MOVIELENS_SHA256: &str = "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490";

/// The name of the cached rating log.
const MOVIELENS_FILENAME: &str = "movielens_100k_ratings.tsv";

/// Number of ratings in the log.
const N_SAMPLES: usize = 100_000;

/// The four column names, in source order.
const COLUMN_NAMES: [&str; 4] = ["user_id", "item_id", "rating", "timestamp"];

/// Ratings per star value, from `1` star to `5` stars.
const RATING_COUNTS: [usize; 5] = [6_110, 11_370, 27_145, 34_174, 21_201];

/// Sum of every rating.
const RATING_SUM: u64 = 352_986;

/// The earliest and the latest timestamp in the log.
const FIRST_TIMESTAMP: i64 = 874_724_710;
const LAST_TIMESTAMP: i64 = 893_286_638;

/// Assert that the table names and types its columns as the docs claim.
fn assert_movielens_layout(table: &Table) {
    // Every name in the loader's constants reaches a real column.
    assert!(
        table.column(MovieLens100k::TARGET).is_some(),
        "TARGET `{}` names no column",
        MovieLens100k::TARGET
    );
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 4);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    // The named constant agrees with the source column order: `rating` is the
    // third of the four columns.
    assert_eq!(MovieLens100k::TARGET, COLUMN_NAMES[2]);
    assert_eq!(
        table.column(MovieLens100k::TARGET).unwrap().name(),
        "rating"
    );
    assert_eq!(table.column("timestamp").unwrap().name(), "timestamp");

    for name in ["user_id", "item_id", MovieLens100k::TARGET] {
        assert!(
            matches!(table.column(name).unwrap().data(), ColumnData::Integer(_)),
            "column {name} should be an integer"
        );
    }
    assert!(matches!(
        table.column("timestamp").unwrap().data(),
        ColumnData::Integer(_)
    ));
}

/// Assert the MovieLens invariants: the four lengths, the identifier ranges, the
/// rating balance, and the pinned records.
fn assert_movielens_semantics(table: &Table) {
    assert_movielens_layout(table);

    let users = table.column("user_id").unwrap().as_integer().unwrap();
    let items = table.column("item_id").unwrap().as_integer().unwrap();
    let ratings = table
        .column(MovieLens100k::TARGET)
        .unwrap()
        .as_integer()
        .unwrap();
    let timestamps = table.column("timestamp").unwrap().as_integer().unwrap();

    assert_eq!(users.len(), N_SAMPLES);
    assert_eq!(items.len(), N_SAMPLES);
    assert_eq!(ratings.len(), N_SAMPLES);
    assert_eq!(timestamps.len(), N_SAMPLES);

    let mut rating_counts = [0usize; 5];
    let mut rating_sum = 0u64;
    let mut seen_users: HashSet<i64> = HashSet::new();
    let mut seen_items: HashSet<i64> = HashSet::new();
    let mut pairs: HashSet<(i64, i64)> = HashSet::with_capacity(N_SAMPLES);

    for i in 0..N_SAMPLES {
        let user = users[i];
        let item = items[i];
        let rating = ratings[i];
        let timestamp = timestamps[i];

        assert!(
            user >= 1 && user as usize <= MovieLens100k::N_USERS,
            "row {} user {} is outside 1..=943",
            i,
            user
        );
        assert!(
            item >= 1 && item as usize <= MovieLens100k::N_ITEMS,
            "row {} item {} is outside 1..=1682",
            i,
            item
        );
        assert!(
            (1..=5).contains(&rating),
            "row {} rating {} is outside 1..=5",
            i,
            rating
        );
        assert!(
            (FIRST_TIMESTAMP..=LAST_TIMESTAMP).contains(&timestamp),
            "row {} timestamp {} is outside the collection period",
            i,
            timestamp
        );

        rating_counts[rating as usize - 1] += 1;
        rating_sum += rating as u64;
        seen_users.insert(user);
        seen_items.insert(item);
        assert!(
            pairs.insert((user, item)),
            "user {} rated movie {} twice",
            user,
            item
        );
    }

    // Every user and every movie of the two ranges appears at least once.
    assert_eq!(seen_users.len(), MovieLens100k::N_USERS);
    assert_eq!(seen_items.len(), MovieLens100k::N_ITEMS);

    assert_eq!(
        rating_counts, RATING_COUNTS,
        "the star balance should match the published counts"
    );
    assert_eq!(rating_sum, RATING_SUM, "the ratings should sum to 352,986");

    // The collection period runs from 1997-09-20 to 1998-04-22 in UTC.
    assert_eq!(*timestamps.iter().min().unwrap(), FIRST_TIMESTAMP);
    assert_eq!(*timestamps.iter().max().unwrap(), LAST_TIMESTAMP);

    // The first and the last record of the source, pinned field by field.
    assert_eq!(
        (users[0], items[0], ratings[0], timestamps[0]),
        (196, 242, 3, 881_250_949)
    );
    assert_eq!(
        (
            users[N_SAMPLES - 1],
            items[N_SAMPLES - 1],
            ratings[N_SAMPLES - 1],
            timestamps[N_SAMPLES - 1]
        ),
        (12, 203, 3, 879_959_583)
    );
}

#[test]
// Verifies that the MovieLens 100K rating log loads with the correct lengths,
// identifier ranges, and star balance.
fn test_load_movielens_100k() {
    let download_dir = "./test_load_movielens_100k"; // the loader creates this directory if it is missing

    let dataset = MovieLens100k::new(download_dir);
    assert_movielens_semantics(dataset.data().unwrap());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the identifier ranges match the two constants, and that the log
// keeps the source order rather than a sorted one.
fn test_movielens_100k_constants_and_order() {
    let download_dir = "./test_movielens_100k_constants_and_order";

    assert_eq!(MovieLens100k::N_USERS, 943);
    assert_eq!(MovieLens100k::N_ITEMS, 1682);

    let dataset = MovieLens100k::new(download_dir);
    let table = dataset.data().unwrap();
    let users = table.column("user_id").unwrap().as_integer().unwrap();
    let timestamps = table.column("timestamp").unwrap().as_integer().unwrap();

    // The rows keep the source order, which is neither by user nor by time.
    assert!(
        (0..N_SAMPLES - 1).any(|i| users[i] > users[i + 1]),
        "the users should not arrive sorted"
    );
    assert!(
        (0..N_SAMPLES - 1).any(|i| timestamps[i] > timestamps[i + 1]),
        "the timestamps should not arrive sorted"
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that n_samples() counts the ratings, not the users or the movies.
fn test_movielens_100k_n_samples() {
    let download_dir = "./test_movielens_100k_n_samples";

    let dataset = MovieLens100k::new(download_dir);
    assert_eq!(MovieLens100k::NAME, "movielens_100k");
    assert!(!dataset.is_loaded());
    assert_eq!(dataset.n_samples().unwrap(), N_SAMPLES);
    assert!(dataset.is_loaded());

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that MovieLens 100K reuses a cached file instead of a new download.
fn test_movielens_100k_no_need_download() {
    let download_dir = "./test_movielens_100k_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load primes the cache. The second instance then reuses it.
    MovieLens100k::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join(MOVIELENS_FILENAME),
            MOVIELENS_SHA256
        )
        .unwrap(),
        "the cached rating log should match the expected SHA256"
    );

    let dataset = MovieLens100k::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake rating log and overwrites it
// with the real one.
fn test_movielens_100k_overwrite() {
    let download_dir = "./test_movielens_100k_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let mut fake = File::create(download_dir_path.join(MOVIELENS_FILENAME)).unwrap();
        fake.write_all(b"fake data").unwrap();
    }

    let dataset = MovieLens100k::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(
            &download_dir_path.join(MOVIELENS_FILENAME),
            MOVIELENS_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_movielens_100k_into_data() {
    let download_dir = "./test_movielens_100k_into_data";

    let dataset = MovieLens100k::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The table is now fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 4);

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Integer(values)) = table.column_mut("rating").map(|c| c.data_mut()) {
        values[0] = 5;
    }
    assert_eq!(table.column("rating").unwrap().as_integer().unwrap()[0], 5);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the instance reusable.
fn test_movielens_100k_take_data() {
    let download_dir = "./test_movielens_100k_take_data";

    let mut dataset = MovieLens100k::new(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 4);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file and yields the same length.
    assert!(!dataset.is_loaded());
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_movielens_100k_get_data() {
    let download_dir = "./test_movielens_100k_get_data";

    let dataset = MovieLens100k::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), 4);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place.
fn test_movielens_100k_get_data_mut() {
    let download_dir = "./test_movielens_100k_get_data_mut";

    let mut dataset = MovieLens100k::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() clamps the first rating in place, with no clone and no
    // reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Integer(values)) = table.column_mut("rating").map(|c| c.data_mut())
    {
        values[0] = 1;
    }

    // The change persisted in the cache: a later access observes it.
    let ratings = dataset
        .data()
        .unwrap()
        .column("rating")
        .unwrap()
        .as_integer()
        .unwrap();
    assert_eq!(ratings[0], 1);
    assert_eq!(ratings[1], 3, "the second rating should stay untouched");

    remove_dir_all(download_dir).unwrap();
}
