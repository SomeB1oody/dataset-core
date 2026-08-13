#![cfg(feature = "dataset")]

//! Integration tests for the `MlDataset` trait.
//!
//! Most assertions here need no network access. The trait's inspection methods
//! (`storage_dir`, `is_loaded`, `peek`) never trigger a load. The tests that do
//! load use Iris, the smallest dataset in the crate.

use dataset_ml::table::ColumnData;
use dataset_ml::traits::MlDataset;
use dataset_ml::{Digits, Iris, SmsSpam, Titanic};
use std::fs::remove_dir_all;

/// A generic summary, written against the trait rather than a concrete loader.
/// Its existence is the point of the trait: it compiles for every dataset.
fn summarize<D: MlDataset>(dataset: &D) -> String {
    format!("{} in {}", D::NAME, dataset.storage_dir())
}

#[test]
fn inspection_methods_do_not_load() {
    let dataset = Iris::new("./test_traits_no_load");

    assert_eq!(dataset.storage_dir(), "./test_traits_no_load");
    assert!(!dataset.is_loaded());
    assert!(dataset.peek().is_none());
    assert_eq!(Iris::NAME, "iris");

    assert!(!std::path::Path::new("./test_traits_no_load").exists());
}

#[test]
fn a_generic_function_accepts_every_loader() {
    // A numeric table, a text table, a mixed-type table, and an image table.
    assert_eq!(summarize(&Iris::new("./a")), "iris in ./a");
    assert_eq!(summarize(&SmsSpam::new("./b")), "sms_spam in ./b");
    assert_eq!(summarize(&Titanic::new("./c")), "titanic in ./c");
    assert_eq!(summarize(&Digits::new("./d")), "digits in ./d");
}

#[test]
fn load_peek_and_invalidate_cycle() {
    let download_dir = "./test_traits_load_cycle";

    let mut dataset = Iris::new(download_dir);
    assert!(dataset.peek().is_none());

    // `load` populates the cache. `peek` then sees the same value, with no reload.
    let table = dataset.load().unwrap();
    assert_eq!(table.n_samples(), 150);
    assert_eq!(table.n_columns(), 5);
    assert!(dataset.is_loaded());
    assert!(dataset.peek().is_some());

    assert_eq!(dataset.n_samples().unwrap(), 150);

    // `invalidate` drops the cache but leaves the loader usable.
    dataset.invalidate();
    assert!(!dataset.is_loaded());
    assert!(dataset.peek().is_none());

    // The next access re-reads the cached file on disk and yields the same data.
    assert_eq!(dataset.n_samples().unwrap(), 150);
    assert!(dataset.is_loaded());

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn n_samples_agrees_with_every_column() {
    let download_dir = "./test_traits_n_samples";

    let dataset = Iris::new(download_dir);
    let table = dataset.load().unwrap();

    // The table's guarantee: every column holds `n_samples` values.
    assert_eq!(dataset.n_samples().unwrap(), table.n_samples());
    for column in table.columns() {
        assert_eq!(
            column.len(),
            table.n_samples(),
            "column {} holds a different number of samples",
            column.name()
        );
    }

    // The column split is what the loader documents: four numeric inputs and one
    // string label.
    assert_eq!(Iris::FEATURE_NAMES.len(), 4);
    for name in Iris::FEATURE_NAMES {
        assert!(
            matches!(table.column(name).unwrap().data(), ColumnData::Numeric(_)),
            "feature {name} should be numeric"
        );
    }
    assert!(matches!(
        table.column(Iris::TARGET).unwrap().data(),
        ColumnData::String(_)
    ));

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn load_mut_and_unload_move_data_without_cloning() {
    let download_dir = "./test_traits_load_mut_unload";

    let mut dataset = Iris::new(download_dir);

    // `load_mut` loads on demand, unlike the inherent `get_data_mut`.
    let table = dataset.load_mut().unwrap();
    if let Some(ColumnData::Numeric(values)) =
        table.column_mut("sepal_length").map(|c| c.data_mut())
    {
        values[0] = 42.0;
    }

    assert_eq!(
        dataset
            .load()
            .unwrap()
            .column("sepal_length")
            .unwrap()
            .as_numeric()
            .unwrap()[0],
        42.0
    );

    // `unload` moves the owned table out and resets the loader.
    let owned = dataset.unload().unwrap();
    assert_eq!(
        owned.column("sepal_length").unwrap().as_numeric().unwrap()[0],
        42.0
    );
    assert_eq!(owned.n_samples(), 150);
    assert!(!dataset.is_loaded());

    // The reset loader reads the file again from disk, so the edit is gone.
    assert_eq!(
        dataset
            .load()
            .unwrap()
            .column("sepal_length")
            .unwrap()
            .as_numeric()
            .unwrap()[0],
        5.1
    );

    // `unload` on an unloaded instance returns None. It does not load data first.
    dataset.invalidate();
    assert!(dataset.unload().is_none());

    remove_dir_all(download_dir).unwrap();
}

#[test]
fn into_dataset_yields_the_underlying_container() {
    let dataset = Iris::new("./test_traits_into_dataset");

    let container = dataset.into_dataset();

    assert_eq!(container.storage_dir(), "./test_traits_into_dataset");
    assert!(!container.is_loaded());
}
