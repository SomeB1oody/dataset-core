//! Integration tests for the `MlDataset` trait.
//!
//! Most assertions here need no network: the trait's inspection methods
//! (`storage_dir`, `is_loaded`, `peek`) are defined never to load. The two tests
//! that do load use Iris, the smallest dataset in the crate.

use dataset_ml::traits::{MlDataset, NumSamples};
use dataset_ml::{Digits, Iris, SmsSpam, Titanic};
use ndarray::{Array1, Array2, array};
use std::fs::remove_dir_all;

/// A generic summary, written against the trait rather than a concrete loader.
/// Its existence is the point of the trait: it compiles for every dataset.
fn summarize<D: MlDataset>(dataset: &D) -> String {
    format!("{} in {}", D::NAME, dataset.storage_dir())
}

#[test]
// Verifies that the trait's inspection methods never trigger a download.
fn inspection_methods_do_not_load() {
    let dataset = Iris::new("./test_traits_no_load");

    assert_eq!(dataset.storage_dir(), "./test_traits_no_load");
    assert!(!dataset.is_loaded());
    assert!(dataset.peek().is_none());
    assert_eq!(Iris::NAME, "iris");

    // Nothing above should have created the storage directory.
    assert!(!std::path::Path::new("./test_traits_no_load").exists());
}

#[test]
// Verifies that one generic function works across loaders with different data shapes.
fn a_generic_function_accepts_every_loader() {
    // Tabular pair, text pair, and mixed-type triple respectively.
    assert_eq!(summarize(&Iris::new("./a")), "iris in ./a");
    assert_eq!(summarize(&SmsSpam::new("./b")), "sms_spam in ./b");
    assert_eq!(summarize(&Titanic::new("./c")), "titanic in ./c");
    assert_eq!(summarize(&Digits::new("./d")), "digits in ./d");
}

#[test]
// Verifies that NumSamples reads the leading axis for pair- and triple-shaped data.
fn num_samples_reads_the_leading_axis() {
    // (features, labels): a 3-row feature matrix.
    let pair: (Array2<f64>, Array1<u8>) = (Array2::zeros((3, 7)), Array1::zeros(3));
    assert_eq!(pair.num_samples(), 3);

    // (texts, labels): the text loaders' shape, where both arrays are 1-D.
    let text_pair: (Array1<String>, Array1<&str>) =
        (array!["a".to_string(), "b".to_string()], array!["x", "y"]);
    assert_eq!(text_pair.num_samples(), 2);

    // (categorical, numeric, labels): the mixed-type loaders' triple.
    let triple: (Array2<String>, Array2<f64>, Array1<f64>) = (
        Array2::from_elem((5, 2), String::new()),
        Array2::zeros((5, 4)),
        Array1::zeros(5),
    );
    assert_eq!(triple.num_samples(), 5);
}

#[test]
// Verifies the load/peek/n_samples/invalidate cycle against a real dataset.
fn load_peek_and_invalidate_cycle() {
    let download_dir = "./test_traits_load_cycle";

    let mut dataset = Iris::new(download_dir);
    assert!(dataset.peek().is_none());

    // `load` populates the cache; `peek` then sees the same value without reloading.
    let (features, labels) = dataset.load().unwrap();
    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);
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

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that load_mut edits persist and that unload hands the data back.
fn load_mut_and_unload_move_data_without_cloning() {
    let download_dir = "./test_traits_load_mut_unload";

    let mut dataset = Iris::new(download_dir);

    // `load_mut` loads on demand, unlike the inherent `get_data_mut`.
    let (features, _labels) = dataset.load_mut().unwrap();
    features[[0, 0]] = 42.0;

    // The edit stayed in the cache.
    assert_eq!(dataset.load().unwrap().0[[0, 0]], 42.0);

    // `unload` moves the owned arrays out and resets the loader.
    let (owned_features, owned_labels) = dataset.unload().unwrap();
    assert_eq!(owned_features[[0, 0]], 42.0);
    assert_eq!(owned_labels.len(), 150);
    assert!(!dataset.is_loaded());

    // Reset means reloaded from disk, so the edit is gone.
    assert_eq!(dataset.load().unwrap().0[[0, 0]], 5.1);

    // `unload` on an unloaded instance returns None rather than loading.
    dataset.invalidate();
    assert!(dataset.unload().is_none());

    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_dataset hands back the underlying container.
fn into_dataset_yields_the_underlying_container() {
    let dataset = Iris::new("./test_traits_into_dataset");

    let container = dataset.into_dataset();

    // The container carries the storage directory and is still unloaded.
    assert_eq!(container.storage_dir(), "./test_traits_into_dataset");
    assert!(!container.is_loaded());
}
