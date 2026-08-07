//! Integration tests for the `Dataset<T, E>` container.
//!
//! These tests use only the public API. They do not need the `utils` feature.
//! They test the lazy-loading contract, the cache-invalidating operations, and
//! the guarantee that `load` runs the loader at most once across threads.

use dataset_core::Dataset;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A dataset whose loader counts its own calls. Tests can check the exact
/// number of runs. The loader shares its counter with the caller.
fn counting_dataset() -> (Dataset<usize, std::convert::Infallible>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let loader_calls = Arc::clone(&calls);

    let dataset = Dataset::new("./unused_dir", move |_| {
        Ok(loader_calls.fetch_add(1, Ordering::SeqCst) + 1)
    });

    (dataset, calls)
}

#[test]
fn load_runs_the_loader_once_and_caches() {
    let (dataset, calls) = counting_dataset();

    assert!(!dataset.is_loaded());
    assert_eq!(calls.load(Ordering::SeqCst), 0); // construction performs no I/O

    let first = dataset.load().unwrap();
    assert_eq!(*first, 1);
    assert!(dataset.is_loaded());

    let second = dataset.load().unwrap();
    assert!(std::ptr::eq(first, second));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn concurrent_loads_run_the_loader_once() {
    const THREADS: usize = 16;

    let calls = Arc::new(AtomicUsize::new(0));
    let loader_calls = Arc::clone(&calls);

    let dataset = Arc::new(Dataset::<usize, std::convert::Infallible>::new(
        "./unused_dir",
        move |_| {
            loader_calls.fetch_add(1, Ordering::SeqCst);
            // The sleep widens the window in which a second thread could start its own load.
            std::thread::sleep(std::time::Duration::from_millis(50));
            Ok(42)
        },
    ));

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let dataset = Arc::clone(&dataset);
            std::thread::spawn(move || *dataset.load().unwrap())
        })
        .collect();

    for handle in handles {
        assert_eq!(handle.join().unwrap(), 42);
    }

    // Without serialization, every thread would start its own load. For a real
    // loader, that means downloading the same file `THREADS` times.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn failed_load_is_not_cached() {
    let calls = Arc::new(AtomicUsize::new(0));
    let loader_calls = Arc::clone(&calls);

    let dataset = Dataset::<usize, String>::new("./unused_dir", move |_| {
        if loader_calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Err("transient failure".to_string())
        } else {
            Ok(7)
        }
    });

    assert_eq!(dataset.load().unwrap_err(), "transient failure");
    assert!(!dataset.is_loaded());

    assert_eq!(*dataset.load().unwrap(), 7);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn load_mut_loads_then_edits_in_place() {
    let (mut dataset, calls) = counting_dataset();

    // Unlike `get_mut`, this loads rather than returning `None`.
    *dataset.load_mut().unwrap() = 99;
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    assert_eq!(dataset.get(), Some(&99));
    assert_eq!(*dataset.load().unwrap(), 99);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn get_and_get_mut_never_load() {
    let (mut dataset, calls) = counting_dataset();

    assert!(dataset.get().is_none());
    assert!(dataset.get_mut().is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    dataset.load().unwrap();
    assert_eq!(dataset.get(), Some(&1));
}

#[test]
fn invalidate_drops_the_cache_and_keeps_the_loader() {
    let (mut dataset, calls) = counting_dataset();

    assert_eq!(*dataset.load().unwrap(), 1);
    dataset.invalidate();
    assert!(!dataset.is_loaded());

    assert_eq!(*dataset.load().unwrap(), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn set_loader_swaps_the_loader_and_invalidates() {
    let (mut dataset, calls) = counting_dataset();

    assert_eq!(*dataset.load().unwrap(), 1);

    dataset.set_loader(|_| Ok(1000));
    assert!(!dataset.is_loaded());

    assert_eq!(*dataset.load().unwrap(), 1000);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn take_returns_the_value_and_resets_the_container() {
    let (mut dataset, calls) = counting_dataset();

    assert!(dataset.take().is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    dataset.load().unwrap();
    assert_eq!(dataset.take(), Some(1));
    assert!(!dataset.is_loaded());

    assert_eq!(*dataset.load().unwrap(), 2);
}

#[test]
fn into_inner_consumes_the_container() {
    let (dataset, _calls) = counting_dataset();

    assert_eq!(dataset.into_inner(), None);

    let (dataset, _calls) = counting_dataset();
    dataset.load().unwrap();
    assert_eq!(dataset.into_inner(), Some(1));
}

#[test]
fn storage_dir_is_reported_verbatim() {
    let dataset: Dataset<u8, std::convert::Infallible> = Dataset::new("./some/dir", |_| Ok(0));

    assert_eq!(dataset.storage_dir(), "./some/dir");
}

#[test]
fn loader_receives_the_storage_dir() {
    let dataset: Dataset<String, std::convert::Infallible> =
        Dataset::new("./expected/dir", |dir| Ok(dir.to_string()));

    assert_eq!(dataset.load().unwrap(), "./expected/dir");
}

#[test]
fn debug_reports_storage_dir_and_load_state() {
    let dataset: Dataset<u8, std::convert::Infallible> = Dataset::new("./debug/dir", |_| Ok(0));

    let before = format!("{:?}", dataset);
    assert!(before.contains("./debug/dir"), "{before}");
    assert!(before.contains("data_loaded: false"), "{before}");

    dataset.load().unwrap();
    assert!(format!("{:?}", dataset).contains("data_loaded: true"));
}
