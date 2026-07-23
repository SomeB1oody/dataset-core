//! Integration tests for the `Dataset<T, E>` container itself.
//!
//! These exercise the container through its public API only (no `utils` feature
//! needed): the lazy-loading contract, the cache-invalidating operations, and the
//! "loader runs at most once" guarantee `load` makes across threads.

use dataset_core::Dataset;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A dataset whose loader counts its own invocations, so tests can assert exactly
/// how many times it ran. The returned counter is shared with the loader.
fn counting_dataset() -> (Dataset<usize, std::convert::Infallible>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let loader_calls = Arc::clone(&calls);

    let dataset = Dataset::new("./unused_dir", move |_| {
        Ok(loader_calls.fetch_add(1, Ordering::SeqCst) + 1)
    });

    (dataset, calls)
}

#[test]
// Verifies that the loader runs on first access only and the value is cached.
fn load_runs_the_loader_once_and_caches() {
    let (dataset, calls) = counting_dataset();

    assert!(!dataset.is_loaded());
    assert_eq!(calls.load(Ordering::SeqCst), 0); // construction performs no I/O

    let first = dataset.load().unwrap();
    assert_eq!(*first, 1);
    assert!(dataset.is_loaded());

    // Repeated loads hand back the very same reference without re-running the loader.
    let second = dataset.load().unwrap();
    assert!(std::ptr::eq(first, second));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
// Verifies that concurrent loads run the loader exactly once and all observe it.
fn concurrent_loads_run_the_loader_once() {
    const THREADS: usize = 16;

    let calls = Arc::new(AtomicUsize::new(0));
    let loader_calls = Arc::clone(&calls);

    let dataset = Arc::new(Dataset::<usize, std::convert::Infallible>::new(
        "./unused_dir",
        move |_| {
            loader_calls.fetch_add(1, Ordering::SeqCst);
            // Widen the window in which a second thread could start its own load.
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

    // Without serialization every thread would have started its own load — for a
    // real loader, that means the same file downloaded `THREADS` times.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
// Verifies that a failing loader is not cached and is retried on the next load.
fn failed_load_is_not_cached() {
    let calls = Arc::new(AtomicUsize::new(0));
    let loader_calls = Arc::clone(&calls);

    // Fails on the first attempt, succeeds afterwards.
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
// Verifies that load_mut loads on demand and its edits persist in the cache.
fn load_mut_loads_then_edits_in_place() {
    let (mut dataset, calls) = counting_dataset();

    // Unlike `get_mut`, this loads rather than returning `None`.
    *dataset.load_mut().unwrap() = 99;
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // The edit stayed in the cache, and no reload happened.
    assert_eq!(dataset.get(), Some(&99));
    assert_eq!(*dataset.load().unwrap(), 99);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
// Verifies that get/get_mut never trigger loading.
fn get_and_get_mut_never_load() {
    let (mut dataset, calls) = counting_dataset();

    assert!(dataset.get().is_none());
    assert!(dataset.get_mut().is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    dataset.load().unwrap();
    assert_eq!(dataset.get(), Some(&1));
}

#[test]
// Verifies that invalidate drops the cache but keeps the loader.
fn invalidate_drops_the_cache_and_keeps_the_loader() {
    let (mut dataset, calls) = counting_dataset();

    assert_eq!(*dataset.load().unwrap(), 1);
    dataset.invalidate();
    assert!(!dataset.is_loaded());

    // Same loader, run a second time: the counter it returns has advanced.
    assert_eq!(*dataset.load().unwrap(), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
// Verifies that set_loader swaps the loader and invalidates the cached value.
fn set_loader_swaps_the_loader_and_invalidates() {
    let (mut dataset, calls) = counting_dataset();

    assert_eq!(*dataset.load().unwrap(), 1);

    dataset.set_loader(|_| Ok(1000));
    assert!(!dataset.is_loaded());

    // The new loader is used, and the old one is never called again.
    assert_eq!(*dataset.load().unwrap(), 1000);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
// Verifies that take moves the value out and leaves the container reusable.
fn take_returns_the_value_and_resets_the_container() {
    let (mut dataset, calls) = counting_dataset();

    // Nothing cached yet, so there is nothing to take — and no load is triggered.
    assert!(dataset.take().is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    dataset.load().unwrap();
    assert_eq!(dataset.take(), Some(1));
    assert!(!dataset.is_loaded());

    // Reusable: the next load runs the loader again.
    assert_eq!(*dataset.load().unwrap(), 2);
}

#[test]
// Verifies that into_inner consumes the container and yields the cached value.
fn into_inner_consumes_the_container() {
    let (dataset, _calls) = counting_dataset();

    // A container that was never loaded yields `None` without loading.
    assert_eq!(dataset.into_inner(), None);

    let (dataset, _calls) = counting_dataset();
    dataset.load().unwrap();
    assert_eq!(dataset.into_inner(), Some(1));
}

#[test]
// Verifies that storage_dir reports the path given at construction.
fn storage_dir_is_reported_verbatim() {
    let dataset: Dataset<u8, std::convert::Infallible> = Dataset::new("./some/dir", |_| Ok(0));

    assert_eq!(dataset.storage_dir(), "./some/dir");
}

#[test]
// Verifies that the loader receives the storage directory it was constructed with.
fn loader_receives_the_storage_dir() {
    let dataset: Dataset<String, std::convert::Infallible> =
        Dataset::new("./expected/dir", |dir| Ok(dir.to_string()));

    assert_eq!(dataset.load().unwrap(), "./expected/dir");
}

#[test]
// Verifies that Debug reports the storage directory and the load state.
fn debug_reports_storage_dir_and_load_state() {
    let dataset: Dataset<u8, std::convert::Infallible> = Dataset::new("./debug/dir", |_| Ok(0));

    let before = format!("{:?}", dataset);
    assert!(before.contains("./debug/dir"), "{before}");
    assert!(before.contains("data_loaded: false"), "{before}");

    dataset.load().unwrap();
    assert!(format!("{:?}", dataset).contains("data_loaded: true"));
}
