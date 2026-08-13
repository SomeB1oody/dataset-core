//! The [`MlDataset`] trait shared by every loader in this crate.
//!
//! Every loader parses into a [`Table`], so the trait needs no type parameter and
//! no per-loader accessor. It covers the container operations that stay the same
//! whatever the table holds:
//!
//! - [`invalidate`](MlDataset::invalidate) drops the in-memory cache and forces the
//!   next access to re-read (and re-verify) the file on disk.
//! - [`is_loaded`](MlDataset::is_loaded) and [`storage_dir`](MlDataset::storage_dir)
//!   let you inspect a loader without touching the data.
//! - [`n_samples`](MlDataset::n_samples) gives the sample count.
//!
//! This trait names its data accessors [`load`](MlDataset::load),
//! [`peek`](MlDataset::peek), and [`unload`](MlDataset::unload) rather than reusing
//! `data`, `get_data`, and `take_data`. This way, a trait method never silently
//! shadows the inherent method of the same name, and the inherent method never
//! shadows the trait method either. Both sets are always available and always
//! agree. Use whichever reads better where you are.
//!
//! This module is always available. The `dataset` feature only decides whether the
//! built-in loaders that implement the trait compile with it.
//!
//! # Example
//!
//! This example needs the `dataset` feature, because it uses two built-in loaders.
//!
//! ```no_run
//! use dataset_ml::traits::MlDataset;
//! use dataset_ml::{Iris, SmsSpam};
//!
//! // One function works for any loader, including the text corpora, whose table
//! // holds entirely different columns from Iris's.
//! fn describe<D: MlDataset>(dataset: &D) -> String {
//!     format!("{} ({} samples)", D::NAME, dataset.n_samples().unwrap())
//! }
//!
//! assert_eq!(describe(&Iris::new("./data")), "iris (150 samples)");
//! assert_eq!(describe(&SmsSpam::new("./data")), "sms_spam (5574 samples)");
//! ```

use crate::table::Table;
use dataset_core::{Dataset, DatasetError};

/// The lazy-loading behavior every dataset loader in this crate shares.
///
/// Implementors wrap a [`Dataset<Table, DatasetError>`](dataset_core::Dataset) and
/// only need to expose it through the three needed methods. The trait provides
/// everything else.
///
/// # Implementing it for your own loader
///
/// ```rust
/// use dataset_core::{Dataset, DatasetError};
/// use dataset_ml::table::Table;
/// use dataset_ml::traits::MlDataset;
///
/// struct MyDataset {
///     dataset: Dataset<Table, DatasetError>,
/// }
///
/// impl MlDataset for MyDataset {
///     const NAME: &'static str = "my_dataset";
///
///     fn dataset(&self) -> &Dataset<Table, DatasetError> {
///         &self.dataset
///     }
///
///     fn dataset_mut(&mut self) -> &mut Dataset<Table, DatasetError> {
///         &mut self.dataset
///     }
///
///     fn into_dataset(self) -> Dataset<Table, DatasetError> {
///         self.dataset
///     }
/// }
/// ```
pub trait MlDataset: Sized {
    /// The dataset's identifier, matching the one used in its error messages
    /// (for example, `"iris"`, `"sms_spam"`).
    const NAME: &'static str;

    /// Borrow the underlying container.
    fn dataset(&self) -> &Dataset<Table, DatasetError>;

    /// Borrow the underlying container mutably.
    fn dataset_mut(&mut self) -> &mut Dataset<Table, DatasetError>;

    /// Consume the loader and return the underlying container.
    fn into_dataset(self) -> Dataset<Table, DatasetError>;

    /// Load the dataset if needed and borrow the table.
    ///
    /// The generic equivalent of each loader's inherent `data()`: it downloads and
    /// parses on first call, then returns the cached value. Concurrent calls run
    /// the loader once and share the result.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if the download, file I/O, or parsing fails.
    fn load(&self) -> Result<&Table, DatasetError> {
        self.dataset().load()
    }

    /// Load the dataset if needed and borrow the table **mutably**.
    ///
    /// If you edit the table in place through the returned reference, the change
    /// persists in the cache, so later accesses observe it. Unlike the inherent
    /// `get_data_mut()`, this loads rather than returning `None` when nothing is
    /// cached yet.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if the download, file I/O, or parsing fails.
    fn load_mut(&mut self) -> Result<&mut Table, DatasetError> {
        self.dataset_mut().load_mut()
    }

    /// Borrow the table **without** triggering loading.
    ///
    /// The generic equivalent of each loader's inherent `get_data()`. Returns
    /// `None`, rather than downloading, when the dataset has not loaded yet.
    fn peek(&self) -> Option<&Table> {
        self.dataset().get()
    }

    /// Move the table out, leaving the loader reusable and unloaded.
    ///
    /// The generic equivalent of each loader's inherent `take_data()`, except that
    /// it never loads: it returns `None` if nothing is cached.
    fn unload(&mut self) -> Option<Table> {
        self.dataset_mut().take()
    }

    /// Whether the cache currently holds the table.
    ///
    /// Never triggers loading, so this is the cheap way to ask whether an accessor
    /// already has the value or must start a download.
    fn is_loaded(&self) -> bool {
        self.dataset().is_loaded()
    }

    /// The directory this loader stores its files in.
    fn storage_dir(&self) -> &str {
        self.dataset().storage_dir()
    }

    /// Drop the cached table, keeping the loader usable.
    ///
    /// The next access re-reads the file from `storage_dir`. This re-runs the
    /// SHA-256 check and the parser, and re-downloads if the file is gone or no
    /// longer matches. Use it to reclaim the memory a dataset occupies
    /// (`covtype`, `kddcup99`), or to read a file that changed on disk.
    ///
    /// To retrieve the table rather than discard it, use [`unload`](Self::unload).
    fn invalidate(&mut self) {
        self.dataset_mut().invalidate();
    }

    /// The number of samples in the dataset, loading it if needed.
    ///
    /// Every column of the table holds this many values.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if the download, file I/O, or parsing fails.
    fn n_samples(&self) -> Result<usize, DatasetError> {
        Ok(self.load()?.n_samples())
    }
}

/// Implement [`MlDataset`] for a loader that stores its container in a field named
/// `dataset`.
///
/// Every loader in this crate has that exact shape, so the implementation is
/// entirely mechanical. This macro writes the three needed methods and leaves the
/// rest to the trait's defaults. It is crate-internal. Downstream loaders
/// implement the trait directly (see [`MlDataset`]'s own example).
#[cfg(feature = "dataset")]
macro_rules! impl_ml_dataset {
    ($struct_name:ident, $name:literal) => {
        impl $crate::traits::MlDataset for $struct_name {
            const NAME: &'static str = $name;

            fn dataset(
                &self,
            ) -> &::dataset_core::Dataset<$crate::table::Table, ::dataset_core::DatasetError> {
                &self.dataset
            }

            fn dataset_mut(
                &mut self,
            ) -> &mut ::dataset_core::Dataset<$crate::table::Table, ::dataset_core::DatasetError>
            {
                &mut self.dataset
            }

            fn into_dataset(
                self,
            ) -> ::dataset_core::Dataset<$crate::table::Table, ::dataset_core::DatasetError> {
                self.dataset
            }
        }
    };
}

#[cfg(feature = "dataset")]
pub(crate) use impl_ml_dataset;
