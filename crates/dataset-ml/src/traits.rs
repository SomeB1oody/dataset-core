//! The [`MlDataset`] trait shared by every loader in this crate.
//!
//! Each loader has its own inherent accessors, named for what it holds.
//! Tabular loaders use `features()`/`labels()`, regression loaders use
//! `targets()`, and text corpora use `texts()`. Those names make each loader
//! easy to use directly. Before this trait existed, though, no code could work
//! with datasets *generically*.
//!
//! [`MlDataset`] is the common denominator: the container operations that are the
//! same whatever the loader parses into. It adds three capabilities the inherent
//! APIs never exposed:
//!
//! - [`invalidate`](MlDataset::invalidate) drops the in-memory cache and forces the
//!   next access to re-read (and re-verify) the file on disk.
//! - [`is_loaded`](MlDataset::is_loaded) and [`storage_dir`](MlDataset::storage_dir)
//!   let you inspect a loader without touching the data.
//! - [`n_samples`](MlDataset::n_samples) gives a uniform sample count that works
//!   across the single-array, pair-shaped, and triple-shaped datasets alike.
//!
//! This trait deliberately names its data accessors [`load`](MlDataset::load),
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
//! // One function works for any loader, including the text corpora, whose data has an
//! // entirely different shape from Iris's.
//! fn describe<D: MlDataset>(dataset: &D) -> String {
//!     format!("{} ({} samples)", D::NAME, dataset.n_samples().unwrap())
//! }
//!
//! assert_eq!(describe(&Iris::new("./data")), "iris (150 samples)");
//! assert_eq!(describe(&SmsSpam::new("./data")), "sms_spam (5574 samples)");
//! ```

use dataset_core::{Dataset, DatasetError};
use ndarray::{Array, Axis, Dimension};

/// A parsed dataset whose samples can be counted.
///
/// This crate implements it for the three shapes its loaders parse into:
///
/// - A single array, for a dataset with no target. `features` is the one example.
/// - An array pair, such as `(features, labels)`, `(features, targets)`, or
///   `(texts, labels)`.
/// - An array triple, such as `(categorical, numeric, labels)` or
///   `(texts, sources, labels)`.
///
/// In all of them, the first array's leading axis is the sample axis, so this
/// counts that axis. A zero-dimensional array has no leading axis, so do not use
/// one as the first array.
///
/// You only need this trait directly to call [`MlDataset::n_samples`] in a generic
/// function. If you want the same from a loader of your own, implement it for your
/// own data type.
pub trait NumSamples {
    /// The number of samples the parsed data holds.
    fn num_samples(&self) -> usize;
}

impl<A, D> NumSamples for Array<A, D>
where
    D: Dimension,
{
    fn num_samples(&self) -> usize {
        self.len_of(Axis(0))
    }
}

impl<A, DA, B, DB> NumSamples for (Array<A, DA>, Array<B, DB>)
where
    DA: Dimension,
    DB: Dimension,
{
    fn num_samples(&self) -> usize {
        self.0.len_of(Axis(0))
    }
}

impl<A, DA, B, DB, C, DC> NumSamples for (Array<A, DA>, Array<B, DB>, Array<C, DC>)
where
    DA: Dimension,
    DB: Dimension,
    DC: Dimension,
{
    fn num_samples(&self) -> usize {
        self.0.len_of(Axis(0))
    }
}

/// The lazy-loading behavior every dataset loader in this crate shares.
///
/// Implementors wrap a [`Dataset<Self::Data, DatasetError>`](dataset_core::Dataset)
/// and only need to expose it through the three needed methods. The trait
/// provides everything else.
///
/// # Implementing it for your own loader
///
/// ```rust
/// use dataset_core::{Dataset, DatasetError};
/// use dataset_ml::traits::MlDataset;
/// use ndarray::{Array1, Array2};
///
/// type MyData = (Array2<f64>, Array1<u8>);
///
/// struct MyDataset {
///     dataset: Dataset<MyData, DatasetError>,
/// }
///
/// impl MlDataset for MyDataset {
///     type Data = MyData;
///     const NAME: &'static str = "my_dataset";
///
///     fn dataset(&self) -> &Dataset<Self::Data, DatasetError> {
///         &self.dataset
///     }
///
///     fn dataset_mut(&mut self) -> &mut Dataset<Self::Data, DatasetError> {
///         &mut self.dataset
///     }
///
///     fn into_dataset(self) -> Dataset<Self::Data, DatasetError> {
///         self.dataset
///     }
/// }
/// ```
pub trait MlDataset: Sized {
    /// What this loader parses into: the module's `…Data` type alias.
    ///
    /// It must implement [`NumSamples`], which the single arrays, array pairs,
    /// and array triples every loader here produces already satisfy. This trait
    /// needs that bound up front, rather than placing it on
    /// [`n_samples`](Self::n_samples) itself.
    /// That way, a generic `fn f<D: MlDataset>(d: &D)` can call `d.n_samples()`
    /// without repeating the bound.
    type Data: NumSamples;

    /// The dataset's identifier, matching the one used in its error messages
    /// (for example, `"iris"`, `"sms_spam"`).
    const NAME: &'static str;

    /// Borrow the underlying container.
    fn dataset(&self) -> &Dataset<Self::Data, DatasetError>;

    /// Borrow the underlying container mutably.
    fn dataset_mut(&mut self) -> &mut Dataset<Self::Data, DatasetError>;

    /// Consume the loader and return the underlying container.
    fn into_dataset(self) -> Dataset<Self::Data, DatasetError>;

    /// Load the dataset if needed and borrow the parsed data.
    ///
    /// The generic equivalent of each loader's inherent `data()`: it downloads and
    /// parses on first call, then returns the cached value. Concurrent calls run
    /// the loader once and share the result.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if the download, file I/O, or parsing fails.
    fn load(&self) -> Result<&Self::Data, DatasetError> {
        self.dataset().load()
    }

    /// Load the dataset if needed and borrow the parsed data **mutably**.
    ///
    /// If you edit the data in place through the returned reference, the change
    /// persists in the cache, so later accesses observe it. Unlike the inherent
    /// `get_data_mut()`, this loads rather than returning `None` when nothing is
    /// cached yet.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if the download, file I/O, or parsing fails.
    fn load_mut(&mut self) -> Result<&mut Self::Data, DatasetError> {
        self.dataset_mut().load_mut()
    }

    /// Borrow the parsed data **without** triggering loading.
    ///
    /// The generic equivalent of each loader's inherent `get_data()`. Returns
    /// `None`, rather than downloading, when the dataset has not loaded yet.
    fn peek(&self) -> Option<&Self::Data> {
        self.dataset().get()
    }

    /// Move the parsed data out, leaving the loader reusable and unloaded.
    ///
    /// The generic equivalent of each loader's inherent `take_data()`, except that
    /// it never loads: it returns `None` if nothing is cached.
    fn unload(&mut self) -> Option<Self::Data> {
        self.dataset_mut().take()
    }

    /// Whether the cache currently holds the data.
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

    /// Drop the cached data, keeping the loader usable.
    ///
    /// The next access re-reads the file from `storage_dir`. This re-runs the
    /// SHA-256 check and the parser, and re-downloads if the file is gone or no
    /// longer matches. Use it to reclaim the memory a dataset occupies
    /// (`covtype`, `kddcup99`), or to read a file that changed on disk.
    ///
    /// To retrieve the data rather than discard it, use [`unload`](Self::unload).
    fn invalidate(&mut self) {
        self.dataset_mut().invalidate();
    }

    /// The number of samples in the dataset, loading it if needed.
    ///
    /// Reads the leading axis of the data's first array: the row count for the
    /// tabular loaders, the document count for the text ones.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if the download, file I/O, or parsing fails.
    fn n_samples(&self) -> Result<usize, DatasetError> {
        Ok(self.load()?.num_samples())
    }
}

/// Implement [`MlDataset`] for a loader that stores its container in a field named
/// `dataset`.
///
/// Every loader in this crate has that exact shape, so the implementation is
/// entirely mechanical. This macro writes the three needed methods and leaves
/// the rest to the trait's defaults. It is crate-internal. Downstream loaders
/// implement the trait directly (see [`MlDataset`]'s own example).
#[cfg(feature = "dataset")]
macro_rules! impl_ml_dataset {
    ($struct_name:ident, $data_type:ty, $name:literal) => {
        impl $crate::traits::MlDataset for $struct_name {
            type Data = $data_type;
            const NAME: &'static str = $name;

            fn dataset(
                &self,
            ) -> &::dataset_core::Dataset<Self::Data, ::dataset_core::DatasetError> {
                &self.dataset
            }

            fn dataset_mut(
                &mut self,
            ) -> &mut ::dataset_core::Dataset<Self::Data, ::dataset_core::DatasetError> {
                &mut self.dataset
            }

            fn into_dataset(
                self,
            ) -> ::dataset_core::Dataset<Self::Data, ::dataset_core::DatasetError> {
                self.dataset
            }
        }
    };
}

#[cfg(feature = "dataset")]
pub(crate) use impl_ml_dataset;
