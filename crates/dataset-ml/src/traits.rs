//! The [`MlDataset`] trait shared by every loader in this crate.
//!
//! Each loader has its own inherent accessors named for what it actually holds —
//! `features()`/`labels()` for the tabular ones, `targets()` for the regression
//! ones, `texts()` for the text corpora. Those names are the reason the loaders are
//! pleasant to use directly, but they are also why nothing could be written
//! *generically* over a dataset before this trait existed.
//!
//! [`MlDataset`] is the common denominator: the container operations that are the
//! same whatever the loader parses into. It adds three capabilities the inherent
//! APIs never exposed —
//!
//! - [`invalidate`](MlDataset::invalidate), to drop the in-memory cache and force
//!   the next access to re-read (and re-verify) the file on disk;
//! - [`is_loaded`](MlDataset::is_loaded) and [`storage_dir`](MlDataset::storage_dir),
//!   to inspect a loader without touching the data;
//! - [`n_samples`](MlDataset::n_samples), a uniform sample count that works across
//!   the pair-shaped and triple-shaped datasets alike.
//!
//! The trait's data accessors are deliberately named [`load`](MlDataset::load) /
//! [`peek`](MlDataset::peek) / [`unload`](MlDataset::unload) rather than reusing
//! `data` / `get_data` / `take_data`, so a trait method never silently shadows —
//! or is shadowed by — the inherent method of the same name. Both sets are always
//! available and always agree; use whichever reads better where you are.
//!
//! # Example
//!
//! ```no_run
//! use dataset_ml::traits::MlDataset;
//! use dataset_ml::{Iris, SmsSpam};
//!
//! // One function, any loader — including the text corpora, whose data has an
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
/// Implemented for the array pairs and triples every loader in this crate parses
/// into — `(features, labels)`, `(features, targets)`, `(texts, labels)`,
/// `(categorical, numeric, labels)`, `(texts, sources, labels)`. In all of them the
/// first array's leading axis is the sample axis, so that is what gets counted.
///
/// You only need this trait directly to call [`MlDataset::n_samples`] in a generic
/// function; implement it for your own data type if you want the same from a
/// loader of your own.
pub trait NumSamples {
    /// The number of samples the parsed data holds.
    fn num_samples(&self) -> usize;
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

/// The lazy-loading behaviour every dataset loader in this crate shares.
///
/// Implementors wrap a [`Dataset<Self::Data, DatasetError>`](dataset_core::Dataset)
/// and only have to hand it out through the three required methods; everything
/// else is provided.
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
    /// What this loader parses into — the module's `…Data` type alias.
    ///
    /// It must implement [`NumSamples`], which the array pairs and triples every
    /// loader here produces already do. Requiring it up front (rather than as a
    /// bound on [`n_samples`](Self::n_samples)) is what lets a generic
    /// `fn f<D: MlDataset>(d: &D)` call `d.n_samples()` without repeating the
    /// bound itself.
    type Data: NumSamples;

    /// The dataset's identifier, matching the one used in its error messages
    /// (e.g. `"iris"`, `"sms_spam"`).
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
    /// Edits are made in place and persist in the cache, so later accesses observe
    /// them. Unlike the inherent `get_data_mut()`, this loads rather than returning
    /// `None` when nothing is cached yet.
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
    /// `None` — rather than downloading — when the dataset is not loaded yet.
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

    /// Whether the data is currently held in memory.
    ///
    /// Never triggers loading, so this is the cheap way to ask whether an accessor
    /// would return instantly or start a download.
    fn is_loaded(&self) -> bool {
        self.dataset().is_loaded()
    }

    /// The directory this loader stores its files in.
    fn storage_dir(&self) -> &str {
        self.dataset().storage_dir()
    }

    /// Drop the cached data, keeping the loader usable.
    ///
    /// The next access re-reads the file from `storage_dir` — re-running the
    /// SHA-256 check and the parser, and re-downloading if the file is gone or no
    /// longer matches. Use it to reclaim the memory a large dataset occupies
    /// (`covtype`, `kddcup99`) or to pick up a file that changed on disk.
    ///
    /// To get the data back rather than discard it, use [`unload`](Self::unload).
    fn invalidate(&mut self) {
        self.dataset_mut().invalidate();
    }

    /// The number of samples in the dataset, loading it if needed.
    ///
    /// Reads the leading axis of the data's first array, so it is the row count for
    /// the tabular loaders and the document count for the text ones.
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
/// entirely mechanical — this macro writes the three required methods and leaves
/// the rest to the trait's defaults. It is crate-internal; downstream loaders
/// implement the trait directly (see [`MlDataset`]'s own example).
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

pub(crate) use impl_ml_dataset;
