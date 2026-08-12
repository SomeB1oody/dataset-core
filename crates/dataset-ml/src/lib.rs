//! Built-in dataset implementations for machine learning.
//!
//! `dataset-ml` provides ready-to-use loaders for classic ML datasets, built on top
//! of [`dataset_core::Dataset`]. Every loader lives in a module under `dataset`.
//! Each one is a worked example that shows how to wrap `Dataset<T, E>` for one
//! data source. The steps are the same each time:
//!
//! 1. Download from a URL.
//! 2. Verify a SHA-256 hash.
//! 3. Parse the source: CSV records, raw documents from an archive, or binary IDX images.
//! 4. Expose typed accessors backed by [`ndarray`].
//!
//! # Feature flags
//!
//! Both features are on by default. Turn one off to leave out what you do not use.
//!
//! | Feature         | What it enables |
//! |-----------------|-----------------|
//! | `dataset`       | The `dataset` module and its loaders, the crate-root re-export of every loader struct, and `DOWNLOAD_RETRIES`. It adds the `csv`, `serde`, and `tempfile` dependencies. |
//! | `preprocessing` | The `preprocessing` module: seeded train/test and k-fold splits, feature scaling, one-hot encoding, and label encoding. It adds no dependencies. |
//!
//! The `traits` module is always available, whichever features you pick. It holds
//! [`MlDataset`] and [`NumSamples`], so you can write a loader of your own against
//! the same interface with both features off.
//!
//! The `dataset` module lists every built-in dataset with its sample count,
//! feature count, and task type.
//!
//! # Example
//!
//! This example needs the `dataset` feature.
//!
//! ```no_run
//! use dataset_ml::Iris;
//!
//! let iris = Iris::new("./data");
//! let (features, labels) = iris.data().unwrap();
//! assert_eq!(features.shape(), &[150, 4]);
//! ```
//!
//! The crate root re-exports every loader struct, so `dataset_ml::Iris` and
//! `dataset_ml::dataset::iris::Iris` name the same type. Use whichever path reads
//! better.
//!
//! All loaders are lazy. The first call downloads and parses the file. Every
//! later call returns a cached reference. See the individual module docs for
//! features, target, sample count, and source.
//!
//! # Beyond the loaders
//!
//! Two modules apply to every dataset here rather than to one of them:
//!
//! - `preprocessing`: seeded train/test and k-fold splits (plain or
//!   class-stratified), feature scaling, one-hot encoding, and label encoding. You
//!   can feed the arrays a loader returns straight to a model, without writing
//!   that glue code by hand.
//! - [`traits`]: the [`MlDataset`] trait every loader implements. It lets you
//!   write code generically over "some dataset": cache inspection and
//!   invalidation, plus a uniform `n_samples()`.
//!
//! This example needs the `dataset` and `preprocessing` features.
//!
//! ```no_run
//! use dataset_ml::preprocessing::{stratified_split, standardize};
//! use dataset_ml::traits::MlDataset;
//! use dataset_ml::Iris;
//! use ndarray::Axis;
//!
//! let iris = Iris::new("./data");
//! let (features, labels) = iris.data().unwrap();
//!
//! // Split with each species proportionally represented on both sides.
//! let (train, test) = stratified_split(labels.as_slice().unwrap(), 0.2, 42).unwrap();
//! let (scaled_train, scaler) = standardize(&features.select(Axis(0), &train)).unwrap();
//!
//! assert_eq!(scaled_train.nrows(), 120);
//! assert_eq!(iris.n_samples().unwrap(), 150); // from the `MlDataset` trait
//! ```

/// How many extra download tries every loader in this crate makes before it
/// stops retrying.
///
/// The datasets are hosted on university archives and personal pages that
/// intermittently time out or reset a connection. A run that fails for that
/// reason is not a bug in the data. Every loader therefore fetches through
/// [`download_to_with_retries`](dataset_core::download_to_with_retries) with
/// this many retries. It waits 500 ms, then 1 s, between tries. A single blip
/// does not surface as a `DownloadError`.
///
/// The loader returns errors that retrying cannot fix right away. A genuinely
/// unreachable host costs at most 1.5 s of waiting before it fails.
#[cfg(feature = "dataset")]
pub const DOWNLOAD_RETRIES: u32 = 2;

/// Every built-in dataset loader, one module per data source.
///
/// The modules run from [`iris`](dataset::iris), the smallest, to
/// [`kddcup99`](dataset::kddcup99), the largest. Each one documents its features,
/// target, sample count, and source.
///
/// The crate root re-exports every loader struct, so `dataset_ml::Iris` is a
/// shorter name for `dataset_ml::dataset::iris::Iris`.
///
/// Needs the `dataset` feature, which is on by default.
#[cfg(feature = "dataset")]
pub mod dataset;

/// Preprocessing helpers.
///
/// Turns what the loaders return into what a model consumes. It covers seeded
/// train/test and k-fold splits (plain or class-stratified), feature scaling,
/// one-hot encoding of the categorical matrices, and label encoding. Everything
/// is deterministic given a seed and depends on no extra crates.
///
/// Needs the `preprocessing` feature, which is on by default.
#[cfg(feature = "preprocessing")]
pub mod preprocessing;

/// The [`traits::MlDataset`] trait implemented by every loader in this crate.
///
/// Provides the container operations that stay the same whatever a loader parses
/// into: lazy access, cache inspection, cache invalidation, and a uniform sample
/// count. With it, you can write code generically over "some dataset" instead of
/// one concrete struct.
pub mod traits;

#[cfg(feature = "dataset")]
pub use dataset::{
    abalone::Abalone, adult::Adult, bank_marketing::BankMarketing,
    banknote_authentication::BanknoteAuthentication,
    bike_sharing::bike_sharing_daily::BikeSharingDaily,
    bike_sharing::bike_sharing_hourly::BikeSharingHourly, boston_housing::BostonHousing,
    breast_cancer::BreastCancer, california_housing::CaliforniaHousing,
    car_evaluation::CarEvaluation, covtype::Covtype, diabetes::Diabetes, digits::Digits,
    fashion_mnist::FashionMnist, heart_disease::HeartDisease, ionosphere::Ionosphere, iris::Iris,
    kddcup99::Kddcup99, letter_recognition::LetterRecognition, linnerud::Linnerud, mnist::Mnist,
    movie_review_polarity::MovieReviewPolarity, mushroom::Mushroom, newsgroups20::Newsgroups20,
    palmer_penguins::PalmerPenguins, sentiment_sentences::SentimentSentences, sms_spam::SmsSpam,
    spambase::Spambase, titanic::Titanic, wholesale_customers::WholesaleCustomers,
    wine_quality::red_wine_quality::RedWineQuality,
    wine_quality::white_wine_quality::WhiteWineQuality, wine_recognition::WineRecognition,
    youtube_spam::YoutubeSpam,
};
pub use traits::{MlDataset, NumSamples};
