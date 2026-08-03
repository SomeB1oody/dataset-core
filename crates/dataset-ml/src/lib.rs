//! Built-in dataset implementations for machine learning.
//!
//! `dataset-ml` provides ready-to-use loaders for classic ML datasets, built on top
//! of [`dataset_core::Dataset`]. Every loader lives in a module under [`dataset`],
//! and each one is a worked example that shows how to wrap `Dataset<T, E>` for one
//! data source. The steps are the same each time:
//!
//! 1. Download from a URL.
//! 2. Verify a SHA-256 hash.
//! 3. Parse CSV records, or extract raw documents from an archive.
//! 4. Expose typed accessors backed by [`ndarray`].
//!
//! # Datasets
//!
//! | Module                                                | Samples | Features | Task Type      |
//! |-------------------------------------------------------|---------|----------|----------------|
//! | [`abalone`](dataset::abalone)                         | 4,177   | 8        | Regression     |
//! | [`adult`](dataset::adult)                             | 32,561  | 14       | Classification |
//! | [`bank_marketing`](dataset::bank_marketing)           | 45,211  | 16       | Classification |
//! | [`banknote_authentication`](dataset::banknote_authentication) | 1,372 | 4 | Classification |
//! | [`iris`](dataset::iris)                               | 150     | 4        | Classification |
//! | [`breast_cancer`](dataset::breast_cancer)             | 569     | 30       | Classification |
//! | [`boston_housing`](dataset::boston_housing)           | 506     | 13       | Regression     |
//! | [`california_housing`](dataset::california_housing)   | 20,640  | 8        | Regression     |
//! | [`car_evaluation`](dataset::car_evaluation)           | 1,728   | 6        | Classification |
//! | [`covtype`](dataset::covtype)                         | 581,012 | 54       | Classification |
//! | [`diabetes`](dataset::diabetes)                       | 442     | 10       | Regression     |
//! | [`digits`](dataset::digits)                           | 1,797   | 64       | Classification |
//! | [`heart_disease`](dataset::heart_disease)             | 303     | 13       | Classification |
//! | [`ionosphere`](dataset::ionosphere)                   | 351     | 34       | Classification |
//! | [`kddcup99`](dataset::kddcup99)                       | 494,021 / 4,898,431 | 41 | Classification |
//! | [`letter_recognition`](dataset::letter_recognition)   | 20,000  | 16       | Classification (26 classes) |
//! | [`linnerud`](dataset::linnerud)                       | 20      | 3        | Regression (multi-output) |
//! | [`mushroom`](dataset::mushroom)                       | 8,124   | 22       | Classification |
//! | [`spambase`](dataset::spambase)                       | 4,601   | 57       | Classification |
//! | [`titanic`](dataset::titanic)                         | 891     | 11       | Classification |
//! | [`palmer_penguins`](dataset::palmer_penguins)         | 344     | 7        | Classification |
//! | [`sms_spam`](dataset::sms_spam)                       | 5,574   | text     | Classification |
//! | [`wine_recognition`](dataset::wine_recognition)       | 178     | 13       | Classification |
//! | [`wine_quality::red_wine_quality`](dataset::wine_quality::red_wine_quality) | 1,599 | 11 | Regression |
//! | [`wine_quality::white_wine_quality`](dataset::wine_quality::white_wine_quality) | 4,898 | 11 | Regression |
//! | [`youtube_spam`](dataset::youtube_spam)               | 1,956   | text     | Classification |
//! | [`sentiment_sentences`](dataset::sentiment_sentences) | 3,000   | text     | Classification |
//! | [`newsgroups20`](dataset::newsgroups20)               | 11,314 / 18,846 | text | Classification |
//! | [`movie_review_polarity`](dataset::movie_review_polarity) | 2,000 | text   | Classification |
//!
//! # Example
//!
//! ```no_run
//! use dataset_ml::Iris;
//!
//! let iris = Iris::new("./data");
//! let (features, labels) = iris.data().unwrap();
//! assert_eq!(features.shape(), &[150, 4]);
//! ```
//!
//! Every loader struct is re-exported at the crate root, so `dataset_ml::Iris` and
//! `dataset_ml::dataset::iris::Iris` name the same type. Use whichever path reads
//! better.
//!
//! All loaders are lazy: the first call downloads and parses the file, every
//! subsequent call returns a cached reference. See the individual module docs
//! for features, target, sample count, and source.
//!
//! # Beyond the loaders
//!
//! Two modules apply to every dataset here rather than to one of them:
//!
//! - [`preprocessing`]: seeded train/test and k-fold splits (plain or
//!   class-stratified), feature scaling, one-hot encoding, and label encoding. You
//!   can feed the arrays a loader returns straight to a model, without writing
//!   that glue code by hand.
//! - [`traits`]: the [`MlDataset`] trait every loader implements. It lets you
//!   write code generically over "some dataset": cache inspection and
//!   invalidation, plus a uniform `n_samples()`.
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

/// How many extra download attempts every loader in this crate makes before it
/// stops retrying.
///
/// The datasets are hosted on university archives and personal pages that
/// intermittently time out or reset a connection. A run that fails for that
/// reason is not a bug in the data. Every loader therefore fetches through
/// [`download_to_with_retries`](dataset_core::download_to_with_retries) with this
/// many retries, waiting 500 ms then 1 s between attempts. A single blip does
/// not surface as a `DownloadError`.
///
/// The loader returns errors that retrying cannot fix right away, and a
/// genuinely unreachable host costs at most 1.5 s of waiting before it fails.
pub const DOWNLOAD_RETRIES: u32 = 2;

/// Every built-in dataset loader, one module per data source.
///
/// The modules run from [`iris`](dataset::iris), the smallest, to
/// [`kddcup99`](dataset::kddcup99), the largest. Each one documents its features,
/// target, sample count, and source.
///
/// The crate root re-exports every loader struct, so `dataset_ml::Iris` is a
/// shorter name for `dataset_ml::dataset::iris::Iris`.
pub mod dataset;

/// Preprocessing helpers.
///
/// Turns what the loaders return into what a model consumes. It covers seeded
/// train/test and k-fold splits (plain or class-stratified), feature scaling,
/// one-hot encoding of the categorical matrices, and label encoding. Everything
/// is deterministic given a seed and depends on no extra crates.
pub mod preprocessing;

/// The [`traits::MlDataset`] trait implemented by every loader in this crate.
///
/// Provides the container operations that stay the same whatever a loader parses
/// into: lazy access, cache inspection, cache invalidation, and a uniform sample
/// count. With it, you can write code generically over "some dataset" instead of
/// one concrete struct.
pub mod traits;

pub use dataset::abalone::Abalone;
pub use dataset::adult::Adult;
pub use dataset::bank_marketing::BankMarketing;
pub use dataset::banknote_authentication::BanknoteAuthentication;
pub use dataset::boston_housing::BostonHousing;
pub use dataset::breast_cancer::BreastCancer;
pub use dataset::california_housing::CaliforniaHousing;
pub use dataset::car_evaluation::CarEvaluation;
pub use dataset::covtype::Covtype;
pub use dataset::diabetes::Diabetes;
pub use dataset::digits::Digits;
pub use dataset::heart_disease::HeartDisease;
pub use dataset::ionosphere::Ionosphere;
pub use dataset::iris::Iris;
pub use dataset::kddcup99::Kddcup99;
pub use dataset::letter_recognition::LetterRecognition;
pub use dataset::linnerud::Linnerud;
pub use dataset::movie_review_polarity::MovieReviewPolarity;
pub use dataset::mushroom::Mushroom;
pub use dataset::newsgroups20::Newsgroups20;
pub use dataset::palmer_penguins::PalmerPenguins;
pub use dataset::sentiment_sentences::SentimentSentences;
pub use dataset::sms_spam::SmsSpam;
pub use dataset::spambase::Spambase;
pub use dataset::titanic::Titanic;
pub use dataset::wine_quality::{
    red_wine_quality::RedWineQuality, white_wine_quality::WhiteWineQuality,
};
pub use dataset::wine_recognition::WineRecognition;
pub use dataset::youtube_spam::YoutubeSpam;
pub use traits::{MlDataset, NumSamples};
