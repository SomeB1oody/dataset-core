//! Built-in dataset loaders.
//!
//! Every module here wraps one data source in a [`Dataset`](dataset_core::Dataset).
//! Each module is a worked example of the same four steps:
//!
//! 1. Download from a URL.
//! 2. Verify a SHA-256 hash.
//! 3. Parse CSV records, or extract raw documents from an archive.
//! 4. Expose typed accessors backed by [`ndarray`].
//!
//! Every loader struct is also re-exported at the crate root, so
//! [`dataset_ml::Iris`](crate::Iris) and
//! [`dataset_ml::dataset::iris::Iris`](crate::dataset::iris::Iris) name the same
//! type. Use whichever path reads better.
//!
//! The crate root documents the samples, the features, and the task type of
//! every dataset. Each module documents its own source and column layout.

/// Abalone dataset module.
///
/// Contains the Abalone dataset (UCI, Nash et al. 1994) for **regression**. It
/// predicts an abalone's `rings` (age in years is `rings + 1.5`) from 8 mixed
/// features: 1 categorical `sex` feature and 7 numeric physical measurements.
/// Unlike the other mixed-type loaders (which are classification tasks), its
/// target is an `Array1<f64>` regression target via `targets()`.
pub mod abalone;

/// Adult / Census Income dataset module.
///
/// Contains the Adult dataset (also called "Census Income") for binary
/// classification. It predicts whether a person earns over $50K/year from 14
/// mixed features: 8 categorical and 6 numeric, covering demographic and
/// employment attributes. Extracted from the 1994 US Census. Uses the canonical
/// `adult.data` training partition.
pub mod adult;

/// Bank Marketing dataset module.
///
/// Contains the Bank Marketing dataset for binary classification. It predicts
/// whether a client subscribes to a term deposit from 16 mixed features: 9
/// categorical and 7 numeric, covering client, contact, and campaign attributes.
/// Recorded from a Portuguese bank's phone campaigns. Uses the full
/// `bank-full.csv` partition. Sourced from a ZIP archive (like `digits`).
pub mod bank_marketing;

/// Banknote Authentication dataset module.
///
/// Contains the Banknote Authentication dataset (UCI, Lohweg 2012) for binary
/// classification. It tells genuine banknote specimens from forged ones, using 4
/// continuous statistics (variance, skewness, curtosis, entropy) of
/// Wavelet-transformed banknote images. This is the crate's most compact
/// pure-numeric benchmark. Its target is the source's raw `0`/`1` code as an
/// `Array1<u8>`, because UCI does not document which code means which.
pub mod banknote_authentication;

/// Boston Housing dataset module.
///
/// Contains the Boston Housing dataset for predicting median house values
/// in Boston suburbs, based on features like crime rate, room count,
/// and accessibility to highways.
pub mod boston_housing;

/// Breast Cancer Wisconsin (Diagnostic) dataset module.
///
/// Contains the Breast Cancer Wisconsin dataset for binary classification of
/// tumors as malignant or benign. It uses 30 features computed from digitized
/// images of cell nuclei.
pub mod breast_cancer;

/// California Housing dataset module.
///
/// Contains the California Housing dataset for predicting median house values
/// in California districts. Reproduces the eight derived features of
/// scikit-learn's `fetch_california_housing`. A modern replacement for Boston
/// Housing.
pub mod california_housing;

/// Car Evaluation dataset module.
///
/// Contains the Car Evaluation dataset (UCI, Bohanec 1988) for multi-class
/// classification. It predicts a car's overall acceptability (`unacc`, `acc`,
/// `good`, `vgood`) from 6 categorical price and technical attributes. Like
/// [`mushroom`], it is **all-categorical**: `features()` returns a single
/// `Array2<String>`.
pub mod car_evaluation;

/// Forest Cover Type dataset module.
///
/// Contains the scikit-learn Forest CoverType dataset (`fetch_covtype`) for
/// multi-class classification: predicting one of seven forest cover types from 54
/// cartographic features of 30×30 metre cells. Sourced from a gzip-compressed file,
/// it is the first loader to decompress its source with `gunzip`.
pub mod covtype;

/// Diabetes dataset module.
///
/// Contains the scikit-learn diabetes dataset (`load_diabetes`) for regression:
/// predicting disease progression from 10 standardized physiological features.
pub mod diabetes;

/// Optical Recognition of Handwritten Digits dataset module.
///
/// Contains the scikit-learn digits dataset (`load_digits`) for multi-class
/// classification: recognizing handwritten digits (`0`–`9`) from 8×8 grayscale
/// images flattened into 64 integer pixel intensities.
pub mod digits;

/// Heart Disease (Cleveland) dataset module.
///
/// Contains the Cleveland Heart Disease dataset (UCI, Janosi et al. 1988) for
/// classification: predicting the presence of heart disease (`num`, `0`–`4`) from
/// 13 clinical features. The loader maps the `?` missing values in `ca`/`thal` to
/// `NaN` (like [`titanic`]/[`palmer_penguins`]). The target is an `Array1<u8>`.
pub mod heart_disease;

/// Ionosphere dataset module.
///
/// Contains the Ionosphere dataset (UCI, Sigillito et al. 1989) for binary
/// classification. It predicts whether a radar return shows structure in the
/// ionosphere (`good`) or passes through it (`bad`), from 34 continuous
/// autocorrelation features. A compact pure-numeric benchmark like
/// [`breast_cancer`].
pub mod ionosphere;

/// Iris flower dataset module.
///
/// Contains the classic Iris dataset for classifying iris flowers into
/// three species (setosa, versicolor, virginica) based on sepal and petal
/// measurements.
pub mod iris;

/// KDD Cup 1999 network-intrusion dataset module.
///
/// Contains the scikit-learn KDD Cup 1999 dataset (`fetch_kddcup99`) for
/// multi-class classification: detecting network intrusions from 41 mixed
/// (3 categorical + 38 numeric) connection features. `Kddcup99::new` loads the
/// default 10% subset (494,021 samples) and `Kddcup99::new_full` the full set
/// (4,898,431 samples). Like `covtype`, it is sourced from a gzip-compressed file
/// and decompressed with `gunzip`.
pub mod kddcup99;

/// Letter Recognition dataset module.
///
/// Contains the Letter Recognition dataset (UCI, Slate 1991) for multi-class
/// classification. It identifies which of the 26 capital letters a distorted
/// glyph shows, from 16 integer statistics of its pixel image. This is the
/// crate's widest classification problem by class count, and the only loader
/// whose label is an `Array1<char>`. A one-letter class is naturally a `char`,
/// so it needs no lookup table.
pub mod letter_recognition;

/// Linnerud dataset module.
///
/// Contains the scikit-learn Linnerud dataset (`load_linnerud`) for multi-output
/// regression. It predicts three physiological variables (`Weight`, `Waist`,
/// `Pulse`) from three exercise variables (`Chins`, `Situps`, `Jumps`), measured
/// on 20 middle-aged men.
pub mod linnerud;

/// Movie Review Polarity dataset module.
///
/// Contains the Cornell Movie Review Polarity dataset (Pang and Lee 2004,
/// polarity dataset v2.0) for binary **text** classification. It labels 2,000
/// full IMDb movie reviews as `positive` or `negative` (1,000 each). Like
/// [`sms_spam`], it is a text-modality loader (document accessor `texts()`, not
/// `features()`) and complements the sentence-level [`sentiment_sentences`] with
/// full-document reviews. Sourced from a `.tar.gz` archive (decompressed with
/// `untar_gz`).
pub mod movie_review_polarity;

/// Mushroom dataset module.
///
/// Contains the Mushroom dataset (UCI `agaricus-lepiota`) for binary
/// classification: predicting whether a mushroom is edible or poisonous from 22
/// categorical attributes. This is the first **all-categorical** loader: every
/// feature is a single-letter string code, so `features()` returns a single
/// `Array2<String>`.
pub mod mushroom;

/// 20 Newsgroups dataset module.
///
/// Contains the classic 20 Newsgroups dataset (Lang 1995, the `bydate` version)
/// for multi-class **text** classification: labeling ~18,846 Usenet posts with
/// one of 20 newsgroups. It is the framework-agnostic analogue of scikit-learn's
/// `fetch_20newsgroups`, and the crate's first **multi-class** text loader. Like
/// [`sms_spam`], it is a text-modality loader (document accessor `texts()`, not
/// `features()`). `new`/`new_test`/`new_all` mirror scikit-learn's train/test/all
/// subsets. Sourced from a `.tar.gz` archive (decompressed with `untar_gz`).
pub mod newsgroups20;

/// Palmer Penguins dataset module.
///
/// Contains the Palmer Penguins dataset for classifying penguins into three
/// species (Adelie, Chinstrap, Gentoo). It uses bill and flipper measurements,
/// body mass, and categorical island/sex features. A modern alternative to Iris.
pub mod palmer_penguins;

/// Sentiment Labelled Sentences dataset module.
///
/// Contains the Sentiment Labelled Sentences dataset (UCI, Kotzias et al. 2015)
/// for binary **text** classification. It labels 3,000 review sentences from
/// three sites (Amazon, IMDb, Yelp) as `positive` or `negative`. Like
/// [`sms_spam`] and [`youtube_spam`], it is a text-modality loader (document
/// accessor `texts()`, not `features()`). It also carries per-sample
/// **metadata**, which site each sentence came from, via a `sources()` accessor.
/// This makes `SentimentSentencesData` a `(texts, sources, labels)` triple.
/// Sourced from a ZIP archive of three per-site files.
pub mod sentiment_sentences;

/// SMS Spam Collection dataset module.
///
/// Contains the SMS Spam Collection dataset (UCI, Almeida and Hidalgo 2011) for
/// binary **text** classification: labeling 5,574 SMS messages as `ham` or
/// `spam`. This is the crate's first text-modality loader. There is no feature
/// matrix, so the document accessor is `texts()` (an `Array1<String>` of raw
/// messages) rather than `features()`. Sourced from a ZIP archive.
pub mod sms_spam;

/// Spambase dataset module.
///
/// Contains the Spambase dataset (UCI, Hopkins et al. 1999) for binary
/// classification. It labels 4,601 emails as `ham` or `spam` from 57 numeric
/// features: word and character frequencies, plus capital-run-length statistics.
/// This is the feature-engineered counterpart to the crate's raw-text spam
/// corpora ([`sms_spam`], [`youtube_spam`]). Those loaders leave vectorization to
/// you, but Spambase already does it, so it drops straight into a numeric model.
pub mod spambase;

/// Titanic dataset module.
///
/// Contains data about Titanic passengers for predicting survival based
/// on features like passenger class, sex, age, and fare.
pub mod titanic;

/// Wine Quality dataset module.
///
/// Contains wine quality assessment data for predicting quality scores
/// based on physicochemical properties like acidity, sugar content, and
/// alcohol percentage.
pub mod wine_quality;

/// Wine Recognition dataset module.
///
/// Contains the scikit-learn Wine recognition dataset for classifying wines
/// into three cultivars based on 13 chemical constituents. Distinct from
/// [`wine_quality`], which is a regression task on quality scores.
pub mod wine_recognition;

/// YouTube Spam Collection dataset module.
///
/// Contains the YouTube Spam Collection dataset (UCI, Alberto, Lochter, and
/// Almeida 2017) for binary **text** classification. It labels 1,956 comments
/// from five popular music videos as `ham` or `spam`. Like [`sms_spam`] (a
/// sibling by the same authors), it is a text-modality loader. There is no
/// feature matrix, so the document accessor is `texts()` (an `Array1<String>` of
/// raw comments) rather than `features()`. Sourced from a ZIP archive of five
/// per-video CSVs.
pub mod youtube_spam;
