//! Built-in dataset loaders.
//!
//! Every module here wraps one data source in a [`Dataset`](dataset_core::Dataset).
//! Each module is a worked example of the same four steps:
//!
//! 1. Download from a URL.
//! 2. Verify a SHA-256 hash.
//! 3. Parse the source: CSV records, raw documents from an archive, or binary IDX images.
//! 4. Return a [`Table`](crate::table::Table) of named, typed columns.
//!
//! The crate root also re-exports every loader struct, so
//! [`dataset_ml::Iris`](crate::Iris) and
//! [`dataset_ml::dataset::iris::Iris`](crate::dataset::iris::Iris) name the same
//! type. Use whichever path reads better.
//!
//! This module needs the `dataset` feature, which is on by default.
//!
//! # Datasets
//!
//! | Module | Samples | Features | Task Type |
//! |--------|---------|----------|-----------|
//! | [`abalone`](crate::dataset::abalone) | 4,177 | 8 | Regression |
//! | [`adult`](crate::dataset::adult) | 32,561 | 14 | Classification |
//! | [`bank_marketing`](crate::dataset::bank_marketing) | 45,211 | 16 | Classification |
//! | [`banknote_authentication`](crate::dataset::banknote_authentication) | 1,372 | 4 | Classification |
//! | [`bike_sharing_hourly`](crate::dataset::bike_sharing::bike_sharing_hourly) | 17,379 | 12 | Regression (multi-output) |
//! | [`bike_sharing_daily`](crate::dataset::bike_sharing::bike_sharing_daily) | 731 | 11 | Regression (multi-output) |
//! | [`iris`](crate::dataset::iris) | 150 | 4 | Classification |
//! | [`breast_cancer`](crate::dataset::breast_cancer) | 569 | 30 | Classification |
//! | [`boston_housing`](crate::dataset::boston_housing) | 506 | 13 | Regression |
//! | [`california_housing`](crate::dataset::california_housing) | 20,640 | 8 | Regression |
//! | [`car_evaluation`](crate::dataset::car_evaluation) | 1,728 | 6 | Classification |
//! | [`covtype`](crate::dataset::covtype) | 581,012 | 54 | Classification |
//! | [`diabetes`](crate::dataset::diabetes) | 442 | 10 | Regression |
//! | [`digits`](crate::dataset::digits) | 1,797 | 64 | Classification |
//! | [`fashion_mnist`](crate::dataset::fashion_mnist) | 60,000 / 10,000 / 70,000 | 784 (28×28 pixels) | Classification (10 classes) |
//! | [`heart_disease`](crate::dataset::heart_disease) | 303 | 13 | Classification |
//! | [`ionosphere`](crate::dataset::ionosphere) | 351 | 34 | Classification |
//! | [`kddcup99`](crate::dataset::kddcup99) | 494,021 / 4,898,431 | 41 | Classification |
//! | [`letter_recognition`](crate::dataset::letter_recognition) | 20,000 | 16 | Classification (26 classes) |
//! | [`linnerud`](crate::dataset::linnerud) | 20 | 3 | Regression (multi-output) |
//! | [`mnist`](crate::dataset::mnist) | 60,000 / 10,000 / 70,000 | 784 (28×28 pixels) | Classification (10 classes) |
//! | [`movielens_100k`](crate::dataset::movielens_100k) | 100,000 ratings | 943 users × 1,682 movies | Recommendation |
//! | [`mushroom`](crate::dataset::mushroom) | 8,124 | 22 | Classification |
//! | [`spambase`](crate::dataset::spambase) | 4,601 | 57 | Classification |
//! | [`titanic`](crate::dataset::titanic) | 891 | 11 | Classification |
//! | [`palmer_penguins`](crate::dataset::palmer_penguins) | 344 | 7 | Classification |
//! | [`sms_spam`](crate::dataset::sms_spam) | 5,574 | text | Classification |
//! | [`wholesale_customers`](crate::dataset::wholesale_customers) | 440 | 8 | Clustering (no target) |
//! | [`wine_recognition`](crate::dataset::wine_recognition) | 178 | 13 | Classification |
//! | [`red_wine_quality`](crate::dataset::wine_quality::red_wine_quality) | 1,599 | 11 | Regression |
//! | [`white_wine_quality`](crate::dataset::wine_quality::white_wine_quality) | 4,898 | 11 | Regression |
//! | [`youtube_spam`](crate::dataset::youtube_spam) | 1,956 | text | Classification |
//! | [`sentiment_sentences`](crate::dataset::sentiment_sentences) | 3,000 | text | Classification |
//! | [`newsgroups20`](crate::dataset::newsgroups20) | 11,314 / 18,846 | text | Classification |
//! | [`movie_review_polarity`](crate::dataset::movie_review_polarity) | 2,000 | text | Classification |
//!
//! Each module documents its own source and column layout.

/// Reader for the IDX binary format, shared by [`mnist`] and [`fashion_mnist`].
///
/// The two datasets ship the same four-file layout and the same 28×28 image
/// shape. This module is internal to the crate, unlike every other module here.
mod idx;

/// Abalone dataset module.
///
/// Contains the Abalone dataset (UCI, Nash et al. 1994) for **regression**. It
/// predicts an abalone's `rings` (age in years is `rings + 1.5`) from 8 mixed
/// features: 1 categorical `sex` feature and 7 numeric physical measurements.
/// `Abalone::FEATURE_NAMES` names the 8 inputs. Unlike the other mixed-type
/// loaders, which are classification tasks, its target column
/// `Abalone::TARGET` holds numeric values.
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
/// pure-numeric benchmark. Its target column `BanknoteAuthentication::TARGET`
/// holds the source's raw `0`/`1` code, because UCI does not document which code
/// means which.
pub mod banknote_authentication;

/// Bike Sharing dataset module.
///
/// Contains the Bike Sharing dataset (UCI, Fanaee-T 2013) for **regression**:
/// predicting the rental count of the Capital Bikeshare system in Washington,
/// D.C., over 2011 and 2012 from the calendar attributes and the weather. Each
/// sample carries its calendar date in the `dteday` column, and the rows stay in
/// chronological order, so a split by time is possible. One ZIP archive holds
/// two aggregations of the same rental log, and each one has its own loader:
/// `bike_sharing_hourly::BikeSharingHourly` (17,379 records) and
/// `bike_sharing_daily::BikeSharingDaily` (731 records). Both use the
/// multi-output target `(casual, registered, cnt)`, which each loader names in
/// its own `TARGET_NAMES` constant.
pub mod bike_sharing;

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
/// scikit-learn's `fetch_california_housing`.
pub mod california_housing;

/// Car Evaluation dataset module.
///
/// Contains the Car Evaluation dataset (UCI, Bohanec 1988) for multi-class
/// classification. It predicts a car's overall acceptability (`unacc`, `acc`,
/// `good`, `vgood`) from 6 categorical price and technical attributes. Like
/// [`mushroom`], it is **all-categorical**: every feature column is
/// `ColumnData::String`.
pub mod car_evaluation;

/// Forest Cover Type dataset module.
///
/// Contains the scikit-learn Forest CoverType dataset (`fetch_covtype`) for
/// multi-class classification: predicting one of seven forest cover types from 54
/// cartographic features of 30×30 meter cells. Its source is a gzip-compressed
/// file. The `Cover_Type` column holds the source's codes `1` to `7`, and the
/// `Covtype::CLASS_NAMES` constant names each one.
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

/// Fashion-MNIST dataset module.
///
/// Contains the Fashion-MNIST dataset (Zalando, Xiao et al. 2017) for
/// multi-class classification: recognizing one of 10 garment classes from a
/// 28×28 grayscale image. It matches [`mnist`] in image size, class count,
/// partition sizes, and IDX file format, but its classes overlap more. They are
/// exactly balanced: 6,000 images per class for training and 1,000 for test.
/// Like `mnist`, it holds one `pixels` column of `ColumnData::Bytes`, plus a
/// `FashionMnist::CLASS_NAMES` constant that names each label code. It offers
/// `new`/`new_test`/`new_all` subset constructors.
pub mod fashion_mnist;

/// Heart Disease (Cleveland) dataset module.
///
/// Contains the Cleveland Heart Disease dataset (UCI, Janosi et al. 1988) for
/// classification: predicting the presence of heart disease (`num`, `0`–`4`) from
/// 13 clinical features. The loader maps the `?` missing values in `ca`/`thal` to
/// `NaN` (like [`titanic`]/[`palmer_penguins`]). `HeartDisease::TARGET` names the
/// `num` column, which holds one integer code per patient.
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
/// (4,898,431 samples). Like `covtype`, the loader downloads a gzip-compressed
/// file and decompresses it with `gunzip`.
pub mod kddcup99;

/// Letter Recognition dataset module.
///
/// Contains the Letter Recognition dataset (UCI, Slate 1991) for multi-class
/// classification. It identifies which of the 26 capital letters a distorted
/// glyph shows, from 16 integer statistics of its pixel image. This is the
/// crate's widest classification problem by class count. Its label column
/// `LetterRecognition::TARGET` holds one capital letter per sample, as a
/// one-character string, so it needs no lookup table.
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
/// [`sms_spam`], its document column is `ColumnData::String`. It complements the
/// sentence-level [`sentiment_sentences`] with full-document reviews. Sourced
/// from a `.tar.gz` archive (decompressed with `untar_gz`).
pub mod movie_review_polarity;

/// MNIST handwritten digits dataset module.
///
/// Contains the MNIST database (LeCun et al. 1998) for multi-class
/// classification: recognizing a handwritten digit (`0`-`9`) from a 28×28
/// grayscale image. Its source is the binary IDX format, read from four
/// gzip-compressed files. One `pixels` column holds `ColumnData::Bytes` of shape
/// `(n_samples, 784)`, which a `(n_samples, 28, 28)` view reads at no copy. It
/// offers `new`/`new_test`/`new_all` subset constructors.
pub mod mnist;

/// MovieLens 100K dataset module.
///
/// Contains the MovieLens 100K dataset (GroupLens, Harper & Konstan 2015) for
/// **recommendation**: 100,000 ratings that 943 users gave to 1,682 movies
/// between September 1997 and April 1998. One sample is one rating. The table
/// holds the `user_id` and `item_id` columns, the `rating` column
/// (`MovieLens100k::TARGET`), and a `timestamp` column of Unix seconds.
/// `MovieLens100k::N_USERS` and `MovieLens100k::N_ITEMS` give the two identifier
/// ranges. GroupLens permits research use under conditions that this crate's MIT
/// license does not cover, so read the struct docs before you use the data.
pub mod movielens_100k;

/// Mushroom dataset module.
///
/// Contains the Mushroom dataset (UCI `agaricus-lepiota`) for binary
/// classification: predicting whether a mushroom is edible or poisonous from 22
/// categorical attributes. It is **all-categorical**: every feature is a
/// single-letter string code in a `ColumnData::String` column.
pub mod mushroom;

/// 20 Newsgroups dataset module.
///
/// Contains the classic 20 Newsgroups dataset (Lang 1995, the `bydate` version)
/// for multi-class **text** classification: labeling ~18,846 Usenet posts with
/// one of 20 newsgroups. It is the framework-agnostic analogue of scikit-learn's
/// `fetch_20newsgroups`. It is the **multi-class** text corpus, with 20 classes.
/// Like [`sms_spam`], its document column is `ColumnData::String`.
/// `new`/`new_test`/`new_all` mirror scikit-learn's train/test/all subsets.
/// Sourced from a `.tar.gz` archive (decompressed with `untar_gz`).
pub mod newsgroups20;

/// Palmer Penguins dataset module.
///
/// Contains the Palmer Penguins dataset for classifying penguins into three
/// species (Adelie, Chinstrap, Gentoo). It uses bill and flipper measurements,
/// body mass, and categorical island and sex features.
pub mod palmer_penguins;

/// Sentiment Labelled Sentences dataset module.
///
/// Contains the Sentiment Labelled Sentences dataset (UCI, Kotzias et al. 2015)
/// for binary **text** classification. It labels 3,000 review sentences from
/// three sites (Amazon, IMDb, Yelp) as `positive` or `negative`. Like
/// [`sms_spam`] and [`youtube_spam`], it is a text-modality loader: its document
/// column is `ColumnData::String`. A third column, `source`, names the review
/// site of each sentence. Sourced from a ZIP archive of three per-site files.
pub mod sentiment_sentences;

/// SMS Spam Collection dataset module.
///
/// Contains the SMS Spam Collection dataset (UCI, Almeida and Hidalgo 2011) for
/// binary **text** classification: labeling 5,574 SMS messages as `ham` or
/// `spam`. Its document column is `ColumnData::String`, one raw message per
/// sample. Sourced from a ZIP archive.
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

/// Wholesale Customers dataset module.
///
/// Contains the Wholesale Customers dataset (UCI, Cardoso 2013) for
/// **clustering**: segmenting 440 clients of a Portuguese wholesale distributor
/// by their annual spending across six product categories, plus a sales-channel
/// code and a region code. The dataset has **no target column**, so the loader
/// has no target constant: all 8 columns are numeric features.
/// `WholesaleCustomers::COLUMN_NAMES` names them in source order.
pub mod wholesale_customers;

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
/// sibling by the same authors), it is a text-modality loader: its document
/// column is `ColumnData::String`, one raw comment per sample. Sourced from a ZIP
/// archive of five per-video CSVs.
pub mod youtube_spam;
