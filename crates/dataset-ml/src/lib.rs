//! Built-in dataset implementations for machine learning.
//!
//! `dataset-ml` provides ready-to-use loaders for classic ML datasets built on top
//! of [`dataset_core::Dataset`]. Each module is a worked example showing how to wrap
//! `Dataset<T, E>` for a concrete data source: downloading from a URL, verifying a
//! SHA-256 hash, parsing CSV records, and exposing typed accessors backed by
//! [`ndarray`].
//!
//! # Datasets
//!
//! | Module                                                | Samples | Features | Task Type      |
//! |-------------------------------------------------------|---------|----------|----------------|
//! | [`abalone`]                                           | 4,177   | 8        | Regression     |
//! | [`adult`]                                             | 32,561  | 14       | Classification |
//! | [`bank_marketing`]                                    | 45,211  | 16       | Classification |
//! | [`iris`]                                              | 150     | 4        | Classification |
//! | [`breast_cancer`]                                     | 569     | 30       | Classification |
//! | [`boston_housing`]                                    | 506     | 13       | Regression     |
//! | [`california_housing`]                                | 20,640  | 8        | Regression     |
//! | [`car_evaluation`]                                    | 1,728   | 6        | Classification |
//! | [`covtype`]                                           | 581,012 | 54       | Classification |
//! | [`diabetes`]                                          | 442     | 10       | Regression     |
//! | [`digits`]                                            | 1,797   | 64       | Classification |
//! | [`heart_disease`]                                     | 303     | 13       | Classification |
//! | [`ionosphere`]                                        | 351     | 34       | Classification |
//! | [`kddcup99`]                                          | 494,021 / 4,898,431 | 41 | Classification |
//! | [`linnerud`]                                          | 20      | 3        | Regression     |
//! | [`mushroom`]                                          | 8,124   | 22       | Classification |
//! | [`titanic`]                                           | 891     | 11       | Classification |
//! | [`palmer_penguins`]                                   | 344     | 7        | Classification |
//! | [`sms_spam`]                                          | 5,574   | text     | Classification |
//! | [`wine_recognition`]                                  | 178     | 13       | Classification |
//! | [`wine_quality::red_wine_quality`]                    | 1,599   | 11       | Regression     |
//! | [`wine_quality::white_wine_quality`]                  | 4,898   | 11       | Regression     |
//! | [`youtube_spam`]                                      | 1,956   | text     | Classification |
//!
//! # Example
//!
//! ```no_run
//! use dataset_ml::iris::Iris;
//!
//! let iris = Iris::new("./data");
//! let (features, labels) = iris.data().unwrap();
//! assert_eq!(features.shape(), &[150, 4]);
//! ```
//!
//! All loaders are lazy: the first call downloads and parses the file, every
//! subsequent call returns a cached reference. See the individual module docs
//! for features, target, sample count, and source.

/// Abalone dataset module.
///
/// Contains the Abalone dataset (UCI, Nash et al. 1994) for **regression**:
/// predicting an abalone's `rings` (age in years is `rings + 1.5`) from 8 mixed
/// (1 categorical `sex` + 7 numeric) physical measurements. Unlike the other
/// mixed-type loaders (which are classification), its target is an
/// `Array1<f64>` regression target via `targets()`.
pub mod abalone;

/// Adult / Census Income dataset module.
///
/// Contains the Adult dataset (also called "Census Income") for binary
/// classification: predicting whether a person earns over $50K/year from 14 mixed
/// (8 categorical + 6 numeric) demographic and employment features. Extracted from
/// the 1994 US Census; uses the canonical `adult.data` training partition.
pub mod adult;

/// Bank Marketing dataset module.
///
/// Contains the Bank Marketing dataset for binary classification: predicting
/// whether a client subscribes a term deposit from 16 mixed (9 categorical +
/// 7 numeric) client, contact, and campaign features. Recorded from a Portuguese
/// bank's phone campaigns; uses the full `bank-full.csv` partition. Sourced from a
/// ZIP archive (like `digits`).
pub mod bank_marketing;

/// Boston Housing dataset module.
///
/// Contains the Boston Housing dataset for predicting median house values
/// in Boston suburbs based on various features like crime rate, room count,
/// and accessibility to highways.
pub mod boston_housing;

/// Breast Cancer Wisconsin (Diagnostic) dataset module.
///
/// Contains the Breast Cancer Wisconsin dataset for binary classification of
/// tumors as malignant or benign based on 30 features computed from digitized
/// images of cell nuclei.
pub mod breast_cancer;

/// California Housing dataset module.
///
/// Contains the California Housing dataset for predicting median house values
/// in California districts. Reproduces scikit-learn's `fetch_california_housing`
/// eight derived features. A modern replacement for Boston Housing.
pub mod california_housing;

/// Car Evaluation dataset module.
///
/// Contains the Car Evaluation dataset (UCI, Bohanec 1988) for multi-class
/// classification: predicting a car's overall acceptability (`unacc`, `acc`,
/// `good`, `vgood`) from 6 categorical price and technical attributes. Like
/// [`mushroom`], it is **all-categorical** — `features()` returns a single
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
/// 13 clinical features. The `?` missing values in `ca`/`thal` are mapped to
/// `NaN` (like [`titanic`]/[`palmer_penguins`]); the target is an `Array1<u8>`.
pub mod heart_disease;

/// Ionosphere dataset module.
///
/// Contains the Ionosphere dataset (UCI, Sigillito et al. 1989) for binary
/// classification: predicting whether a radar return shows structure in the
/// ionosphere (`good`) or passes through it (`bad`) from 34 continuous
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

/// Linnerud dataset module.
///
/// Contains the scikit-learn Linnerud dataset (`load_linnerud`) for multi-output
/// regression: predicting three physiological variables (`Weight`, `Waist`,
/// `Pulse`) from three exercise variables (`Chins`, `Situps`, `Jumps`) measured
/// on 20 middle-aged men.
pub mod linnerud;

/// Mushroom dataset module.
///
/// Contains the Mushroom dataset (UCI `agaricus-lepiota`) for binary
/// classification: predicting whether a mushroom is edible or poisonous from 22
/// categorical attributes. The first **all-categorical** loader — every feature is
/// a single-letter string code, so `features()` returns a single `Array2<String>`.
pub mod mushroom;

/// Palmer Penguins dataset module.
///
/// Contains the Palmer Penguins dataset for classifying penguins into three
/// species (Adelie, Chinstrap, Gentoo) based on bill and flipper measurements,
/// body mass, and categorical island/sex features. A modern alternative to Iris.
pub mod palmer_penguins;

/// SMS Spam Collection dataset module.
///
/// Contains the SMS Spam Collection dataset (UCI, Almeida & Hidalgo 2011) for
/// binary **text** classification: labelling 5,574 SMS messages as `ham` or
/// `spam`. The crate's first text-modality loader — there is no feature matrix,
/// so the document accessor is `texts()` (an `Array1<String>` of raw messages)
/// rather than `features()`. Sourced from a ZIP archive.
pub mod sms_spam;

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
/// Contains the YouTube Spam Collection dataset (UCI, Alberto, Lochter & Almeida
/// 2017) for binary **text** classification: labelling 1,956 comments from five
/// popular music videos as `ham` or `spam`. Like [`sms_spam`] (a sibling by the
/// same authors) it is a text-modality loader — there is no feature matrix, so
/// the document accessor is `texts()` (an `Array1<String>` of raw comments)
/// rather than `features()`. Sourced from a ZIP archive of five per-video CSVs.
pub mod youtube_spam;

pub use abalone::Abalone;
pub use adult::Adult;
pub use bank_marketing::BankMarketing;
pub use boston_housing::BostonHousing;
pub use breast_cancer::BreastCancer;
pub use california_housing::CaliforniaHousing;
pub use car_evaluation::CarEvaluation;
pub use covtype::Covtype;
pub use diabetes::Diabetes;
pub use digits::Digits;
pub use heart_disease::HeartDisease;
pub use ionosphere::Ionosphere;
pub use iris::Iris;
pub use kddcup99::Kddcup99;
pub use linnerud::Linnerud;
pub use mushroom::Mushroom;
pub use palmer_penguins::PalmerPenguins;
pub use sms_spam::SmsSpam;
pub use titanic::Titanic;
pub use wine_quality::{red_wine_quality::RedWineQuality, white_wine_quality::WhiteWineQuality};
pub use wine_recognition::WineRecognition;
pub use youtube_spam::YoutubeSpam;
