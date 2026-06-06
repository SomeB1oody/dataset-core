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
//! | [`iris`]                                              | 150     | 4        | Classification |
//! | [`breast_cancer`]                                     | 569     | 30       | Classification |
//! | [`boston_housing`]                                    | 506     | 13       | Regression     |
//! | [`california_housing`]                                | 20,640  | 8        | Regression     |
//! | [`diabetes`]                                          | 442     | 10       | Regression     |
//! | [`titanic`]                                           | 891     | 11       | Classification |
//! | [`palmer_penguins`]                                   | 344     | 7        | Classification |
//! | [`wine_recognition`]                                  | 178     | 13       | Classification |
//! | [`wine_quality::red_wine_quality`]                    | 1,599   | 11       | Regression     |
//! | [`wine_quality::white_wine_quality`]                  | 4,898   | 11       | Regression     |
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

/// Diabetes dataset module.
///
/// Contains the scikit-learn diabetes dataset (`load_diabetes`) for regression:
/// predicting disease progression from 10 standardized physiological features.
pub mod diabetes;

/// Iris flower dataset module.
///
/// Contains the classic Iris dataset for classifying iris flowers into
/// three species (setosa, versicolor, virginica) based on sepal and petal
/// measurements.
pub mod iris;

/// Palmer Penguins dataset module.
///
/// Contains the Palmer Penguins dataset for classifying penguins into three
/// species (Adelie, Chinstrap, Gentoo) based on bill and flipper measurements,
/// body mass, and categorical island/sex features. A modern alternative to Iris.
pub mod palmer_penguins;

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

pub use boston_housing::BostonHousing;
pub use breast_cancer::BreastCancer;
pub use california_housing::CaliforniaHousing;
pub use diabetes::Diabetes;
pub use iris::Iris;
pub use palmer_penguins::PalmerPenguins;
pub use titanic::Titanic;
pub use wine_quality::{red_wine_quality::RedWineQuality, white_wine_quality::WhiteWineQuality};
pub use wine_recognition::WineRecognition;
