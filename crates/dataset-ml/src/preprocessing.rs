//! Preprocessing helpers for the loaded datasets.
//!
//! Every loader in this crate hands back raw [`ndarray`] arrays: numbers exactly as
//! the source published them, and categorical values as strings. Turning those into
//! model input almost always means the same four steps — split off an evaluation
//! set, scale the numeric columns, encode the categorical ones, and encode the
//! labels. This module provides those steps, so a user does not have to reimplement
//! them (or pull in a framework) just to run a baseline.
//!
//! # Splitting is index-based
//!
//! The splitting functions ([`train_test_split`](crate::preprocessing::train_test_split),
//! [`stratified_split`](crate::preprocessing::stratified_split),
//! [`k_fold_indices`](crate::preprocessing::k_fold_indices),
//! [`shuffled_indices`](crate::preprocessing::shuffled_indices)) return **row indices**, not arrays.
//! That is deliberate: a sample is spread across two or three parallel arrays
//! (`features` + `labels`, or `categorical` + `numeric` + `labels`, or
//! `texts` + `sources` + `labels`), and one index list keeps them aligned. Turn
//! indices into arrays with ndarray's own `select`:
//!
//! ```no_run
//! use dataset_ml::iris::Iris;
//! use dataset_ml::preprocessing::train_test_split;
//! use ndarray::Axis;
//!
//! let dataset = Iris::new("./data");
//! let (features, labels) = dataset.data().unwrap();
//!
//! let (train, test) = train_test_split(features.nrows(), 0.2, 42).unwrap();
//!
//! let train_x = features.select(Axis(0), &train);
//! let train_y = labels.select(Axis(0), &train);
//! let test_x = features.select(Axis(0), &test);
//! let test_y = labels.select(Axis(0), &test);
//!
//! assert_eq!(train_x.nrows(), 120);
//! assert_eq!(test_x.nrows(), 30);
//! ```
//!
//! # Determinism
//!
//! Everything that shuffles takes an explicit `u64` seed and uses a
//! [SplitMix64](https://doi.org/10.1145/2714064.2660195) generator built into this
//! crate — no `rand` dependency, and no hidden global state. The same seed and the
//! same inputs always produce the same split, on every platform and every release
//! of this crate.
//!
//! # Missing values
//!
//! Several loaders encode a missing number as `NaN` (`titanic`, `palmer_penguins`,
//! `heart_disease`). The scalers here compute their statistics over the **finite**
//! values of each column and leave non-finite entries untouched, so a missing value
//! stays missing instead of poisoning the whole column. Decide how to impute it
//! yourself.

use dataset_core::DatasetError;
use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;

/// The name used to tag errors raised by this module.
const MODULE_NAME: &str = "preprocessing";

/// A pair of disjoint row-index lists produced by a splitting function.
///
/// The first list is the training side, the second is the held-out side (the test
/// set for [`train_test_split`] / [`stratified_split`], the validation fold for
/// [`k_fold_indices`]). Index them into your arrays with ndarray's
/// `select(Axis(0), &indices)`.
pub type IndexSplit = (Vec<usize>, Vec<usize>);

/// A small, fast, fully deterministic pseudo-random number generator.
///
/// This is Steele, Lea & Flood's SplitMix64 — the finalizer used to seed many
/// modern generators. It is not cryptographically secure and is not meant to be:
/// it exists so that shuffling and splitting are reproducible across platforms
/// without taking on a dependency, and it is more than good enough for choosing
/// which rows land in which fold.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// The odd constant SplitMix64 advances its state by (the 64-bit golden ratio).
    const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

    /// Seed a generator. Every `u64` is a valid seed, including `0`.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Produce the next 64-bit output and advance the state.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::GAMMA);

        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        z ^ (z >> 31)
    }

    /// Produce a uniformly distributed value in `0..bound`.
    ///
    /// Uses rejection sampling rather than a plain modulo, so every value in the
    /// range is equally likely (a modulo would over-represent the low values
    /// whenever `bound` does not divide 2^64).
    ///
    /// # Panics
    ///
    /// Panics if `bound` is 0. Callers in this module guarantee it is not.
    fn below(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "bound must be positive");

        // The largest multiple of `bound` that fits in a u64; draws at or above it
        // would bias the result, so they are discarded and redrawn.
        let limit = u64::MAX - (u64::MAX % bound) - (bound - 1);

        loop {
            let value = self.next_u64();
            if value <= limit {
                return value % bound;
            }
        }
    }

    /// Shuffle a slice in place with an unbiased Fisher-Yates pass.
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

/// Return `0..n_samples` in a deterministic pseudo-random order.
///
/// This is the shuffling primitive the other functions here are built on, exposed
/// for when you want to reorder a dataset without splitting it — for instance
/// before a sequential pass over data that arrived grouped by class (as `iris`,
/// `covtype`, and `movie_review_polarity` do).
///
/// # Parameters
///
/// - `n_samples` - How many indices to produce.
/// - `seed` - Seed for the internal generator; the same seed always yields the same order.
///
/// # Returns
///
/// - `Vec<usize>` - A permutation of `0..n_samples`.
///
/// # Example
/// ```rust
/// use dataset_ml::preprocessing::shuffled_indices;
///
/// let order = shuffled_indices(5, 42);
/// assert_eq!(order.len(), 5);
///
/// // Every index appears exactly once...
/// let mut sorted = order.clone();
/// sorted.sort_unstable();
/// assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
///
/// // ...and the same seed reproduces the same order.
/// assert_eq!(shuffled_indices(5, 42), order);
/// ```
pub fn shuffled_indices(n_samples: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n_samples).collect();
    SplitMix64::new(seed).shuffle(&mut indices);

    indices
}

/// Split `0..n_samples` into shuffled train and test index lists.
///
/// The test set gets `round(n_samples * test_ratio)` rows, clamped so that neither
/// side is empty whenever there are at least two samples; the train set gets the
/// rest. Both lists are in shuffled order, so a dataset stored grouped by class
/// (the common case) does not produce a train set missing a class entirely.
///
/// To keep each class's proportion intact, use [`stratified_split`] instead.
///
/// # Parameters
///
/// - `n_samples` - Total number of samples to split.
/// - `test_ratio` - Fraction of samples to place in the test set, in `0.0..=1.0`.
/// - `seed` - Seed for the internal generator; the same seed always yields the same split.
///
/// # Returns
///
/// - `IndexSplit` - The `(train, test)` row indices. Together they are
///   a permutation of `0..n_samples`, and they never overlap.
///
/// # Errors
///
/// - `DatasetError::ValidationError` - Returned when `n_samples` is 0, or when
///   `test_ratio` is not a finite value in `0.0..=1.0`.
///
/// # Example
/// ```rust
/// use dataset_ml::preprocessing::train_test_split;
///
/// let (train, test) = train_test_split(150, 0.2, 42).unwrap();
/// assert_eq!(train.len(), 120);
/// assert_eq!(test.len(), 30);
///
/// // The two sides are disjoint and cover everything.
/// let mut all: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
/// all.sort_unstable();
/// assert_eq!(all, (0..150).collect::<Vec<_>>());
/// ```
pub fn train_test_split(
    n_samples: usize,
    test_ratio: f64,
    seed: u64,
) -> Result<IndexSplit, DatasetError> {
    if n_samples == 0 {
        return Err(DatasetError::empty_dataset(MODULE_NAME));
    }
    validate_ratio(test_ratio)?;

    let n_test = test_size(n_samples, test_ratio);

    let mut indices = shuffled_indices(n_samples, seed);
    let test = indices.split_off(n_samples - n_test);

    Ok((indices, test))
}

/// Split into train and test index lists that preserve each class's proportion.
///
/// Like [`train_test_split`], but the split is drawn **within** each class rather
/// than over the dataset as a whole, so a class holding 10% of the samples holds
/// about 10% of the train set and 10% of the test set. This matters for the
/// imbalanced loaders — `sms_spam` (13% spam), `covtype` (its rarest cover type is
/// under 0.5%), `kddcup99` — where an unstratified split can leave a rare class out
/// of the test set altogether.
///
/// Every class with at least two members contributes at least one row to each side.
/// A class with a single member contributes it to the train set.
///
/// # Parameters
///
/// - `labels` - The per-sample class labels; any comparable, hashable type
///   (`&str`, `String`, `u8`, `char`, … — every label type this crate produces).
/// - `test_ratio` - Fraction of each class to place in the test set, in `0.0..=1.0`.
/// - `seed` - Seed for the internal generator; the same seed always yields the same split.
///
/// # Returns
///
/// - `IndexSplit` - The `(train, test)` row indices, each in shuffled
///   order. Together they are a permutation of `0..labels.len()`.
///
/// # Errors
///
/// - `DatasetError::ValidationError` - Returned when `labels` is empty, or when
///   `test_ratio` is not a finite value in `0.0..=1.0`.
///
/// # Example
/// ```rust
/// use dataset_ml::preprocessing::stratified_split;
///
/// // Nine samples of class "a", one of class "b".
/// let labels = ["a", "a", "a", "a", "a", "a", "a", "a", "a", "b"];
/// let (train, test) = stratified_split(&labels, 0.5, 7).unwrap();
///
/// // The lone "b" cannot be in both sides, so it stays in the train set.
/// assert!(train.contains(&9));
/// assert_eq!(train.len() + test.len(), 10);
/// ```
pub fn stratified_split<T: std::hash::Hash + Eq>(
    labels: &[T],
    test_ratio: f64,
    seed: u64,
) -> Result<IndexSplit, DatasetError> {
    if labels.is_empty() {
        return Err(DatasetError::empty_dataset(MODULE_NAME));
    }
    validate_ratio(test_ratio)?;

    // Group row indices by class, keeping first-appearance order so the result does
    // not depend on `HashMap` iteration order.
    let mut order: Vec<&T> = Vec::new();
    let mut groups: HashMap<&T, Vec<usize>> = HashMap::new();
    for (index, label) in labels.iter().enumerate() {
        groups.entry(label).or_insert_with(|| {
            order.push(label);
            Vec::new()
        });
        // The entry was just created if it was missing, so this cannot fail.
        groups.get_mut(label).expect("group exists").push(index);
    }

    let mut rng = SplitMix64::new(seed);
    let mut train = Vec::with_capacity(labels.len());
    let mut test = Vec::new();

    for label in order {
        let mut group = groups.remove(label).expect("group exists");
        rng.shuffle(&mut group);

        // A single-member class cannot be split, so it goes to the train set.
        let n_test = if group.len() < 2 {
            0
        } else {
            test_size(group.len(), test_ratio)
        };

        let group_test = group.split_off(group.len() - n_test);
        train.extend(group);
        test.extend(group_test);
    }

    // Re-shuffle so the output is not ordered class by class.
    rng.shuffle(&mut train);
    rng.shuffle(&mut test);

    Ok((train, test))
}

/// Partition `0..n_samples` into `k` cross-validation folds.
///
/// The samples are shuffled once and dealt into `k` folds whose sizes differ by at
/// most one. Each entry of the result is the `(train, validation)` pair for one
/// fold: the validation list is that fold, and the train list is everything else.
/// Every sample is used for validation exactly once across the `k` rounds.
///
/// # Parameters
///
/// - `n_samples` - Total number of samples to partition.
/// - `k` - Number of folds; must be at least 2 and at most `n_samples`.
/// - `seed` - Seed for the internal generator; the same seed always yields the same folds.
///
/// # Returns
///
/// - `Vec<IndexSplit>` - `k` pairs of `(train, validation)` row indices.
///
/// # Errors
///
/// - `DatasetError::ValidationError` - Returned when `n_samples` is 0, or when `k`
///   is less than 2 or greater than `n_samples`.
///
/// # Example
/// ```rust
/// use dataset_ml::preprocessing::k_fold_indices;
///
/// let folds = k_fold_indices(10, 5, 42).unwrap();
/// assert_eq!(folds.len(), 5);
///
/// for (train, validation) in &folds {
///     assert_eq!(validation.len(), 2);
///     assert_eq!(train.len(), 8);
/// }
///
/// // Each sample is validated exactly once.
/// let mut validated: Vec<usize> = folds.iter().flat_map(|(_, v)| v.clone()).collect();
/// validated.sort_unstable();
/// assert_eq!(validated, (0..10).collect::<Vec<_>>());
/// ```
pub fn k_fold_indices(
    n_samples: usize,
    k: usize,
    seed: u64,
) -> Result<Vec<IndexSplit>, DatasetError> {
    if n_samples == 0 {
        return Err(DatasetError::empty_dataset(MODULE_NAME));
    }
    if k < 2 || k > n_samples {
        return Err(DatasetError::ValidationError(format!(
            "[{MODULE_NAME}] k must be between 2 and n_samples ({n_samples}), got {k}"
        )));
    }

    let indices = shuffled_indices(n_samples, seed);

    // Deal into folds of size `n / k`, giving the first `n % k` folds one extra so
    // every sample is used and no fold is more than one larger than another.
    let base = n_samples / k;
    let remainder = n_samples % k;

    let mut folds = Vec::with_capacity(k);
    let mut start = 0;
    for fold in 0..k {
        let len = base + usize::from(fold < remainder);
        let end = start + len;

        let validation = indices[start..end].to_vec();
        let train = indices[..start]
            .iter()
            .chain(indices[end..].iter())
            .copied()
            .collect();

        folds.push((train, validation));
        start = end;
    }

    Ok(folds)
}

/// Map labels of any type to consecutive integer codes.
///
/// Turns the label vector a loader produces — `&'static str` species names,
/// `String` categories, `char` letters — into the `0..n_classes` codes most
/// training code expects, plus the class list needed to read a prediction back.
/// Classes are numbered in **sorted** order, so the encoding depends only on the
/// set of labels present and never on their order in the file.
///
/// # Parameters
///
/// - `labels` - The per-sample labels to encode.
///
/// # Returns
///
/// - `(Array1<usize>, Vec<T>)` - The per-sample codes, and the sorted class list
///   where index `i` is the class encoded as `i`.
///
/// # Errors
///
/// - `DatasetError::ValidationError` - Returned when `labels` is empty.
///
/// # Example
/// ```rust
/// use dataset_ml::preprocessing::label_encode;
/// use ndarray::array;
///
/// let labels = array!["virginica", "setosa", "setosa", "versicolor"];
/// let (codes, classes) = label_encode(&labels).unwrap();
///
/// // Classes are numbered alphabetically, not in order of appearance.
/// assert_eq!(classes, vec!["setosa", "versicolor", "virginica"]);
/// assert_eq!(codes, array![2, 0, 0, 1]);
///
/// // Read a prediction back through the class list.
/// assert_eq!(classes[codes[0]], "virginica");
/// ```
pub fn label_encode<T: Clone + Ord>(
    labels: &Array1<T>,
) -> Result<(Array1<usize>, Vec<T>), DatasetError> {
    if labels.is_empty() {
        return Err(DatasetError::empty_dataset(MODULE_NAME));
    }

    let mut classes: Vec<T> = labels.iter().cloned().collect();
    classes.sort_unstable();
    classes.dedup();

    let codes = labels.mapv(|label| {
        classes
            .binary_search(&label)
            .expect("every label is in the class list it was built from")
    });

    Ok((codes, classes))
}

/// Count how many samples carry each label.
///
/// A quick way to see how balanced a dataset is before choosing between
/// [`train_test_split`] and [`stratified_split`]. Counts are returned in sorted
/// class order, matching the numbering [`label_encode`] assigns.
///
/// # Parameters
///
/// - `labels` - The per-sample labels to count.
///
/// # Returns
///
/// - `Vec<(T, usize)>` - Each distinct class and its sample count, sorted by class.
///
/// # Example
/// ```rust
/// use dataset_ml::preprocessing::class_counts;
/// use ndarray::array;
///
/// let labels = array!["spam", "ham", "ham", "ham"];
/// assert_eq!(class_counts(&labels), vec![("ham", 3), ("spam", 1)]);
/// ```
pub fn class_counts<T: Clone + Ord>(labels: &Array1<T>) -> Vec<(T, usize)> {
    let mut sorted: Vec<T> = labels.iter().cloned().collect();
    sorted.sort_unstable();

    let mut counts: Vec<(T, usize)> = Vec::new();
    for label in sorted {
        match counts.last_mut() {
            Some((class, count)) if *class == label => *count += 1,
            _ => counts.push((label, 1)),
        }
    }

    counts
}

/// Per-column statistics produced by fitting a scaler, so the same transform can be
/// replayed on new data.
///
/// A scaler must be fitted on the **training** rows only and then applied unchanged
/// to the test rows; fitting it on everything leaks information about the test set
/// into training. That is why [`standardize`] and [`min_max_scale`] hand back this
/// struct: keep it, and pass it to [`apply_scaler`] for every later batch.
///
/// The two fields are named for the general shape of the transform,
/// `(value - center) / scale`:
///
/// - [`standardize`] sets `center` to the column mean and `scale` to its standard
///   deviation.
/// - [`min_max_scale`] sets `center` to the column minimum and `scale` to its range.
#[derive(Debug, Clone, PartialEq)]
pub struct Scaler {
    /// Per-column value subtracted before scaling (mean, or minimum).
    pub center: Array1<f64>,
    /// Per-column divisor (standard deviation, or range). Never 0: a constant
    /// column gets a scale of 1, so it maps to all-zeros instead of `NaN`.
    pub scale: Array1<f64>,
}

/// Standardize each feature column to zero mean and unit variance.
///
/// The classic z-score transform, `(value - mean) / std_dev`, applied per column.
/// It is what distance- and gradient-based models want from the raw numeric matrices
/// these loaders return, whose columns routinely differ by orders of magnitude
/// (`adult`'s `fnlwgt` runs to the hundreds of thousands while `education-num` tops
/// out at 16).
///
/// The mean and standard deviation (population, i.e. divided by `n`) are computed
/// over the **finite** values of each column; non-finite entries are copied through
/// untouched, so a `NaN` marking a missing value stays a `NaN`. A column with no
/// variation gets a scale of 1 and maps to all zeros rather than dividing by 0.
///
/// # Parameters
///
/// - `features` - The numeric feature matrix, shape `(n_samples, n_features)`.
///
/// # Returns
///
/// - `(Array2<f64>, Scaler)` - The standardized matrix, and the fitted per-column
///   statistics to replay on later data with [`apply_scaler`].
///
/// # Errors
///
/// - `DatasetError::ValidationError` - Returned when `features` has no rows or no columns.
///
/// # Example
/// ```rust
/// use dataset_ml::preprocessing::standardize;
/// use ndarray::array;
///
/// let features = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
/// let (scaled, scaler) = standardize(&features).unwrap();
///
/// assert_eq!(scaler.center, array![2.0, 20.0]);
/// assert_eq!(scaled[[1, 0]], 0.0); // the mean row maps to 0
/// assert!((scaled[[0, 0]] + scaled[[2, 0]]).abs() < 1e-12); // symmetric about it
/// ```
pub fn standardize(features: &Array2<f64>) -> Result<(Array2<f64>, Scaler), DatasetError> {
    let scaler = fit_scaler(features.view(), |column| {
        let (sum, count) = finite_sum(column);
        if count == 0 {
            // Nothing finite to learn from: leave the column alone.
            return (0.0, 1.0);
        }

        let mean = sum / count as f64;
        let variance = column
            .iter()
            .filter(|value| value.is_finite())
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / count as f64;

        (mean, variance.sqrt())
    })?;

    let scaled = apply_scaler(features, &scaler)?;

    Ok((scaled, scaler))
}

/// Rescale each feature column into the `[0, 1]` range.
///
/// Min-max scaling, `(value - min) / (max - min)`, applied per column. Prefer it
/// over [`standardize`] when a bounded range matters more than a comparable spread —
/// for pixel-like features (`digits`), or as input to a model that expects `[0, 1]`.
///
/// As with [`standardize`], the minimum and maximum come from the **finite** values
/// of each column, non-finite entries pass through untouched, and a constant column
/// maps to all zeros rather than dividing by 0.
///
/// # Parameters
///
/// - `features` - The numeric feature matrix, shape `(n_samples, n_features)`.
///
/// # Returns
///
/// - `(Array2<f64>, Scaler)` - The rescaled matrix, and the fitted per-column
///   statistics to replay on later data with [`apply_scaler`].
///
/// # Errors
///
/// - `DatasetError::ValidationError` - Returned when `features` has no rows or no columns.
///
/// # Example
/// ```rust
/// use dataset_ml::preprocessing::min_max_scale;
/// use ndarray::array;
///
/// let features = array![[1.0, -5.0], [3.0, 5.0]];
/// let (scaled, _scaler) = min_max_scale(&features).unwrap();
///
/// assert_eq!(scaled, array![[0.0, 0.0], [1.0, 1.0]]);
/// ```
pub fn min_max_scale(features: &Array2<f64>) -> Result<(Array2<f64>, Scaler), DatasetError> {
    let scaler = fit_scaler(features.view(), |column| {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &value in column {
            if value.is_finite() {
                min = min.min(value);
                max = max.max(value);
            }
        }

        if min > max {
            // No finite values at all: leave the column alone.
            return (0.0, 1.0);
        }

        (min, max - min)
    })?;

    let scaled = apply_scaler(features, &scaler)?;

    Ok((scaled, scaler))
}

/// Apply an already-fitted [`Scaler`] to a feature matrix.
///
/// This is how a scaler fitted on the training rows is replayed on the test rows —
/// or on data that arrives later — without refitting, which would give the two sets
/// different transforms and leak test statistics into training.
///
/// Non-finite entries are copied through untouched, matching the fitting functions.
///
/// # Parameters
///
/// - `features` - The numeric feature matrix to transform, shape `(n_samples, n_features)`.
/// - `scaler` - Statistics from a previous [`standardize`] or [`min_max_scale`] call.
///
/// # Returns
///
/// - `Array2<f64>` - The transformed matrix, with the same shape as `features`.
///
/// # Errors
///
/// - `DatasetError::LengthMismatch` - Returned when the scaler was fitted on a
///   different number of columns than `features` has.
///
/// # Example
/// ```rust
/// use dataset_ml::preprocessing::{apply_scaler, standardize};
/// use ndarray::array;
///
/// let train = array![[1.0], [2.0], [3.0]];
/// let (_scaled_train, scaler) = standardize(&train).unwrap();
///
/// // The test rows are transformed with the training statistics, not their own.
/// let test = array![[2.0], [4.0]];
/// let scaled_test = apply_scaler(&test, &scaler).unwrap();
/// assert_eq!(scaled_test[[0, 0]], 0.0); // 2.0 was the training mean
/// ```
pub fn apply_scaler(features: &Array2<f64>, scaler: &Scaler) -> Result<Array2<f64>, DatasetError> {
    if features.ncols() != scaler.center.len() {
        return Err(DatasetError::length_mismatch(
            MODULE_NAME,
            "scaler columns",
            scaler.center.len(),
            features.ncols(),
        ));
    }

    let mut scaled = features.clone();
    for (column_index, mut column) in scaled.columns_mut().into_iter().enumerate() {
        let center = scaler.center[column_index];
        let scale = scaler.scale[column_index];

        for value in column.iter_mut() {
            // A missing or infinite value has no meaningful scaled counterpart, so
            // it is preserved rather than turned into a different kind of nonsense.
            if value.is_finite() {
                *value = (*value - center) / scale;
            }
        }
    }

    Ok(scaled)
}

/// One-hot encode a matrix of categorical string features.
///
/// The mixed-type loaders (`adult`, `titanic`, `bank_marketing`, `abalone`,
/// `kddcup99`, `palmer_penguins`) and the all-categorical ones (`mushroom`,
/// `car_evaluation`) hand back their categorical columns as an `Array2<String>`,
/// which no numeric model can consume. This expands each column into one indicator
/// column per level it takes: a row gets `1.0` in the column for its own level and
/// `0.0` everywhere else.
///
/// Levels within a column are sorted, so the output layout depends only on the
/// values present. The returned names identify the columns as `<column>=<level>`,
/// using `column_names` when supplied and `column_0`, `column_1`, … otherwise.
///
/// Note that this widens the matrix by however many distinct levels the data holds —
/// harmless for `mushroom` (22 columns become 117), but worth checking before
/// running it on `kddcup99`'s `service` column (70 levels over millions of rows).
///
/// # Parameters
///
/// - `categorical` - The categorical matrix, shape `(n_samples, n_features)`.
/// - `column_names` - Optional names for the source columns, used to build the
///   output names. Must have one entry per column when supplied.
///
/// # Returns
///
/// - `(Array2<f64>, Vec<String>)` - The indicator matrix, shape
///   `(n_samples, total_levels)`, and one name per output column.
///
/// # Errors
///
/// - `DatasetError::ValidationError` - Returned when `categorical` has no rows or no columns.
/// - `DatasetError::LengthMismatch` - Returned when `column_names` is supplied but
///   does not have one entry per column.
///
/// # Example
/// ```rust
/// use dataset_ml::preprocessing::one_hot_encode;
/// use ndarray::array;
///
/// let categorical = array![
///     ["male".to_string(), "S".to_string()],
///     ["female".to_string(), "C".to_string()],
///     ["male".to_string(), "C".to_string()],
/// ];
/// let (encoded, names) = one_hot_encode(&categorical, Some(&["sex", "port"])).unwrap();
///
/// assert_eq!(names, vec!["sex=female", "sex=male", "port=C", "port=S"]);
/// assert_eq!(encoded.row(0).to_vec(), vec![0.0, 1.0, 0.0, 1.0]); // male, S
/// ```
pub fn one_hot_encode(
    categorical: &Array2<String>,
    column_names: Option<&[&str]>,
) -> Result<(Array2<f64>, Vec<String>), DatasetError> {
    let n_samples = categorical.nrows();
    let n_columns = categorical.ncols();

    if n_samples == 0 || n_columns == 0 {
        return Err(DatasetError::empty_dataset(MODULE_NAME));
    }
    if let Some(names) = column_names
        && names.len() != n_columns
    {
        return Err(DatasetError::length_mismatch(
            MODULE_NAME,
            "column_names",
            n_columns,
            names.len(),
        ));
    }

    // Collect each column's sorted levels first, so the output width is known before
    // any allocation of the result matrix.
    let mut levels_per_column: Vec<Vec<&String>> = Vec::with_capacity(n_columns);
    for column_index in 0..n_columns {
        // `into_iter` on the view yields references borrowed from `categorical`
        // itself, so the collected levels outlive this loop iteration.
        let mut levels: Vec<&String> = categorical.column(column_index).into_iter().collect();
        levels.sort_unstable();
        levels.dedup();
        levels_per_column.push(levels);
    }

    let total_levels: usize = levels_per_column.iter().map(Vec::len).sum();

    let mut names = Vec::with_capacity(total_levels);
    for (column_index, levels) in levels_per_column.iter().enumerate() {
        let column_name = match column_names {
            Some(supplied) => supplied[column_index].to_string(),
            None => format!("column_{column_index}"),
        };
        for level in levels {
            names.push(format!("{column_name}={level}"));
        }
    }

    let mut encoded = Array2::<f64>::zeros((n_samples, total_levels));
    let mut offset = 0;
    for (column_index, levels) in levels_per_column.iter().enumerate() {
        for row in 0..n_samples {
            let value = &categorical[[row, column_index]];
            let level_index = levels
                .binary_search(&value)
                .expect("every value is in the level list it was built from");
            encoded[[row, offset + level_index]] = 1.0;
        }
        offset += levels.len();
    }

    Ok((encoded, names))
}

/// Reject a ratio that is not a finite fraction.
fn validate_ratio(ratio: f64) -> Result<(), DatasetError> {
    if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
        return Err(DatasetError::ValidationError(format!(
            "[{MODULE_NAME}] test_ratio must be a finite value in 0.0..=1.0, got {ratio}"
        )));
    }

    Ok(())
}

/// How many of `n` samples the test side gets, keeping both sides non-empty when
/// there is more than one sample to go around.
fn test_size(n: usize, ratio: f64) -> usize {
    let requested = (n as f64 * ratio).round() as usize;

    requested.clamp(usize::from(n > 1), n.saturating_sub(1))
}

/// Sum a column's finite values, and count how many there were.
fn finite_sum(column: &ndarray::ArrayView1<f64>) -> (f64, usize) {
    column
        .iter()
        .filter(|value| value.is_finite())
        .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1))
}

/// Build a [`Scaler`] by applying `statistics` to every column of `features`.
///
/// The closure returns that column's `(center, scale)`; a scale of 0 (a constant
/// column) is replaced by 1 so the transform maps it to zeros instead of `NaN`.
fn fit_scaler(
    features: ArrayView2<f64>,
    statistics: impl Fn(&ndarray::ArrayView1<f64>) -> (f64, f64),
) -> Result<Scaler, DatasetError> {
    if features.nrows() == 0 || features.ncols() == 0 {
        return Err(DatasetError::empty_dataset(MODULE_NAME));
    }

    let mut center = Vec::with_capacity(features.ncols());
    let mut scale = Vec::with_capacity(features.ncols());

    for column in features.columns() {
        let (column_center, column_scale) = statistics(&column);
        center.push(column_center);
        scale.push(if column_scale > 0.0 {
            column_scale
        } else {
            1.0
        });
    }

    Ok(Scaler {
        center: Array1::from_vec(center),
        scale: Array1::from_vec(scale),
    })
}
