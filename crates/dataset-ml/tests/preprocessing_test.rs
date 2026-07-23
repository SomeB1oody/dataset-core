//! Integration tests for the `preprocessing` module.
//!
//! These need no network: the module operates on arrays the caller supplies, so
//! every property here is checked against hand-built inputs.

use dataset_core::DatasetError;
use dataset_ml::preprocessing::{
    apply_scaler, class_counts, k_fold_indices, label_encode, min_max_scale, one_hot_encode,
    shuffled_indices, standardize, stratified_split, train_test_split,
};
use ndarray::{Array2, array};

/// Assert that `indices` is exactly a permutation of `0..n`.
fn assert_is_permutation(indices: &[usize], n: usize) {
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..n).collect::<Vec<_>>());
}

#[test]
// Verifies that shuffled_indices permutes every index and is seed-deterministic.
fn shuffled_indices_permutes_and_is_deterministic() {
    let order = shuffled_indices(100, 42);
    assert_is_permutation(&order, 100);

    // Same seed, same order; a different seed gives a different one.
    assert_eq!(shuffled_indices(100, 42), order);
    assert_ne!(shuffled_indices(100, 43), order);

    // Degenerate sizes are handled rather than panicking.
    assert!(shuffled_indices(0, 1).is_empty());
    assert_eq!(shuffled_indices(1, 1), vec![0]);
}

#[test]
// Verifies that shuffled_indices actually reorders rather than returning 0..n.
fn shuffled_indices_does_not_return_sorted_order() {
    let order = shuffled_indices(1000, 7);

    assert_ne!(order, (0..1000).collect::<Vec<_>>());
}

#[test]
// Verifies the train/test sizes, disjointness, and determinism of train_test_split.
fn train_test_split_partitions_by_ratio() {
    let (train, test) = train_test_split(150, 0.2, 42).unwrap();

    assert_eq!(train.len(), 120);
    assert_eq!(test.len(), 30);

    // Together they cover every row exactly once, so the sides cannot overlap.
    let combined: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
    assert_is_permutation(&combined, 150);

    assert_eq!(train_test_split(150, 0.2, 42).unwrap(), (train, test));
}

#[test]
// Verifies that train_test_split keeps both sides non-empty at extreme ratios.
fn train_test_split_keeps_both_sides_non_empty() {
    // A ratio of 0 would round to an empty test set; one row is kept back instead.
    let (train, test) = train_test_split(10, 0.0, 1).unwrap();
    assert_eq!((train.len(), test.len()), (9, 1));

    // Likewise a ratio of 1 leaves one row for training.
    let (train, test) = train_test_split(10, 1.0, 1).unwrap();
    assert_eq!((train.len(), test.len()), (1, 9));

    // A single sample cannot be split, so it stays in the train set.
    let (train, test) = train_test_split(1, 0.5, 1).unwrap();
    assert_eq!((train.len(), test.len()), (1, 0));
}

#[test]
// Verifies that train_test_split rejects an empty dataset and out-of-range ratios.
fn train_test_split_rejects_invalid_arguments() {
    assert!(matches!(
        train_test_split(0, 0.2, 1),
        Err(DatasetError::DataFormatError(_))
    ));

    for bad_ratio in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                train_test_split(10, bad_ratio, 1),
                Err(DatasetError::ValidationError(_))
            ),
            "ratio {bad_ratio} should be rejected"
        );
    }
}

#[test]
// Verifies that stratified_split preserves each class's proportion on both sides.
fn stratified_split_preserves_class_proportions() {
    // 80 of class "a", 20 of class "b".
    let labels: Vec<&str> = (0..100).map(|i| if i < 80 { "a" } else { "b" }).collect();

    let (train, test) = stratified_split(&labels, 0.25, 42).unwrap();

    assert_eq!(test.len(), 25);
    assert_eq!(train.len(), 75);
    assert_is_permutation(
        &train.iter().chain(test.iter()).copied().collect::<Vec<_>>(),
        100,
    );

    // Each class contributes a quarter of its own members to the test set.
    let test_b = test.iter().filter(|&&i| labels[i] == "b").count();
    assert_eq!(test_b, 5);
    assert_eq!(test.len() - test_b, 20);
}

#[test]
// Verifies that stratified_split keeps a rare class present in the test set.
fn stratified_split_keeps_rare_classes_in_the_test_set() {
    // A class holding 2% of the data would often vanish from an unstratified split.
    let labels: Vec<&str> = (0..100)
        .map(|i| if i < 98 { "common" } else { "rare" })
        .collect();

    let (_train, test) = stratified_split(&labels, 0.2, 3).unwrap();

    assert!(test.iter().any(|&i| labels[i] == "rare"));
}

#[test]
// Verifies that a single-member class is assigned to the train set.
fn stratified_split_sends_singleton_classes_to_train() {
    let labels = ["a", "a", "a", "a", "b"];

    let (train, test) = stratified_split(&labels, 0.5, 7).unwrap();

    assert!(train.contains(&4));
    assert!(!test.contains(&4));
    assert_eq!(train.len() + test.len(), 5);
}

#[test]
// Verifies that stratified_split rejects empty labels and out-of-range ratios.
fn stratified_split_rejects_invalid_arguments() {
    let empty: [&str; 0] = [];
    assert!(matches!(
        stratified_split(&empty, 0.2, 1),
        Err(DatasetError::DataFormatError(_))
    ));

    assert!(matches!(
        stratified_split(&["a", "b"], 2.0, 1),
        Err(DatasetError::ValidationError(_))
    ));
}

#[test]
// Verifies that k_fold_indices validates each sample exactly once across folds.
fn k_fold_indices_covers_every_sample_once() {
    let folds = k_fold_indices(100, 5, 42).unwrap();
    assert_eq!(folds.len(), 5);

    let mut validated = Vec::new();
    for (train, validation) in &folds {
        assert_eq!(validation.len(), 20);
        assert_eq!(train.len(), 80);

        // Within a fold, train and validation partition the whole dataset.
        assert_is_permutation(
            &train
                .iter()
                .chain(validation.iter())
                .copied()
                .collect::<Vec<_>>(),
            100,
        );
        validated.extend(validation.iter().copied());
    }

    assert_is_permutation(&validated, 100);
}

#[test]
// Verifies that uneven fold sizes differ by at most one and still cover everything.
fn k_fold_indices_balances_uneven_folds() {
    let folds = k_fold_indices(10, 3, 42).unwrap();

    let sizes: Vec<usize> = folds
        .iter()
        .map(|(_, validation)| validation.len())
        .collect();
    assert_eq!(sizes, vec![4, 3, 3]);
    assert_eq!(sizes.iter().sum::<usize>(), 10);
}

#[test]
// Verifies that k_fold_indices rejects a fold count it cannot honour.
fn k_fold_indices_rejects_invalid_k() {
    assert!(matches!(
        k_fold_indices(10, 1, 1),
        Err(DatasetError::ValidationError(_))
    ));
    assert!(matches!(
        k_fold_indices(10, 11, 1),
        Err(DatasetError::ValidationError(_))
    ));
    assert!(matches!(
        k_fold_indices(0, 2, 1),
        Err(DatasetError::DataFormatError(_))
    ));

    // The boundary cases are accepted.
    assert!(k_fold_indices(10, 2, 1).is_ok());
    assert!(k_fold_indices(10, 10, 1).is_ok());
}

#[test]
// Verifies that label_encode numbers classes in sorted order and round-trips.
fn label_encode_numbers_classes_in_sorted_order() {
    let labels = array!["virginica", "setosa", "setosa", "versicolor"];

    let (codes, classes) = label_encode(&labels).unwrap();

    assert_eq!(classes, vec!["setosa", "versicolor", "virginica"]);
    assert_eq!(codes, array![2, 0, 0, 1]);

    // Every code indexes back to the label it came from.
    for (code, label) in codes.iter().zip(labels.iter()) {
        assert_eq!(classes[*code], *label);
    }
}

#[test]
// Verifies that label_encode works for the non-string label types loaders produce.
fn label_encode_handles_numeric_and_char_labels() {
    let (codes, classes) = label_encode(&array![3u8, 1, 1, 7]).unwrap();
    assert_eq!(classes, vec![1, 3, 7]);
    assert_eq!(codes, array![1, 0, 0, 2]);

    let (codes, classes) = label_encode(&array!['C', 'A', 'B']).unwrap();
    assert_eq!(classes, vec!['A', 'B', 'C']);
    assert_eq!(codes, array![2, 0, 1]);
}

#[test]
// Verifies that label_encode rejects an empty label vector.
fn label_encode_rejects_empty_labels() {
    let empty = ndarray::Array1::<u8>::zeros(0);

    assert!(matches!(
        label_encode(&empty),
        Err(DatasetError::DataFormatError(_))
    ));
}

#[test]
// Verifies that class_counts totals each class in sorted order.
fn class_counts_totals_each_class() {
    let labels = array!["spam", "ham", "ham", "ham"];

    assert_eq!(class_counts(&labels), vec![("ham", 3), ("spam", 1)]);
    assert!(class_counts(&ndarray::Array1::<u8>::zeros(0)).is_empty());
}

#[test]
// Verifies that standardize centres each column and scales it to unit variance.
fn standardize_centres_and_scales_each_column() {
    let features = array![[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]];

    let (scaled, scaler) = standardize(&features).unwrap();

    assert_eq!(scaler.center, array![2.0, 200.0]);

    for column in 0..2 {
        let values: Vec<f64> = (0..3).map(|row| scaled[[row, column]]).collect();
        let mean = values.iter().sum::<f64>() / 3.0;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 3.0;

        assert!(mean.abs() < 1e-12, "column {column} mean is {mean}");
        assert!(
            (variance - 1.0).abs() < 1e-12,
            "column {column} variance is {variance}"
        );
    }
}

#[test]
// Verifies that a constant column maps to zeros instead of dividing by zero.
fn standardize_maps_a_constant_column_to_zeros() {
    let features = array![[5.0], [5.0], [5.0]];

    let (scaled, scaler) = standardize(&features).unwrap();

    assert_eq!(scaler.scale, array![1.0]);
    assert_eq!(scaled, array![[0.0], [0.0], [0.0]]);
}

#[test]
// Verifies that missing values are excluded from the statistics and left in place.
fn standardize_ignores_nan_and_preserves_it() {
    let features = array![[1.0], [f64::NAN], [3.0]];

    let (scaled, scaler) = standardize(&features).unwrap();

    // The mean is over the finite values only: (1 + 3) / 2, not a NaN-poisoned one.
    assert_eq!(scaler.center, array![2.0]);
    assert!(scaled[[1, 0]].is_nan());
    assert_eq!(scaled[[0, 0]], -1.0);
    assert_eq!(scaled[[2, 0]], 1.0);
}

#[test]
// Verifies that min_max_scale maps each column onto [0, 1].
fn min_max_scale_maps_columns_onto_unit_range() {
    let features = array![[1.0, -5.0], [2.0, 0.0], [3.0, 5.0]];

    let (scaled, scaler) = min_max_scale(&features).unwrap();

    assert_eq!(scaler.center, array![1.0, -5.0]);
    assert_eq!(scaler.scale, array![2.0, 10.0]);
    assert_eq!(scaled, array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]);
}

#[test]
// Verifies that min_max_scale also leaves missing values alone.
fn min_max_scale_ignores_nan_and_preserves_it() {
    let features = array![[0.0], [f64::NAN], [10.0]];

    let (scaled, _scaler) = min_max_scale(&features).unwrap();

    assert_eq!(scaled[[0, 0]], 0.0);
    assert!(scaled[[1, 0]].is_nan());
    assert_eq!(scaled[[2, 0]], 1.0);
}

#[test]
// Verifies that the scalers reject an empty feature matrix.
fn scalers_reject_an_empty_matrix() {
    let empty = Array2::<f64>::zeros((0, 3));
    assert!(matches!(
        standardize(&empty),
        Err(DatasetError::DataFormatError(_))
    ));

    let no_columns = Array2::<f64>::zeros((3, 0));
    assert!(matches!(
        min_max_scale(&no_columns),
        Err(DatasetError::DataFormatError(_))
    ));
}

#[test]
// Verifies that a fitted scaler transforms new data with the training statistics.
fn apply_scaler_replays_training_statistics() {
    let train = array![[1.0], [2.0], [3.0]];
    let (_scaled_train, scaler) = standardize(&train).unwrap();

    let test = array![[2.0], [4.0]];
    let scaled_test = apply_scaler(&test, &scaler).unwrap();

    // 2.0 was the training mean, so it maps to 0 — not to the test set's own mean.
    assert_eq!(scaled_test[[0, 0]], 0.0);
    assert!(scaled_test[[1, 0]] > 0.0);

    // Re-applying the scaler to the training data reproduces the fitted output.
    assert_eq!(apply_scaler(&train, &scaler).unwrap(), _scaled_train);
}

#[test]
// Verifies that apply_scaler rejects a matrix with the wrong number of columns.
fn apply_scaler_rejects_a_column_count_mismatch() {
    let (_scaled, scaler) = standardize(&array![[1.0, 2.0], [3.0, 4.0]]).unwrap();

    assert!(matches!(
        apply_scaler(&array![[1.0], [2.0]], &scaler),
        Err(DatasetError::DataFormatError(_))
    ));
}

#[test]
// Verifies the layout, naming, and one-hot property of the encoded matrix.
fn one_hot_encode_expands_each_column_into_indicators() {
    let categorical = array![
        ["male".to_string(), "S".to_string()],
        ["female".to_string(), "C".to_string()],
        ["male".to_string(), "C".to_string()],
    ];

    let (encoded, names) = one_hot_encode(&categorical, Some(&["sex", "port"])).unwrap();

    // Levels are sorted within each column, columns keep their original order.
    assert_eq!(names, vec!["sex=female", "sex=male", "port=C", "port=S"]);
    assert_eq!(encoded.shape(), &[3, 4]);
    assert_eq!(encoded.row(0).to_vec(), vec![0.0, 1.0, 0.0, 1.0]); // male, S
    assert_eq!(encoded.row(1).to_vec(), vec![1.0, 0.0, 1.0, 0.0]); // female, C
    assert_eq!(encoded.row(2).to_vec(), vec![0.0, 1.0, 1.0, 0.0]); // male, C

    // Exactly one indicator is set per source column, in every row.
    for row in encoded.rows() {
        assert_eq!(row.sum(), 2.0);
    }
}

#[test]
// Verifies the generated column names when no names are supplied.
fn one_hot_encode_generates_names_when_none_are_given() {
    let categorical = array![["x".to_string()], ["y".to_string()]];

    let (_encoded, names) = one_hot_encode(&categorical, None).unwrap();

    assert_eq!(names, vec!["column_0=x", "column_0=y"]);
}

#[test]
// Verifies that one_hot_encode rejects empty input and a mismatched name list.
fn one_hot_encode_rejects_invalid_input() {
    let categorical = array![["x".to_string(), "y".to_string()]];
    assert!(matches!(
        one_hot_encode(&categorical, Some(&["only_one"])),
        Err(DatasetError::DataFormatError(_))
    ));

    let empty = Array2::<String>::from_shape_vec((0, 0), vec![]).unwrap();
    assert!(matches!(
        one_hot_encode(&empty, None),
        Err(DatasetError::DataFormatError(_))
    ));
}
