# rustyml-dataset

A collection of classic machine learning datasets with automatic download, caching, and ndarray integration for Rust.

[![Rust Version](https://img.shields.io/badge/Rust-v.1.85-brown)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/rustyml-dataset.svg)](https://crates.io/crates/rustyml-dataset)

## Overview

`rustyml-dataset` is an extension crate of [`rustyml`](https://crates.io/crates/rustyml). Before rustyml v0.12.0, the database module was built-in. Later, to prevent rustyml from becoming overly complex, it was separated into `rustyml-dataset`. It can be used independently of [`rustyml`](https://crates.io/crates/rustyml), but works best and is most convenient when used in conjunction with [`rustyml`](https://crates.io/crates/rustyml).

`rustyml-dataset` provides easy access to popular machine learning datasets with built-in support for the `ndarray` crate. Datasets are automatically downloaded from their original sources on first use and cached in memory using thread-safe `OnceLock` for optimal performance, ensuring data is loaded only once and cached for subsequent calls.

## Features

- **Automatic downloading**: Datasets are fetched from original sources on demand
- **Thread-safe memoization**: Uses `OnceLock` for lazy initialization and caching
- **ndarray integration**: All data returned as `ndarray` types (`Array1`, `Array2`)
- **Struct-based API**: Each dataset is a struct with lazy-loading accessor methods
- **Local storage**: Downloaded datasets are stored locally for offline access
- **Minimal binary size**: Datasets are not embedded in your binary

## Supported Datasets

| Dataset                  | Samples | Features | Task Type      | Source                    |
|--------------------------|---------|----------|----------------|---------------------------|
| Iris                     | 150     | 4        | Classification | UCI ML Repository         |
| Boston Housing           | 506     | 13       | Regression     | UCI ML Repository         |
| Diabetes                 | 768     | 8        | Classification | Kaggle                    |
| Titanic                  | 891     | 11       | Classification | Kaggle                    |
| Wine Quality (Red)       | 1599    | 11       | Regression     | UCI ML Repository         |
| Wine Quality (White)     | 4898    | 11       | Regression     | UCI ML Repository         |

### Dataset Information

Each dataset is automatically downloaded from its original source on first use. The documentation comments for each struct and method provide detailed information about the features, labels/targets, and data format.

## Getting Started

Add this to your `Cargo.toml`:

```toml
[dependencies]
rustyml-dataset = "*" # use the latest version
```

Then, in your Rust code:

```rust
use rustyml_dataset::iris::load_iris;

fn main() {
    // Specify where to store downloaded datasets
    let download_dir = "./datasets";

    // Create a dataset instance (no I/O happens yet)
    let dataset = Iris::new(download_dir);

    // Access data — downloads on first call, then cached in memory
    let (features, labels) = dataset.data().unwrap();

    println!("Dataset shape: {:?}", features.shape()); // [150, 4]
    println!("First sample: {:?}", features.row(0));
    println!("First label: {}", labels[0]);
}
```

**Note**: The dataset will be automatically downloaded to the specified directory on first use. Subsequent calls will use the cached in-memory version for instant access.

## Modifying Data / Owned References

If you need to modify the data or owned references, call `.to_owned()` on the returned reference:

```rust
use rustyml_dataset::iris::load_iris_owned;

fn main() {
    let dataset = Iris::new(download_dir);
    let (features, labels) = dataset.data().unwrap();

    // Clone into owned arrays that can be modified
    let mut features_owned = features.to_owned();
    let mut labels_owned = labels.to_owned();

    // Now you can modify the data
    features_owned[[0, 0]] = 5.5;
    labels_owned[0] = "setosa-modified";
}
```

## Dataset Details

### Iris
- **Samples**: 150 | **Features**: 4 | **Task**: Classification
- **Struct**: `rustyml_dataset::iris::Iris`
- **Description**: Classic flower species classification (setosa, versicolor, virginica)
- **Features**: Sepal length, sepal width, petal length, petal width
- **Labels** (`labels()`): Species name as `&str` — `"setosa"`, `"versicolor"`, `"virginica"`

### Boston Housing
- **Samples**: 506 | **Features**: 13 | **Task**: Regression
- **Struct**: `rustyml_dataset::boston_housing::BostonHousing`
- **Description**: Predict median home values in Boston suburbs
- **Features**: Crime rate, zoning, industrial proportion, Charles River proximity, NOx concentration, rooms, age, employment distance, highway access, tax rate, pupil-teacher ratio, demographic metrics, lower status percentage
- **Targets** (`targets()`): Median value of owner-occupied homes in $1000's (MEDV)

### Diabetes
- **Samples**: 768 | **Features**: 8 | **Task**: Classification
- **Struct**: `rustyml_dataset::diabetes::Diabetes`
- **Description**: Pima Indians diabetes binary classification
- **Features**: Pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree, age
- **Labels** (`labels()`): Outcome — `0.0` or `1.0`

### Titanic
- **Samples**: 891 | **Task**: Classification
- **Struct**: `rustyml_dataset::titanic::Titanic`
- **Description**: Predict passenger survival on the Titanic
- **Features** (`features()`): Returns two matrices —
  - String features `(891, 5)`: `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`
  - Numeric features `(891, 6)`: `PassengerId`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`
- **Labels** (`labels()`): `Survived` — `0.0` or `1.0` (`NaN` if missing)
- **Data** (`data()`): Returns `(string_features, numeric_features, labels)`

### Wine Quality
- **Samples**: 1599 (red) / 4898 (white) | **Features**: 11 | **Task**: Regression
- **Structs**: `rustyml_dataset::wine_quality::RedWineQuality` / `WhiteWineQuality`
- **Description**: Predict wine quality ratings based on physicochemical properties
- **Features**: Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- **Targets** (`targets()`): Quality score (0–10)

## Performance Considerations

### Download and Caching

The first call to any accessor method on a dataset struct:
1. Downloads the dataset from the original source (if not already on disk)
2. Validates the file via SHA256 hash
3. Parses the data and caches it in memory using `OnceLock`

Subsequent calls return references to the cached data with zero overhead:

```rust
use rustyml_dataset::iris::load_iris;

fn main() {
    let download_dir = "./datasets";
  
    let dataset = Iris::new(download_dir);

    // First call: downloads, parses, and caches data 
    let (features, labels) = dataset.data().unwrap();

    // Subsequent calls: instant access to cached data (no I/O) 
    let (features2, labels2) = dataset.data().unwrap(); 
}
```

### References vs Owned Copies

- **Accessor methods** (`features()`, `labels()`, `targets()`, `data()`): Return references to the in-memory cache — zero allocation after the first call.
- **`.to_owned()`**: Clones the cached data into a new owned array — use only when you need to mutate the data.

### Offline Usage

Once downloaded, datasets are stored locally in the specified directory. Your application can work offline as long as the files exist on disk. The in-memory cache persists for the lifetime of the dataset struct.

## API Reference

Each dataset module provides a struct with the following methods:

| Method            | Returns                      | Use Case                              |
|-------------------|------------------------------|---------------------------------------|
| `new(path)`       | Dataset struct               | Create instance (no I/O)              |
| `features()`      | Reference(s) to feature data | Read feature matrix                   |
| `labels()`        | Reference to label vector    | Read labels (classification datasets) |
| `targets()`       | Reference to target vector   | Read targets (regression datasets)    |
| `data()`          | All references at once       | Read features + labels/targets        |

> **Note**: The Titanic dataset's `features()` returns a tuple of `(&Array2<String>, &Array2<f64>)`, and `data()` returns a triple `(&Array2<String>, &Array2<f64>, &Array1<f64>)`.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Datasets Attribution

The datasets included in this crate are classic machine learning datasets widely used for educational and research purposes:

- **Iris**: Fisher's Iris dataset (1936)
- **Boston Housing**: Harrison & Rubinfeld (1978)
- **Diabetes**: Pima Indians Diabetes Database
- **Titanic**: Kaggle Titanic dataset
- **Wine Quality**: UCI Machine Learning Repository

## Author

**SomeB1oody** – [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
