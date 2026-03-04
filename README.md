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
- **Flexible API**: Both reference-based and owned data access patterns
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

Each dataset is automatically downloaded from its original source on first use. The documentation comments for each function provide detailed information about the features, labels, and data format. You can also find dataset descriptions in the module documentation.

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

    // Load the Iris dataset (downloads on first call, then cached)
    let (features, labels) = load_iris(download_dir).unwrap();

    println!("Dataset shape: {:?}", features.shape()); // [150, 4]
    println!("First sample: {:?}", features.row(0));
    println!("First label: {}", labels[0]);
}
```

**Note**: The dataset will be automatically downloaded to the specified directory on first use. Subsequent calls will use the cached in-memory version for instant access.

## Owned Data

If you need to modify the data, use the `load_*_owned` functions:

```rust
use rustyml_dataset::iris::load_iris_owned;

fn main() {
    let download_dir = "./datasets";

    // Returns owned copies that can be modified
    let (mut features, mut labels) = load_iris_owned(download_dir).unwrap();

    // Now you can modify the data
    features[[0, 0]] = 5.5;
    labels[0] = "setosa-modified";
}
```

## Dataset Details

### Iris
- **Samples**: 150 | **Features**: 4 | **Task**: Classification
- **Description**: Classic flower species classification (setosa, versicolor, virginica)
- **Features**: Sepal length, sepal width, petal length, petal width

### Boston Housing
- **Samples**: 506 | **Features**: 13 | **Task**: Regression
- **Description**: Predict median home values in Boston suburbs
- **Features**: Crime rate, zoning, industrial proportion, Charles River proximity, NOx concentration, rooms, age, employment distance, highway access, tax rate, pupil-teacher ratio, demographic metrics

### Diabetes
- **Samples**: 768 | **Features**: 8 | **Task**: Classification
- **Description**: Pima Indians diabetes binary classification
- **Features**: Pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree, age

### Titanic
- **Samples**: 891 | **Features**: 11 | **Task**: Classification
- **Description**: Predict passenger survival on the Titanic
- **Features**: Returned as two matrices:
  string features (`Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`) and numeric features (`PassengerId`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`)

### Wine Quality
- **Samples**: 1599 (red) / 4898 (white) | **Features**: 11 | **Task**: Regression
- **Description**: Predict wine quality ratings based on physicochemical properties
- **Features**: Acidity levels, sugar, chlorides, sulfur dioxide, density, pH, sulphates, alcohol

## Performance Considerations

### Download and Caching

The first call to any `load_*` function:
1. Downloads the dataset from the original source (if not already on disk)
2. Extracts and parses the data
3. Caches it in memory using `OnceLock`

Subsequent calls return references to the cached data with zero overhead:

``` rust
let download_dir = "./datasets";

// First call: downloads, parses, and caches data
let (features, labels) = load_iris(download_dir).unwrap();

// Subsequent calls: instant access to cached data (no I/O)
let (features2, labels2) = load_iris(download_dir).unwrap();
```

### Reference vs Owned

- **Reference functions** (`load_*`): Return static references, zero allocation after caching
- **Owned functions** (`load_*_owned`): Clone the cached data, suitable for mutation

Choose reference functions when possible for better performance.

### Offline Usage

Once downloaded, datasets are stored locally in the specified directory. Your application can work offline as long as the files exist on disk. The in-memory cache persists for the lifetime of your program.

## API Reference

Each dataset module provides two functions:

| Function          | Returns            | Use Case                              |
|-------------------|--------------------|---------------------------------------|
| `load_*()`        | Static references  | Read-only access, optimal performance |
| `load_*_owned()`  | Owned copies       | Mutable data, independent copies      |

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
