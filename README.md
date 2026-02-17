# rustyml-dataset

A collection of classic machine learning datasets with ndarray integration and memoization support for Rust.

[![Rust Version](https://img.shields.io/badge/Rust-v.1.85-brown)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/rustyml-dataset.svg)](https://crates.io/crates/rustyml-dataset)

## Overview

`rustyml-dataset` is an extension crate of [`rustyml`](https://crates.io/crates/rustyml). Before rustyml v0.12.0, the database module was built-in. Later, to prevent rustyml from becoming overly complex, it was separated into `rustyml-dataset`. It can be used independently of [`rustyml`](https://crates.io/crates/rustyml), but works best and is most convenient when used in conjunction with [`rustyml`](https://crates.io/crates/rustyml).

`rustyml-dataset` provides easy access to popular machine learning datasets with built-in support for the `ndarray` crate. All datasets are embedded at compile time and use thread-safe memoization for optimal performance, ensuring data is loaded only once and cached for subsequent calls.

## Features

- **Zero-cost abstractions**: Datasets are embedded at compile time
- **Thread-safe memoization**: Uses `OnceLock` for lazy initialization and caching
- **ndarray integration**: All data returned as `ndarray` types (`Array1`, `Array2`)
- **Flexible API**: Both reference-based and owned data access patterns
- **No external dependencies**: Datasets are compiled directly into your binary

## Supported Datasets

| Dataset                  | Samples | Features | Task Type              |
|--------------------------|---------|----------|------------------------|
| Iris                     | 150     | 4        | Classification         |
| Boston Housing           | 506     | 13       | Regression             |
| Diabetes                 | 768     | 8        | Classification         |
| Titanic                  | 891     | 12       | Classification         |
| Wine Quality (Red)       | 1599    | 12       | Regression             |
| Wine Quality (White)     | 4898    | 12       | Regression             |

### What is the format of the data? Where can I find more information about it?

The documentation comments for each function clearly indicate the type of the return value and the corresponding meaning. You can find this information there.

## Getting Started

Add this to your `Cargo.toml`:

```toml
[dependencies]
rustyml-dataset = "*" # use the latest version
```
Then, in your Rust code, write:
```rust
use rustyml_dataset::iris::load_iris;

fn main() {
    // Load the Iris dataset (returns static references)
    let (headers, features, labels) = load_iris();

    println!("Dataset shape: {:?}", features.shape()); // [150, 4]
    println!("Headers: {:?}", headers);
    println!("First sample: {:?}", features.row(0));
    println!("First label: {}", labels[0]);
}
```

## Owned Data

If you need to modify the data, use the `load_*_owned` functions:

```rust
use rustyml_dataset::iris::load_iris_owned;

fn main() {
    let (headers, mut features, mut labels) = load_iris_owned();

    // Now you can modify the data
    features[[0, 0]] = 5.5;
    labels[0] = "Modified-setosa";
}
```

## Available Datasets

- **Iris**: Classic flower classification with 3 species (150 samples, 4 features)
- **Boston Housing**: Housing price prediction for Boston suburbs (506 samples, 13 features)
- **Diabetes**: Pima Indians Diabetes binary classification (768 samples, 8 features)
- **Titanic**: Passenger survival prediction with mixed feature types (891 samples, 12 features)
- **Wine Quality**: Red and white wine quality ratings (1599/4898 samples, 12 features)

## Performance Considerations

### Memoization

All datasets use `OnceLock` for thread-safe lazy initialization. The first call to any `load_*` function parses and caches the data. Subsequent calls return references to the cached data with zero overhead.

``` rust
// First call: loads and caches data
let (headers, features, labels) = load_iris();

// Subsequent calls: instant access to cached data
let (headers2, features2, labels2) = load_iris();
```

### Reference vs Owned

- **Reference functions** (`load_*`): Return static references, zero allocation
- **Owned functions** (`load_*_owned`): Clone the data, suitable for mutation

Choose reference functions when possible for better performance.

## API Reference

Each dataset module provides two functions:

| Function          | Returns            | Use Case                              |
|-------------------|--------------------|---------------------------------------|
| `load_*()`        | Static references  | Read-only access, optimal performance |
| `load_*_owned()`  | Owned copies       | Mutable data, independent copies      |

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/SomeB1oody/RustyML-dataset/blob/master/README.md) file for details.

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
