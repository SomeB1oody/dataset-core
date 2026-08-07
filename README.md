[简体中文](https://github.com/SomeB1oody/dataset-core/blob/master/README.zh-CN.md) | English

# dataset-core workspace

A Cargo workspace for building and consuming Rust dataset loaders. This workspace splits the architecture layer and the built-in dataset implementations into two crates. You depend only on what you need.

[![rustc](https://img.shields.io/badge/rustc-1.88%2B-brown)](https://www.rust-lang.org/) [![edition](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/) [![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/dataset-core/blob/master/LICENSE)

[![CI](https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/ci.yml?branch=master&label=CI)](https://github.com/SomeB1oody/dataset-core/actions/workflows/ci.yml)

## Crates

| Crate                                        | Path                       | What it provides                                                                                                                                       |
|----------------------------------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`dataset-core`](crates/dataset-core)        | `crates/dataset-core`      | The `Dataset<T, E>` type (thread-safe, lazy, cached) and the optional `utils` module (download, unzip, gunzip, tar / tar.gz extraction, SHA-256, etc.) |
| [`dataset-ml`](crates/dataset-ml)            | `crates/dataset-ml`        | Ready-to-use loaders for 29 classic ML datasets (Iris, Adult, Titanic, Covtype, Abalone, SMS Spam, 20 Newsgroups, …), built on `dataset-core`          |

```
dataset-core (workspace root)
├── crates/
│   ├── dataset-core/    architecture: Dataset<T, E>, utils, error
│   └── dataset-ml/      implementations: Iris, Titanic, Wine Quality, ...
├── Cargo.toml           workspace manifest
└── README.md            this file
```

## Which crate do I want?

- **Just need lazy caching for your own data?** Depend on [`dataset-core`](crates/dataset-core).
- **Want ready-to-use loaders for the classic ML datasets?** Depend on [`dataset-ml`](crates/dataset-ml). It includes `dataset-core` automatically.

``` toml
# Minimal: just Dataset<T, E>
[dependencies]
dataset-core = "0.5"

# Need download / unzip / gunzip / untar / SHA-256 helpers too
[dependencies]
dataset-core = { version = "0.5", features = ["utils"] }

# Built-in ML datasets (Iris, Titanic, ...) plus the preprocessing helpers
[dependencies]
dataset-ml = "0.4"

# Only the loaders, without the preprocessing helpers
[dependencies]
dataset-ml = { version = "0.4", default-features = false, features = ["dataset"] }
```

## Development

This workspace uses Rust edition 2024, MSRV 1.88.0.

```bash
# Build everything
cargo build --workspace --all-features

# Check a single crate
cargo check -p dataset-core
cargo check -p dataset-core --features utils
cargo check -p dataset-ml

# Run tests (most dataset-ml tests perform real network downloads)
cargo test -p dataset-core --features utils
cargo test -p dataset-ml
cargo test --workspace --all-features

# Docs
cargo doc --workspace --all-features --no-deps --open

# Lint & format
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo fmt --all
```

## Changelogs

Each crate has its own changelog:

- [`crates/dataset-core/CHANGELOG.md`](crates/dataset-core/CHANGELOG.md)
- [`crates/dataset-ml/CHANGELOG.md`](crates/dataset-ml/CHANGELOG.md)

## License

This project uses the MIT License. See [LICENSE](LICENSE) for details.

## Code of Conduct

Review the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Contributing

Contributions are welcome. Open an issue or a pull request on [SomeB1oody/dataset-core](https://github.com/SomeB1oody/dataset-core).

## Author

**SomeB1oody**: [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
