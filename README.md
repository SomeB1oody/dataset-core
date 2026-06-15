[简体中文](https://github.com/SomeB1oody/dataset-core/blob/master/README.zh-CN.md) | English

# dataset-core workspace

A Cargo workspace for building and consuming Rust dataset loaders. The architecture layer and the built-in dataset implementations are split into two crates so you only depend on what you actually need.

<p align="center">
  <a href="https://www.rust-lang.org/"><img alt="rustc" src="https://img.shields.io/badge/rustc-1.88%2B-brown"></a>
  <a href="https://doc.rust-lang.org/edition-guide/"><img alt="edition" src="https://img.shields.io/badge/edition-2024-orange"></a>
  <a href="https://github.com/SomeB1oody/dataset-core/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a>
  <br>
  <a href="https://github.com/SomeB1oody/dataset-core/actions/workflows/fmt.yml"><img alt="fmt" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/fmt.yml?branch=master&label=fmt"></a>
  <a href="https://github.com/SomeB1oody/dataset-core/actions/workflows/clippy.yml"><img alt="clippy" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/clippy.yml?branch=master&label=clippy"></a>
  <a href="https://github.com/SomeB1oody/dataset-core/actions/workflows/test.yml"><img alt="test" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/test.yml?branch=master&label=test"></a>
  <a href="https://github.com/SomeB1oody/dataset-core/actions/workflows/doc.yml"><img alt="doc" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/doc.yml?branch=master&label=doc"></a>
</p>

## Crates

| Crate                                        | Path                       | What it provides                                                                                                  |
|----------------------------------------------|----------------------------|-------------------------------------------------------------------------------------------------------------------|
| [`dataset-core`](crates/dataset-core)        | `crates/dataset-core`      | The `Dataset<T, E>` type (thread-safe, lazy, cached) and the optional `utils` module (download, unzip, SHA-256, etc.) |
| [`dataset-ml`](crates/dataset-ml)            | `crates/dataset-ml`        | Ready-to-use loaders for 12 classic ML datasets (Iris, Breast Cancer, Boston/California Housing, Diabetes, Digits, Linnerud, Titanic, Palmer Penguins, Wine Recognition, Red/White Wine Quality), built on `dataset-core` |

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
- **Want the classic ML datasets out of the box?** Depend on [`dataset-ml`](crates/dataset-ml) — it pulls in `dataset-core` automatically.

``` toml
# Minimal: just Dataset<T, E>
[dependencies]
dataset-core = "0.3"

# Need download / unzip / SHA-256 helpers too
[dependencies]
dataset-core = { version = "0.3", features = ["utils"] }

# Built-in ML datasets (Iris, Titanic, ...)
[dependencies]
dataset-ml = "0.2"
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

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Code of Conduct

Please review the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Contributing

Contributions are welcome! Please open an issue or pull request on [SomeB1oody/dataset-core](https://github.com/SomeB1oody/dataset-core).

## Author

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
