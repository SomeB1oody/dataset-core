简体中文 | [English](https://github.com/SomeB1oody/dataset-core/blob/master/README.md)

# dataset-core 工作区

一个用于构建和使用 Rust 数据集加载器的 Cargo 工作区。架构层与内置数据集实现拆分为两个 crate，按需依赖。

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

## 包含的 crate

| Crate                                     | 路径                       | 提供的内容                                                                                                          |
|-------------------------------------------|----------------------------|---------------------------------------------------------------------------------------------------------------------|
| [`dataset-core`](crates/dataset-core)     | `crates/dataset-core`      | `Dataset<T, E>` 类型（线程安全、惰性、缓存）以及可选的 `utils` 模块（下载、解压、SHA-256 等）                       |
| [`dataset-ml`](crates/dataset-ml)         | `crates/dataset-ml`        | 基于 `dataset-core` 的 23 个经典 ML 数据集开箱即用加载器：Iris、Adult、Titanic、Covtype、Abalone、SMS Spam 等 |

```
dataset-core （工作区根目录）
├── crates/
│   ├── dataset-core/    架构层：Dataset<T, E>、utils、error
│   └── dataset-ml/      实现层：Iris、Titanic、Wine Quality 等
├── Cargo.toml           工作区清单
└── README.md            本文件
```

## 我应该使用哪个 crate？

- **只需要为自己的数据做惰性缓存？** 依赖 [`dataset-core`](crates/dataset-core)。
- **想直接使用经典 ML 数据集？** 依赖 [`dataset-ml`](crates/dataset-ml)——它会自动引入 `dataset-core`。

``` toml
# 最小依赖：仅 Dataset<T, E>
[dependencies]
dataset-core = "0.3"

# 还需要下载 / 解压 / SHA-256 辅助函数
[dependencies]
dataset-core = { version = "0.3", features = ["utils"] }

# 内置机器学习数据集（Iris、Titanic 等）
[dependencies]
dataset-ml = "0.2"
```

## 开发

本工作区使用 Rust edition 2024，MSRV 1.88.0。

```bash
# 构建整个工作区
cargo build --workspace --all-features

# 单独检查某个 crate
cargo check -p dataset-core
cargo check -p dataset-core --features utils
cargo check -p dataset-ml

# 运行测试（大多数 dataset-ml 测试会进行真实的网络下载）
cargo test -p dataset-core --features utils
cargo test -p dataset-ml
cargo test --workspace --all-features

# 生成文档
cargo doc --workspace --all-features --no-deps --open

# Lint 与格式化
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo fmt --all
```

## 变更日志

每个 crate 各自维护变更日志：

- [`crates/dataset-core/CHANGELOG.md`](crates/dataset-core/CHANGELOG.md)
- [`crates/dataset-ml/CHANGELOG.md`](crates/dataset-ml/CHANGELOG.md)

## 许可证

本项目采用 MIT 许可证——详见 [LICENSE](LICENSE)。

## 行为准则

参与贡献前请阅读 [Code of Conduct](CODE_OF_CONDUCT.md)。

## 贡献

欢迎贡献！请在 [SomeB1oody/dataset-core](https://github.com/SomeB1oody/dataset-core) 提交 issue 或 pull request。

## 作者

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
