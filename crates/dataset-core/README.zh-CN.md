简体中文 | [English](https://github.com/SomeB1oody/dataset-core/blob/master/crates/dataset-core/README.md)

# dataset-core

一个通用的、线程安全的数据集容器，支持惰性加载和缓存，适用于 Rust。

[![Rust Version](https://img.shields.io/badge/Rust-1.88+-brown)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/dataset-core.svg)](https://crates.io/crates/dataset-core)

## 概述

`dataset-core` 提供了 `Dataset<T, E>`，一个轻量级封装，将存储目录与任意类型 `T` 的惰性初始化值配对。实际的加载逻辑由调用者通过在构造时存入的闭包提供，因此 `Dataset<T, E>` 可以与任何数据源配合使用——本地文件、远程 URL、数据库或内存生成。（`E` 为加载器的错误类型，由调用者自由选择。）

第一次调用 `load()` 会执行闭包并通过 `OnceLock` 缓存结果；之后的每次调用都会返回缓存值的引用，即使在多线程环境下也是零开销。

在此核心类型之上，还有一个**可选**的特性门控模块：

- **`utils`** — 用于下载文件、解压归档、验证 SHA-256 哈希值和管理临时目录的辅助工具。

需要经典机器学习数据集（Iris、Breast Cancer、Boston/California Housing、Diabetes、Titanic、Palmer Penguins、Wine Recognition、Wine Quality）的开箱即用加载器？请参见同一工作区中的配套 crate [`dataset-ml`](https://crates.io/crates/dataset-ml)，它在启用 `utils` 特性的前提下依赖 `dataset-core`。

## 安装

**仅核心功能**（零依赖）：

```toml
[dependencies]
dataset-core = "0.3"
```

**包含工具函数**：

```toml
[dependencies]
dataset-core = { version = "0.3", features = ["utils"] }
```

## 特性标志

| 特性     | 启用的功能                                          | 额外依赖                                  |
|----------|-----------------------------------------------------|-------------------------------------------|
| *（无）* | 仅 `Dataset<T, E>`                                  | 无                                        |
| `utils`  | 下载、解压、临时目录、SHA-256 验证、错误类型        | ureq, zip, tempfile, sha2, thiserror      |

## 核心用法

```rust
use dataset_core::Dataset;

fn my_loader(dir: &str) -> Result<Vec<String>, std::io::Error> {
    // 在实际使用中，你会从 `dir` 读取/下载文件。
    Ok(vec!["hello".to_string(), "world".to_string()])
}

fn main() {
    // 加载器在构造时一次性传入。
    let ds: Dataset<Vec<String>, std::io::Error> = Dataset::new("./my_data", my_loader);

    // 第一次调用会运行加载器并缓存结果。
    let data = ds.load().unwrap();
    assert_eq!(data.len(), 2);

    // 之后的调用会即时返回缓存的引用。
    let data_again = ds.load().unwrap();
    assert!(std::ptr::eq(data, data_again)); // 相同的引用，无需重新加载
}
```

### `Dataset<T, E>` API

| 方法                 | 返回值          | 描述                                              |
|----------------------|-----------------|---------------------------------------------------|
| `new(dir, loader)`   | `Dataset<T, E>` | 创建实例并存入加载器（无 I/O 操作）              |
| `load()`             | `Result<&T, E>` | 首次调用时运行存好的加载器，之后返回缓存的 `&T`  |
| `set_loader(loader)` | `()`            | 替换加载器并使缓存失效（下次访问惰性重新解析）   |
| `invalidate()`       | `()`            | 丢弃缓存值、保留加载器（下次 `load` 用它重载）   |
| `is_loaded()`        | `bool`          | 数据是否已加载                                    |
| `storage_dir()`      | `&str`          | 存储目录路径                                      |

## 工具函数（特性 `utils`）

| 函数                  | 用途                                                                             |
|-----------------------|----------------------------------------------------------------------------------|
| `download_to`         | 将远程文件下载到目录                                                             |
| `unzip`               | 解压 ZIP 归档                                                                    |
| `acquire_dataset`     | 缓存感知的数据集获取：复用有效本地文件、临时目录准备、哈希校验、移动到最终位置   |

## 构建自己的数据集

`Dataset<T, E>` 设计成被封装使用。配套 crate [`dataset-ml`](https://crates.io/crates/dataset-ml) 展示了推荐的模式；以下是一个简化的大纲：

```rust,ignore
use dataset_core::Dataset;

pub struct MyDataset {
    inner: Dataset<(Vec<f64>, Vec<String>), MyError>,
}

impl MyDataset {
    pub fn new(storage_dir: &str) -> Self {
        Self {
            inner: Dataset::new(storage_dir, |dir| {
                // 从 `dir` 下载/读取/解析文件……
                Ok((vec![1.0, 2.0], vec!["a".into(), "b".into()]))
            }),
        }
    }

    pub fn data(&self) -> Result<&(Vec<f64>, Vec<String>), MyError> {
        self.inner.load()
    }
}
```

参见 [`dataset-ml`](https://crates.io/crates/dataset-ml) 源码，了解包含下载、CSV 解析、SHA-256 验证和 ndarray 集成的完整实际示例。

## 性能考量

- **首次访问**：运行一次加载器（可能涉及网络请求和解析），缓存结果。
- **后续访问**：返回缓存数据的引用——零分配、零 I/O。
- **跨线程安全**：只要 `T` 是 `Send + Sync`，`Dataset<T, E>` 就是 `Send + Sync`（存入的加载器始终是 `Send + Sync`）；内部 `OnceLock` 保证即使在并发调用下加载器也最多执行一次。

## 许可证

本项目采用 MIT 许可证——详见 [LICENSE](../../LICENSE)。

## 作者

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
