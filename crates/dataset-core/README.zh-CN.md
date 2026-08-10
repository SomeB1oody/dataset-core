简体中文 | [English](https://github.com/SomeB1oody/dataset-core/blob/master/crates/dataset-core/README.md)

# dataset-core

一个通用的、线程安全的数据集容器，支持惰性加载和缓存，适用于 Rust。

[![rustc](https://img.shields.io/badge/rustc-1.88%2B-brown)](https://www.rust-lang.org/) [![edition](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/) [![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/dataset-core/blob/master/LICENSE) [![crates.io](https://img.shields.io/crates/v/dataset-core.svg)](https://crates.io/crates/dataset-core)

[![CI](https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/ci.yml?branch=master&label=CI)](https://github.com/SomeB1oody/dataset-core/actions/workflows/ci.yml)

## 概述

`dataset-core` 提供了 `Dataset<T, E>`，一个轻量级封装，将存储目录与任意类型 `T` 的惰性初始化值配对。加载逻辑由调用者通过构造时存入的闭包提供。因此，`Dataset<T, E>` 可以配合任何数据源使用：本地文件、远程 URL、数据库，或内存生成。（`E` 是加载器的错误类型，由调用者自由选择。）

第一次调用 `load()` 会执行闭包，并通过 `OnceLock` 缓存结果。之后的每次调用都会返回缓存值的引用，即使在多线程环境下也是零开销。

在此核心类型之上，本 crate 还提供一个**可选**的特性门控模块：

- **`utils`**：用于下载文件、解压归档、验证 SHA-256 哈希值和管理临时目录的辅助工具。

如需经典机器学习数据集的开箱即用加载器，参见配套 crate [`dataset-ml`](https://crates.io/crates/dataset-ml)。

## 安装

**仅核心功能**（零依赖）：

```toml
[dependencies]
dataset-core = "0.5"
```

**包含工具函数**：

```toml
[dependencies]
dataset-core = { version = "0.5", features = ["utils"] }
```

## 特性标志

| 特性     | 启用的功能                                                                                                 | 额外依赖                                          |
|----------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| *（无）* | 仅 `Dataset<T, E>`                                                                                         | 无                                                |
| `utils`  | 下载（可选自动重试）、unzip、gunzip、untar、untar_gz、临时目录、SHA-256 计算与验证、Latin-1 读取、错误类型 | ureq, zip, flate2, tar, tempfile, sha2, thiserror |

## 核心用法

```rust
use dataset_core::Dataset;

fn my_loader(dir: &str) -> Result<Vec<String>, std::io::Error> {
    // 实际的加载器会从 `dir` 读取或下载文件。
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

| 方法                  | 返回值                          | 描述                                            |
|-----------------------|---------------------------------|-------------------------------------------------|
| `new(dir, loader)`    | `Dataset<T, E>`                 | 创建实例并存入加载器（无 I/O 操作）             |
| `load()`              | `Result<&T, E>`                 | 首次调用时运行存好的加载器，之后返回缓存的 `&T` |
| `load_mut()`          | `Result<&mut T, E>`             | 按需加载后可变借用缓存值，便于原地修改          |
| `get()` / `get_mut()` | `Option<&T>` / `Option<&mut T>` | **不触发加载**地借用缓存值                      |
| `take()`              | `Option<T>`                     | 取出缓存值，容器保持可复用                      |
| `into_inner()`        | `Option<T>`                     | 消耗容器并返回缓存值                            |
| `set_loader(loader)`  | `()`                            | 替换加载器并使缓存失效（下次访问惰性重新解析）  |
| `invalidate()`        | `()`                            | 丢弃缓存值、保留加载器（下次 `load` 用它重载）  |
| `is_loaded()`         | `bool`                          | 该数据集是否已加载数据                          |
| `storage_dir()`       | `&str`                          | 存储目录路径                                    |

## 工具函数（特性 `utils`）

| 函数                       | 用途                                                                                           |
|----------------------------|------------------------------------------------------------------------------------------------|
| `download_to`              | 将远程文件下载到目录                                                                           |
| `download_to_with_retries` | 与 `download_to` 类似，但会以指数退避重试瞬时失败                                              |
| `unzip`                    | 解压 ZIP 归档                                                                                  |
| `gunzip`                   | 将 gzip（`.gz`）文件解压为单个输出文件                                                         |
| `untar`                    | 将 tar（`.tar`）归档解压到目录                                                                 |
| `untar_gz`                 | 将 gzip 压缩的 tar（`.tar.gz` / `.tgz`）归档解压到目录，并以流式方式处理，使中间数据不落盘     |
| `sha256_file`              | 计算文件的 SHA-256 摘要（十六进制格式）。可用它为新数据集固定哈希值                            |
| `verify_sha256`            | 用已有的哈希值校验文件                                                                         |
| `read_latin1`              | 以 Latin-1 读取文件文本：无损，且不会因非 UTF-8 字节而失败                                     |
| `acquire_dataset`          | 缓存感知的数据集获取：复用有效的本地文件、在临时目录中准备文件、校验哈希值，然后移动到最终位置 |

## 构建自己的数据集

你可以将 `Dataset<T, E>` 封装进自己的类型中。配套 crate [`dataset-ml`](https://crates.io/crates/dataset-ml) 展示了推荐的模式。以下是一个简化的大纲：

```rust,ignore
use dataset_core::Dataset;

pub struct MyDataset {
    inner: Dataset<(Vec<f64>, Vec<String>), MyError>,
}

impl MyDataset {
    pub fn new(storage_dir: &str) -> Self {
        Self {
            inner: Dataset::new(storage_dir, |dir| {
                // 在这里从 `dir` 下载、读取或解析文件。
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
- **后续访问**：返回缓存数据的引用，零分配、零 I/O。
- **跨线程安全**：只要 `T` 是 `Send + Sync`，`Dataset<T, E>` 就是 `Send + Sync`（存入的加载器始终是 `Send + Sync`）。即使在并发调用下，加载器也最多执行一次。内部互斥锁会将首次加载串行化。中途到达的线程会等待结果并共享它，而不会各自发起下载。

## 许可证

本项目使用 MIT 许可证。详见 [LICENSE](../../LICENSE)。

## 作者

**SomeB1oody**：[stanyin64@gmail.com](mailto:stanyin64@gmail.com)
