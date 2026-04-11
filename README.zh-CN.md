# dataset-core

一个通用的、线程安全的数据集容器，支持惰性加载和缓存，适用于 Rust。

[![Rust Version](https://img.shields.io/badge/Rust-1.88+-brown)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/dataset-core.svg)](https://crates.io/crates/dataset-core)

## 概述

`dataset-core` 提供了 `Dataset<T>`，一个轻量级封装，将存储目录与任意类型 `T` 的惰性初始化值配对。实际的加载逻辑由调用者通过闭包提供，因此 `Dataset<T>` 可以与任何数据源配合使用——本地文件、远程 URL、数据库或内存生成。

第一次调用 `load()` 会执行闭包并通过 `OnceLock` 缓存结果；之后的每次调用都会返回缓存值的引用，即使在多线程环境下也是零开销。

在此核心类型之上，还有两个**可选的**特性门控模块：

- **`utils`** — 用于下载文件、解压归档、验证 SHA-256 哈希值和管理临时目录的辅助工具。
- **`datasets`** — 开箱即用的经典机器学习数据集加载器，同时也作为展示如何封装 `Dataset<T>` 的参考实现。

## 安装

**仅核心功能**（零依赖）：

```toml
[dependencies]
dataset-core = "*"
```

**包含工具函数**：

```toml
[dependencies]
dataset-core = { version = "*", features = ["utils"] }
```

**包含内置数据集**（隐含启用 `utils`）：

```toml
[dependencies]
dataset-core = { version = "*", features = ["datasets"] }
```

## 特性标志

| 特性       | 启用的功能                                                     | 额外依赖                                  |
|------------|----------------------------------------------------------------|-------------------------------------------|
| *（无）*   | 仅 `Dataset<T>`                                                | 无                                        |
| `utils`    | 下载、解压、临时目录、SHA-256 验证、错误类型                   | ureq, zip, tempfile, sha2           |
| `datasets` | 所有内置数据集加载器（隐含启用 `utils`）                       | ndarray, csv（+ `utils` 中的所有依赖）    |

## 核心用法

```rust
use dataset_core::Dataset;

fn my_loader(dir: &str) -> Result<Vec<String>, std::io::Error> {
    // 在实际使用中，你会从 `dir` 读取/下载文件。
    Ok(vec!["hello".to_string(), "world".to_string()])
}

fn main() {
    let ds: Dataset<Vec<String>> = Dataset::new("./my_data");

    // 第一次调用会运行加载器并缓存结果。
    let data = ds.load(my_loader).unwrap();
    assert_eq!(data.len(), 2);

    // 之后的调用会即时返回缓存的引用。
    let data_again = ds.load(my_loader).unwrap();
    assert!(std::ptr::eq(data, data_again)); // 相同的引用，无需重新加载    
}
```

### `Dataset<T>` API

| 方法            | 返回值        | 描述                                                     |
|-----------------|---------------|----------------------------------------------------------|
| `new(dir)`      | `Dataset<T>`  | 创建实例（无 I/O 操作）                                 |
| `load(loader)`  | `Result<&T, E>` | 首次调用时运行 `loader`，之后返回缓存的 `&T`           |
| `is_loaded()`   | `bool`        | 数据是否已加载                                           |
| `storage_dir()` | `&str`        | 存储目录路径                                             |

## 内置数据集（特性 `datasets`）

| 数据集               | 样本数  | 特征数   | 任务类型       | 来源                |
|----------------------|---------|----------|----------------|---------------------|
| Iris                 | 150     | 4        | 分类           | UCI ML Repository   |
| Boston Housing       | 506     | 13       | 回归           | UCI ML Repository   |
| Diabetes             | 768     | 8        | 分类           | Kaggle              |
| Titanic              | 891     | 11       | 分类           | Kaggle              |
| Wine Quality (Red)   | 1,599   | 11       | 回归           | UCI ML Repository   |
| Wine Quality (White) | 4,898   | 11       | 回归           | UCI ML Repository   |

```rust
use dataset_core::datasets::iris::Iris;

fn main() {
    let iris = Iris::new("./data");

    // 惰性加载：首次访问时下载并解析，之后使用缓存。
    let features = iris.features().unwrap();  // &Array2<f64>
    let labels   = iris.labels().unwrap();    // &Array1<String>

    // 或者一次性获取全部数据：
    let (features, labels) = iris.data().unwrap();

    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);

    // 当需要可变副本时使用 .to_owned()。
    let mut owned = features.to_owned();
    owned[[0, 0]] = 5.5;
}
```

每个内置数据集结构体都遵循相同的模式：

- `new(storage_dir)` — 创建实例（无 I/O 操作）
- `features()` — 特征矩阵的引用
- `labels()` / `targets()` — 标签/目标向量的引用
- `data()` — 一次性获取所有引用

> **注意**：Titanic 的 `features()` 返回 `(&Array2<String>, &Array2<f64>)`（字符串特征 + 数值特征），`data()` 返回三元组。

## 工具函数（特性 `utils`）

| 函数                   | 用途                                                          |
|------------------------|---------------------------------------------------------------|
| `download_to`          | 将远程文件下载到目录                                          |
| `unzip`                | 解压 ZIP 归档                                                 |
| `create_temp_dir`      | 创建自动清理的临时目录                                        |
| `file_sha256_matches`  | 验证文件的 SHA-256 哈希值                                     |
| `acquire_dataset`      | 缓存感知的数据集获取：复用有效本地文件、临时目录准备、哈希校验、移动 |

## 构建自己的数据集

`datasets` 模块中的内置数据集展示了封装 `Dataset<T>` 的推荐模式。以下是一个简化的大纲：

```rust,ignore
use dataset_core::Dataset;

pub struct MyDataset {
    inner: Dataset<(Vec<f64>, Vec<String>)>,
}

impl MyDataset {
    pub fn new(storage_dir: &str) -> Self {
        Self { inner: Dataset::new(storage_dir) }
    }

    pub fn data(&self) -> Result<&(Vec<f64>, Vec<String>), MyError> {
        self.inner.load(|dir| {
            // 从 `dir` 下载/读取/解析文件……
            Ok((vec![1.0, 2.0], vec!["a".into(), "b".into()]))
        })
    }
}
```

参见 `src/datasets/iris.rs` 及其他文件，了解包含下载、CSV 解析、SHA-256 验证和 ndarray 集成的完整实际示例。

## 性能考量

- **首次访问**：下载文件（如果磁盘上不存在）、验证 SHA-256、解析并缓存到内存。
- **后续访问**：返回缓存数据的引用——零分配、零 I/O。
- **`.to_owned()`**：将缓存数据克隆为新的拥有值——仅在需要修改时使用。
- **离线使用**：下载后数据集存储在磁盘上；后续运行无需网络连接。

## 许可证

本项目采用 MIT 许可证——详见 [LICENSE](LICENSE)。

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 数据集归属

内置数据集是广泛用于教学和研究目的的经典机器学习数据集：

- **Iris**：Fisher 的鸢尾花数据集（1936）
- **Boston Housing**：Harrison & Rubinfeld（1978）
- **Diabetes**：Pima 印第安人糖尿病数据库
- **Titanic**：Kaggle Titanic 数据集
- **Wine Quality**：UCI 机器学习数据库

## 作者

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
