简体中文 | [English](https://github.com/SomeB1oody/dataset-core/blob/master/crates/dataset-ml/README.md)

# dataset-ml

构建于 [`dataset-core`](https://crates.io/crates/dataset-core) 之上的经典机器学习数据集开箱即用加载器。

[![Rust Version](https://img.shields.io/badge/Rust-1.88+-brown)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/dataset-ml.svg)](https://crates.io/crates/dataset-ml)

## 概述

`dataset-ml` 内置了十个经典 ML 数据集的加载器。每个加载器会：

- 在首次访问时下载源文件（通过 `ureq`）。
- 校验预设的 SHA-256 哈希值，以检测损坏或上游变化。
- 使用 [`ndarray`](https://crates.io/crates/ndarray) 将 CSV 解析为 `Array1` / `Array2`。
- 通过 `dataset_core::Dataset<T, E>` 在内存中缓存解析结果——后续访问会直接返回 `&` 引用，零 I/O。

每个模块同时也是封装 `Dataset<T, E>` 处理具体数据源的完整参考实现。

## 安装

```toml
[dependencies]
dataset-ml = "0.2"
```

## 数据集

| 结构体                                     | 模块路径                                            | 样本数  | 特征数 | 任务类型 | 来源              |
|--------------------------------------------|-----------------------------------------------------|---------|--------|----------|-------------------|
| `Iris`                                     | `dataset_ml::iris`                                  | 150     | 4      | 分类     | UCI ML Repository |
| `BreastCancer`                             | `dataset_ml::breast_cancer`                         | 569     | 30     | 分类     | UCI ML Repository |
| `BostonHousing`                            | `dataset_ml::boston_housing`                        | 506     | 13     | 回归     | UCI ML Repository |
| `CaliforniaHousing`                        | `dataset_ml::california_housing`                    | 20,640  | 8      | 回归     | StatLib（1990 普查） |
| `Diabetes`                                 | `dataset_ml::diabetes`                              | 768     | 8      | 分类     | Kaggle            |
| `Titanic`                                  | `dataset_ml::titanic`                               | 891     | 11     | 分类     | Kaggle            |
| `PalmerPenguins`                           | `dataset_ml::palmer_penguins`                       | 344     | 7      | 分类     | palmerpenguins    |
| `WineRecognition`                          | `dataset_ml::wine_recognition`                      | 178     | 13     | 分类     | UCI ML Repository |
| `RedWineQuality`                           | `dataset_ml::wine_quality::red_wine_quality`        | 1,599   | 11     | 回归     | UCI ML Repository |
| `WhiteWineQuality`                         | `dataset_ml::wine_quality::white_wine_quality`      | 4,898   | 11     | 回归     | UCI ML Repository |

所有结构体也在 crate 根部重新导出，所以 `dataset_ml::Iris`、`dataset_ml::RedWineQuality` 等也可以使用。

## 用法

```rust
use dataset_ml::iris::Iris;

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

每个数据集结构体都遵循相同的模式：

- `new(storage_dir)` — 创建实例（无 I/O 操作）
- `features()` — 特征矩阵的引用
- `labels()` / `targets()` — 标签/目标向量的引用
- `data()` — 一次性获取所有引用

> **注意**：Titanic 和 Palmer Penguins 是混合类型数据集：`features()` 返回 `(&Array2<String>, &Array2<f64>)`（字符串特征 + 数值特征），`data()` 返回三元组。Palmer Penguins 还会把缺失值表示为 `NaN`（数值）或 `""`（字符串）。
>
> **注意**：California Housing 复现了 scikit-learn `fetch_california_housing` 的特征——从原始普查列派生（例如 `AveRooms = total_rooms / households`），并把目标缩放 `1/100000`。源文件中 207 个缺失的 `total_bedrooms` 会让派生特征 `AveBedrms` 出现 `NaN`。

## 从 `dataset-core` 0.1.x 迁移

如果你之前使用 `dataset-core` 0.1.x 的 `datasets` 特性，请切换到本 crate：

```diff
- dataset-core = { version = "0.1", features = ["datasets"] }
+ dataset-ml = "0.2"
```

| 旧路径                                                                         | 新路径                                                         |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|
| `dataset_core::datasets::iris::Iris`                                           | `dataset_ml::iris::Iris`                                       |
| `dataset_core::datasets::boston_housing::BostonHousing`                        | `dataset_ml::boston_housing::BostonHousing`                    |
| `dataset_core::datasets::diabetes::Diabetes`                                   | `dataset_ml::diabetes::Diabetes`                               |
| `dataset_core::datasets::titanic::Titanic`                                     | `dataset_ml::titanic::Titanic`                                 |
| `dataset_core::datasets::wine_quality::red_wine_quality::RedWineQuality`       | `dataset_ml::wine_quality::red_wine_quality::RedWineQuality`   |
| `dataset_core::datasets::wine_quality::white_wine_quality::WhiteWineQuality`   | `dataset_ml::wine_quality::white_wine_quality::WhiteWineQuality` |

`dataset_core::utils::*` 和 `dataset_core::DatasetError` 保持不变——它们仍位于 [`dataset-core`](https://crates.io/crates/dataset-core) 的 `utils` 特性之下。

## 性能考量

- **首次访问**：下载文件（如果磁盘上不存在）、校验 SHA-256、解析并缓存到内存。
- **后续访问**：返回缓存数据的引用——零分配、零 I/O。
- **`.to_owned()`**：将缓存数据克隆为新的拥有值——仅在需要修改时使用。
- **离线使用**：下载后数据集存储在磁盘上；后续运行无需网络连接。

## 许可证

本项目采用 MIT 许可证——详见 [LICENSE](../../LICENSE)。

## 数据集归属

内置数据集是广泛用于教学和研究目的的经典机器学习数据集：

- **Iris**：Fisher 的鸢尾花数据集（1936）
- **Breast Cancer Wisconsin（诊断）**：Wolberg、Mangasarian、Street & Street（1995）
- **Boston Housing**：Harrison & Rubinfeld（1978）
- **California Housing**：Pace & Barry（1997），源自 1990 年美国普查
- **Diabetes**：Pima 印第安人糖尿病数据库
- **Titanic**：Kaggle Titanic 数据集
- **Palmer Penguins**：Horst、Hill & Gorman（2020）；原始数据 Gorman、Williams & Fraser（2014）
- **Wine Recognition**：Aeberhard & Forina（1991），UCI 机器学习数据库
- **Wine Quality**：UCI 机器学习数据库

## 作者

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
