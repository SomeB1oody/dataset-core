简体中文 | [English](https://github.com/SomeB1oody/dataset-core/blob/master/crates/dataset-ml/README.md)

# dataset-ml

构建于 [`dataset-core`](https://crates.io/crates/dataset-core) 之上的经典机器学习数据集开箱即用加载器。

<p align="center">
  <a href="https://www.rust-lang.org/"><img alt="rustc" src="https://img.shields.io/badge/rustc-1.88%2B-brown"></a>
  <a href="https://doc.rust-lang.org/edition-guide/"><img alt="edition" src="https://img.shields.io/badge/edition-2024-orange"></a>
  <a href="https://github.com/SomeB1oody/dataset-core/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a>
  <a href="https://crates.io/crates/dataset-ml"><img alt="crates.io" src="https://img.shields.io/crates/v/dataset-ml.svg"></a>
  <br>
  <a href="https://github.com/SomeB1oody/dataset-core/actions/workflows/fmt.yml"><img alt="fmt" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/fmt.yml?branch=master&label=fmt"></a>
  <a href="https://github.com/SomeB1oody/dataset-core/actions/workflows/clippy.yml"><img alt="clippy" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/clippy.yml?branch=master&label=clippy"></a>
  <a href="https://github.com/SomeB1oody/dataset-core/actions/workflows/test.yml"><img alt="test" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/test.yml?branch=master&label=test"></a>
  <a href="https://github.com/SomeB1oody/dataset-core/actions/workflows/doc.yml"><img alt="doc" src="https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/doc.yml?branch=master&label=doc"></a>
</p>

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
| `Adult`                                    | `dataset_ml::adult`                                 | 32,561  | 14     | 分类     | UCI ML Repository |
| `Iris`                                     | `dataset_ml::iris`                                  | 150     | 4      | 分类     | UCI ML Repository |
| `BreastCancer`                             | `dataset_ml::breast_cancer`                         | 569     | 30     | 分类     | UCI ML Repository |
| `BostonHousing`                            | `dataset_ml::boston_housing`                        | 506     | 13     | 回归     | UCI ML Repository |
| `CaliforniaHousing`                        | `dataset_ml::california_housing`                    | 20,640  | 8      | 回归     | StatLib（1990 普查） |
| `Covtype`                                  | `dataset_ml::covtype`                               | 581,012 | 54     | 分类     | UCI ML Repository |
| `Diabetes`                                 | `dataset_ml::diabetes`                              | 442     | 10     | 回归     | Efron et al.（2004） |
| `Digits`                                   | `dataset_ml::digits`                                | 1,797   | 64     | 分类     | UCI ML Repository |
| `Kddcup99`                                 | `dataset_ml::kddcup99`                              | 494,021 / 4,898,431 | 41 | 分类  | UCI KDD Archive   |
| `Linnerud`                                 | `dataset_ml::linnerud`                              | 20      | 3      | 回归（多输出） | scikit-learn |
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

> **注意**：Titanic、Palmer Penguins、Adult 和 Kddcup99 是混合类型数据集：`features()` 返回 `(&Array2<String>, &Array2<f64>)`（字符串特征 + 数值特征），`data()` 返回三元组。Palmer Penguins 还会把缺失值表示为 `NaN`（数值）或 `""`（字符串）。
>
> **注意**：Adult（人口普查收入）复现了经典的 UCI 数据集，用于预测年收入是否超过 5 万美元：8 个类别特征（`workclass`、`education`、`marital-status`、`occupation`、`relationship`、`race`、`sex`、`native-country`），6 个数值特征（`age`、`fnlwgt`、`education-num`、`capital-gain`、`capital-loss`、`hours-per-week`），以及保持原样的 `Array1<String>` 收入标签（`<=50K` 或 `>50K`）。它加载标准的 `adult.data` 训练分区（32,561 条记录）；源文件中的 `?` 缺失类别标记被映射为空字符串 `""`。
>
> **注意**：California Housing 复现了 scikit-learn `fetch_california_housing` 的特征——从原始普查列派生（例如 `AveRooms = total_rooms / households`），并把目标缩放 `1/100000`。源文件中 207 个缺失的 `total_bedrooms` 会让派生特征 `AveBedrms` 出现 `NaN`。
>
> **注意**：Diabetes 复现了 scikit-learn `load_diabetes`（默认输出）：10 个特征列经过标准化（均值中心化，再除以各自的 L2 范数，使每列的平方和为 1），回归目标保持未缩放。
>
> **注意**：Digits 复现了 scikit-learn `load_digits`：64 个特征均为 `0..=16` 范围内的整数像素强度（一张 8×8 图像按行主序展平），`labels()` 返回数字类别（`0`–`9`）的 `Array1<u8>`。数据来自 UCI 静态 ZIP 包，使用其中的 `optdigits.tes` 测试分区（即 scikit-learn 所用的同一批 1,797 条样本）。
>
> **注意**：Linnerud 复现了 scikit-learn `load_linnerud`（多输出回归）：`features()` 返回三个运动量变量（`Chins`、`Situps`、`Jumps`），`targets()` 返回三个生理量变量（`Weight`、`Waist`、`Pulse`），两者都是形状为 `(20, 3)` 的 `Array2<f64>`。数据来自 scikit-learn 附带的两个以空白分隔的文件。
>
> **注意**：Covtype 复现了 scikit-learn `fetch_covtype`：581,012 条样本，54 个特征（10 个数量型变量、4 个独热编码的 `Wilderness_Area` 列、40 个独热编码的 `Soil_Type` 列），以及 `Array1<u8>` 的森林覆盖类型标签（`1`–`7`）。它是首个以 gzip 压缩文件为源的加载器：下载 `covtype.data.gz` 并用 `dataset-core` 的 `gunzip` 辅助函数解压。
>
> **注意**：Kddcup99 复现了 scikit-learn `fetch_kddcup99`。与 scikit-learn 一样，`Kddcup99::new` 加载**默认的 10% 子集**（494,021 条连接，`percent10=True`），`Kddcup99::new_full` 加载**全量**（4,898,431 条连接，`percent10=False`）；两者共享相同的 41 特征结构与 23 个类别。与 Titanic 一样是混合类型：`features()` 返回 `(&Array2<String>, &Array2<f64>)`——3 个类别特征（`protocol_type`、`service`、`flag`）和 38 个数值特征，`labels()` 返回 `Array1<String>`，标签保持原样（含末尾句点，如 `"normal."`、`"smurf."`）。与 Covtype 一样，源文件经 gzip 压缩并用 `gunzip` 解压。**提示**：全量解压后的源文件约 743 MB，解析后的内存数组达数 GB，`new_full` 会耗费可观的时间与内存；默认子集约小 10 倍。

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

- **Adult / 人口普查收入**：Becker & Kohavi（1996），UCI 机器学习数据库，源自 1994 年美国人口普查
- **Iris**：Fisher 的鸢尾花数据集（1936）
- **Breast Cancer Wisconsin（诊断）**：Wolberg、Mangasarian、Street & Street（1995）
- **Boston Housing**：Harrison & Rubinfeld（1978）
- **California Housing**：Pace & Barry（1997），源自 1990 年美国普查
- **Forest Cover Type**：Blackard & Dean（1999），UCI 机器学习数据库，通过 scikit-learn 的 `fetch_covtype`
- **KDD Cup 1999**：Stolfo、Fan、Lee、Prodromidis & Chan（1999/2000），UCI KDD 数据库，通过 scikit-learn 的 `fetch_kddcup99`
- **Diabetes**：Efron、Hastie、Johnstone & Tibshirani（2004），通过 scikit-learn 的 `load_diabetes`
- **Linnerud**：A. C. Linnerud（北卡罗来纳州立大学），通过 scikit-learn 的 `load_linnerud`
- **Titanic**：Kaggle Titanic 数据集
- **Palmer Penguins**：Horst、Hill & Gorman（2020）；原始数据 Gorman、Williams & Fraser（2014）
- **Wine Recognition**：Aeberhard & Forina（1991），UCI 机器学习数据库
- **Wine Quality**：UCI 机器学习数据库

## 作者

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
