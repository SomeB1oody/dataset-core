简体中文 | [English](https://github.com/SomeB1oody/dataset-core/blob/master/crates/dataset-ml/README.md)

# dataset-ml

构建于 [`dataset-core`](https://crates.io/crates/dataset-core) 之上的经典机器学习数据集开箱即用加载器。

[![rustc](https://img.shields.io/badge/rustc-1.88%2B-brown)](https://www.rust-lang.org/) [![edition](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/) [![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/SomeB1oody/dataset-core/blob/master/LICENSE) [![crates.io](https://img.shields.io/crates/v/dataset-ml.svg)](https://crates.io/crates/dataset-ml)

[![CI](https://img.shields.io/github/actions/workflow/status/SomeB1oody/dataset-core/ci.yml?branch=master&label=CI)](https://github.com/SomeB1oody/dataset-core/actions/workflows/ci.yml)

## 概述

`dataset-ml` 内置了一系列经典 ML 数据集的加载器。每个加载器会：

- 在首次访问时下载源文件（通过 `ureq`），并对瞬时网络故障自动重试。
- 校验预设的 SHA-256 哈希值，以检测损坏或上游变化。
- 将源数据解析为 `Table`：每个源列对应一个具名、有类型的 `Column`。
- 通过 `dataset_core::Dataset<T, E>` 在内存中缓存解析出的 `Table`。后续访问会直接返回 `&` 引用，零 I/O。

每个模块同时也是封装 `Dataset<T, E>` 处理具体数据源的完整参考实现。

另有两个模块面向所有数据集，而不是某一个：

- [`preprocessing`](#预处理)：带随机种子的训练/测试集划分与 k 折划分（普通或按类别分层）、特征缩放、独热编码与标签编码。
- [`traits`](#mldataset-trait)：所有加载器都实现的 `MlDataset` trait，用于编写泛型于“某个数据集”的代码。

## 安装

```toml
[dependencies]
dataset-ml = "0.5"
```

## 特性开关

| 特性            | 默认 | 启用内容                                                             |
|-----------------|------|----------------------------------------------------------------------|
| `dataset`       | 是   | `dataset` 模块及其加载器、所有加载器结构体在 crate 根部的重新导出    |
| `preprocessing` | 是   | `preprocessing` 模块：带随机种子的划分、特征缩放、独热编码与标签编码 |

无论选择哪些特性，`table` 和 `traits` 模块始终可用。它们提供 `Table` 和 `MlDataset`，因此即使两个特性都关闭，你也可以按同一套接口编写自己的加载器。

只取所需时，请关闭默认特性：

```toml
[dependencies]
dataset-ml = { version = "0.5", default-features = false, features = ["dataset"] }
```

关闭 `dataset` 后，直接依赖只剩 `dataset-core` 和 `ndarray`。

## 数据集

完整列表见 [docs.rs 上的数据集总览](https://docs.rs/dataset-ml/latest/dataset_ml/dataset/index.html#datasets)，其中列出了每个数据集的样本数、特征数与任务类型。

## 用法

```rust
use dataset_ml::Iris;

fn main() {
    let iris = Iris::new("./data");

    // 惰性加载：首次访问时下载并解析，之后使用缓存。
    let table = iris.data().unwrap();

    assert_eq!(table.n_samples(), 150);
    assert_eq!(table.n_columns(), 5);

    // 需要矩阵时再物化，并给出想要的列名。
    let features = table.numeric_matrix(&Iris::FEATURE_NAMES).unwrap();
    assert_eq!(features.shape(), &[150, 4]);

    // 也可以按列名取某一列，与它的位置无关。
    let species = table.column(Iris::TARGET).unwrap().as_string().unwrap();
    assert_eq!(species[0], "setosa");
}
```

无论装的是什么，每个数据集结构体都提供同样的六个方法：

- `new(storage_dir)` — 创建实例（无 I/O 操作）。部分数据集另有 `new_test` / `new_all` / `new_full` 用于选择子集
- `data()` — 解析出的 `Table` 的引用
- `get_data()` / `get_data_mut()` — **不触发加载**地借用缓存中的 `Table`
- `into_data()` / `take_data()` — 零克隆地取出拥有所有权的 `Table`

## `Table`

所有加载器统一返回 `Table`：每个源列一个 `Column`，各自带有自己的名字和源数据本来的类型。

`Table::new` 会校验列，加载器不可能交给你错位的数据：

- 至少有一列
- 所有列的样本数相同
- 列名不重复

| `ColumnData` | 一列装的内容                            |
|--------------|-----------------------------------------|
| `Numeric`    | 每个样本一个 `f64`。缺失值为 `NaN`      |
| `Integer`    | 每个样本一个 `i64`                      |
| `String`     | 每个样本一个 `String`，保持源数据的拼写 |
| `Bytes`      | 每个样本一行定宽 `u8`，例如图像的像素   |

每个加载器都用关联常量给出列名。`FEATURE_NAMES` 列出源数据指定为模型输入的那些列，`TARGET` 给出标签列的列名。如果源数据指定了不止一个标签列，则用 `TARGET_NAMES` 代替 `TARGET`。没有标签的数据集这两个常量都没有。其余的列都按列名访问。

```rust
use dataset_ml::Iris;

fn main() {
    let iris = Iris::new("./data");
    let table = iris.data().unwrap();

    // 遍历每一列的名字和类型。
    for column in table.columns() {
        println!("{} {}", column.name(), column.data().kind());
    }

    // 需要矩阵时才物化。
    let features = table.numeric_matrix(&Iris::FEATURE_NAMES).unwrap();

    // 矩阵按你给出的列名顺序排列，不一定是源列顺序。
    let petals = table.numeric_matrix(&["petal_width", "petal_length"]).unwrap();

    // 字符串保持源数据原样，编码与否由你决定。
    let species = table.column(Iris::TARGET).unwrap().as_string().unwrap();
}
```

`String` 列没有数值形式，因此把这样的列名传给 `numeric_matrix` 会返回错误。`numeric_matrix` 每次调用都会分配，取一次留着用即可。

## `MlDataset` trait

所有加载器都实现了 `dataset_ml::traits::MlDataset`。它涵盖了与具体解析结果无关的那部分容器操作，因此你可以针对“某个数据集”而不是某个具体结构体来写函数：

```rust
use dataset_ml::traits::MlDataset;
use dataset_ml::{Iris, SmsSpam};

fn describe<D: MlDataset>(dataset: &D) -> String {
    format!("{} ({} samples)", D::NAME, dataset.n_samples().unwrap())
}

fn main() {
    println!("{}", describe(&Iris::new("./data")));     // iris (150 samples)
    println!("{}", describe(&SmsSpam::new("./data")));  // sms_spam (5574 samples)
}
```

| 方法                            | 描述                                              |
|---------------------------------|---------------------------------------------------|
| `load()` / `load_mut()`         | 按需加载后借用解析结果（`load_mut` 用于原地修改） |
| `peek()`                        | **不触发加载**地借用解析结果                      |
| `unload()`                      | 取出解析结果，加载器保持可复用                    |
| `n_samples()`                   | 样本数；不论加载器解析成何种形态，用法都一致      |
| `is_loaded()` / `storage_dir()` | 在不接触数据的前提下检视加载器                    |
| `invalidate()`                  | 丢弃内存缓存——可回收大数据集占用的内存            |

trait 方法的命名刻意与固有方法 `data()` / `get_data()` / `take_data()` 区分开，因此这两套接口不会互相遮蔽。两者始终可用，且结果始终一致。

## 预处理

`dataset_ml::preprocessing` 负责把加载器返回的数据转换成模型可以直接使用的输入。所有结果在给定种子下完全确定，且不需要任何额外的 crate。

```rust
use dataset_ml::preprocessing::{label_encode, standardize, stratified_split};
use dataset_ml::Iris;
use ndarray::Axis;

fn main() {
    let iris = Iris::new("./data");
    let table = iris.data().unwrap();

    let features = table.numeric_matrix(&Iris::FEATURE_NAMES).unwrap();
    let species = table.column(Iris::TARGET).unwrap().as_string().unwrap();

    // 按类别分层划分，使每个物种在两侧的占比保持一致。
    let (train, test) = stratified_split(species.as_slice().unwrap(), 0.2, 42).unwrap();

    // 只在训练集上拟合缩放器，再原样应用到测试集。
    let (train_x, scaler) = standardize(&features.select(Axis(0), &train)).unwrap();
    let (codes, classes) = label_encode(&species.select(Axis(0), &train)).unwrap();

    assert_eq!(train_x.nrows(), 120);
    assert_eq!(classes.len(), 3);
}
```

| 函数                                    | 用途                                                   |
|-----------------------------------------|--------------------------------------------------------|
| `train_test_split(n, ratio, seed)`      | 打乱后的训练/测试行索引                                |
| `stratified_split(labels, ratio, seed)` | 同上，但保持各类别占比——适用于类别不平衡的数据集       |
| `k_fold_indices(n, k, seed)`            | `k` 组 `(训练, 验证)` 索引；每个样本恰好被验证一次     |
| `shuffled_indices(n, seed)`             | `0..n` 的确定性随机排列                                |
| `standardize` / `min_max_scale`         | 按列缩放，并返回拟合好的 `Scaler`                      |
| `apply_scaler(features, &scaler)`       | 用已拟合的缩放器处理新数据，不重新拟合                 |
| `one_hot_encode(categorical, names)`    | 把类别型 `Array2<String>` 展开为指示列                 |
| `label_encode(labels)` / `class_counts` | 把标签映射为 `0..n_classes` 编码；统计每个类别的样本数 |

划分函数返回的是**行索引**，而不是数组，因为一条样本的数据分散在表的每一列中。用同一份索引列表，就能让所有列的样本保持对齐。如需得到数组，可以使用 ndarray 的 `select(Axis(0), &indices)`。缩放器只在每列的**有限值**上计算统计量。因此，`Titanic`、`PalmerPenguins`、`HeartDisease` 中标记缺失值的 `NaN` 会继续保持缺失，不会影响该列的统计结果。

## 性能考量

- **首次访问**：下载文件（如果磁盘上不存在）、校验 SHA-256、解析并缓存到内存。
- **后续访问**：返回缓存数据的引用，零分配、零 I/O。
- **`numeric_matrix()`**：按你给出的列名分配一个新矩阵。取一次留着用即可。
- **`take_data()` / `into_data()`**：零克隆地取出拥有所有权的 `Table`；`get_data_mut()` 可原地修改。
- **离线使用**：首次下载完成后，数据集会保留在磁盘上。后续运行不需要联网。

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../../LICENSE)。

## 作者

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
