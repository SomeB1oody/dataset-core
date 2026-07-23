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

`dataset-ml` 内置了 29 个经典 ML 数据集的加载器。每个加载器会：

- 在首次访问时下载源文件（通过 `ureq`），并对瞬时网络故障自动重试。
- 校验预设的 SHA-256 哈希值，以检测损坏或上游变化。
- 将源数据（CSV，文本语料则为从归档中解出的原始文档）用 [`ndarray`](https://crates.io/crates/ndarray) 解析为 `Array1` / `Array2`。
- 通过 `dataset_core::Dataset<T, E>` 在内存中缓存解析结果——后续访问会直接返回 `&` 引用，零 I/O。

每个模块同时也是封装 `Dataset<T, E>` 处理具体数据源的完整参考实现。

另有两个模块面向**所有**数据集，而非某一个：

- [`preprocessing`](#预处理) —— 带随机种子的训练/测试集划分与 k 折划分（普通或按类别分层）、特征缩放、独热编码与标签编码。
- [`traits`](#mldataset-trait) —— 所有加载器都实现的 `MlDataset` trait，用于编写泛型于“某个数据集”的代码。

## 安装

```toml
[dependencies]
dataset-ml = "0.3"
```

## 数据集

| 结构体                                     | 模块路径                                            | 样本数  | 特征数 | 任务类型 | 来源              |
|--------------------------------------------|-----------------------------------------------------|---------|--------|----------|-------------------|
| `Abalone`                                  | `dataset_ml::abalone`                               | 4,177   | 8      | 回归     | UCI ML Repository |
| `Adult`                                    | `dataset_ml::adult`                                 | 32,561  | 14     | 分类     | UCI ML Repository |
| `BankMarketing`                            | `dataset_ml::bank_marketing`                        | 45,211  | 16     | 分类     | UCI ML Repository |
| `BanknoteAuthentication`                   | `dataset_ml::banknote_authentication`               | 1,372   | 4      | 分类     | UCI ML Repository |
| `Iris`                                     | `dataset_ml::iris`                                  | 150     | 4      | 分类     | UCI ML Repository |
| `BreastCancer`                             | `dataset_ml::breast_cancer`                         | 569     | 30     | 分类     | UCI ML Repository |
| `BostonHousing`                            | `dataset_ml::boston_housing`                        | 506     | 13     | 回归     | UCI ML Repository |
| `CaliforniaHousing`                        | `dataset_ml::california_housing`                    | 20,640  | 8      | 回归     | StatLib（1990 普查） |
| `CarEvaluation`                            | `dataset_ml::car_evaluation`                        | 1,728   | 6      | 分类     | UCI ML Repository |
| `Covtype`                                  | `dataset_ml::covtype`                               | 581,012 | 54     | 分类     | UCI ML Repository |
| `Diabetes`                                 | `dataset_ml::diabetes`                              | 442     | 10     | 回归     | Efron et al.（2004） |
| `Digits`                                   | `dataset_ml::digits`                                | 1,797   | 64     | 分类     | UCI ML Repository |
| `HeartDisease`                             | `dataset_ml::heart_disease`                         | 303     | 13     | 分类     | UCI ML Repository |
| `Ionosphere`                               | `dataset_ml::ionosphere`                            | 351     | 34     | 分类     | UCI ML Repository |
| `Kddcup99`                                 | `dataset_ml::kddcup99`                              | 494,021 / 4,898,431 | 41 | 分类  | UCI KDD Archive   |
| `LetterRecognition`                        | `dataset_ml::letter_recognition`                    | 20,000  | 16     | 分类（26 类） | UCI ML Repository |
| `Linnerud`                                 | `dataset_ml::linnerud`                              | 20      | 3      | 回归（多输出） | scikit-learn |
| `Mushroom`                                 | `dataset_ml::mushroom`                              | 8,124   | 22     | 分类     | UCI ML Repository |
| `Spambase`                                 | `dataset_ml::spambase`                              | 4,601   | 57     | 分类     | UCI ML Repository |
| `Titanic`                                  | `dataset_ml::titanic`                               | 891     | 11     | 分类     | Kaggle            |
| `PalmerPenguins`                           | `dataset_ml::palmer_penguins`                       | 344     | 7      | 分类     | palmerpenguins    |
| `SmsSpam`                                  | `dataset_ml::sms_spam`                              | 5,574   | 文本   | 分类     | UCI ML Repository |
| `WineRecognition`                          | `dataset_ml::wine_recognition`                      | 178     | 13     | 分类     | UCI ML Repository |
| `RedWineQuality`                           | `dataset_ml::wine_quality::red_wine_quality`        | 1,599   | 11     | 回归     | UCI ML Repository |
| `WhiteWineQuality`                         | `dataset_ml::wine_quality::white_wine_quality`      | 4,898   | 11     | 回归     | UCI ML Repository |
| `YoutubeSpam`                              | `dataset_ml::youtube_spam`                          | 1,956   | 文本   | 分类     | UCI ML Repository |
| `SentimentSentences`                       | `dataset_ml::sentiment_sentences`                   | 3,000   | 文本   | 分类     | UCI ML Repository |
| `Newsgroups20`                             | `dataset_ml::newsgroups20`                          | 11,314 / 18,846 | 文本 | 分类  | Jason Rennie / 20 Newsgroups |
| `MovieReviewPolarity`                      | `dataset_ml::movie_review_polarity`                 | 2,000   | 文本   | 分类     | Cornell (Pang & Lee) |

所有结构体也在 crate 根部重新导出，所以 `dataset_ml::Iris`、`dataset_ml::RedWineQuality` 等也可以使用。

## 用法

```rust
use dataset_ml::iris::Iris;

fn main() {
    let iris = Iris::new("./data");

    // 惰性加载：首次访问时下载并解析，之后使用缓存。
    let features = iris.features().unwrap();  // &Array2<f64>
    let labels   = iris.labels().unwrap();    // &Array1<&'static str>

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

> 文本加载器 **SmsSpam**、**YoutubeSpam**、**SentimentSentences**、**Newsgroups20** 和 **MovieReviewPolarity** 是例外：它们用 `texts()`（原始文档的 `Array1<String>`）代替 `features()`，因为文本语料没有固定的特征矩阵。**SentimentSentences** 还额外提供 `sources()`（每条句子来自哪个评论站点）；**Newsgroups20** 是唯一的**多分类**文本加载器（20 类），并提供 `new`/`new_test`/`new_all` 三个子集构造函数。

## `MlDataset` trait

所有加载器都实现了 `dataset_ml::traits::MlDataset`。它涵盖了与具体解析结果无关的那部分容器操作，因此你可以针对“某个数据集”而不是某个具体结构体来写函数：

```rust
use dataset_ml::traits::MlDataset;
use dataset_ml::{Iris, SmsSpam};

fn describe<D: MlDataset>(dataset: &D) -> String {
    format!("{} ({} 条样本)", D::NAME, dataset.n_samples().unwrap())
}

fn main() {
    println!("{}", describe(&Iris::new("./data")));     // iris (150 条样本)
    println!("{}", describe(&SmsSpam::new("./data")));  // sms_spam (5574 条样本)
}
```

| 方法                            | 描述                                                          |
|---------------------------------|---------------------------------------------------------------|
| `load()` / `load_mut()`         | 按需加载后借用解析结果（`load_mut` 用于原地修改）            |
| `peek()`                        | **不触发加载**地借用解析结果                                  |
| `unload()`                      | 取出解析结果，加载器保持可复用                                |
| `n_samples()`                   | 样本数；二元组与三元组形态的数据集用法一致                    |
| `is_loaded()` / `storage_dir()` | 在不接触数据的前提下检视加载器                                |
| `invalidate()`                  | 丢弃内存缓存——可回收大数据集占用的内存                       |

trait 的方法名刻意与固有方法 `data()` / `get_data()` / `take_data()` 区分开，因此两套 API 不会互相遮蔽，且始终一致可用。

## 预处理

`dataset_ml::preprocessing` 负责把加载器返回的数组变成模型能吃的输入。所有结果在给定种子下完全确定，且不引入任何额外依赖。

```rust
use dataset_ml::preprocessing::{stratified_split, standardize, label_encode};
use dataset_ml::Iris;
use ndarray::Axis;

fn main() {
    let iris = Iris::new("./data");
    let (features, labels) = iris.data().unwrap();

    // 按类别分层划分，使每个物种在两侧的占比保持一致。
    let (train, test) = stratified_split(labels.as_slice().unwrap(), 0.2, 42).unwrap();

    // 只在训练集上拟合缩放器，再原样应用到测试集。
    let (train_x, scaler) = standardize(&features.select(Axis(0), &train)).unwrap();
    let (codes, classes) = label_encode(&labels.select(Axis(0), &train)).unwrap();

    assert_eq!(train_x.nrows(), 120);
    assert_eq!(classes.len(), 3);
}
```

| 函数                                        | 用途                                                          |
|---------------------------------------------|---------------------------------------------------------------|
| `train_test_split(n, ratio, seed)`          | 打乱后的训练/测试行索引                                       |
| `stratified_split(labels, ratio, seed)`     | 同上，但保持各类别占比——适用于类别不平衡的数据集             |
| `k_fold_indices(n, k, seed)`                | `k` 组 `(训练, 验证)` 索引；每个样本恰好被验证一次            |
| `shuffled_indices(n, seed)`                 | `0..n` 的确定性随机排列                                       |
| `standardize` / `min_max_scale`             | 按列缩放，并返回拟合好的 `Scaler`                             |
| `apply_scaler(features, &scaler)`           | 用已拟合的缩放器处理新数据，不重新拟合                        |
| `one_hot_encode(categorical, names)`        | 把类别型 `Array2<String>` 展开为指示列                        |
| `label_encode(labels)` / `class_counts`     | 把标签映射为 `0..n_classes` 编码；统计每个类别的样本数        |

划分函数返回的是**行索引**而非数组：一条样本分散在两到三个平行数组中，一份索引列表才能让它们保持对齐——用 ndarray 的 `select(Axis(0), &indices)` 取出即可。缩放器只在每列的**有限值**上统计，因此 `Titanic`、`PalmerPenguins`、`HeartDisease` 中标记缺失的 `NaN` 会保持缺失，而不会污染整列。

## 从 `dataset-core` 0.1.x 迁移

如果你之前使用 `dataset-core` 0.1.x 的 `datasets` 特性，请切换到本 crate：

```diff
- dataset-core = { version = "0.1", features = ["datasets"] }
+ dataset-ml = "0.3"
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

## 作者

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
