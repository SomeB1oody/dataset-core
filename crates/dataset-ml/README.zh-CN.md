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

`dataset-ml` 内置了 26 个经典 ML 数据集的加载器。每个加载器会：

- 在首次访问时下载源文件（通过 `ureq`）。
- 校验预设的 SHA-256 哈希值，以检测损坏或上游变化。
- 将源数据（CSV，文本语料则为从归档中解出的原始文档）用 [`ndarray`](https://crates.io/crates/ndarray) 解析为 `Array1` / `Array2`。
- 通过 `dataset_core::Dataset<T, E>` 在内存中缓存解析结果——后续访问会直接返回 `&` 引用，零 I/O。

每个模块同时也是封装 `Dataset<T, E>` 处理具体数据源的完整参考实现。

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
| `Linnerud`                                 | `dataset_ml::linnerud`                              | 20      | 3      | 回归（多输出） | scikit-learn |
| `Mushroom`                                 | `dataset_ml::mushroom`                              | 8,124   | 22     | 分类     | UCI ML Repository |
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
