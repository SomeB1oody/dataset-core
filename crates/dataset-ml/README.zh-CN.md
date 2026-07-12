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

`dataset-ml` 内置了 24 个经典 ML 数据集的加载器。每个加载器会：

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

> 文本加载器 **SmsSpam**、**YoutubeSpam** 和 **SentimentSentences** 是例外：它们用 `texts()`（原始文档的 `Array1<String>`）代替 `features()`，因为文本语料没有固定的特征矩阵。**SentimentSentences** 还额外提供 `sources()`（每条句子来自哪个评论站点）。

> **注意**：Titanic、Palmer Penguins、Adult、BankMarketing、Kddcup99 和 Abalone 是混合类型数据集：`features()` 返回 `(&Array2<String>, &Array2<f64>)`（字符串特征 + 数值特征），`data()` 返回三元组。除 **Abalone** 外都是分类（`labels()` 访问器）；Abalone 是回归——三元组的第三个元素是通过 `targets()` 暴露的 `Array1<f64>` 目标。Palmer Penguins 还会把缺失值表示为 `NaN`（数值）或 `""`（字符串）。
>
> **注意**：Abalone 是首个**混合类型回归**加载器：单个类别特征 `sex`（`M`/`F`/`I`，即 `(4177, 1)` 的 `&Array2<String>`），加上 7 个数值测量（`length`、`diameter`、`height`、`whole_weight`、`shucked_weight`、`viscera_weight`、`shell_weight`，即 `(4177, 7)` 的 `&Array2<f64>`），预测 `rings`——`Array1<f64>` 回归目标（年龄为 `rings + 1.5` 岁）。它没有缺失值。
>
> **注意**：Adult（人口普查收入）复现了经典的 UCI 数据集，用于预测年收入是否超过 5 万美元：8 个类别特征（`workclass`、`education`、`marital-status`、`occupation`、`relationship`、`race`、`sex`、`native-country`），6 个数值特征（`age`、`fnlwgt`、`education-num`、`capital-gain`、`capital-loss`、`hours-per-week`），以及保持原样的 `Array1<String>` 收入标签（`<=50K` 或 `>50K`）。它加载标准的 `adult.data` 训练分区（32,561 条记录）；源文件中的 `?` 缺失类别标记被映射为空字符串 `""`。
>
> **注意**：BankMarketing 复现了经典的 UCI Bank Marketing 数据集（葡萄牙某银行的电话营销活动），用于预测客户是否会订购定期存款：9 个类别特征（`job`、`marital`、`education`、`default`、`housing`、`loan`、`contact`、`month`、`poutcome`），7 个数值特征（`age`、`balance`、`day`、`duration`、`campaign`、`pdays`、`previous`），以及保持原样的 `Array1<String>` 标签（`yes` 或 `no`）。它从 ZIP 压缩包中加载完整的 `bank-full.csv` 分区（45,211 条记录）；类别中的 `unknown` 标记保持原样（它是文档化的取值，例如 `poutcome = unknown` 表示此前没有联系过），而不映射为空字符串。
>
> **注意**：Mushroom 是**全类别型**数据集（与 Car Evaluation 相同）：全部 22 个特征都是单字母字符串编码，因此 `features()` 返回单个 `&Array2<String>`（没有数值矩阵），`data()` 返回 `(特征, 标签)` 二元组。`Array1<String>` 标签保持原样（`e` = 可食用，`p` = 有毒）。源文件中的 `?` 缺失标记（仅出现在 `stalk-root`）被映射为空字符串 `""`。
>
> **注意**：Car Evaluation 同样是**全类别型**数据集（与 Mushroom 相同）：全部 6 个特征都是字符串编码（`buying`、`maint`、`doors`、`persons`、`lug_boot`、`safety`），因此 `features()` 返回单个 `&Array2<String>`，`data()` 返回 `(特征, 标签)` 二元组。`Array1<String>` 标签保持原样，是四个可接受度类别 `unacc` / `acc` / `good` / `vgood` 之一。1,728 条记录穷举了六个属性的全笛卡尔积，因此没有缺失值。
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
>
> **注意**：Heart Disease（Cleveland）是带缺失值的全数值数据集：`features()` 返回单个 `&Array2<f64>`，形状 `(303, 13)`（其中若干列是以 `f64` 保存的整数编码类别），源文件中 `ca`（4 个）与 `thal`（2 个）的 `?` 标记被映射为 `NaN`（与 Titanic / Palmer Penguins 相同）。`labels()` 返回 `Array1<u8>` 诊断标签 `num`，取值 `0..=4`（`0` = 无病，`1`–`4` = 病情递增），通常二值化为 `0` 与 `> 0`。它加载标准的 `processed.cleveland.data` 分区（几乎所有已发表实验所用的 14 列子集）。
>
> **注意**：SMS Spam 是首个**文本**数据集。它没有特征矩阵：`texts()` 返回 5,574 条原始 SMS 消息正文的 `Array1<String>`（需自行向量化——词袋、TF-IDF、词向量等），`labels()` 返回 `"ham"` / `"spam"` 的 `Array1<&'static str>`，`data()` 返回 `(texts, labels)` 二元组。它从 **ZIP 压缩包**加载（与 Digits/BankMarketing 相同）：下载 `smsspamcollection.zip`，解压其中制表符分隔的 `SMSSpamCollection` 文件（缓存为 `sms_spam.csv`），并在关闭引号处理的情况下解析（消息是可能包含 `"` 与 `,` 的自由文本）。
>
> **注意**：YouTube Spam 是第二个**文本**数据集，也是 SMS Spam 的姊妹集（同一批作者）。与 SMS Spam 一样它没有特征矩阵：`texts()` 返回 1,956 条原始 YouTube 评论正文的 `Array1<String>`（源文件的 `CONTENT` 列），`labels()` 返回 `"ham"` / `"spam"` 的 `Array1<&'static str>`（由源文件的 `CLASS` 编码 `0` / `1` 映射而来），`data()` 返回 `(texts, labels)` 二元组。它从**五个**按视频划分的 CSV 组成的 **ZIP 压缩包**加载（Psy、Katy Perry、LMFAO、Eminem、Shakira 的音乐视频评论）；加载器按固定顺序将它们拼接为单个 `youtube_spam.csv`，用一个预设的 SHA-256 覆盖全部数据，然后以标准逗号分隔 CSV 解析，并**启用**引号处理（与 SMS Spam 不同——评论是规范加引号的，其中一条甚至跨越换行），同时跳过每个文件重复的表头行。每条评论的元数据列（`COMMENT_ID`、`AUTHOR`、`DATE`）不予暴露。
>
> **注意**：Sentiment Labelled Sentences 是第三个**文本**数据集，也是首个带有每样本**元数据**的加载器。它包含 3,000 条评论句子（Amazon、IMDb、Yelp 各 1,000 条；每个站点 500 正 + 500 负，因此完全均衡）。`texts()` 返回句子的 `Array1<String>`，`labels()` 返回 `"positive"` / `"negative"` 的 `Array1<&'static str>`（由源标签 `1` / `0` 映射而来）；此外 `sources()` 返回 `"amazon"` / `"imdb"` / `"yelp"` 的 `Array1<&'static str>`，可据此按领域切分语料（或搭建跨领域迁移实验）。因为多了这一列，`SentimentSentencesData` 是 `(texts, sources, labels)` **三元组**，`data()` 一并返回三者。它从三个按站点划分的 `sentence<TAB>label` 文件组成的 **ZIP 压缩包**加载；由于这些文件本身没有来源列，加载器为每行打上其站点标签，合并为单个 `source<TAB>sentence<TAB>label` 语料（`sentiment_sentences.csv`，由一个预设 SHA-256 覆盖），并在关闭引号处理的情况下按制表符分隔解析（与 SMS Spam 相同）。

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

- **Abalone**：Nash、Sellers、Talbot、Cawthorn & Ford（1994），UCI 机器学习数据库，源自塔斯马尼亚黑唇鲍鱼的研究
- **Adult / 人口普查收入**：Becker & Kohavi（1996），UCI 机器学习数据库，源自 1994 年美国人口普查
- **Bank Marketing**：Moro、Rita & Cortez（2012），UCI 机器学习数据库，源自葡萄牙某银行的直接营销活动
- **Iris**：Fisher 的鸢尾花数据集（1936）
- **Breast Cancer Wisconsin（诊断）**：Wolberg、Mangasarian、Street & Street（1995）
- **Boston Housing**：Harrison & Rubinfeld（1978）
- **California Housing**：Pace & Barry（1997），源自 1990 年美国普查
- **Forest Cover Type**：Blackard & Dean（1999），UCI 机器学习数据库，通过 scikit-learn 的 `fetch_covtype`
- **Heart Disease**：Janosi、Steinbrunn、Pfisterer & Detrano（1988），UCI 机器学习数据库，克利夫兰临床基金会分区
- **Ionosphere**：Sigillito、Wing、Hutton & Baker（1989），UCI 机器学习数据库，源自在拉布拉多 Goose Bay 采集的雷达回波
- **Car Evaluation**：Bohanec（1988），UCI 机器学习数据库，源自 DEX 层次化决策模型
- **KDD Cup 1999**：Stolfo、Fan、Lee、Prodromidis & Chan（1999/2000），UCI KDD 数据库，通过 scikit-learn 的 `fetch_kddcup99`
- **Diabetes**：Efron、Hastie、Johnstone & Tibshirani（2004），通过 scikit-learn 的 `load_diabetes`
- **Linnerud**：A. C. Linnerud（北卡罗来纳州立大学），通过 scikit-learn 的 `load_linnerud`
- **Mushroom**：UCI 机器学习数据库（1987），源自《奥杜邦协会北美蘑菇野外指南》（1981）
- **Titanic**：Kaggle Titanic 数据集
- **Palmer Penguins**：Horst、Hill & Gorman（2020）；原始数据 Gorman、Williams & Fraser（2014）
- **SMS Spam Collection**：Almeida & Hidalgo（2011），UCI 机器学习数据库，源自 Grumbletext、NUS SMS 语料库以及一篇博士论文的收集
- **YouTube Spam Collection**：Alberto、Lochter & Almeida（2017），UCI 机器学习数据库，源自五个热门音乐视频的评论
- **Sentiment Labelled Sentences**：Kotzias、Denil、de Freitas & Smyth（2015），UCI 机器学习数据库，源自 Amazon、IMDb 与 Yelp 评论的句子
- **Wine Recognition**：Aeberhard & Forina（1991），UCI 机器学习数据库
- **Wine Quality**：UCI 机器学习数据库

## 作者

**SomeB1oody** — [stanyin64@gmail.com](mailto:stanyin64@gmail.com)
