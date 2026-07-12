# Changelog — `dataset-ml`

All notable changes to the `dataset-ml` crate will be documented in this file.

This crate provides ready-to-use loaders for classic machine learning datasets (Iris, Breast Cancer, Boston/California Housing, Diabetes, Titanic, Palmer Penguins, Wine Recognition, Wine Quality), built on top of [`dataset-core`](https://crates.io/crates/dataset-core).

Please view [SomeB1oody/dataset-core](https://github.com/SomeB1oody/dataset-core) for more info.

## [Unreleased] - 2026-07-12
### Added
- `youtube_spam::YoutubeSpam` loader for the YouTube Spam Collection dataset (UCI Machine Learning Repository, Alberto, Lochter & Almeida 2017): 1,956 YouTube comments (951 ham, 1,005 spam) for binary classification. It is the crate's **second text loader** and a sibling of `SmsSpam` (same authors), so like `SmsSpam` it has no feature matrix — `YoutubeSpamData` is `(Array1<String>, Array1<&'static str>)`, the document accessor is **`texts()`** (a `(1956,)` `Array1<String>` of raw comment bodies, the source `CONTENT` column) rather than `features()`, `labels()` returns an `Array1<&'static str>` of `"ham"`/`"spam"` (mapped from the source `CLASS` codes `0`/`1`), and `data()` returns the `(texts, labels)` pair. It is sourced from a **ZIP archive of five** per-video CSVs (comments on music videos by Psy, Katy Perry, LMFAO, Eminem, and Shakira): the loader downloads `YouTube-Spam-Collection-v1.zip`, concatenates the five CSVs in a fixed order into a single `youtube_spam.csv` covered by one pinned SHA-256, then parses standard comma-separated CSV with csv quote handling **enabled** (`quoting` left on — unlike `SmsSpam` the comments are properly quoted, one even spanning an embedded newline) and skips each file's repeated header row. The per-comment metadata columns (`COMMENT_ID`, `AUTHOR`, `DATE`) are not exposed. The struct is also re-exported at the crate root as `dataset_ml::YoutubeSpam`.
- `sms_spam::SmsSpam` loader for the SMS Spam Collection dataset (UCI Machine Learning Repository, Almeida & Hidalgo 2011): 5,574 SMS messages (4,827 ham, 747 spam) for binary classification. This is the crate's first **text** loader: there is no feature matrix, so `SmsSpamData` is `(Array1<String>, Array1<&'static str>)` and the document accessor is **`texts()`** (a `(5574,)` `Array1<String>` of raw message bodies) rather than `features()`; `labels()` returns an `Array1<&'static str>` of `"ham"`/`"spam"` and `data()` returns the `(texts, labels)` pair. It is sourced from a **ZIP archive** (like `Digits`/`BankMarketing`): the loader downloads `smsspamcollection.zip`, extracts the tab-separated `SMSSpamCollection` file (cached as `sms_spam.csv`, SHA-256 verified), and parses it with csv quote handling **disabled** (`quoting(false)`), since the free-text messages can contain `"` and `,`. The struct is also re-exported at the crate root as `dataset_ml::SmsSpam`.
- `abalone::Abalone` loader for the Abalone dataset (UCI Machine Learning Repository, Nash, Sellers, Talbot, Cawthorn & Ford 1994): 4,177 samples, 8 mixed features — 1 categorical string feature (`sex`: `M`/`F`/`I`) and 7 numeric measurements (`length`, `diameter`, `height`, `whole_weight`, `shucked_weight`, `viscera_weight`, `shell_weight`) — and an `Array1<f64>` regression target `rings` (age in years is `rings + 1.5`). It is the first **mixed-type regression** loader: like `Titanic`/`Adult` it is mixed-type, so `features()` returns `(&Array2<String>, &Array2<f64>)` and `data()` returns a triple `AbaloneData = (Array2<String>, Array2<f64>, Array1<f64>)`, but unlike those classification loaders the target is a regression target exposed via `targets()` (like `BostonHousing`/`CaliforniaHousing`/`Diabetes`) rather than a `labels()` class vector. The dataset has no missing values. Sourced from `abalone.data` (plain comma-separated, no header) with SHA-256 verification. The struct is also re-exported at the crate root as `dataset_ml::Abalone`.
- `heart_disease::HeartDisease` loader for the Heart Disease (Cleveland) dataset (UCI Machine Learning Repository, Janosi, Steinbrunn, Pfisterer & Detrano 1988): 303 samples, 13 numeric clinical features (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal` — several are integer-coded categoricals kept as `f64`), and an `Array1<u8>` diagnosis target `num` in `0..=4` (`0` = absence, `1`–`4` = increasing presence) for classification. `HeartDiseaseData` is `(Array2<f64>, Array1<u8>)`. It is the first loader that is **all-numeric yet has missing values**: the source's `?` tokens — 4 in `ca`, 2 in `thal` — are mapped to `NaN` (reusing the missing-numeric convention of `Titanic`/`PalmerPenguins`) rather than to empty strings (as `Adult`/`Mushroom` do for categorical `?`). It loads the canonical `processed.cleveland.data` partition (the 14-column subset used by virtually all published experiments) from a plain comma-separated, headerless file with SHA-256 verification. The struct is also re-exported at the crate root as `dataset_ml::HeartDisease`.
- `car_evaluation::CarEvaluation` loader for the Car Evaluation dataset (UCI Machine Learning Repository, Bohanec 1988): 1,728 samples, 6 categorical features (`buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`), and an `Array1<String>` label kept verbatim (one of `unacc`, `acc`, `good`, `vgood`) for multi-class classification. Like `Mushroom` it is **all-categorical**, so `CarEvaluationData` is `(Array2<String>, Array1<String>)` — `features()` returns a single `&Array2<String>` (there is no numeric matrix) and `data()` returns a `(features, labels)` pair. Unlike `Mushroom` (whose label is the first column), the label is the **last** source column, and the 1,728 records enumerate the full cartesian product of the six attributes, so there are no missing values. Sourced from `car.data` (plain comma-separated, no header) with SHA-256 verification. The struct is also re-exported at the crate root as `dataset_ml::CarEvaluation`.

## [Unreleased] - 2026-07-11
### Added
- `ionosphere::Ionosphere` loader for the Ionosphere dataset (UCI Machine Learning Repository, Sigillito, Wing, Hutton & Baker 1989): 351 samples, 34 continuous features (17 radar pulses × real/imaginary autocorrelation components), and an `Array1<&'static str>` label for binary classification. `IonosphereData` is `(Array2<f64>, Array1<&'static str>)` — a compact **pure-numeric** loader like `Iris`/`BreastCancer`: `features()` returns a single `&Array2<f64>` and `data()` returns a `(features, labels)` pair. The source's single-letter `class` codes are mapped to readable names (`g` → `"good"`, `b` → `"bad"`, matching how `BreastCancer` maps `M`/`B`). The feature values are normalized to `[-1, 1]`; the first two columns are degenerate in this collection (column `0` is `0`/`1`, column `1` is constant `0`) but are kept verbatim so the schema matches the source. Sourced from `ionosphere.data` (plain comma-separated, no header) with SHA-256 verification. The struct is also re-exported at the crate root as `dataset_ml::Ionosphere`.

## [Unreleased] - 2026-06-23
### Added
- `mushroom::Mushroom` loader for the Mushroom dataset (UCI Machine Learning Repository, `agaricus-lepiota`, 1987): 8,124 samples, 22 categorical features, and an `Array1<String>` label kept verbatim (`e` = edible, `p` = poisonous) for binary classification. This is the first **all-categorical** loader: every feature is a single-letter string code, so `MushroomData` is `(Array2<String>, Array1<String>)` — `features()` returns a single `&Array2<String>` (there is no numeric matrix) and `data()` returns a `(features, labels)` pair. The label is the **first** source column (unlike the mixed-type loaders, where it is last). It parses raw `StringRecord`s by position; the `?` missing token (only in `stalk-root`, 2,480 samples) is mapped to empty strings `""` (like `Adult`'s `?`). The struct is also re-exported at the crate root as `dataset_ml::Mushroom`.

## [Unreleased] - 2026-06-22
### Added
- `bank_marketing::BankMarketing` loader for the Bank Marketing dataset (UCI Machine Learning Repository, Moro, Rita & Cortez 2012): 45,211 samples, 16 mixed features — 9 categorical string features (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`) and 7 numeric features (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`) — and an `Array1<String>` label kept verbatim (`yes` or `no`, whether the client subscribed a term deposit) for binary classification. Like `Titanic`/`Adult` it is mixed-type, so `features()` returns `(&Array2<String>, &Array2<f64>)` and `data()` is a triple. It loads the full `bank-full.csv` partition from a **ZIP archive** (like `Digits`): its `prepare_file` closure downloads `bank.zip`, unzips it, and uses `bank-full.csv` (cached as `bank_marketing.csv`), with SHA-256 verification of the decompressed file. The source is **semicolon-separated** with double-quoted string fields and a header row. The categorical `unknown` label is kept **verbatim** as a documented category level (unlike `Adult`'s `?`, which is mapped to `""`), since e.g. `poutcome = unknown` means no previous contact and is informative. The struct is also re-exported at the crate root as `dataset_ml::BankMarketing`.

## [Unreleased] - 2026-06-21
### Added
- `adult::Adult` loader for the Adult / Census Income dataset (UCI Machine Learning Repository, Becker & Kohavi 1996): 32,561 samples, 14 mixed features — 8 categorical string features (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`) and 6 numeric features (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) — and an `Array1<String>` income label kept verbatim (`<=50K` or `>50K`) for binary classification. Like `Titanic` it is mixed-type, so `features()` returns `(&Array2<String>, &Array2<f64>)` and `data()` is a triple. It loads the canonical `adult.data` training partition (the `adult.test` partition is not bundled, as it carries a non-data header line and trailing periods on its labels); the source is comma-and-space separated, so fields are trimmed, and the `?` missing categorical token is mapped to empty strings `""`. The struct is also re-exported at the crate root as `dataset_ml::Adult`.

## [Unreleased] - 2026-06-17
### Added
- `kddcup99::Kddcup99` loader for the KDD Cup 1999 network-intrusion dataset (scikit-learn's `fetch_kddcup99`): 41 mixed features — 3 categorical string features (`protocol_type`, `service`, `flag`) and 38 numeric connection features — and an `Array1<String>` label (the connection class, kept verbatim including the trailing period, e.g. `"normal."`, `"smurf."`; 23 distinct classes) for multi-class classification. Mirroring scikit-learn's `percent10` switch, it has two constructors: `Kddcup99::new` loads the **default 10% subset** (494,021 samples, `percent10=True`) and `Kddcup99::new_full` loads the **full set** (4,898,431 samples, `percent10=False`); the two partitions share the same schema and 23 classes and cache to distinct filenames (`kddcup99_10_percent.csv` / `kddcup99.csv`), each with its own pinned SHA-256. Like `Titanic` it is mixed-type, so `features()` returns `(&Array2<String>, &Array2<f64>)` and `data()` is a triple; like `Covtype` it is sourced from a **gzip-compressed** file decompressed with `dataset-core`'s `gunzip` helper, with SHA-256 verification of the decompressed file. The full set is large: the decompressed source is ~743 MB and the parsed in-memory arrays are several GB. The struct is also re-exported at the crate root as `dataset_ml::Kddcup99`.

## [Unreleased] - 2026-06-14
### Added
- `covtype::Covtype` loader for the Forest Cover Type dataset (scikit-learn's `fetch_covtype`): 581,012 samples, 54 cartographic features (10 quantitative variables, 4 one-hot `Wilderness_Area` columns, 40 one-hot `Soil_Type` columns, all stored as `f64`), and a `u8` cover-type label (`1`–`7`) for multi-class classification. It is the first loader sourced from a **gzip-compressed** file: its `prepare_file` closure downloads `covtype.data.gz` and decompresses it with `dataset-core`'s new `gunzip` helper before parsing the plain comma-separated data, with SHA-256 verification of the decompressed file. The struct is also re-exported at the crate root as `dataset_ml::Covtype`. Requires `dataset-core` with the `gunzip` helper (the `utils` feature now pulls in `flate2`).

## [Unreleased] - 2026-06-08
### Added
- `linnerud::Linnerud` loader for the Linnerud dataset (scikit-learn's `load_linnerud`): 20 samples, **multi-output regression**. `features()` returns the three exercise variables (`Chins`, `Situps`, `Jumps`) and `targets()` returns the three physiological variables (`Weight`, `Waist`, `Pulse`), so both are `Array2<f64>` with shape `(20, 3)` — the first loader whose target is an `Array2<f64>` rather than a 1-D vector. It is the first loader to acquire **two** source files (the whitespace-separated `linnerud_exercise.csv` and `linnerud_physiological.csv` distributed with scikit-learn), each downloaded and SHA-256 verified independently. The struct is also re-exported at the crate root as `dataset_ml::Linnerud`.

## [Unreleased] - 2026-06-07
### Added
- `digits::Digits` loader for the Optical Recognition of Handwritten Digits dataset (scikit-learn's `load_digits`): 1,797 samples, 64 numeric pixel features (an 8×8 grayscale image flattened in row-major order, each intensity an integer in `0..=16`), and a `u8` digit label (`0`–`9`) for multi-class classification. This is the first loader with an `Array1<u8>` target and the first to extract its source from a **ZIP archive**: it downloads the UCI static package, unzips it, and uses the `optdigits.tes` test partition (the same partition scikit-learn uses, hence the 1,797 sample count), with SHA-256 verification. The struct is also re-exported at the crate root as `dataset_ml::Digits`.

## [Unreleased] - 2026-06-06
### Changed
- **BREAKING**: `diabetes::Diabetes` now loads the **scikit-learn `load_diabetes`** dataset (Efron, Hastie, Johnstone & Tibshirani, 2004) instead of the Pima Indians Diabetes dataset — changing it from **768 samples × 8 features, binary classification** to **442 samples × 10 features, regression**. The ten features are standardized to reproduce scikit-learn's default `load_diabetes()` output (each column mean-centered and divided by its L2 norm, so its sum of squares is 1); the target is the unscaled measure of disease progression one year after baseline (integer-valued, range 25–346). The label accessor is renamed `labels()` → `targets()` to match the other regression loaders (`CaliforniaHousing`, `BostonHousing`); `DiabetesData` remains `(Array2<f64>, Array1<f64>)` but its `.1` is now the regression target. The source URL and pinned SHA-256 now point at the original tab-separated `diabetes.tab` file, with SHA-256 verification.

## [0.2.0] - 2026-06-05
### Added
- `california_housing::CaliforniaHousing` loader for the California Housing dataset: 20,640 samples, 8 numeric features, and an `f64` regression target (`MedHouseVal`, median house value in units of $100,000). Unlike the other loaders, it does **feature engineering** rather than exposing raw columns — it reproduces scikit-learn's `fetch_california_housing` features (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`) by deriving the per-household ratios from Géron's `housing.csv` and scaling the target by `1/100000`. The source's 207 missing `total_bedrooms` values surface as `NaN` in `AveBedrms` (sklearn's complete upstream has none). A modern replacement for Boston Housing. Sourced with SHA-256 verification; the struct is also re-exported at the crate root as `dataset_ml::CaliforniaHousing`.

## [0.2.0] - 2026-06-04
### Added
- `palmer_penguins::PalmerPenguins` loader for the Palmer Penguins dataset: 344 samples with mixed features — 2 categorical string features (`island`, `sex`) and 5 numeric features (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `year`) — and a `&'static str` species label (`"Adelie"` / `"Chinstrap"` / `"Gentoo"`) for multi-class classification. Like `Titanic`, `features()` returns `(&Array2<String>, &Array2<f64>)` and `data()` is a triple. The source encodes missing values as the literal token `NA`, which becomes `NaN` (numeric) or `""` (string). Sourced from the palmerpenguins R package (`penguins.csv`) with SHA-256 verification. The struct is also re-exported at the crate root as `dataset_ml::PalmerPenguins`.

## [0.2.0] - 2026-06-03
### Added
- `wine_recognition::WineRecognition` loader for the Wine recognition dataset (scikit-learn's `load_wine`): 178 samples, 13 numeric features (the chemical constituents), and a `&'static str` cultivar label (`"class_1"` / `"class_2"` / `"class_3"`) for multi-class classification. Sourced from the UCI Machine Learning Repository (`wine.data`) with SHA-256 verification. This is **distinct** from the `wine_quality` datasets, which are a regression task on quality scores. The struct is also re-exported at the crate root as `dataset_ml::WineRecognition`.

## [0.2.0] - 2026-06-02
### Added
- `breast_cancer::BreastCancer` loader for the Breast Cancer Wisconsin (Diagnostic) dataset: 569 samples, 30 numeric features (the `mean`, `se`, and `worst` statistics for 10 cell-nucleus measurements), and a `&'static str` diagnosis label (`"malignant"` / `"benign"`) for binary classification. Sourced from the UCI Machine Learning Repository (`wdbc.data`) with SHA-256 verification. The struct is also re-exported at the crate root as `dataset_ml::BreastCancer`.

## [0.2.0] - 2026-05-29
### Changed
- Adapted to `dataset-core`'s loader-on-construction API: the loader is now stored on the `Dataset` and supplied at construction. Every loader's `Dataset<XData>` field becomes `Dataset<XData, DatasetError>`, `new` passes `Self::load_data` to `Dataset::new`, and the accessor methods call `self.dataset.load()` (no argument). The public API of each loader (`Iris::new(dir)`, `features()`, `labels()`, `data()`, etc.) is unchanged.
- Refactored CSV parsing in every dataset loader to use Serde. Each loader now defines a `#[derive(Deserialize)]` record struct and parses with `csv::Reader::deserialize()`, replacing the manual per-field `record[i].parse()` loops, the `num_features` inference, and the explicit column-count checks (the `csv` reader now enforces a consistent column count). Records are deserialized **positionally**, so parsing is independent of the exact CSV header spelling and of any byte-order mark on the header row. Missing Titanic numeric fields deserialize to `None` and are mapped to `NaN` exactly as before.
- The error type and the download → SHA-256 verify → cache → reuse workflow are unchanged; the Serde change is an internal parsing refactor.
- Each loader's content type now has a named alias (`IrisData`, `BostonHousingData`, `DiabetesData`, `TitanicData`, and the shared `WineData` for both wine subsets), used for the `Dataset<…>` field and the owned/borrowed accessor return types.
- `data()` on every loader now returns a reference to the cached data tuple (`&IrisData`, `&TitanicData`, …) instead of a tuple of references (`(&Array2, &Array1)`). Call-site destructuring (`let (features, labels) = ds.data()?`) is unchanged thanks to match ergonomics.

### Added
- `into_data(self)` and `take_data(&mut self)` on every dataset loader (`Iris`, `BostonHousing`, `Diabetes`, `Titanic`, `RedWineQuality`, `WhiteWineQuality`), returning **owned** arrays instead of borrows — no `to_owned()` clone needed. `into_data` consumes the loader; `take_data` leaves it reusable (a later accessor call reloads). Built on the new `Dataset::into_inner` / `Dataset::take` in `dataset-core`.
- `get_data(&self) -> Option<&XData>` and `get_data_mut(&mut self) -> Option<&mut XData>` on every dataset loader, returning a reference to the cached data tuple **without** triggering loading (they return `None` if the data has not been loaded yet). `get_data` borrows it; `get_data_mut` allows editing it in place — no `to_owned()` clone, no reload, and the change persists in the cache. Built on the new `Dataset::get` / `Dataset::get_mut` in `dataset-core`.
- `serde` (with the `derive` feature) as a direct dependency, used for record deserialization.

## [0.1.0] - 2026-05-27
### Added
- Initial release as a standalone crate.
- Split out from `dataset-core` v0.1.x as part of the workspace reorganization. The dataset loaders previously lived behind the `datasets` feature in `dataset-core`; they are now their own crate that depends on `dataset-core` with the `utils` feature.
- Public surface: `iris::Iris`, `boston_housing::BostonHousing`, `diabetes::Diabetes`, `titanic::Titanic`, `wine_quality::red_wine_quality::RedWineQuality`, `wine_quality::white_wine_quality::WhiteWineQuality`. The structs themselves are also re-exported at the crate root.

### Migration from `dataset-core` 0.1.x with `datasets` feature

| Old path (`dataset-core` 0.1.x)                                              | New path (`dataset-ml` 0.1.0)                                    |
|------------------------------------------------------------------------------|------------------------------------------------------------------|
| `dataset_core::datasets::iris::Iris`                                         | `dataset_ml::iris::Iris`                                         |
| `dataset_core::datasets::boston_housing::BostonHousing`                      | `dataset_ml::boston_housing::BostonHousing`                      |
| `dataset_core::datasets::diabetes::Diabetes`                                 | `dataset_ml::diabetes::Diabetes`                                 |
| `dataset_core::datasets::titanic::Titanic`                                   | `dataset_ml::titanic::Titanic`                                   |
| `dataset_core::datasets::wine_quality::red_wine_quality::RedWineQuality`     | `dataset_ml::wine_quality::red_wine_quality::RedWineQuality`     |
| `dataset_core::datasets::wine_quality::white_wine_quality::WhiteWineQuality` | `dataset_ml::wine_quality::white_wine_quality::WhiteWineQuality` |

## History prior to the split

The following entries trace the development of the dataset loaders while they were part of `dataset-core` 0.1.x. They are reproduced here for continuity.

### 2026-04-14
- Remove formatting of record data in dataset validations.

### 2026-04-13
- Add comprehensive semantic tests for datasets, including value constraints, consistency checks.

### 2026-04-12
- Add detailed module-level dataset descriptions for all dataset submodules.

### 2026-04-11
- Split Wine Quality dataset into red and white datasets.

### 2026-04-07
- Add feature gating for `datasets`, update examples and tests.

### 2026-04-06
- Rename all dataset test files to `dataset_*_test.rs`.

### 2026-04-05
- Move dataset implementations to the `datasets` module.

### 2026-04-02
- Streamline dataset loader methods so that downloading and parsing happen in one method.

### 2026-04-01
- Refactor dataset modules to use `Dataset` for unified lazy loading and caching logic.

### 2026-03-29
- Refactor dataset modules to return `PathBuf` from `download_dataset` and accept it in `parse_dataset` for improved API clarity.

### 2026-03-28
- Separate downloading and parsing logic for improved clarity and maintainability.

### 2026-03-27
- Remove hardcoded feature counts from dataset modules and infer dynamically.

### 2026-03-26
- Refactor dataset error handling to use `empty_dataset` helper for improved clarity.

### 2026-03-25
- Remove hardcoded dataset sample sizes for dynamic determination.

### 2026-03-18
- Implement `Clone` and `Debug` traits for dataset structs.

### 2026-03-17
- Use type alias for Titanic dataset.

### 2026-03-16
- Replace manual CSV parsing with the `csv` crate for Titanic and Wine Quality datasets.

### 2026-03-14
- Replace manual CSV parsing with the `csv` crate for Iris.

### 2026-03-13
- Replace manual CSV parsing with the `csv` crate for Diabetes.

### 2026-03-12
- Replace manual CSV parsing with the `csv` crate for Boston Housing.

### 2026-03-11
- Improve error handling and code readability across dataset modules.

### 2026-03-09
- Refactor Wine Quality dataset handling to enable lazy loading and improve modularity.

### 2026-03-08
- Refactor Titanic dataset handling to enable lazy loading and improve modularity.

### 2026-03-07
- Refactor Iris dataset handling to enable lazy loading and improve modularity.

### 2026-03-06
- Refactor Diabetes dataset handling to enable lazy loading and improve modularity.

### 2026-03-05
- Refactor Boston Housing dataset handling to enable lazy loading and improve modularity.

### 2026-03-04
- Update dataset documentation for clarity and consistency.

### 2026-03-02
- Add SHA-256 validation for Wine Quality dataset download.

### 2026-03-01
- Add SHA-256 validation for Titanic dataset download.

### 2026-02-28
- Add SHA-256 validation for Iris dataset download.

### 2026-02-27
- Add SHA-256 validation for Diabetes dataset download.

### 2026-02-26
- Add SHA-256 validation for Boston Housing dataset download.

### 2026-02-24
- Replace hardcoded dataset configurations with reusable constants.

### 2026-02-23
- Replace hardcoded Wine Quality dataset with dynamic download and processing.

### 2026-02-22
- Replace hardcoded Titanic dataset with dynamic download and processing.

### 2026-02-20
- Replace hardcoded Boston Housing dataset with dynamic download and processing.

### 2026-02-19
- Replace hardcoded Diabetes dataset with dynamic download and processing.

### 2026-02-18
- Replace hardcoded Iris dataset with dynamic download and processing.

### 2026-02-14
- Initial project scaffolding with starter datasets and documentation.
