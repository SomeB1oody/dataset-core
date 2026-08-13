//! Bike Sharing dataset.
//!
//! Rental counts of the Capital Bikeshare system in Washington, D.C., over the
//! two years 2011 and 2012. Each record also carries the weather and the season
//! of its period. The task is to predict the rental count from the calendar and
//! weather attributes. Each sample carries its calendar date, and the rows stay
//! in chronological order.
//!
//! The one source archive holds two aggregations of the same rental log. Each
//! one has its own loader:
//!
//! - `bike_sharing_hourly::BikeSharingHourly` - 17,379 hourly records
//! - `bike_sharing_daily::BikeSharingDaily` - 731 daily records
//!
//! **Columns (16 hourly, 15 daily):**
//!
//! | Name         | Type      | Description                     |
//! |--------------|-----------|----------------------------------|
//! | `dteday`     | `String`  | calendar date as `YYYY-MM-DD`   |
//! | `season`     | `Numeric` | `1` = winter, `2` = spring, `3` = summer, `4` = fall |
//! | `yr`         | `Numeric` | `0` = 2011, `1` = 2012          |
//! | `mnth`       | `Numeric` | month, `1` to `12`              |
//! | `hr`         | `Numeric` | hour, `0` to `23`. The hourly subset alone holds it |
//! | `holiday`    | `Numeric` | `1` on a holiday, else `0`      |
//! | `weekday`    | `Numeric` | `0` = Sunday to `6` = Saturday  |
//! | `workingday` | `Numeric` | `1` on a day that is neither a weekend nor a holiday, else `0` |
//! | `weathersit` | `Numeric` | `1` = clear, `2` = mist, `3` = light rain or snow, `4` = heavy rain or snow |
//! | `temp`       | `Numeric` | temperature in Celsius, divided by 41 |
//! | `atemp`      | `Numeric` | apparent temperature in Celsius, divided by 50 |
//! | `hum`        | `Numeric` | humidity, divided by 100        |
//! | `windspeed`  | `Numeric` | wind speed, divided by 67       |
//! | `casual`     | `Numeric` | rentals by users without a membership |
//! | `registered` | `Numeric` | rentals by members              |
//! | `cnt`        | `Numeric` | total rentals, the sum of `casual` and `registered` |
//!
//! The source designates the weather and calendar columns as the inputs
//! (`FEATURE_NAMES` on each loader) and `casual`, `registered`, and `cnt` as
//! the labels (`TARGET_NAMES` on each loader).
//!
//! Both subsets span `2011-01-01` to `2012-12-31`. The source normalizes
//! `temp`, `atemp`, `hum`, and `windspeed` to `[0, 1]`. The three targets make
//! a multi-output target.
//!
//! **Samples:**
//! - Hourly subset: 17,379
//! - Daily subset: 731
//!
//! **Application:** Regression / demand forecasting
//!
//! **Missing values:** none. No field is empty in either subset. The hourly
//! subset holds 17,379 of the 17,544 hours of the two years, because the source
//! omits the hours with no rental activity. The daily subset covers all 731
//! days.
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C5W894>

pub mod bike_sharing_daily;
pub mod bike_sharing_hourly;

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use csv::ReaderBuilder;
use dataset_core::{DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::Array1;
use std::fs::File;
use std::path::PathBuf;

/// The URL for the Bike Sharing dataset (the ZIP archive that holds both
/// `hour.csv` and `day.csv`).
///
/// # Citation
///
/// Fanaee-T, H. (2013). Bike Sharing \[Dataset\]. UCI Machine Learning
/// Repository. <https://doi.org/10.24432/C5W894>
const BIKE_DATA_URL: &str =
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip";

/// The filename used for the downloaded ZIP archive inside the temp directory.
const BIKE_ZIP_FILENAME: &str = "bike+sharing+dataset.zip";

/// Source column index of the date (`dteday`). Column 0 is `instant`, a 1-based
/// row counter that carries no information, so no loader keeps it.
const DATE_COLUMN: usize = 1;

/// Number of target columns (`casual`, `registered`, `cnt`), in source order.
const N_TARGETS: usize = 3;

/// The names of the target columns, in source order.
const TARGET_NAMES: [&str; N_TARGETS] = ["casual", "registered", "cnt"];

/// Number of columns that come before the features (`instant` and `dteday`).
const N_LEADING_COLUMNS: usize = 2;

/// Get one CSV member of the shared Bike Sharing ZIP archive.
///
/// Both subsets come from the same archive, under different member names. Each
/// one caches its own member as a separate file with its own SHA256 hash.
///
/// # Parameters
///
/// - `dir` - The directory that stores the dataset.
/// - `cache_filename` - The name of the cached file (for example,
///   `"bike_sharing_hourly.csv"`).
/// - `dataset_name` - The dataset name for error messages.
/// - `expected_sha256` - The expected SHA256 hash of the CSV member.
/// - `member_filename` - The name of the member inside the archive
///   (`"hour.csv"` or `"day.csv"`).
///
/// # Returns
///
/// - `PathBuf` - Path to the cached CSV file.
///
/// # Errors
///
/// Returns `DatasetError` if the download, the extraction, or the hash check
/// fails.
fn acquire_bike_csv(
    dir: &str,
    cache_filename: &str,
    dataset_name: &str,
    expected_sha256: &str,
    member_filename: &str,
) -> Result<PathBuf, DatasetError> {
    acquire_dataset(
        dir,
        cache_filename,
        dataset_name,
        Some(expected_sha256),
        |temp_path| {
            download_to_with_retries(
                BIKE_DATA_URL,
                temp_path,
                Some(BIKE_ZIP_FILENAME),
                DOWNLOAD_RETRIES,
            )?;
            unzip(&temp_path.join(BIKE_ZIP_FILENAME), temp_path)?;
            Ok(temp_path.join(member_filename))
        },
    )
}

/// Parse one Bike Sharing CSV (hourly or daily) into a [`Table`].
///
/// The two subsets share one column layout: `instant`, `dteday`, the feature
/// columns, then `casual`, `registered`, and `cnt`. Only the feature list
/// differs, so `feature_names` selects the subset. The file is comma-separated
/// and starts with a header row.
///
/// The parser drops `instant`, a 1-based row counter. It keeps `dteday`
/// verbatim as a `YYYY-MM-DD` string.
///
/// # Parameters
///
/// - `dataset_name` - The dataset name for error messages.
/// - `file_path` - Path to the CSV file.
/// - `feature_names` - The feature column names, in source order.
/// - `n_samples` - Expected number of records, used to reserve capacity.
///
/// # Returns
///
/// - `Table` - One `dteday` column, one column per feature name, and the three
///   target columns.
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - The file cannot be read
/// - A row has an unexpected number of columns
/// - A date field is empty
/// - A feature or target value does not parse as `f64`
/// - The file holds no records
fn parse_bike_data(
    dataset_name: &'static str,
    file_path: &std::path::Path,
    feature_names: &[&'static str],
    n_samples: usize,
) -> Result<Table, DatasetError> {
    let n_features = feature_names.len();
    let n_columns = N_LEADING_COLUMNS + n_features + N_TARGETS;
    let first_target_column = n_columns - N_TARGETS;

    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .from_reader(file);

    let mut dates: Vec<String> = Vec::with_capacity(n_samples);
    let mut features: Vec<Vec<f64>> = (0..n_features)
        .map(|_| Vec::with_capacity(n_samples))
        .collect();
    let mut targets: Vec<Vec<f64>> = (0..N_TARGETS)
        .map(|_| Vec::with_capacity(n_samples))
        .collect();

    for (idx, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| DatasetError::csv_read_error(dataset_name, e))?;
        let line_num = idx + 2; // +1 for the header, +1 for 1-based lines

        // Skip blank lines, such as a trailing newline at the end of the file.
        if record.iter().all(|f| f.is_empty()) {
            continue;
        }

        if record.len() != n_columns {
            return Err(DatasetError::invalid_column_count(
                dataset_name,
                n_columns,
                record.len(),
                line_num,
            ));
        }

        let date = &record[DATE_COLUMN];
        if date.is_empty() {
            return Err(DatasetError::invalid_value(
                dataset_name,
                "dteday",
                date,
                line_num,
            ));
        }
        dates.push(date.to_string());

        for col in N_LEADING_COLUMNS..N_LEADING_COLUMNS + n_features {
            let value: f64 = record[col]
                .parse()
                .map_err(|e| DatasetError::parse_failed(dataset_name, "features", line_num, e))?;
            features[col - N_LEADING_COLUMNS].push(value);
        }

        for col in first_target_column..n_columns {
            let value: f64 = record[col]
                .parse()
                .map_err(|e| DatasetError::parse_failed(dataset_name, "targets", line_num, e))?;
            targets[col - first_target_column].push(value);
        }
    }

    let mut columns: Vec<Column> = Vec::with_capacity(1 + n_features + N_TARGETS);
    columns.push(Column::new(
        "dteday",
        ColumnData::String(Array1::from_vec(dates)),
    ));
    for (name, values) in feature_names.iter().copied().zip(features) {
        columns.push(Column::new(
            name,
            ColumnData::Numeric(Array1::from_vec(values)),
        ));
    }
    for (name, values) in TARGET_NAMES.iter().copied().zip(targets) {
        columns.push(Column::new(
            name,
            ColumnData::Numeric(Array1::from_vec(values)),
        ));
    }

    Table::new(dataset_name, columns)
}
