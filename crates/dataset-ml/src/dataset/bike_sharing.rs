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
//! **Dates:** `dteday`, the calendar date of the record, as `YYYY-MM-DD`. Both
//! subsets span `2011-01-01` to `2012-12-31`.
//!
//! **Features:** 12 for the hourly subset, 11 for the daily subset. The daily
//! subset has no `hr` column. Every feature is numeric. The source normalizes
//! the weather readings `temp`, `atemp`, `hum`, and `windspeed` to `[0, 1]`.
//! The calendar attributes are integer codes.
//!
//! **Targets (3, multi-output):** `casual`, `registered`, and `cnt`. The count
//! `cnt` is the sum of `casual` and `registered`. Use `casual` and `registered`
//! to model the two rider groups apart.
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
use csv::ReaderBuilder;
use dataset_core::{DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::{Array1, Array2};
use std::fs::File;
use std::path::PathBuf;

/// Type alias shared by both Bike Sharing subsets: (dates, features, targets).
pub type BikeSharingData = (Array1<String>, Array2<f64>, Array2<f64>);

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

/// Parse one Bike Sharing CSV (hourly or daily) into `(dates, features, targets)`.
///
/// The two subsets share one column layout: `instant`, `dteday`, the feature
/// columns, then `casual`, `registered`, and `cnt`. Only the feature count
/// differs, so `n_features` selects the subset. The file is comma-separated and
/// starts with a header row.
///
/// The parser drops `instant`, a 1-based row counter. It keeps `dteday`
/// verbatim as a `YYYY-MM-DD` string.
///
/// # Parameters
///
/// - `dataset_name` - The dataset name for error messages.
/// - `file_path` - Path to the CSV file.
/// - `n_features` - Number of feature columns (12 hourly, 11 daily).
/// - `n_samples` - Expected number of records, used to reserve capacity.
///
/// # Returns
///
/// - `Array1<String>` - Date vector with length `n_samples`.
/// - `Array2<f64>` - Feature matrix with shape `(n_samples, n_features)`.
/// - `Array2<f64>` - Target matrix with shape `(n_samples, 3)`.
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
    dataset_name: &str,
    file_path: &std::path::Path,
    n_features: usize,
    n_samples: usize,
) -> Result<BikeSharingData, DatasetError> {
    let n_columns = N_LEADING_COLUMNS + n_features + N_TARGETS;

    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .from_reader(file);

    let mut dates: Vec<String> = Vec::with_capacity(n_samples);
    let mut features: Vec<f64> = Vec::with_capacity(n_samples * n_features);
    let mut targets: Vec<f64> = Vec::with_capacity(n_samples * N_TARGETS);

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
            features.push(value);
        }

        for col in n_columns - N_TARGETS..n_columns {
            let value: f64 = record[col]
                .parse()
                .map_err(|e| DatasetError::parse_failed(dataset_name, "targets", line_num, e))?;
            targets.push(value);
        }
    }

    let parsed_samples = dates.len();
    if parsed_samples == 0 {
        return Err(DatasetError::empty_dataset(dataset_name));
    }

    let dates_array = Array1::from_vec(dates);

    let features_array = Array2::from_shape_vec((parsed_samples, n_features), features)
        .map_err(|e| DatasetError::array_shape_error(dataset_name, "features", e))?;

    let targets_array = Array2::from_shape_vec((parsed_samples, N_TARGETS), targets)
        .map_err(|e| DatasetError::array_shape_error(dataset_name, "targets", e))?;

    Ok((dates_array, features_array, targets_array))
}
