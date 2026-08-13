#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::bike_sharing::bike_sharing_hourly::BikeSharingHourly;
use dataset_ml::table::{ColumnData, Table};
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached hourly Bike Sharing dataset file (`bike_sharing_hourly.csv`).
const BIKE_HOURLY_SHA256: &str = "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f";

/// The hourly subset holds this many records.
const N_SAMPLES: usize = 17_379;

/// Number of feature columns.
const N_FEATURES: usize = 12;

/// Number of target columns (`casual`, `registered`, `cnt`).
const N_TARGETS: usize = 3;

/// Number of columns in the table: the date, the features, and the targets.
const N_COLUMNS: usize = 1 + N_FEATURES + N_TARGETS;

/// The two years span this many distinct dates.
const N_DATES: usize = 731;

/// Sum of the `cnt` column over the whole subset. The daily subset sums to the
/// same total, because both aggregate one rental log.
const TOTAL_CNT: f64 = 3_292_679.0;

/// The 16 column names, in source order.
const COLUMN_NAMES: [&str; N_COLUMNS] = [
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
];

/// The feature column names, in source order.
const FEATURE_NAMES: [&str; N_FEATURES] = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
];

/// The target column names, in source order.
const TARGET_NAMES: [&str; N_TARGETS] = ["casual", "registered", "cnt"];

/// Assert that the table names and types its columns as the docs claim.
fn assert_bike_hourly_layout(table: &Table) {
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), N_COLUMNS);
    assert_eq!(table.names().collect::<Vec<_>>(), COLUMN_NAMES);

    // The named constants agree with the source column order.
    assert_eq!(BikeSharingHourly::FEATURE_NAMES, FEATURE_NAMES);
    assert_eq!(BikeSharingHourly::TARGET_NAMES, TARGET_NAMES);

    // `dteday` holds a date string.
    let date_column = table.column("dteday").unwrap();
    assert!(matches!(date_column.data(), ColumnData::String(_)));

    // The 12 features and the 3 targets are numeric, in source order.
    for name in FEATURE_NAMES.iter().chain(TARGET_NAMES.iter()) {
        let column = table.column(name).unwrap();
        assert!(
            matches!(column.data(), ColumnData::Numeric(_)),
            "column {name} should be numeric"
        );
    }
}

/// Assert the hourly Bike Sharing invariants: the shapes, the date range, the
/// per-feature domains, and the target identity `cnt = casual + registered`.
fn assert_bike_hourly_semantics(table: &Table) {
    assert_bike_hourly_layout(table);

    let dates = table.column("dteday").unwrap().as_string().unwrap();
    let features = table
        .numeric_matrix(&BikeSharingHourly::FEATURE_NAMES)
        .unwrap();
    let targets = table
        .numeric_matrix(&BikeSharingHourly::TARGET_NAMES)
        .unwrap();

    assert_eq!(dates.len(), N_SAMPLES);
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(targets.shape(), &[N_SAMPLES, N_TARGETS]);

    // The dates run over the two full years, in chronological order. The source
    // omits the hours with no rental activity, so a date repeats once per
    // recorded hour of that day.
    assert_eq!(dates[0], "2011-01-01");
    assert_eq!(dates[N_SAMPLES - 1], "2012-12-31");
    let unique_dates: HashSet<&str> = dates.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        unique_dates.len(),
        N_DATES,
        "the two years should hold 731 distinct dates"
    );
    for (row, date) in dates.iter().enumerate() {
        assert_eq!(
            date.len(),
            10,
            "date {} = {:?} is not YYYY-MM-DD",
            row,
            date
        );
        if row > 0 {
            assert!(
                dates[row - 1].as_str() <= date.as_str(),
                "row {} date {:?} breaks the chronological order",
                row,
                date
            );
        }
    }

    // Every feature is finite and stays inside its documented domain.
    for row in 0..features.nrows() {
        for col in 0..features.ncols() {
            assert!(
                features[[row, col]].is_finite(),
                "feature[{}, {}] = {} is not finite",
                row,
                col,
                features[[row, col]]
            );
        }

        let season = features[[row, 0]];
        let year = features[[row, 1]];
        let month = features[[row, 2]];
        let hour = features[[row, 3]];
        let holiday = features[[row, 4]];
        let weekday = features[[row, 5]];
        let workingday = features[[row, 6]];
        let weathersit = features[[row, 7]];

        assert!(
            (1.0..=4.0).contains(&season) && season.fract() == 0.0,
            "row {} season {} is outside 1..=4",
            row,
            season
        );
        assert!(
            year == 0.0 || year == 1.0,
            "row {} yr {} is not 0/1",
            row,
            year
        );
        assert!(
            (1.0..=12.0).contains(&month) && month.fract() == 0.0,
            "row {} mnth {} is outside 1..=12",
            row,
            month
        );
        assert!(
            (0.0..=23.0).contains(&hour) && hour.fract() == 0.0,
            "row {} hr {} is outside 0..=23",
            row,
            hour
        );
        assert!(
            holiday == 0.0 || holiday == 1.0,
            "row {} holiday {} is not 0/1",
            row,
            holiday
        );
        assert!(
            (0.0..=6.0).contains(&weekday) && weekday.fract() == 0.0,
            "row {} weekday {} is outside 0..=6",
            row,
            weekday
        );
        assert!(
            workingday == 0.0 || workingday == 1.0,
            "row {} workingday {} is not 0/1",
            row,
            workingday
        );
        assert!(
            (1.0..=4.0).contains(&weathersit) && weathersit.fract() == 0.0,
            "row {} weathersit {} is outside 1..=4",
            row,
            weathersit
        );

        // The source normalizes `temp`, `atemp`, `hum`, and `windspeed` to [0, 1].
        for col in 8..N_FEATURES {
            assert!(
                (0.0..=1.0).contains(&features[[row, col]]),
                "row {} normalized feature col {} = {} is outside [0, 1]",
                row,
                col,
                features[[row, col]]
            );
        }
    }

    // `cnt` is the exact sum of `casual` and `registered`. Every count is a
    // non-negative integer.
    let mut total_cnt = 0.0;
    for row in 0..targets.nrows() {
        let casual = targets[[row, 0]];
        let registered = targets[[row, 1]];
        let cnt = targets[[row, 2]];

        for (name, value) in [("casual", casual), ("registered", registered), ("cnt", cnt)] {
            assert!(
                value.is_finite() && value >= 0.0 && value.fract() == 0.0,
                "row {} {} = {} is not a non-negative integer count",
                row,
                name,
                value
            );
        }
        assert_eq!(
            cnt,
            casual + registered,
            "row {} cnt should equal casual + registered",
            row
        );
        total_cnt += cnt;
    }
    assert_eq!(total_cnt, TOTAL_CNT, "the total rental count should match");

    // The `weathersit` code 4 (heavy rain or snow) is rare: 3 of the 17,379 records.
    let weathersit = table.column("weathersit").unwrap().as_numeric().unwrap();
    let heavy_weather = weathersit.iter().filter(|&&v| v == 4.0).count();
    assert_eq!(
        heavy_weather, 3,
        "exactly 3 records should carry weathersit = 4"
    );

    // The first record of `hour.csv`, pinned value by value.
    assert_eq!(dates[0], "2011-01-01");
    let first_features: Vec<f64> = features.row(0).to_vec();
    assert_eq!(
        first_features,
        vec![
            1.0, 0.0, 1.0, 0.0, 0.0, 6.0, 0.0, 1.0, 0.24, 0.2879, 0.81, 0.0
        ]
    );
    assert_eq!(targets.row(0).to_vec(), vec![3.0, 13.0, 16.0]);

    // The last record of `hour.csv`.
    assert_eq!(
        features.row(N_SAMPLES - 1).to_vec(),
        vec![
            1.0, 1.0, 12.0, 23.0, 0.0, 1.0, 1.0, 1.0, 0.26, 0.2727, 0.65, 0.1343
        ]
    );
    assert_eq!(targets.row(N_SAMPLES - 1).to_vec(), vec![12.0, 37.0, 49.0]);

    // A column reached by name agrees with its position in the feature matrix.
    for (col, name) in FEATURE_NAMES.iter().enumerate() {
        let column = table.column(name).unwrap().as_numeric().unwrap();
        for row in [0usize, 1, 9_000, N_SAMPLES - 1] {
            assert_eq!(
                column[row],
                features[[row, col]],
                "column {name} disagrees with matrix column {col} at row {row}"
            );
        }
    }
}

#[test]
// Verifies that the hourly Bike Sharing dataset loads with the correct shapes,
// date range, feature domains, and target identity.
fn test_load_bike_sharing_hourly() {
    let download_dir = "./test_load_bike_sharing_hourly"; // the loader creates this directory if it is missing

    let dataset = BikeSharingHourly::new(download_dir);
    let table = dataset.data().unwrap();

    assert_bike_hourly_semantics(table);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the hourly loader reuses a cached file instead of a new download.
fn test_bike_sharing_hourly_no_need_download() {
    let download_dir = "./test_bike_sharing_hourly_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load primes the cache. The second instance then reuses it.
    BikeSharingHourly::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join("bike_sharing_hourly.csv"),
            BIKE_HOURLY_SHA256
        )
        .unwrap(),
        "cached bike_sharing_hourly.csv should match the expected SHA256"
    );

    let dataset = BikeSharingHourly::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake hourly data file and
// overwrites it with the real dataset.
fn test_bike_sharing_hourly_overwrite() {
    let download_dir = "./test_bike_sharing_hourly_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let bike_path = download_dir_path.join("bike_sharing_hourly.csv");
        let mut fake_bike = File::create(bike_path).unwrap();
        fake_bike.write_all(b"fake data").unwrap();
    }

    let dataset = BikeSharingHourly::new(download_dir);
    assert_eq!(dataset.data().unwrap().n_samples(), N_SAMPLES);

    assert!(
        file_sha256_matches(
            &download_dir_path.join("bike_sharing_hourly.csv"),
            BIKE_HOURLY_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns the owned table and consumes the dataset.
fn test_bike_sharing_hourly_into_data() {
    let download_dir = "./test_bike_sharing_hourly_into_data";

    let dataset = BikeSharingHourly::new(download_dir);
    let mut table = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The table is now fully owned.

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), N_COLUMNS);

    // The caller can mutate the owned table directly, with no clone.
    if let Some(ColumnData::Numeric(values)) = table.column_mut("season").map(|c| c.data_mut()) {
        values[0] = 4.0;
    }
    assert_eq!(
        table.column("season").unwrap().as_numeric().unwrap()[0],
        4.0
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns the owned table and leaves the instance reusable.
fn test_bike_sharing_hourly_take_data() {
    let download_dir = "./test_bike_sharing_hourly_take_data";

    let mut dataset = BikeSharingHourly::new(download_dir);
    let table = dataset.take_data().unwrap();

    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), N_COLUMNS);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file and yields the same shape.
    let table = dataset.data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), N_COLUMNS);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached reference after.
fn test_bike_sharing_hourly_get_data() {
    let download_dir = "./test_bike_sharing_hourly_get_data";

    let dataset = BikeSharingHourly::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    dataset.data().unwrap();
    let table = dataset.get_data().unwrap();
    assert_eq!(table.n_samples(), N_SAMPLES);
    assert_eq!(table.n_columns(), N_COLUMNS);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached table in place.
fn test_bike_sharing_hourly_get_data_mut() {
    let download_dir = "./test_bike_sharing_hourly_get_data_mut";

    let mut dataset = BikeSharingHourly::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() scales `temp` back to degrees Celsius in place, with no
    // clone and no reload.
    dataset.data().unwrap();
    if let Some(table) = dataset.get_data_mut()
        && let Some(ColumnData::Numeric(values)) = table.column_mut("temp").map(|c| c.data_mut())
    {
        values[0] *= 41.0;
    }

    // The change persisted in the cache: a later access observes it.
    let table = dataset.data().unwrap();
    assert_eq!(
        table.column("temp").unwrap().as_numeric().unwrap()[0],
        0.24 * 41.0
    );
    assert_eq!(
        table.column("atemp").unwrap().as_numeric().unwrap()[0],
        0.2879,
        "the other columns should stay untouched"
    );

    remove_dir_all(download_dir).unwrap();
}
