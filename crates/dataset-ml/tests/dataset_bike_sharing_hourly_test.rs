#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::bike_sharing::bike_sharing_hourly::BikeSharingHourly;
use ndarray::{Array1, Array2};
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

/// The two years span this many distinct dates.
const N_DATES: usize = 731;

/// Sum of the `cnt` column over the whole subset. The daily subset sums to the
/// same total, because both aggregate one rental log.
const TOTAL_CNT: f64 = 3_292_679.0;

/// Assert the hourly Bike Sharing invariants: the shapes, the date range, the
/// per-feature domains, and the target identity `cnt = casual + registered`.
fn assert_bike_hourly_semantics(
    dates: &Array1<String>,
    features: &Array2<f64>,
    targets: &Array2<f64>,
) {
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
    let heavy_weather = features.column(7).iter().filter(|&&v| v == 4.0).count();
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
}

#[test]
// Verifies that the hourly Bike Sharing dataset loads with the correct shapes,
// date range, feature domains, and target identity.
fn test_load_bike_sharing_hourly() {
    let download_dir = "./test_load_bike_sharing_hourly"; // the loader creates this directory if it is missing

    let dataset = BikeSharingHourly::new(download_dir);
    let dates = dataset.dates().unwrap();
    let features = dataset.features().unwrap();
    let targets = dataset.targets().unwrap();

    assert_bike_hourly_semantics(dates, features, targets);

    // `data()` returns the same three arrays at once.
    let (dates, features, targets) = dataset.data().unwrap();
    assert_eq!(dates.len(), N_SAMPLES);
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(targets.shape(), &[N_SAMPLES, N_TARGETS]);

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
    let (dates, _features, _targets) = dataset.data().unwrap();
    assert_eq!(dates.len(), N_SAMPLES);

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
    let (dates, _features, _targets) = dataset.data().unwrap();
    assert_eq!(dates.len(), N_SAMPLES);

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
// Verifies that into_data() returns owned arrays and consumes the dataset.
fn test_bike_sharing_hourly_into_data() {
    let download_dir = "./test_bike_sharing_hourly_into_data";

    let dataset = BikeSharingHourly::new(download_dir);
    let (dates, mut features, targets) = dataset.into_data().unwrap();
    // into_data() consumed `dataset`. The three arrays are now fully owned.

    assert_eq!(dates.len(), N_SAMPLES);
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(targets.shape(), &[N_SAMPLES, N_TARGETS]);

    // The caller can mutate the owned data directly, with no `to_owned()` clone.
    features[[0, 0]] = 4.0;
    assert_eq!(features[[0, 0]], 4.0);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that take_data() returns owned arrays and leaves the instance reusable.
fn test_bike_sharing_hourly_take_data() {
    let download_dir = "./test_bike_sharing_hourly_take_data";

    let mut dataset = BikeSharingHourly::new(download_dir);
    let (dates, features, targets) = dataset.take_data().unwrap();

    assert_eq!(dates.len(), N_SAMPLES);
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(targets.shape(), &[N_SAMPLES, N_TARGETS]);

    // After take_data, the instance resets to unloaded but stays usable. The next
    // access reloads it from the cached file and yields the same shapes.
    let (dates, features, targets) = dataset.data().unwrap();
    assert_eq!(dates.len(), N_SAMPLES);
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(targets.shape(), &[N_SAMPLES, N_TARGETS]);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data() returns None before loading and the cached references after.
fn test_bike_sharing_hourly_get_data() {
    let download_dir = "./test_bike_sharing_hourly_get_data";

    let dataset = BikeSharingHourly::new(download_dir);
    // Before loading, get_data() returns None and triggers no download.
    assert!(dataset.get_data().is_none());

    dataset.data().unwrap();
    let (dates, features, targets) = dataset.get_data().unwrap();
    assert_eq!(dates.len(), N_SAMPLES);
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(targets.shape(), &[N_SAMPLES, N_TARGETS]);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that get_data_mut() edits the cached arrays in place.
fn test_bike_sharing_hourly_get_data_mut() {
    let download_dir = "./test_bike_sharing_hourly_get_data_mut";

    let mut dataset = BikeSharingHourly::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() scales `temp` (column 8) back to degrees Celsius in place,
    // with no clone and no reload.
    dataset.data().unwrap();
    if let Some((_dates, features, _targets)) = dataset.get_data_mut() {
        features[[0, 8]] *= 41.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (_dates, features, _targets) = dataset.data().unwrap();
    assert_eq!(features[[0, 8]], 0.24 * 41.0);

    remove_dir_all(download_dir).unwrap();
}
