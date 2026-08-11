#![cfg(feature = "dataset")]

mod common;

use common::file_sha256_matches;
use dataset_ml::dataset::bike_sharing::bike_sharing_daily::BikeSharingDaily;
use ndarray::{Array1, Array2};
use std::collections::HashSet;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;

/// SHA256 of the cached daily Bike Sharing dataset file (`bike_sharing_daily.csv`).
const BIKE_DAILY_SHA256: &str = "a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be";

/// The daily subset holds one record per day of 2011 and 2012.
const N_SAMPLES: usize = 731;

/// Number of feature columns.
const N_FEATURES: usize = 11;

/// Number of target columns (`casual`, `registered`, `cnt`).
const N_TARGETS: usize = 3;

/// Sum of the `cnt` column over the whole subset. The hourly subset sums to the
/// same total, because both aggregate one rental log.
const TOTAL_CNT: f64 = 3_292_679.0;

/// Assert the daily Bike Sharing invariants: the shapes, the gap-free date
/// range, the per-feature domains, and the target identity
/// `cnt = casual + registered`.
fn assert_bike_daily_semantics(
    dates: &Array1<String>,
    features: &Array2<f64>,
    targets: &Array2<f64>,
) {
    assert_eq!(dates.len(), N_SAMPLES);
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(targets.shape(), &[N_SAMPLES, N_TARGETS]);

    // The daily subset covers every day of the two years once, in chronological
    // order and with no gaps.
    assert_eq!(dates[0], "2011-01-01");
    assert_eq!(dates[N_SAMPLES - 1], "2012-12-31");
    let unique_dates: HashSet<&str> = dates.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        unique_dates.len(),
        N_SAMPLES,
        "every daily record should carry a distinct date"
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
                dates[row - 1].as_str() < date.as_str(),
                "row {} date {:?} breaks the strict chronological order",
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
        let holiday = features[[row, 3]];
        let weekday = features[[row, 4]];
        let workingday = features[[row, 5]];
        let weathersit = features[[row, 6]];

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
        // No day carries the heavy-weather code 4, unlike the hourly subset.
        assert!(
            (1.0..=3.0).contains(&weathersit) && weathersit.fract() == 0.0,
            "row {} weathersit {} is outside 1..=3",
            row,
            weathersit
        );

        // The source normalizes `temp`, `atemp`, `hum`, and `windspeed` to [0, 1].
        for col in 7..N_FEATURES {
            assert!(
                (0.0..=1.0).contains(&features[[row, col]]),
                "row {} normalized feature col {} = {} is outside [0, 1]",
                row,
                col,
                features[[row, col]]
            );
        }
    }

    // Exactly 21 days carry the light rain or snow code.
    let rainy_days = features.column(6).iter().filter(|&&v| v == 3.0).count();
    assert_eq!(
        rainy_days, 21,
        "exactly 21 days should carry weathersit = 3"
    );

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
    assert_eq!(
        total_cnt, TOTAL_CNT,
        "the daily total should match the hourly total"
    );

    // The first record of `day.csv`, pinned value by value.
    assert_eq!(
        features.row(0).to_vec(),
        vec![
            1.0, 0.0, 1.0, 0.0, 6.0, 0.0, 2.0, 0.344167, 0.363625, 0.805833, 0.160446
        ]
    );
    assert_eq!(targets.row(0).to_vec(), vec![331.0, 654.0, 985.0]);

    // The last record of `day.csv`.
    assert_eq!(
        features.row(N_SAMPLES - 1).to_vec(),
        vec![
            1.0, 1.0, 12.0, 0.0, 1.0, 1.0, 2.0, 0.215833, 0.223487, 0.5775, 0.154846
        ]
    );
    assert_eq!(
        targets.row(N_SAMPLES - 1).to_vec(),
        vec![439.0, 2290.0, 2729.0]
    );
}

#[test]
// Verifies that the daily Bike Sharing dataset loads with the correct shapes,
// date range, feature domains, and target identity.
fn test_load_bike_sharing_daily() {
    let download_dir = "./test_load_bike_sharing_daily"; // the loader creates this directory if it is missing

    let dataset = BikeSharingDaily::new(download_dir);
    let dates = dataset.dates().unwrap();
    let features = dataset.features().unwrap();
    let targets = dataset.targets().unwrap();

    assert_bike_daily_semantics(dates, features, targets);

    // `data()` returns the same three arrays at once.
    let (dates, features, targets) = dataset.data().unwrap();
    assert_eq!(dates.len(), N_SAMPLES);
    assert_eq!(features.shape(), &[N_SAMPLES, N_FEATURES]);
    assert_eq!(targets.shape(), &[N_SAMPLES, N_TARGETS]);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the daily loader reuses a cached file instead of a new download.
fn test_bike_sharing_daily_no_need_download() {
    let download_dir = "./test_bike_sharing_daily_no_need_download";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();

    // The first load primes the cache. The second instance then reuses it.
    BikeSharingDaily::new(download_dir).data().unwrap();
    assert!(
        file_sha256_matches(
            &download_dir_path.join("bike_sharing_daily.csv"),
            BIKE_DAILY_SHA256
        )
        .unwrap(),
        "cached bike_sharing_daily.csv should match the expected SHA256"
    );

    let dataset = BikeSharingDaily::new(download_dir);
    let (dates, _features, _targets) = dataset.data().unwrap();
    assert_eq!(dates.len(), N_SAMPLES);

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that the loader detects a corrupt or fake daily data file and
// overwrites it with the real dataset.
fn test_bike_sharing_daily_overwrite() {
    let download_dir = "./test_bike_sharing_daily_overwrite";
    let download_dir_path = Path::new(download_dir);
    create_dir_all(download_dir_path).unwrap();
    {
        let bike_path = download_dir_path.join("bike_sharing_daily.csv");
        let mut fake_bike = File::create(bike_path).unwrap();
        fake_bike.write_all(b"fake data").unwrap();
    }

    let dataset = BikeSharingDaily::new(download_dir);
    let (dates, _features, _targets) = dataset.data().unwrap();
    assert_eq!(dates.len(), N_SAMPLES);

    assert!(
        file_sha256_matches(
            &download_dir_path.join("bike_sharing_daily.csv"),
            BIKE_DAILY_SHA256
        )
        .unwrap()
    );

    remove_dir_all(download_dir).unwrap();
}

#[test]
// Verifies that into_data() returns owned arrays and consumes the dataset.
fn test_bike_sharing_daily_into_data() {
    let download_dir = "./test_bike_sharing_daily_into_data";

    let dataset = BikeSharingDaily::new(download_dir);
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
fn test_bike_sharing_daily_take_data() {
    let download_dir = "./test_bike_sharing_daily_take_data";

    let mut dataset = BikeSharingDaily::new(download_dir);
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
fn test_bike_sharing_daily_get_data() {
    let download_dir = "./test_bike_sharing_daily_get_data";

    let dataset = BikeSharingDaily::new(download_dir);
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
fn test_bike_sharing_daily_get_data_mut() {
    let download_dir = "./test_bike_sharing_daily_get_data_mut";

    let mut dataset = BikeSharingDaily::new(download_dir);
    // Before loading, get_data_mut() returns None and triggers no download.
    assert!(dataset.get_data_mut().is_none());

    // get_data_mut() scales `temp` (column 7) back to degrees Celsius in place,
    // with no clone and no reload.
    dataset.data().unwrap();
    if let Some((_dates, features, _targets)) = dataset.get_data_mut() {
        features[[0, 7]] *= 41.0;
    }

    // The change persisted in the cache: a later access observes it.
    let (_dates, features, _targets) = dataset.data().unwrap();
    assert_eq!(features[[0, 7]], 0.344167 * 41.0);

    remove_dir_all(download_dir).unwrap();
}
