//! MovieLens 100K dataset.
//!
//! 100,000 movie ratings that 943 users gave to 1,682 movies, collected by the
//! GroupLens Research Project at the University of Minnesota between September
//! 1997 and April 1998. Every user rated at least 20 movies. The usual task is
//! to predict the rating a user gives a movie, or to recommend movies a user has
//! not rated.
//!
//! One sample is one **rating**, not one user and not one movie.
//!
//! **Columns (4):**
//!
//! | Name        | Type      | Description                     |
//! |-------------|-----------|-----------------------------------|
//! | `user_id`   | `Integer` | the user who gave the rating, `1` to `943` |
//! | `item_id`   | `Integer` | the movie the rating is about, `1` to `1682` |
//! | `rating`    | `Integer` | whole stars, `1` to `5`         |
//! | `timestamp` | `Integer` | the moment of the rating, in Unix seconds |
//!
//! The source designates `rating` as the label ([`MovieLens100k::TARGET`](crate::MovieLens100k::TARGET)).
//!
//! **Samples:** 100,000 ratings
//! **Application:** Recommendation / collaborative filtering
//!
//! **Missing values:** none. The log holds only the ratings that users gave, so
//! the 943×1,682 user-item matrix it describes is 93.7% empty.
//!
//! **Usage license:** GroupLens permits research use under conditions that
//! differ from this crate's MIT license. See the struct docs.
//!
//! **Source:** GroupLens Research, University of Minnesota
//! <https://grouplens.org/datasets/movielens/100k/>

use crate::DOWNLOAD_RETRIES;
use crate::table::{Column, ColumnData, Table};
use crate::traits::impl_ml_dataset;
use csv::ReaderBuilder;
use dataset_core::{Dataset, DatasetError, acquire_dataset, download_to_with_retries, unzip};
use ndarray::Array1;
use std::fs::File;

/// The URL for the MovieLens 100K dataset (the ZIP archive).
///
/// # Citation
///
/// Harper, F. M. & Konstan, J. A. (2015). "The MovieLens Datasets: History and
/// Context." *ACM Transactions on Interactive Intelligent Systems*, 5(4),
/// Article 19. <https://doi.org/10.1145/2827872>
const MOVIELENS_DATA_URL: &str = "https://files.grouplens.org/datasets/movielens/ml-100k.zip";

/// The filename used for the downloaded ZIP archive inside the temp directory.
const MOVIELENS_ZIP_FILENAME: &str = "ml-100k.zip";

/// The path of the rating log inside the archive.
const MOVIELENS_SOURCE_PATH: &str = "ml-100k/u.data";

/// The name of the cached MovieLens 100K rating log.
const MOVIELENS_FILENAME: &str = "movielens_100k_ratings.tsv";

/// The SHA256 hash of the cached rating log (`u.data`).
const MOVIELENS_SHA256: &str = "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490";

/// The name of the dataset.
const MOVIELENS_DATASET_NAME: &str = "movielens_100k";

/// Number of ratings in the log.
const N_SAMPLES: usize = 100_000;

/// Number of fields per record.
const N_COLUMNS: usize = 4;

/// A struct that represents the MovieLens 100K dataset with lazy loading.
///
/// The dataset loads only when you call a data accessor method. After the first
/// load, the dataset caches the data for later accesses.
///
/// # About Dataset
///
/// MovieLens 100K holds 100,000 ratings that 943 users gave to 1,682 movies
/// through the MovieLens web site, between September 19, 1997 and April 22,
/// 1998. GroupLens cleaned the data: every user in it rated at least 20 movies
/// and gave complete demographic information. The usual task is to predict the
/// rating a user gives a movie, or to recommend movies a user has not rated.
///
/// One sample is one **rating**, so [`n_samples`](crate::MlDataset::n_samples)
/// returns 100,000, not the user count or the movie count. Use
/// [`MovieLens100k::N_USERS`] and [`MovieLens100k::N_ITEMS`] for those.
///
/// # Columns
///
/// | Name        | Type      | Description                     |
/// |-------------|-----------|-----------------------------------|
/// | `user_id`   | `Integer` | the user who gave the rating, `1` to `943` |
/// | `item_id`   | `Integer` | the movie the rating is about, `1` to `1682` |
/// | `rating`    | `Integer` | whole stars, `1` to `5`         |
/// | `timestamp` | `Integer` | the moment of the rating, in Unix seconds |
///
/// The source designates `rating` as the label ([`MovieLens100k::TARGET`]).
///
/// The identifiers start at `1`, not at `0`. Subtract `1` to index a matrix of
/// `N_USERS` rows by `N_ITEMS` columns.
///
/// Every identifier in its range appears at least once, so the 943 users and the
/// 1,682 movies are all present. No user rated the same movie twice, so a
/// `(user_id, item_id)` pair identifies one rating.
///
/// The rows keep the order of the source file. That order is neither by user nor
/// by time, so sort by `timestamp` for a split by time.
///
/// # Ratings
///
/// | Rating | Count  |
/// |--------|--------|
/// | `1`    | 6,110  |
/// | `2`    | 11,370 |
/// | `3`    | 27,145 |
/// | `4`    | 34,174 |
/// | `5`    | 21,201 |
///
/// The mean rating is 3.53. A model that always predicts the mean is the usual
/// baseline.
///
/// # Timestamps
///
/// The timestamps run from `874724710` to `893286638`, which is
/// 1997-09-20 to 1998-04-22 in UTC. The source records no time zone.
///
/// # Sparsity
///
/// The log holds 100,000 of the 943 × 1,682 = 1,586,126 possible user-movie
/// pairs, so a dense matrix of the ratings would be 93.7% empty. Build that
/// matrix only if you need it, and expect most of it to hold your own
/// missing-value marker.
///
/// # What this loader does not read
///
/// The archive also holds the movie titles and genres (`u.item`), the user
/// demographics (`u.user`), and five prepared train/test splits (`u1.base` to
/// `u5.test`). Those files describe movies and users rather than ratings, so
/// their row counts differ from the rating count. This loader reads `u.data`
/// alone.
///
/// # Usage license
///
/// GroupLens permits research use of this dataset under conditions that this
/// crate's MIT license does not cover. The archive's own `README` states them:
///
/// - Do not state or imply an endorsement from the University of Minnesota or
///   the GroupLens Research Group.
/// - Acknowledge the dataset in any publication that uses it.
/// - Do not redistribute the data without separate permission.
/// - Do not use the data for a commercial or revenue-bearing purpose without
///   permission from a GroupLens faculty member.
///
/// This loader downloads the archive from GroupLens at run time and redistributes
/// nothing. Read the conditions before you use the data.
///
/// See more information at <https://grouplens.org/datasets/movielens/100k/>.
///
/// # Citation
///
/// Harper, F. M. & Konstan, J. A. (2015). "The MovieLens Datasets: History and
/// Context." *ACM Transactions on Interactive Intelligent Systems*, 5(4),
/// Article 19. <https://doi.org/10.1145/2827872>
///
/// # Thread Safety
///
/// This struct implements `Send` and `Sync` automatically, because all fields
/// implement them. This makes the struct safe to share across threads. The
/// internal [`Dataset`] makes lazy initialization thread-safe.
///
/// # Example
/// ```no_run
/// use dataset_ml::MovieLens100k;
///
/// // the loader creates the directory if it does not exist
/// let download_dir = "./movielens_100k";
///
/// let mut dataset = MovieLens100k::new(download_dir);
/// let table = dataset.data().unwrap();
///
/// assert_eq!(table.n_samples(), 100_000);
/// assert_eq!(table.n_columns(), 4);
///
/// // Reach each column by name.
/// let users = table.column("user_id").unwrap().as_integer().unwrap();
/// let items = table.column("item_id").unwrap().as_integer().unwrap();
/// let ratings = table.column(MovieLens100k::TARGET).unwrap().as_integer().unwrap();
/// let timestamps = table.column("timestamp").unwrap().as_integer().unwrap();
/// assert_eq!(timestamps.len(), 100_000);
///
/// // Build the dense rating matrix. The identifiers start at 1.
/// let mut matrix = vec![0i64; MovieLens100k::N_USERS * MovieLens100k::N_ITEMS];
/// for i in 0..ratings.len() {
///     let row = users[i] as usize - 1;
///     let col = items[i] as usize - 1;
///     matrix[row * MovieLens100k::N_ITEMS + col] = ratings[i];
/// }
///
/// // `get_data_mut()` edits the table in place. This needs no clone and no
/// // reload. The change stays cached.
/// if let Some(table) = dataset.get_data_mut() {
///     if let Some(column) = table.column_mut(MovieLens100k::TARGET) {
///         if let dataset_ml::ColumnData::Integer(values) = column.data_mut() {
///             values[0] = 5;
///         }
///     }
/// }
/// assert!(dataset.get_data().is_some());
///
/// // `take_data()` moves the owned table out with no clone. This leaves the
/// // instance reusable.
/// let owned = dataset.take_data().unwrap();
/// assert_eq!(owned.n_samples(), 100_000);
///
/// // `into_data()` also returns the owned table with no clone, but it consumes
/// // the instance.
/// let owned = dataset.into_data().unwrap();
/// assert_eq!(owned.n_samples(), 100_000);
/// ```
#[derive(Debug)]
pub struct MovieLens100k {
    dataset: Dataset<Table, DatasetError>,
}

impl MovieLens100k {
    /// The number of users the log covers.
    ///
    /// User identifiers run from `1` to this value.
    pub const N_USERS: usize = 943;

    /// The number of movies the log covers.
    ///
    /// Movie identifiers run from `1` to this value.
    pub const N_ITEMS: usize = 1682;

    /// The column the source designates as the label.
    pub const TARGET: &'static str = "rating";

    /// Create a new MovieLens100k instance without loading data.
    ///
    /// The dataset loads lazily, on your first call to a data accessor method.
    /// This is a lightweight operation that only stores the storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - The directory that stores the dataset.
    ///
    /// # Returns
    ///
    /// - `Self` - a `MovieLens100k` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        MovieLens100k {
            dataset: Dataset::new(storage_dir, Self::load_data),
        }
    }

    /// Get and parse the MovieLens 100K rating log.
    fn load_data(dir: &str) -> Result<Table, DatasetError> {
        let file_path = acquire_dataset(
            dir,
            MOVIELENS_FILENAME,
            MOVIELENS_DATASET_NAME,
            Some(MOVIELENS_SHA256),
            |temp_path| {
                download_to_with_retries(
                    MOVIELENS_DATA_URL,
                    temp_path,
                    Some(MOVIELENS_ZIP_FILENAME),
                    DOWNLOAD_RETRIES,
                )?;
                unzip(&temp_path.join(MOVIELENS_ZIP_FILENAME), temp_path)?;
                Ok(temp_path.join(MOVIELENS_SOURCE_PATH))
            },
        )?;

        // The source is tab-separated with no header row: user, item, rating,
        // timestamp.
        let file = File::open(&file_path)?;
        let mut rdr = ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(false)
            .from_reader(file);

        let mut users: Vec<i64> = Vec::with_capacity(N_SAMPLES);
        let mut items: Vec<i64> = Vec::with_capacity(N_SAMPLES);
        let mut ratings: Vec<i64> = Vec::with_capacity(N_SAMPLES);
        let mut timestamps: Vec<i64> = Vec::with_capacity(N_SAMPLES);

        for (idx, result) in rdr.records().enumerate() {
            let record =
                result.map_err(|e| DatasetError::csv_read_error(MOVIELENS_DATASET_NAME, e))?;
            let line_num = idx + 1;

            // Skip blank lines, such as a trailing newline at the end of the file.
            if record.iter().all(|f| f.is_empty()) {
                continue;
            }

            if record.len() != N_COLUMNS {
                return Err(DatasetError::invalid_column_count(
                    MOVIELENS_DATASET_NAME,
                    N_COLUMNS,
                    record.len(),
                    line_num,
                ));
            }

            let user: u16 = record[0].parse().map_err(|e| {
                DatasetError::parse_failed(MOVIELENS_DATASET_NAME, "user_id", line_num, e)
            })?;
            let item: u16 = record[1].parse().map_err(|e| {
                DatasetError::parse_failed(MOVIELENS_DATASET_NAME, "item_id", line_num, e)
            })?;
            let rating: u8 = record[2].parse().map_err(|e| {
                DatasetError::parse_failed(MOVIELENS_DATASET_NAME, "rating", line_num, e)
            })?;
            let timestamp: i64 = record[3].parse().map_err(|e| {
                DatasetError::parse_failed(MOVIELENS_DATASET_NAME, "timestamp", line_num, e)
            })?;

            if user < 1 || usize::from(user) > Self::N_USERS {
                return Err(DatasetError::invalid_value(
                    MOVIELENS_DATASET_NAME,
                    "user_id",
                    &user.to_string(),
                    line_num,
                ));
            }
            if item < 1 || usize::from(item) > Self::N_ITEMS {
                return Err(DatasetError::invalid_value(
                    MOVIELENS_DATASET_NAME,
                    "item_id",
                    &item.to_string(),
                    line_num,
                ));
            }
            if !(1..=5).contains(&rating) {
                return Err(DatasetError::invalid_value(
                    MOVIELENS_DATASET_NAME,
                    Self::TARGET,
                    &rating.to_string(),
                    line_num,
                ));
            }

            users.push(i64::from(user));
            items.push(i64::from(item));
            ratings.push(i64::from(rating));
            timestamps.push(timestamp);
        }

        Table::new(
            MOVIELENS_DATASET_NAME,
            vec![
                Column::new("user_id", ColumnData::Integer(Array1::from_vec(users))),
                Column::new("item_id", ColumnData::Integer(Array1::from_vec(items))),
                Column::new(Self::TARGET, ColumnData::Integer(Array1::from_vec(ratings))),
                Column::new(
                    "timestamp",
                    ColumnData::Integer(Array1::from_vec(timestamps)),
                ),
            ],
        )
    }

    /// Get a reference to the parsed table.
    ///
    /// This method triggers lazy loading on the first call. Later calls return
    /// the cached data.
    ///
    /// # Returns
    ///
    /// - `&Table` - reference to the cached table of 100,000 samples and 4
    ///   columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// value outside its documented range).
    pub fn data(&self) -> Result<&Table, DatasetError> {
        self.dataset.load()
    }

    /// Get a reference to the parsed table **without** triggering loading.
    ///
    /// Unlike [`MovieLens100k::data`], this method never runs the loader. If the
    /// data has not loaded yet, it returns `None` instead of downloading and
    /// parsing it.
    ///
    /// # Returns
    ///
    /// - `Some(&Table)` - reference to the cached table, if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data(&self) -> Option<&Table> {
        self.dataset.get()
    }

    /// Get a mutable reference to the parsed table for **in-place** editing.
    ///
    /// This needs no clone, and it does not remove the data from the cache. The
    /// changes stay in the cache. Later calls to [`MovieLens100k::data`] or
    /// [`MovieLens100k::get_data`] see them.
    ///
    /// Like [`MovieLens100k::get_data`], this does **not** trigger loading.
    ///
    /// # Returns
    ///
    /// - `Some(&mut Table)` - mutable reference to the cached table, if loaded.
    /// - `None` - if the dataset has not loaded yet.
    pub fn get_data_mut(&mut self) -> Option<&mut Table> {
        self.dataset.get_mut()
    }

    /// Consume the dataset and return the **owned** table.
    ///
    /// This **consumes** `self`. If you want owned data but need to keep using
    /// the instance, use [`MovieLens100k::take_data`] instead.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 100,000 samples and 4 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// value outside its documented range).
    pub fn into_data(self) -> Result<Table, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .into_inner()
            .expect("data is present after a successful load"))
    }

    /// Take the **owned** table out of the dataset. This leaves the instance
    /// reusable.
    ///
    /// This resets the instance to its unloaded state. The next accessor call
    /// loads the dataset again.
    ///
    /// # Returns
    ///
    /// - `Table` - the owned table of 100,000 samples and 4 columns.
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if loading fails (network, file I/O, parsing, or a
    /// value outside its documented range).
    pub fn take_data(&mut self) -> Result<Table, DatasetError> {
        self.dataset.load()?;
        Ok(self
            .dataset
            .take()
            .expect("data is present after a successful load"))
    }
}

impl_ml_dataset!(MovieLens100k, "movielens_100k");
