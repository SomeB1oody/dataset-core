use crate::datasets::wine_quality::parse_wine_data_to_array;
use crate::{Dataset, DatasetError, acquire_dataset, download_to};
use ndarray::{Array1, Array2};
use std::fs::File;

/// The URL for the White Wine Quality dataset.
const WHITE_WINE_DATA_URL: &str = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-white.csv";

/// The white wine file of the CSV files inside the zip archive.
const WHITE_WINE_QUALITY_FILENAME: &str = "winequality-white.csv";

/// The SHA256 hash of the white wine quality dataset.
const WHITE_WINE_QUALITY_SHA256: &str =
    "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836";

/// A struct representing the White Wine Quality dataset with lazy loading.
///
/// The dataset is not loaded until you call one of the data accessor methods.
/// Once loaded, the data is cached for subsequent accesses.
///
/// # About Dataset
///
/// The dataset contains physicochemical properties of Portuguese "Vinho Verde"
/// white wine samples and a quality score for each sample.
///
/// Features (11 total, all `f64`):
///   - fixed acidity
///   - volatile acidity
///   - citric acid
///   - residual sugar
///   - chlorides
///   - free sulfur dioxide
///   - total sulfur dioxide
///   - density
///   - pH
///   - sulphates
///   - alcohol
///
/// Targets:
/// - quality (score between 0 and 10, stored as `f64`)
///
/// See more information at <https://archive.ics.uci.edu/dataset/186/wine+quality>
///
/// # Citation
///
/// P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. "Wine Quality," UCI Machine Learning Repository, 2009. \[Online\]. Available: <https://doi.org/10.24432/C56S3T>.
///
/// # Thread Safety
///
/// This struct automatically implements `Send` and `Sync` (All fields implement them), making it safe to share across threads.
/// The internal [`Dataset`] ensures thread-safe lazy initialization.
///
/// # Example
/// ```rust
/// use dataset_core::datasets::WhiteWineQuality;
///
/// let download_dir = "./white_wine"; // the code will create the directory if it doesn't exist
///
/// let dataset = WhiteWineQuality::new(download_dir);
/// let features = dataset.features().unwrap();
/// let targets = dataset.targets().unwrap();
///
/// let (features, targets) = dataset.data().unwrap(); // this is also a way to get features and targets
/// // you can use `.to_owned()` to get owned copies of the data
/// let mut features_owned = features.to_owned();
/// let mut targets_owned = targets.to_owned();
///
/// // Example: Modify feature values
/// features_owned[[0, 0]] = 10.0;
/// targets_owned[0] = 7.0;
///
/// assert_eq!(features.shape(), &[4898, 11]);
/// assert_eq!(targets.len(), 4898);
///
/// // clean up: remove the downloaded files (dispensable)
/// std::fs::remove_dir_all(download_dir).unwrap();
/// ```
#[derive(Debug)]
pub struct WhiteWineQuality {
    dataset: Dataset<(Array2<f64>, Array1<f64>)>,
}

impl WhiteWineQuality {
    /// Create a new WhiteWineQuality instance without loading data.
    ///
    /// The dataset will be loaded lazily when you first call any data accessor method.
    /// This is a lightweight operation that only stores the storage directory.
    ///
    /// # Parameters
    ///
    /// - `storage_dir` - Directory where the dataset will be stored.
    ///
    /// # Returns
    ///
    /// - `Self` - `WhiteWineQuality` instance ready for lazy loading.
    pub fn new(storage_dir: &str) -> Self {
        WhiteWineQuality {
            dataset: Dataset::new(storage_dir),
        }
    }

    /// Acquire and parse the White Wine Quality dataset.
    fn load_data(dir: &str) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
        // Prepare the dataset file
        let file_path = acquire_dataset(
            dir,
            WHITE_WINE_QUALITY_FILENAME,
            "white_wine_quality",
            Some(WHITE_WINE_QUALITY_SHA256),
            |temp_path| {
                download_to(WHITE_WINE_DATA_URL, temp_path, None)?;
                Ok(temp_path.join(WHITE_WINE_QUALITY_FILENAME))
            },
        )?;

        // Parse the file
        let file = File::open(&file_path)?;
        parse_wine_data_to_array("white_wine_quality", file)
    }

    /// Get a reference to the feature matrix.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(4898, 11)` containing:
    ///     - fixed acidity
    ///     - volatile acidity
    ///     - citric acid
    ///     - residual sugar
    ///     - chlorides
    ///     - free sulfur dioxide
    ///     - total sulfur dioxide
    ///     - density
    ///     - pH
    ///     - sulphates
    ///     - alcohol
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (4898 samples, 11 features)
    pub fn features(&self) -> Result<&Array2<f64>, DatasetError> {
        Ok(&self.dataset.load(Self::load_data)?.0)
    }

    /// Get a reference to the target vector.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array1<f64>` - Reference to target vector with shape `(4898,)` containing quality scores (0-10)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (4898 samples)
    pub fn targets(&self) -> Result<&Array1<f64>, DatasetError> {
        Ok(&self.dataset.load(Self::load_data)?.1)
    }

    /// Get both features and targets as references.
    ///
    /// This method triggers lazy loading on first call. Subsequent calls return
    /// the cached data instantly.
    ///
    /// # Returns
    ///
    /// - `&Array2<f64>` - Reference to feature matrix with shape `(4898, 11)` containing:
    ///     - fixed acidity
    ///     - volatile acidity
    ///     - citric acid
    ///     - residual sugar
    ///     - chlorides
    ///     - free sulfur dioxide
    ///     - total sulfur dioxide
    ///     - density
    ///     - pH
    ///     - sulphates
    ///     - alcohol
    /// - `&Array1<f64>` - Reference to target vector with shape `(4898,)` containing quality scores (0-10)
    ///
    /// # Errors
    ///
    /// Returns `DatasetError` if:
    /// - Download fails due to network issues
    /// - File extraction or I/O operations fail
    /// - Data format is invalid (wrong number of columns, unparseable values)
    /// - Dataset size doesn't match expected dimensions (4898 samples, 11 features)
    pub fn data(&self) -> Result<(&Array2<f64>, &Array1<f64>), DatasetError> {
        let data = self.dataset.load(Self::load_data)?;
        Ok((&data.0, &data.1))
    }
}
