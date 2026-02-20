use ndarray::{Array1, Array2};
use std::sync::OnceLock;
use crate::{create_temp_dir, download_to, unzip, DatasetError};
use std::path::Path;
use std::fs::{remove_file, rename, File};
use std::io::Read;

/// A static variable to store the Iris dataset.
///
/// This variable is of type `OnceLock`, which ensures thread-safe, one-time initialization
/// of its contents. It contains a tuple of:
///
/// - `Array2<f64>`: A 2-dimensional array representing the numerical features of the dataset (e.g., sepal length, sepal width, petal length, petal width).
/// - `Array1<&'static str>`: A 1-dimensional array containing the corresponding labels (e.g., species names such as "Iris-setosa", "Iris-versicolor", "Iris-virginica").
///
/// The `OnceLock` ensures that the dataset is initialized only once and is then immutable
/// for the lifetime of the program.
static IRIS_DATA: OnceLock<(Array2<f64>, Array1<&'static str>)> = OnceLock::new();

/// A static string slice containing the URL for the Iris dataset.
///
/// # Citation
///
/// M. Kahn. "Diabetes," UCI Machine Learning Repository,  \[Online\]. Available: <https://doi.org/10.24432/C5T59G>
///
/// # About Dataset
///
/// It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.
///
/// Features:
/// - sepal length in cm
/// - sepal width in cm
/// - petal length in cm
/// - petal width in cm
///
/// labels:
/// - name of the species (in `&str`):
///     - "Iris-setosa"
///     - "Iris-versicolor"
///     - "Iris-virginica"
///
/// See more information at <https://archive.ics.uci.edu/dataset/53/iris>
pub static IRIS_DATA_URL: &str = "https://archive.ics.uci.edu/static/public/53/iris.zip";

/// Internal function to download and process the Iris dataset.
///
/// Downloads the dataset from UCI Machine Learning Repository, extracts it,
/// parses the CSV-like format, and converts it into structured ndarray arrays.
/// The function handles downloading, unzipping, file cleanup, and data parsing.
///
/// # Arguments
///
/// - `path` - Storage directory path where the dataset will be downloaded and extracted
///
/// # Returns
///
/// - `Array2<f64>` - Feature matrix with shape (150, 4) where each row represents a flower sample
/// - `Array1<&'static str>` - Target labels array with shape (150,) containing species names (setosa, versicolor, virginica)
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Download fails due to network issues
/// - File extraction or I/O operations fail
/// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
/// - Dataset size doesn't match expected dimensions (150 samples, 4 features)
fn load_iris_internal(path: &str) -> Result<(Array2<f64>, Array1<&'static str>), DatasetError> {
    // the path the user wants dataset to be stored in
    let path = Path::new(path);
    // temporary directory to store the downloaded zip file
    let temp_dir = create_temp_dir(path, ".tmp-iris-")?;
    let path_temp = temp_dir.path();
    // download and extract iris dataset
    download_to(IRIS_DATA_URL, path_temp)?;
    unzip(&path_temp.join("iris.zip"), path_temp)?;

    // move iris.data out of the temporary directory
    let src = path_temp.join("iris.data");
    let dst = path.join("iris.csv");
    // cover the existing file (if any) with the new one
    if dst.exists() {
        remove_file(&dst).map_err(|e| DatasetError::StdIoError(e))?;
    }
    rename(src, &dst).map_err(|e| DatasetError::StdIoError(e))?;

    let mut file = File::open(dst).map_err(|e| DatasetError::StdIoError(e))?;
    let mut data = String::new();
    file.read_to_string(&mut data).map_err(|e| DatasetError::StdIoError(e))?;
    let lines: Vec<&str> = data.trim().lines().collect();

    let mut features = Vec::with_capacity(150 * 4);
    let mut labels = Vec::with_capacity(150);

    for line in lines {
        if line.is_empty() { continue; }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() != 5 {
            return Err(DatasetError::DataFormatError(
                format!("Expected 5 columns, got {} at line {}",
                        cols.len(),
                        line)
            ));
        }

        for i in 0..4 {
            features.push(cols[i].parse::<f64>().map_err(
                |e| DatasetError::DataFormatError(
                        format!("Failed to parse feature {} at line {}: {}", i, line, e)
                    ))?
            );
        }

        labels.push(match cols[4] {
            "Iris-setosa" => "setosa",
            "Iris-versicolor" => "versicolor",
            "Iris-virginica" => "virginica",
            _ => return Err(DatasetError::DataFormatError(
                format!("Invalid label at line {}", line)
            ))
        });
    }
    if features.len() != 150 * 4 {
        return Err(DatasetError::DataFormatError(
            format!("Expected 150 * 4 elements in features, got {} ", features.len())
        ));
    }
    if labels.len() != 150 {
        return Err(DatasetError::DataFormatError(
            format!("Expected 150 elements in labels, got {} ", labels.len())
        ));
    }

    let features_array = Array2::from_shape_vec((150, 4), features)
        .map_err(|e| DatasetError::DataFormatError(
            format!("Failed to create features array: {}", e)
        ))?;
    let labels_array = Array1::from_vec(labels);

    Ok((features_array, labels_array))
}

/// Loads the Iris dataset with automatic caching
///
/// This function returns references to the cached data, which incurs no additional memory allocation.
/// If you need owned data, prefer [`load_iris_owned()`] which returns owned copies.
///
/// # About Dataset
///
/// It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.
///
/// Features:
/// - sepal length in cm
/// - sepal width in cm
/// - petal length in cm
/// - petal width in cm
///
/// labels:
/// - name of the species (in `&str`):
///     - "Iris-setosa"
///     - "Iris-versicolor"
///     - "Iris-virginica"
///
/// See more information at <https://archive.ics.uci.edu/dataset/53/iris>
///
/// # Arguments
///
/// - `storage_path` - Directory path where the dataset will be stored
///
/// # Returns
///
/// - `&Array2<f64>` - Static reference to feature matrix of shape (150, 4) containing measurements:
///     - sepal length in cm
///     - sepal width in cm
///     - petal length in cm
///     - petal width in cm
/// - `&Array1<&'static str>` - Static reference to labels array of length 150 containing species names
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Download fails due to network issues
/// - File extraction or I/O operations fail
/// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
/// - Dataset size doesn't match expected dimensions (150 samples, 4 features)
///
/// # Example
/// ```rust, no_run
/// use rustyml_dataset::iris::load_iris;
///
/// let download_dir = "./downloads"; // you need to create a directory manually beforehand
///
/// let (features, labels) = load_iris(download_dir).unwrap();
/// assert_eq!(features.shape(), &[150, 4]);
/// assert_eq!(labels.len(), 150);
///
/// // clean up: remove the downloaded files if they exist
/// if let Ok(entries) = std::fs::read_dir(download_dir) {
///     for entry in entries.flatten() {
///         let _ = std::fs::remove_file(entry.path());
///     }
/// }
/// ```
pub fn load_iris(storage_path: &str) -> Result<(&Array2<f64>, &Array1<&'static str>), DatasetError> {
    // if already initialized
    if let Some(cached) = IRIS_DATA.get() {
        return Ok((&cached.0, &cached.1));
    }

    // if not, initialize then store
    let loaded = load_iris_internal(storage_path)?;

    // Try to set the value. If another thread already set it, that's fine - just use the existing value
    let _ = IRIS_DATA.set(loaded);

    let cached = IRIS_DATA
        .get()
        .expect("IRIS_DATA should be initialized after set");
    Ok((&cached.0, &cached.1))
}

/// Loads the Iris dataset and returns owned copies
///
/// Use this function when you need owned data that can be modified.
/// For read-only access, prefer [`load_iris()`] which returns references.
///
/// # About Dataset
///
/// It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.
///
/// Features:
/// - sepal length in cm
/// - sepal width in cm
/// - petal length in cm
/// - petal width in cm
///
/// labels:
/// - name of the species (in `&str`):
///     - "Iris-setosa"
///     - "Iris-versicolor"
///     - "Iris-virginica"
///
/// See more information at <https://archive.ics.uci.edu/dataset/53/iris>
///
/// # Arguments
///
/// - `storage_path` - Directory path where the dataset will be stored
///
/// # Returns
///
/// - `Array2<f64>` - Owned feature matrix with shape (150, 4) containing measurements (sepal length, sepal width, petal length, petal width)
/// - `Array1<&'static str>` - Owned labels array with shape (150,) containing species names (Iris-setosa, Iris-versicolor, Iris-virginica)
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Download fails due to network issues
/// - File extraction or I/O operations fail
/// - Data format is invalid (wrong number of columns, unparseable values, or invalid labels)
/// - Dataset size doesn't match expected dimensions (150 samples, 4 features)
///
/// # Performance
///
/// This function creates owned copies by cloning the cached data, which incurs additional memory allocation.
/// If you only need read-only access to the data, use [`load_iris()`] instead for better performance.
///
/// # Examples
/// ```rust, no_run
/// use rustyml_dataset::iris::load_iris_owned;
///
/// let download_dir = "./downloads"; // you need to create a directory manually beforehand
///
/// let (mut features, labels) = load_iris_owned(download_dir).unwrap();
///
/// // You can now modify the data since these are owned copies
/// assert_eq!(features.shape(), &[150, 4]);
/// assert_eq!(labels.len(), 150);
///
/// // Example: Modify feature values (not possible with references)
/// features[[0, 0]] = 5.5;
///
/// // clean up: remove the downloaded files if they exist
/// if let Ok(entries) = std::fs::read_dir(download_dir) {
///     for entry in entries.flatten() {
///         let _ = std::fs::remove_file(entry.path());
///     }
/// }
pub fn load_iris_owned(storage_path: &str) -> Result<(Array2<f64>, Array1<&'static str>), DatasetError> {
    let (features, labels) = load_iris(storage_path)?;
    Ok((features.clone(), labels.clone()))
}
