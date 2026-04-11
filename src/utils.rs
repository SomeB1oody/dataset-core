use sha2::{Digest, Sha256};
use std::fs::File;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use zip::ZipArchive;
use zip::result::ZipError;
use crate::DatasetError;

/// Download a remote file into the given directory.
///
/// It downloads the content at `url` (using [`ureq`] crate) into `storage_path` using the file name
/// extracted from the last segment of the URL, unless a custom filename is provided.
///
/// # Parameters
///
/// - `url` - The URL to download.
/// - `storage_path` - The directory to store the downloaded file in.
/// - `filename` - Optional custom filename (with extension). If `None`, the filename is extracted
///   from the last segment of the URL.
///
/// # Errors
///
/// - `DatasetError` - Returned when the download fails or URL is invalid.
///
/// # Example
/// ```rust
/// use dataset_core::download_to;
/// use std::path::Path;
///
/// let download_dir = "./download_example";
/// std::fs::create_dir_all(download_dir).unwrap();
///
/// // Download a file from the internet
/// let url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv";
///
/// // Use filename from URL
/// download_to(url, Path::new(download_dir), None).unwrap();
/// assert!(Path::new(download_dir).join("iris.csv").exists());
///
/// // Use custom filename
/// download_to(url, Path::new(download_dir), Some("custom.csv")).unwrap();
/// assert!(Path::new(download_dir).join("custom.csv").exists());
///
/// // Clean up (dispensable)
/// std::fs::remove_dir_all(download_dir).unwrap();
/// ```
pub fn download_to(url: &str, storage_path: &Path, filename: Option<&str>) -> Result<(), DatasetError> {
    // Get the filename: use provided name, or fall back to URL extraction
    let filename = filename
        .or_else(|| url.split('/').last())
        .ok_or_else(|| DatasetError::ValidationError("Invalid URL: cannot extract filename from URL".to_string()))?;

    let save_path = storage_path.join(filename);

    let mut response = ureq::get(url).call()?;
    let mut body = response.body_mut().as_reader();

    // create local file and write body to it
    let mut file = File::create(save_path)?;
    io::copy(&mut body, &mut file)?;

    Ok(())
}

/// Extract a zip archive into a target directory using [`ZipArchive`] in [`zip`] crate.
///
/// # Parameters
///
/// - `file_path` - Path to the `.zip` file to extract.
/// - `extract_dir` - Directory to extract the archive contents into.
///
/// # Errors
///
/// - `DatasetError` - Returned when opening the zip file fails or when extraction fails.
///
/// # Example
/// ```rust
/// use dataset_core::{download_to, unzip};
/// use std::path::Path;
///
/// let work_dir = "./unzip_example";
/// std::fs::create_dir_all(work_dir).unwrap();
///
/// // First download a file
/// let url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv";
/// download_to(url, Path::new(work_dir), None).unwrap();
///
/// // The file is already a CSV (no extraction needed in this example)
/// assert!(Path::new(work_dir).join("iris.csv").exists());
///
/// // Clean up (dispensable)
/// std::fs::remove_dir_all(work_dir).unwrap();
/// ```
pub fn unzip(file_path: &Path, extract_dir: &Path) -> Result<(), DatasetError> {
    let file = File::open(file_path).map_err(|e| DatasetError::from(ZipError::Io(e)))?;

    ZipArchive::new(file)?.extract(extract_dir)?;

    Ok(())
}

/// Create a temporary directory under the given parent directory.
///
/// This is a small wrapper around [`tempfile::Builder`] used by dataset loaders to
/// keep intermediate download/extraction artifacts isolated. The created directory
/// is removed automatically when the returned [`tempfile::TempDir`] is dropped.
///
/// # Parameters
///
/// - `tempdir_in` - The parent directory in which the temporary directory will be created.
///
/// # Errors
///
/// - `DatasetError` - Returned if the temporary directory cannot be created.
///
/// # Example
/// ```rust
/// use dataset_core::create_temp_dir;
/// use std::path::Path;
///
/// let parent_dir = "./temp_dir_example";
/// std::fs::create_dir_all(parent_dir).unwrap();
///
/// // Create a temporary directory
/// let temp_dir = create_temp_dir(Path::new(parent_dir)).unwrap();
/// let temp_path = temp_dir.path();
///
/// // Use the temporary directory for intermediate operations
/// let temp_file = temp_path.join("temp_file.txt");
/// std::fs::write(&temp_file, "temporary content").unwrap();
/// assert!(temp_file.exists());
///
/// // The temporary directory is automatically removed when `temp_dir` is dropped
/// drop(temp_dir);
///
/// // Clean up parent directory
/// std::fs::remove_dir_all(parent_dir).unwrap();
/// ```
pub fn create_temp_dir(tempdir_in: &Path) -> Result<tempfile::TempDir, DatasetError> {
    let temp_dir = tempfile::Builder::new()
        .tempdir_in(tempdir_in)?;

    Ok(temp_dir)
}

/// Verify that a file's SHA256 hash matches an expected value.
///
/// This function computes the SHA256 hash of the file at the given path and compares
/// it with the expected hexadecimal hash string (case-insensitive). It is used by
/// dataset loaders to validate downloaded files before parsing.
///
/// # Parameters
///
/// - `path` - Path to the file to verify.
/// - `expected_hex` - Expected SHA256 hash as a hexadecimal string.
///
/// # Returns
///
/// - `bool` - true if the computed hash matches the expected hash, false if the hashes don't match
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when file I/O operations fail (opening file, reading data).
///
/// # Example
/// ```rust
/// use dataset_core::file_sha256_matches;
/// use std::path::Path;
/// use std::io::Write;
///
/// let test_dir = "./sha256_example";
/// std::fs::create_dir_all(test_dir).unwrap();
///
/// // Create a test file with known content
/// let file_path = Path::new(test_dir).join("test.txt");
/// let mut file = std::fs::File::create(&file_path).unwrap();
/// file.write_all(b"hello world").unwrap();
/// drop(file);
///
/// // SHA256 of "hello world" is:
/// // b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
/// let expected_hash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
///
/// // Verify the hash matches
/// assert!(file_sha256_matches(&file_path, expected_hash).unwrap());
///
/// // Case-insensitive comparison also works
/// let upper_hash = "B94D27B9934D3E08A52E52D7DA7DABFAC484EFE37A5380EE9088F7ACE2EFCDE9";
/// assert!(file_sha256_matches(&file_path, upper_hash).unwrap());
///
/// // Wrong hash returns false
/// assert!(!file_sha256_matches(&file_path, "0000000000000000000000000000000000000000000000000000000000000000").unwrap());
///
/// // Clean up (dispensable)
/// std::fs::remove_dir_all(test_dir).unwrap();
/// ```
pub fn file_sha256_matches(path: &Path, expected_hex: &str) -> Result<bool, DatasetError> {
    let mut file = File::open(path)?;

    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];

    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }

    let digest = hasher.finalize();
    let actual_hex = digest.iter().map(|b| format!("{:02x}", b)).collect::<String>();
    Ok(actual_hex.eq_ignore_ascii_case(expected_hex))
}

/// Prepare a dataset download directory and determine if download/overwrite is needed.
///
/// This helper ensures the target directory exists and checks whether the destination
/// file already matches the expected SHA256 hash. If `expected_sha256` is `None`,
/// the file is accepted if it exists without validation.
///
/// # Parameters
///
/// - `path` - Directory path where the dataset will be stored.
/// - `dst` - Destination file path for the dataset.
/// - `expected_sha256` - Optional expected SHA256 hash for the dataset file. If `None`,
///   any existing file at `dst` is accepted without validation.
///
/// # Returns
///
/// - `(need_download, need_overwrite)` - Flags indicating whether to download and
///   whether an existing file should be overwritten.
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when creating the directory fails or when
///   file I/O operations fail during hash verification.
///
/// # Example
/// ```rust, ignore
/// use dataset_core::prepare_download_dir;
/// use std::path::Path;
/// use std::io::Write;
///
/// let test_dir = "./prepare_download_example";
/// let dir_path = Path::new(test_dir);
/// let file_path = dir_path.join("data.txt");
///
/// // Case 1: Directory doesn't exist yet
/// let (need_download, need_overwrite) = prepare_download_dir(
///     dir_path,
///     &file_path,
///     None,
/// ).unwrap();
/// assert!(need_download);    // File doesn't exist, need to download
/// assert!(!need_overwrite);  // Nothing to overwrite
///
/// // Case 2: File exists with correct hash
/// let mut file = std::fs::File::create(&file_path).unwrap();
/// file.write_all(b"hello world").unwrap();
/// drop(file);
///
/// let correct_hash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
/// let (need_download, need_overwrite) = prepare_download_dir(
///     dir_path,
///     &file_path,
///     Some(correct_hash),
/// ).unwrap();
/// assert!(!need_download);   // File exists with correct hash
/// assert!(!need_overwrite);
///
/// // Case 3: File exists but hash doesn't match
/// let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
/// let (need_download, need_overwrite) = prepare_download_dir(
///     dir_path,
///     &file_path,
///     Some(wrong_hash),
/// ).unwrap();
/// assert!(need_download);    // Hash mismatch, need to download
/// assert!(need_overwrite);   // Existing file needs to be replaced
///
/// // Clean up (dispensable)
/// std::fs::remove_dir_all(test_dir).unwrap();
/// ```
fn prepare_download_dir(
    path: &Path,
    dst: &Path,
    expected_sha256: Option<&str>,
) -> Result<(bool, bool), DatasetError> {
    let mut need_download = true;
    let mut need_overwrite = false;

    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }

    if dst.exists() {
        if let Some(hash) = expected_sha256 {
            // SHA256 validation enabled
            if file_sha256_matches(dst, hash)? {
                need_download = false;
            } else {
                need_overwrite = true;
            }
        } else {
            // No SHA256 validation: accept existing file
            need_download = false;
        }
    }

    Ok((need_download, need_overwrite))
}

/// Generic dataset download framework that handles common download workflow.
///
/// This function manages the complete dataset acquisition workflow: checking if download
/// is needed, creating a temporary directory, delegating file preparation to a user-provided
/// closure, optionally validating the file with SHA256, and moving it to the final destination.
///
/// # Parameters
///
/// - `dir` - Target storage directory path
/// - `filename` - Final dataset filename (will be stored as `dir/filename`).
///    Please give the filename with the extension (e.g., "iris.csv").
/// - `dataset_name` - Dataset name for error messages (e.g., "iris").
/// - `expected_sha256` - Optional expected SHA256 hash of the dataset file. If `None`,
///   any existing file at the destination is accepted without validation, and newly
///   prepared files skip SHA256 verification.
/// - `prepare_file` - Closure that prepares the dataset file in the temporary directory
///   - Input: `temp_dir: &Path` - Path to the temporary directory.
///     It is recommended to execute file operations within this directory, as it will be cleaned up
///     automatically when the closure returns. But it is not required.
///     (Please note that the file will be moved to the final destination, not copied)
///   - Output: `Result<PathBuf, DatasetError>` - Path to the prepared dataset file (will be moved to `dir/filename`).
///   - Responsibility: This closure can perform any operations needed to get the
///     dataset file, such as downloading (you can use [`download_to`] provided in this crate), extracting archives
///     (you can use [`unzip`] provided in this crate), or locating files within extracted folders. The returned
///     `PathBuf` must point to the final dataset file ready for validation.
///
/// # Returns
///
/// - `PathBuf` - Path to the final dataset file (`dir/filename`)
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when directory creation, file operations, or
///   hash verification fails
/// - `DatasetError::Sha256ValidationFailed` - Returned when `expected_sha256` is provided
///   and the prepared file's SHA256 hash does not match it
/// - Any error returned by the `prepare_file` closure
///
/// # Example
/// ```rust
/// // Implement the downloading process for iris dataset
///
/// /// The URL for the Iris dataset.
/// ///
/// /// # Citation
/// ///
/// /// R. A. Fisher. "Iris," UCI Machine Learning Repository, \[Online\].
/// /// Available: <https://doi.org/10.24432/C56C76>
/// const IRIS_DATA_URL: &str = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv";
///
/// /// The name of the Iris dataset file.
/// const IRIS_FILENAME: &str = "iris.csv";
///
/// /// The SHA256 hash of the Iris dataset file.
/// const IRIS_SHA256: &str = "c52742e50315a99f956a383faedf7575552675f6409ef0f9a47076dd08479930";
///
/// /// The name of the dataset
/// const IRIS_DATASET_NAME: &str = "iris";
///
/// use dataset_core::download_dataset_with;
/// use dataset_core::download_to;
/// use dataset_core::unzip;
///
/// fn main() {
///     let dir = "./somewhere";
///
///     let file_path = download_dataset_with(
///             // Target storage directory path
///             dir,
///             // Final dataset filename (will be stored as `dir/filename`)
///             IRIS_FILENAME,
///             // Dataset name for error messages
///             IRIS_DATASET_NAME,
///             // Expected SHA256 hash of the dataset file
///             Some(IRIS_SHA256),
///             // Closure that prepares the dataset file in the temporary directory
///             |temp_path| {
///                 // Download the dataset
///                 download_to(IRIS_DATA_URL, temp_path, None)?;
///                 Ok(temp_path.join(IRIS_FILENAME))
///             },
///         ).unwrap();
///
///     // `file_path` is now the path to the downloaded and extracted Iris dataset file
///     // it can be used to give the path of the dataset or parse data
///
///     // cleanup (dispensable)
///     std::fs::remove_dir_all(dir).unwrap();
/// }
/// ```
pub fn download_dataset_with<F>(
    dir: &str,
    filename: &str,
    dataset_name: &str,
    expected_sha256: Option<&str>,
    prepare_file: F,
) -> Result<PathBuf, DatasetError>
where
    F: FnOnce(&Path) -> Result<PathBuf, DatasetError>,
{
    let dir_path = Path::new(dir);
    let dst = dir_path.join(filename);
    let (need_download, need_overwrite) = prepare_download_dir(dir_path, &dst, expected_sha256)?;

    if need_download {
        let temp_dir = create_temp_dir(dir_path)?;
        let temp_path = temp_dir.path();

        // Call user closure: prepare the dataset file in temporary directory
        let src = prepare_file(temp_path)?;

        // Validate SHA256 hash if provided
        if let Some(hash) = expected_sha256 {
            if !file_sha256_matches(&src, hash)? {
                drop(temp_dir); // Clean up temporary directory
                return Err(DatasetError::sha256_validation_failed(
                    dataset_name,
                    filename,
                ));
            }
        }

        // Move file to final destination
        if need_overwrite {
            std::fs::remove_file(&dst)?;
        }
        std::fs::rename(&src, &dst)?;
    }

    Ok(dst)
}
