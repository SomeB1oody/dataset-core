use crate::DatasetError;
use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use std::fmt::Write;
use std::fs::File;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use zip::ZipArchive;
use zip::result::ZipError;

/// Download a remote file into the given directory.
///
/// It downloads the content at `url` (using [`ureq`] crate) into `storage_path` using the file name
/// extracted from the last segment of the URL, unless a custom filename is provided.
///
/// When the filename is derived from the URL, any trailing `?query` or `#fragment` is stripped
/// (e.g. `.../iris.csv?raw=1` yields `iris.csv`). A URL that ends in `/` has no final segment, so
/// a custom `filename` must be supplied in that case.
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
/// - `DatasetError` - Returned when the download fails or the filename cannot be derived from the URL.
///
/// # Example
/// ```no_run
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
/// ```
pub fn download_to(
    url: &str,
    storage_path: &Path,
    filename: Option<&str>,
) -> Result<(), DatasetError> {
    // Get the filename: use provided name, or fall back to URL extraction
    let filename = match filename {
        Some(name) => name,
        None => filename_from_url(url).ok_or_else(|| {
            DatasetError::ValidationError(
                "Invalid URL: cannot extract filename from URL".to_string(),
            )
        })?,
    };

    let save_path = storage_path.join(filename);

    let mut response = ureq::get(url).call()?;
    let mut body = response.body_mut().as_reader();

    // create local file and write body to it
    let mut file = File::create(save_path)?;
    io::copy(&mut body, &mut file)?;

    Ok(())
}

/// Derive a filename from the last path segment of a URL.
///
/// Strips any `?query` / `#fragment` suffix and returns `None` when the resulting
/// segment is empty (e.g. the URL ends in `/`).
fn filename_from_url(url: &str) -> Option<&str> {
    // `rsplit('/').next()` yields the whole string when there is no '/'.
    let last_segment = url.rsplit('/').next()?;
    let name = last_segment
        .split(['?', '#'])
        .next()
        .unwrap_or(last_segment);
    (!name.is_empty()).then_some(name)
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
/// ```no_run
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
/// ```
pub fn unzip(file_path: &Path, extract_dir: &Path) -> Result<(), DatasetError> {
    let file = File::open(file_path).map_err(|e| DatasetError::from(ZipError::Io(e)))?;

    ZipArchive::new(file)?.extract(extract_dir)?;

    Ok(())
}

/// Decompress a gzip (`.gz`) file into a single output file.
///
/// Unlike [`unzip`], which extracts a multi-entry archive into a directory, a gzip
/// stream wraps exactly **one** file, so this writes the decompressed bytes to a
/// single `output_path`. It streams through [`flate2::read::GzDecoder`], so the
/// whole file is never held in memory at once — suitable for large datasets such as
/// the gzip-compressed `covtype.data.gz`.
///
/// The output file is created (or truncated if it already exists). Any leading
/// directories in `output_path` must already exist.
///
/// # Parameters
///
/// - `file_path` - Path to the `.gz` file to decompress.
/// - `output_path` - Path of the decompressed file to write (including filename).
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when opening the source fails, the gzip
///   stream is malformed, or writing the output fails.
///
/// # Example
/// ```no_run
/// use dataset_core::{download_to, gunzip};
/// use std::path::Path;
///
/// let work_dir = Path::new("./gunzip_example");
/// std::fs::create_dir_all(work_dir).unwrap();
///
/// // Download a gzip-compressed dataset, then decompress it in place.
/// download_to("https://example.com/data.csv.gz", work_dir, Some("data.csv.gz")).unwrap();
/// gunzip(&work_dir.join("data.csv.gz"), &work_dir.join("data.csv")).unwrap();
/// assert!(work_dir.join("data.csv").exists());
/// ```
pub fn gunzip(file_path: &Path, output_path: &Path) -> Result<(), DatasetError> {
    let input = File::open(file_path)?;
    let mut decoder = GzDecoder::new(input);
    let mut output = File::create(output_path)?;
    io::copy(&mut decoder, &mut output)?;

    Ok(())
}

/// Create a temporary directory under the given parent directory.
///
/// A small wrapper around [`tempfile::Builder`] used internally by [`acquire_dataset`] to keep
/// intermediate download/extraction artifacts isolated. The created directory is removed
/// automatically when the returned [`tempfile::TempDir`] is dropped.
fn create_temp_dir(tempdir_in: &Path) -> Result<tempfile::TempDir, DatasetError> {
    let temp_dir = tempfile::Builder::new().tempdir_in(tempdir_in)?;

    Ok(temp_dir)
}

/// Verify that a file's SHA256 hash matches an expected value (case-insensitive).
///
/// Used internally by [`acquire_dataset`] to validate cached and freshly prepared files.
/// Returns `true` when the computed hash matches `expected_hex`.
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when file I/O operations fail (opening file, reading data).
fn file_sha256_matches(path: &Path, expected_hex: &str) -> Result<bool, DatasetError> {
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
    let mut actual_hex = String::with_capacity(digest.len() * 2);
    for b in digest {
        // Writing formatted bytes into a `String` is infallible.
        let _ = write!(actual_hex, "{:02x}", b);
    }
    Ok(actual_hex.eq_ignore_ascii_case(expected_hex))
}

/// State of the destination file relative to the dataset we want to cache.
enum CacheState {
    /// Destination exists and (if a hash was given) matches — reuse it as-is.
    Fresh,
    /// Destination exists but its hash does not match — it must be replaced.
    Stale,
    /// Destination does not exist — a new file must be prepared.
    Missing,
}

/// Ensure `dir` exists, then classify the destination file `dst`.
///
/// When `expected_sha256` is `None`, any existing file counts as [`CacheState::Fresh`].
fn inspect_cache(
    dir: &Path,
    dst: &Path,
    expected_sha256: Option<&str>,
) -> Result<CacheState, DatasetError> {
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
    }

    if !dst.exists() {
        return Ok(CacheState::Missing);
    }

    match expected_sha256 {
        Some(hash) if !file_sha256_matches(dst, hash)? => Ok(CacheState::Stale),
        _ => Ok(CacheState::Fresh),
    }
}

/// Acquire a dataset file using a caller-provided preparation closure.
///
/// This is the single entry point for the dataset acquisition workflow and the
/// recommended way to populate a storage directory. It checks whether the destination
/// file can be reused, creates a temporary directory when a new file is needed,
/// delegates file preparation to a user-provided closure, optionally validates the
/// prepared file with SHA256, and atomically moves it to the final destination.
///
/// The function itself does not perform network I/O. The `prepare_file` closure
/// is responsible for preparing the dataset file, which may include downloading,
/// extracting archives, or locating files within an extracted directory.
///
/// # Parameters
///
/// - `dir` - Target storage directory path.
/// - `filename` - Final dataset filename (stored as `dir/filename`).
///   Please give the filename with the extension (e.g., `"iris.csv"`).
/// - `dataset_name` - Dataset name for error messages (e.g., `"iris"`).
/// - `expected_sha256` - Optional expected SHA256 hash of the dataset file. If `None`,
///   any existing file at the destination is accepted without validation, and newly
///   prepared files skip SHA256 verification.
/// - `prepare_file` - Closure that prepares the dataset file in the temporary directory.
///   - Input: `temp_dir: &Path` - Path to the temporary directory.
///     It is recommended to execute file operations within this directory, as it will be
///     cleaned up automatically when the closure returns. But it is not required.
///     (Please note that the file will be moved to the final destination, not copied.)
///   - Output: `Result<PathBuf, DatasetError>` - Path to the prepared dataset file
///     (which will be moved to `dir/filename`).
///   - Responsibility: This closure can perform any operations needed to prepare the
///     dataset file, such as downloading (you can use [`download_to`] provided in this crate),
///     extracting archives (you can use [`unzip`] provided in this crate), or locating files
///     within extracted folders. The returned `PathBuf` must point to the final dataset file
///     ready for validation.
///
/// # Returns
///
/// - `PathBuf` - Path to the final dataset file (`dir/filename`).
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when directory creation, file operations, or
///   hash verification fails.
/// - `DatasetError::Sha256ValidationFailed` - Returned when `expected_sha256` is provided
///   and the prepared file's SHA256 hash does not match it.
/// - Any error returned by the `prepare_file` closure.
///
/// # Example
/// ```no_run
/// // Implement the file preparation process for the Iris dataset.
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
/// /// The name of the dataset.
/// const IRIS_DATASET_NAME: &str = "iris";
///
/// use dataset_core::acquire_dataset;
/// use dataset_core::download_to;
///
/// fn main() {
///     let dir = "./somewhere";
///
///     let file_path = acquire_dataset(
///         // Target storage directory path
///         dir,
///         // Final dataset filename (will be stored as `dir/filename`)
///         IRIS_FILENAME,
///         // Dataset name for error messages
///         IRIS_DATASET_NAME,
///         // Expected SHA256 hash of the dataset file
///         Some(IRIS_SHA256),
///         // Closure that prepares the dataset file in the temporary directory
///         |temp_path| {
///             // Download the dataset into the temporary directory
///             download_to(IRIS_DATA_URL, temp_path, None)?;
///             Ok(temp_path.join(IRIS_FILENAME))
///         },
///     ).unwrap();
///
///     // `file_path` is now the path to the acquired Iris dataset file.
///     // It can be used to locate or parse the dataset.
/// }
/// ```
pub fn acquire_dataset<F>(
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

    // Reuse a valid cached file without invoking the preparation closure.
    let state = inspect_cache(dir_path, &dst, expected_sha256)?;
    if matches!(state, CacheState::Fresh) {
        return Ok(dst);
    }

    // Prepare the new file inside a temp dir that is cleaned up on drop (including
    // on the early `return` below).
    let temp_dir = create_temp_dir(dir_path)?;
    let src = prepare_file(temp_dir.path())?;

    // Validate the freshly prepared file before it lands at the final path.
    if let Some(hash) = expected_sha256
        && !file_sha256_matches(&src, hash)?
    {
        return Err(DatasetError::sha256_validation_failed(
            dataset_name,
            filename,
        ));
    }

    // A stale file must be removed first: `fs::rename` does not overwrite on all platforms.
    if matches!(state, CacheState::Stale) {
        std::fs::remove_file(&dst)?;
    }
    std::fs::rename(&src, &dst)?;

    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File, create_dir_all, remove_dir_all};
    use std::io::Write;

    /// SHA256 of "hello world"
    const HELLO_WORLD_SHA256: &str =
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";

    /// SHA256 of an empty file
    const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    /// All-zero hash (always wrong)
    const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

    #[test]
    fn create_temp_dir_returns_existing_path() {
        let parent = "./test_create_temp_dir_returns_existing_path";
        create_dir_all(parent).unwrap();

        let temp_dir = create_temp_dir(Path::new(parent)).unwrap();
        assert!(temp_dir.path().exists());

        remove_dir_all(parent).unwrap();
    }

    #[test]
    fn create_temp_dir_cleanup_on_drop() {
        let parent = "./test_create_temp_dir_cleanup_on_drop";
        create_dir_all(parent).unwrap();

        let temp_dir = create_temp_dir(Path::new(parent)).unwrap();
        let temp_path = temp_dir.path().to_path_buf();

        assert!(temp_path.exists());
        drop(temp_dir);
        assert!(!temp_path.exists());

        remove_dir_all(parent).unwrap();
    }

    #[test]
    fn create_temp_dir_nonexistent_parent_errors() {
        let result = create_temp_dir(Path::new("./nonexistent_parent_xyz_abc_123"));
        assert!(result.is_err());
    }

    /// Compress `content` into a gzip file at `path` (test helper).
    fn write_gz(path: &Path, content: &[u8]) {
        use flate2::Compression;
        use flate2::write::GzEncoder;

        let file = File::create(path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(content).unwrap();
        encoder.finish().unwrap();
    }

    #[test]
    fn gunzip_round_trips_content() {
        let dir = "./test_gunzip_round_trips_content";
        create_dir_all(dir).unwrap();
        let dir_path = Path::new(dir);

        let payload = b"col_a,col_b\n1,2\n3,4\n";
        let gz_path = dir_path.join("data.csv.gz");
        let out_path = dir_path.join("data.csv");
        write_gz(&gz_path, payload);

        gunzip(&gz_path, &out_path).unwrap();

        assert_eq!(fs::read(&out_path).unwrap(), payload);

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn gunzip_nonexistent_source_errors() {
        let result = gunzip(
            Path::new("./no_such_file_for_gunzip_test.gz"),
            Path::new("./no_such_file_for_gunzip_test.out"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn file_sha256_matches_correct_hash() {
        let dir = "./test_file_sha256_matches_correct_hash";
        create_dir_all(dir).unwrap();
        let path = Path::new(dir).join("f.txt");
        File::create(&path)
            .unwrap()
            .write_all(b"hello world")
            .unwrap();

        assert!(file_sha256_matches(&path, HELLO_WORLD_SHA256).unwrap());

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn file_sha256_matches_uppercase_hash() {
        let dir = "./test_file_sha256_matches_uppercase_hash";
        create_dir_all(dir).unwrap();
        let path = Path::new(dir).join("f.txt");
        File::create(&path)
            .unwrap()
            .write_all(b"hello world")
            .unwrap();

        assert!(file_sha256_matches(&path, &HELLO_WORLD_SHA256.to_uppercase()).unwrap());

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn file_sha256_matches_wrong_hash_returns_false() {
        let dir = "./test_file_sha256_matches_wrong_hash_returns_false";
        create_dir_all(dir).unwrap();
        let path = Path::new(dir).join("f.txt");
        File::create(&path)
            .unwrap()
            .write_all(b"hello world")
            .unwrap();

        assert!(!file_sha256_matches(&path, ZERO_SHA256).unwrap());

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn file_sha256_matches_empty_file() {
        let dir = "./test_file_sha256_matches_empty_file";
        create_dir_all(dir).unwrap();
        let path = Path::new(dir).join("empty.txt");
        File::create(&path).unwrap();

        assert!(file_sha256_matches(&path, EMPTY_SHA256).unwrap());

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn file_sha256_matches_nonexistent_file_errors() {
        let result = file_sha256_matches(Path::new("./no_such_file_sha256_test.txt"), ZERO_SHA256);
        assert!(result.is_err());
    }

    #[test]
    fn filename_from_url_plain_segment() {
        assert_eq!(
            filename_from_url("https://x.test/a/iris.csv"),
            Some("iris.csv")
        );
    }

    #[test]
    fn filename_from_url_strips_query_and_fragment() {
        assert_eq!(
            filename_from_url("https://x.test/a/iris.csv?raw=1"),
            Some("iris.csv")
        );
        assert_eq!(
            filename_from_url("https://x.test/a/iris.csv#section"),
            Some("iris.csv")
        );
    }

    #[test]
    fn filename_from_url_trailing_slash_is_none() {
        assert_eq!(filename_from_url("https://x.test/a/"), None);
    }

    #[test]
    fn filename_from_url_no_slash() {
        assert_eq!(filename_from_url("iris.csv"), Some("iris.csv"));
    }

    #[test]
    fn inspect_cache_missing_then_fresh_and_stale() {
        let dir = "./test_inspect_cache_states";
        let dir_path = Path::new(dir);
        let dst = dir_path.join("data.txt");
        let _ = remove_dir_all(dir);

        // Missing: directory is created on demand, file absent.
        assert!(matches!(
            inspect_cache(dir_path, &dst, None).unwrap(),
            CacheState::Missing
        ));
        assert!(dir_path.exists());

        // Fresh: file present, hash matches.
        fs::write(&dst, b"hello world").unwrap();
        assert!(matches!(
            inspect_cache(dir_path, &dst, Some(HELLO_WORLD_SHA256)).unwrap(),
            CacheState::Fresh
        ));
        // Fresh: no hash requested.
        assert!(matches!(
            inspect_cache(dir_path, &dst, None).unwrap(),
            CacheState::Fresh
        ));

        // Stale: file present but hash mismatches.
        assert!(matches!(
            inspect_cache(dir_path, &dst, Some(ZERO_SHA256)).unwrap(),
            CacheState::Stale
        ));

        remove_dir_all(dir).unwrap();
    }
}
