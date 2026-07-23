use crate::DatasetError;
use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use std::fmt::Write;
use std::fs::File;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use tar::Archive;
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

/// Download a remote file into the given directory, retrying transient failures.
///
/// A wrapper around [`download_to`] for the unreliable hosts many public datasets
/// live on: if the download fails, it is retried up to `retries` more times, waiting
/// twice as long before each attempt (500 ms, then 1 s, 2 s, …). Passing `retries = 0`
/// makes this exactly equivalent to [`download_to`].
///
/// Only the download is retried. A failure that cannot be fixed by trying again —
/// a filename that cannot be derived from the URL, or a local file that cannot be
/// created — is returned immediately, and the last download error is returned once
/// the attempts are exhausted.
///
/// # Parameters
///
/// - `url` - The URL to download.
/// - `storage_path` - The directory to store the downloaded file in.
/// - `filename` - Optional custom filename (with extension). If `None`, the filename is extracted
///   from the last segment of the URL.
/// - `retries` - How many **additional** attempts to make after the first one fails.
///
/// # Errors
///
/// - `DatasetError` - Returned when every attempt fails (the last error is
///   propagated), or immediately for a non-retryable error.
///
/// # Example
/// ```no_run
/// use dataset_core::download_to_with_retries;
/// use std::path::Path;
///
/// let download_dir = Path::new("./download_retry_example");
/// std::fs::create_dir_all(download_dir).unwrap();
///
/// // Try up to three times in total before giving up.
/// download_to_with_retries(
///     "https://archive.ics.uci.edu/static/public/53/iris.zip",
///     download_dir,
///     Some("iris.zip"),
///     2,
/// )
/// .unwrap();
/// ```
pub fn download_to_with_retries(
    url: &str,
    storage_path: &Path,
    filename: Option<&str>,
    retries: u32,
) -> Result<(), DatasetError> {
    let mut attempt = 0;

    loop {
        match download_to(url, storage_path, filename) {
            Ok(()) => return Ok(()),
            // Retrying cannot make a malformed URL or an unwritable target work.
            Err(e @ (DatasetError::ValidationError(_) | DatasetError::IoError(_))) => {
                return Err(e);
            }
            Err(e) => {
                if attempt >= retries {
                    return Err(e);
                }
                std::thread::sleep(RETRY_BASE_DELAY * 2u32.pow(attempt));
                attempt += 1;
            }
        }
    }
}

/// How long [`download_to_with_retries`] waits before its first retry; each further
/// retry doubles it.
const RETRY_BASE_DELAY: std::time::Duration = std::time::Duration::from_millis(500);

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
/// use dataset_core::unzip;
/// use std::fs::File;
/// use std::io::Write;
/// use std::path::Path;
/// use zip::write::SimpleFileOptions;
///
/// let work_dir = Path::new("./unzip_example");
/// std::fs::create_dir_all(work_dir).unwrap();
///
/// // Build a small `.zip` containing a single `hello.txt` entry.
/// let archive_path = work_dir.join("data.zip");
/// let mut zip = zip::ZipWriter::new(File::create(&archive_path).unwrap());
/// zip.start_file("hello.txt", SimpleFileOptions::default()).unwrap();
/// zip.write_all(b"hello world").unwrap();
/// zip.finish().unwrap();
///
/// // Extract it into `work_dir`.
/// unzip(&archive_path, work_dir).unwrap();
/// assert!(work_dir.join("hello.txt").exists());
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

/// Extract a tar (`.tar`) archive into a target directory using [`tar::Archive`].
///
/// This is the tar analogue of [`unzip`]: a `.tar` bundles a whole directory tree
/// (unlike [`gunzip`], which decompresses a single-file gzip stream). For the very
/// common gzip-compressed tarball (`.tar.gz` / `.tgz`), use [`untar_gz`], which
/// streams the decompression and extraction together without writing an
/// intermediate `.tar` to disk.
///
/// The archive's entries are unpacked **relative to** `extract_dir` (which is
/// created if needed); the [`tar`] crate rejects entries whose paths would escape
/// it.
///
/// # Parameters
///
/// - `file_path` - Path to the `.tar` file to extract.
/// - `extract_dir` - Directory to extract the archive contents into.
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when opening the archive fails or when
///   extraction fails (a malformed archive, or an entry that cannot be written).
///
/// # Example
/// ```no_run
/// use dataset_core::{download_to, untar};
/// use std::path::Path;
///
/// let work_dir = Path::new("./untar_example");
/// std::fs::create_dir_all(work_dir).unwrap();
///
/// // Download a tar archive, then extract it in place.
/// download_to("https://example.com/data.tar", work_dir, Some("data.tar")).unwrap();
/// untar(&work_dir.join("data.tar"), work_dir).unwrap();
/// ```
pub fn untar(file_path: &Path, extract_dir: &Path) -> Result<(), DatasetError> {
    let file = File::open(file_path)?;
    Archive::new(file).unpack(extract_dir)?;

    Ok(())
}

/// Extract a gzip-compressed tar (`.tar.gz` / `.tgz`) archive into a target directory.
///
/// This composes the two layers of a gzipped tarball in one streaming pass: the
/// bytes flow through [`flate2::read::GzDecoder`] (the gzip layer, as in
/// [`gunzip`]) straight into [`tar::Archive`] (the tar layer, as in [`untar`]), so
/// the intermediate uncompressed `.tar` is never written to disk — suitable for
/// large datasets distributed as `.tar.gz`.
///
/// The archive's entries are unpacked **relative to** `extract_dir` (which is
/// created if needed); the [`tar`] crate rejects entries whose paths would escape
/// it.
///
/// # Parameters
///
/// - `file_path` - Path to the `.tar.gz` (or `.tgz`) file to extract.
/// - `extract_dir` - Directory to extract the archive contents into.
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when opening the source fails, the gzip
///   stream is malformed, or the tar extraction fails.
///
/// # Example
/// ```no_run
/// use dataset_core::{download_to, untar_gz};
/// use std::path::Path;
///
/// let work_dir = Path::new("./untar_gz_example");
/// std::fs::create_dir_all(work_dir).unwrap();
///
/// // Download a gzip-compressed tarball, then extract it in place.
/// download_to("https://example.com/data.tar.gz", work_dir, Some("data.tar.gz")).unwrap();
/// untar_gz(&work_dir.join("data.tar.gz"), work_dir).unwrap();
/// ```
pub fn untar_gz(file_path: &Path, extract_dir: &Path) -> Result<(), DatasetError> {
    let input = File::open(file_path)?;
    let decoder = GzDecoder::new(input);
    Archive::new(decoder).unpack(extract_dir)?;

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

/// Compute a file's SHA256 hash and return it as a lowercase hex string.
///
/// The file is streamed in 8 KiB chunks, so hashing a multi-gigabyte dataset costs
/// no more memory than hashing a small one.
///
/// This is the helper to reach for when **pinning** a hash: run it once against a
/// freshly downloaded file and paste the result into the `expected_sha256` you pass
/// to [`acquire_dataset`]. To check a file against a hash you already have, use
/// [`verify_sha256`] instead of comparing strings yourself.
///
/// # Parameters
///
/// - `path` - Path to the file to hash.
///
/// # Returns
///
/// - `String` - The SHA256 digest as 64 lowercase hex characters.
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when the file cannot be opened or read.
///
/// # Example
/// ```no_run
/// use dataset_core::sha256_file;
/// use std::path::Path;
///
/// let digest = sha256_file(Path::new("./data/iris.csv")).unwrap();
/// println!("pin this: {digest}");
/// ```
pub fn sha256_file(path: &Path) -> Result<String, DatasetError> {
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
    let mut hex = String::with_capacity(digest.len() * 2);
    for b in digest {
        // Writing formatted bytes into a `String` is infallible.
        let _ = write!(hex, "{:02x}", b);
    }

    Ok(hex)
}

/// Verify that a file's SHA256 hash matches an expected value (case-insensitive).
///
/// This is the same check [`acquire_dataset`] performs internally on cached and
/// freshly prepared files, exposed for callers that need to validate a file outside
/// that workflow — most commonly a test asserting that the file on disk is the
/// expected one. Returns `true` when the computed hash matches `expected_hex`.
///
/// # Parameters
///
/// - `path` - Path to the file to verify.
/// - `expected_hex` - The expected SHA256 digest, in hex (either case).
///
/// # Returns
///
/// - `bool` - `true` if the file's hash matches `expected_hex`, `false` otherwise.
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when the file cannot be opened or read.
///
/// # Example
/// ```no_run
/// use dataset_core::verify_sha256;
/// use std::path::Path;
///
/// const IRIS_SHA256: &str = "c52742e50315a99f956a383faedf7575552675f6409ef0f9a47076dd08479930";
///
/// assert!(verify_sha256(Path::new("./data/iris.csv"), IRIS_SHA256).unwrap());
/// ```
pub fn verify_sha256(path: &Path, expected_hex: &str) -> Result<bool, DatasetError> {
    Ok(sha256_file(path)?.eq_ignore_ascii_case(expected_hex))
}

/// Read a file as Latin-1 (ISO-8859-1) text.
///
/// Every byte is mapped to the Unicode scalar with the same value, which is exactly
/// what Latin-1 decoding means and what scikit-learn does for the older text
/// corpora. Unlike [`std::fs::read_to_string`], this never fails on non-UTF-8 input
/// and never replaces bytes with `U+FFFD`: the decoding is lossless and reversible,
/// so a corpus whose encoding is unknown or mixed survives the round trip.
///
/// Use it for raw document collections (newsgroup posts, movie reviews, …) that
/// predate UTF-8; for data you know is UTF-8, prefer `std::fs::read_to_string`.
///
/// # Parameters
///
/// - `path` - Path to the file to read.
///
/// # Returns
///
/// - `String` - The file's contents, decoded byte-for-byte as Latin-1.
///
/// # Errors
///
/// - `DatasetError::IoError` - Returned when the file cannot be opened or read.
///
/// # Example
/// ```no_run
/// use dataset_core::read_latin1;
/// use std::path::Path;
///
/// // A byte that is invalid UTF-8 decodes to the matching Latin-1 character
/// // instead of failing the read.
/// let text = read_latin1(Path::new("./corpus/post_00001")).unwrap();
/// assert!(!text.is_empty());
/// ```
pub fn read_latin1(path: &Path) -> Result<String, DatasetError> {
    let bytes = std::fs::read(path)?;

    Ok(bytes.iter().map(|&b| b as char).collect())
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
        Some(hash) if !verify_sha256(dst, hash)? => Ok(CacheState::Stale),
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
        && !verify_sha256(&src, hash)?
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

    /// Build a tar archive at `path` from `(name, content)` entries (test helper).
    fn write_tar(path: &Path, entries: &[(&str, &[u8])]) {
        let file = File::create(path).unwrap();
        let mut builder = tar::Builder::new(file);
        for (name, content) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(content.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, name, *content).unwrap();
        }
        builder.into_inner().unwrap().sync_all().unwrap();
    }

    /// Build a gzip-compressed tar archive at `path` (test helper).
    fn write_tar_gz(path: &Path, entries: &[(&str, &[u8])]) {
        use flate2::Compression;
        use flate2::write::GzEncoder;

        let file = File::create(path).unwrap();
        let encoder = GzEncoder::new(file, Compression::default());
        let mut builder = tar::Builder::new(encoder);
        for (name, content) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(content.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, name, *content).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap();
    }

    #[test]
    fn untar_round_trips_entries() {
        let dir = "./test_untar_round_trips_entries";
        create_dir_all(dir).unwrap();
        let dir_path = Path::new(dir);

        let tar_path = dir_path.join("data.tar");
        write_tar(
            &tar_path,
            &[("a.txt", b"hello"), ("nested/b.txt", b"world")],
        );

        let out_dir = dir_path.join("extracted");
        untar(&tar_path, &out_dir).unwrap();

        assert_eq!(fs::read(out_dir.join("a.txt")).unwrap(), b"hello");
        assert_eq!(fs::read(out_dir.join("nested/b.txt")).unwrap(), b"world");

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn untar_nonexistent_source_errors() {
        let result = untar(
            Path::new("./no_such_file_for_untar_test.tar"),
            Path::new("./no_such_dir_for_untar_test"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn untar_gz_round_trips_entries() {
        let dir = "./test_untar_gz_round_trips_entries";
        create_dir_all(dir).unwrap();
        let dir_path = Path::new(dir);

        let tar_gz_path = dir_path.join("data.tar.gz");
        write_tar_gz(
            &tar_gz_path,
            &[("a.txt", b"hello"), ("nested/b.txt", b"world")],
        );

        let out_dir = dir_path.join("extracted");
        untar_gz(&tar_gz_path, &out_dir).unwrap();

        assert_eq!(fs::read(out_dir.join("a.txt")).unwrap(), b"hello");
        assert_eq!(fs::read(out_dir.join("nested/b.txt")).unwrap(), b"world");

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn untar_gz_nonexistent_source_errors() {
        let result = untar_gz(
            Path::new("./no_such_file_for_untar_gz_test.tar.gz"),
            Path::new("./no_such_dir_for_untar_gz_test"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn verify_sha256_correct_hash() {
        let dir = "./test_verify_sha256_correct_hash";
        create_dir_all(dir).unwrap();
        let path = Path::new(dir).join("f.txt");
        File::create(&path)
            .unwrap()
            .write_all(b"hello world")
            .unwrap();

        assert!(verify_sha256(&path, HELLO_WORLD_SHA256).unwrap());

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn verify_sha256_uppercase_hash() {
        let dir = "./test_verify_sha256_uppercase_hash";
        create_dir_all(dir).unwrap();
        let path = Path::new(dir).join("f.txt");
        File::create(&path)
            .unwrap()
            .write_all(b"hello world")
            .unwrap();

        assert!(verify_sha256(&path, &HELLO_WORLD_SHA256.to_uppercase()).unwrap());

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn verify_sha256_wrong_hash_returns_false() {
        let dir = "./test_verify_sha256_wrong_hash_returns_false";
        create_dir_all(dir).unwrap();
        let path = Path::new(dir).join("f.txt");
        File::create(&path)
            .unwrap()
            .write_all(b"hello world")
            .unwrap();

        assert!(!verify_sha256(&path, ZERO_SHA256).unwrap());

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn verify_sha256_empty_file() {
        let dir = "./test_verify_sha256_empty_file";
        create_dir_all(dir).unwrap();
        let path = Path::new(dir).join("empty.txt");
        File::create(&path).unwrap();

        assert!(verify_sha256(&path, EMPTY_SHA256).unwrap());

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn verify_sha256_nonexistent_file_errors() {
        let result = verify_sha256(Path::new("./no_such_file_sha256_test.txt"), ZERO_SHA256);
        assert!(result.is_err());
    }

    #[test]
    fn sha256_file_returns_lowercase_hex_digest() {
        let dir = "./test_sha256_file_returns_lowercase_hex_digest";
        create_dir_all(dir).unwrap();
        let path = Path::new(dir).join("f.txt");
        File::create(&path)
            .unwrap()
            .write_all(b"hello world")
            .unwrap();

        let digest = sha256_file(&path).unwrap();
        assert_eq!(digest, HELLO_WORLD_SHA256);
        assert_eq!(digest.len(), 64);
        assert!(
            digest
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sha256_file_hashes_larger_than_one_chunk() {
        let dir = "./test_sha256_file_hashes_larger_than_one_chunk";
        create_dir_all(dir).unwrap();
        let path = Path::new(dir).join("big.bin");
        // Larger than the 8 KiB read buffer, so the streaming loop runs several times.
        let payload = vec![0xABu8; 8192 * 3 + 17];
        fs::write(&path, &payload).unwrap();

        // The streamed digest must equal the one-shot digest of the same bytes.
        let expected = {
            let mut hasher = Sha256::new();
            hasher.update(&payload);
            hasher
                .finalize()
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>()
        };
        assert_eq!(sha256_file(&path).unwrap(), expected);

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sha256_file_nonexistent_file_errors() {
        assert!(sha256_file(Path::new("./no_such_file_for_sha256_file_test.bin")).is_err());
    }

    #[test]
    fn read_latin1_decodes_every_byte() {
        let dir = "./test_read_latin1_decodes_every_byte";
        create_dir_all(dir).unwrap();
        let path = Path::new(dir).join("latin1.txt");
        // All 256 byte values, including sequences that are invalid UTF-8.
        let bytes: Vec<u8> = (0u8..=255).collect();
        fs::write(&path, &bytes).unwrap();

        let text = read_latin1(&path).unwrap();

        // One character per source byte, each with the byte's own scalar value.
        assert_eq!(text.chars().count(), 256);
        for (i, c) in text.chars().enumerate() {
            assert_eq!(c as u32, i as u32, "byte {i} decoded to {c:?}");
        }

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn read_latin1_empty_file_is_empty_string() {
        let dir = "./test_read_latin1_empty_file_is_empty_string";
        create_dir_all(dir).unwrap();
        let path = Path::new(dir).join("empty.txt");
        File::create(&path).unwrap();

        assert_eq!(read_latin1(&path).unwrap(), "");

        remove_dir_all(dir).unwrap();
    }

    #[test]
    fn read_latin1_nonexistent_file_errors() {
        assert!(read_latin1(Path::new("./no_such_file_for_read_latin1_test.txt")).is_err());
    }

    #[test]
    fn download_to_with_retries_does_not_retry_unusable_url() {
        let dir = "./test_download_to_with_retries_does_not_retry_unusable_url";
        create_dir_all(dir).unwrap();

        // A URL ending in `/` yields no filename — retrying cannot fix that, so this
        // must fail immediately rather than sleeping through the retry schedule.
        let started = std::time::Instant::now();
        let result = download_to_with_retries("https://x.test/a/", Path::new(dir), None, 5);

        assert!(matches!(result, Err(DatasetError::ValidationError(_))));
        assert!(
            started.elapsed() < RETRY_BASE_DELAY,
            "a non-retryable error must not wait for a retry"
        );

        remove_dir_all(dir).unwrap();
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
