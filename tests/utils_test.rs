#![cfg(feature = "utils")]

use dataset_core::utils::{create_temp_dir, download_dataset_with, download_to, file_sha256_matches, unzip};
use std::fs::{self, File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;
use zip::ZipWriter;
use zip::write::SimpleFileOptions;

/// SHA256 of "hello world"
const HELLO_WORLD_SHA256: &str = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";

/// SHA256 of an empty file
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// All-zero hash (always wrong)
const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[test]
// Verifies that create_temp_dir returns a path that exists within the given parent directory.
fn test_create_temp_dir_returns_existing_path() {
    let parent = "./test_create_temp_dir_returns_existing_path";
    create_dir_all(parent).unwrap();

    let temp_dir = create_temp_dir(Path::new(parent)).unwrap();
    let temp_path = temp_dir.path().to_path_buf();

    assert!(temp_path.exists());

    remove_dir_all(parent).unwrap();
}

#[test]
// Verifies that the temp directory is automatically deleted when the TempDir handle is dropped.
fn test_create_temp_dir_cleanup_on_drop() {
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
// Verifies that files written inside the temp directory are accessible while it exists.
fn test_create_temp_dir_files_written_inside() {
    let parent = "./test_create_temp_dir_files_written_inside";
    create_dir_all(parent).unwrap();

    let temp_dir = create_temp_dir(Path::new(parent)).unwrap();
    let temp_file = temp_dir.path().join("data.txt");
    fs::write(&temp_file, b"content").unwrap();
    assert!(temp_file.exists());

    remove_dir_all(parent).unwrap();
}

#[test]
// Verifies that create_temp_dir returns an error when the parent directory does not exist.
fn test_create_temp_dir_nonexistent_parent_errors() {
    let result = create_temp_dir(Path::new("./nonexistent_parent_xyz_abc_123"));
    assert!(result.is_err());
}

#[test]
// Verifies that file_sha256_matches returns true when the file's hash matches the expected value.
fn test_file_sha256_matches_correct_hash() {
    let dir = "./test_file_sha256_matches_correct_hash";
    create_dir_all(dir).unwrap();
    let path = Path::new(dir).join("f.txt");
    File::create(&path).unwrap().write_all(b"hello world").unwrap();

    assert!(file_sha256_matches(&path, HELLO_WORLD_SHA256).unwrap());

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that hash comparison is case-insensitive (uppercase hex string is accepted).
fn test_file_sha256_matches_uppercase_hash() {
    let dir = "./test_file_sha256_matches_uppercase_hash";
    create_dir_all(dir).unwrap();
    let path = Path::new(dir).join("f.txt");
    File::create(&path).unwrap().write_all(b"hello world").unwrap();

    assert!(file_sha256_matches(&path, &HELLO_WORLD_SHA256.to_uppercase()).unwrap());

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that file_sha256_matches returns false when the hash does not match the file's actual hash.
fn test_file_sha256_matches_wrong_hash_returns_false() {
    let dir = "./test_file_sha256_matches_wrong_hash_returns_false";
    create_dir_all(dir).unwrap();
    let path = Path::new(dir).join("f.txt");
    File::create(&path).unwrap().write_all(b"hello world").unwrap();

    assert!(!file_sha256_matches(&path, ZERO_SHA256).unwrap());

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that file_sha256_matches correctly computes and matches the hash of an empty file.
fn test_file_sha256_matches_empty_file() {
    let dir = "./test_file_sha256_matches_empty_file";
    create_dir_all(dir).unwrap();
    let path = Path::new(dir).join("empty.txt");
    File::create(&path).unwrap();

    assert!(file_sha256_matches(&path, EMPTY_SHA256).unwrap());

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that file_sha256_matches returns an error when the target file does not exist.
fn test_file_sha256_matches_nonexistent_file_errors() {
    let result = file_sha256_matches(Path::new("./no_such_file_sha256_test.txt"), ZERO_SHA256);
    assert!(result.is_err());
}

fn create_zip(zip_path: &Path, entries: &[(&str, &[u8])]) {
    let file = File::create(zip_path).unwrap();
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default();
    for (name, content) in entries {
        zip.start_file(*name, options).unwrap();
        zip.write_all(content).unwrap();
    }
    zip.finish().unwrap();
}

#[test]
// Verifies that unzip correctly extracts a single file from a zip archive.
fn test_unzip_single_file() {
    let dir = "./test_unzip_single_file";
    create_dir_all(dir).unwrap();
    let dir_path = Path::new(dir);

    let zip_path = dir_path.join("archive.zip");
    create_zip(&zip_path, &[("hello.txt", b"hello world")]);

    unzip(&zip_path, dir_path).unwrap();

    assert_eq!(fs::read_to_string(dir_path.join("hello.txt")).unwrap(), "hello world");

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that unzip correctly extracts all files from a multi-entry zip archive.
fn test_unzip_multiple_files() {
    let dir = "./test_unzip_multiple_files";
    create_dir_all(dir).unwrap();
    let dir_path = Path::new(dir);

    let zip_path = dir_path.join("multi.zip");
    create_zip(&zip_path, &[("a.txt", b"file a"), ("b.txt", b"file b")]);

    unzip(&zip_path, dir_path).unwrap();

    assert_eq!(fs::read_to_string(dir_path.join("a.txt")).unwrap(), "file a");
    assert_eq!(fs::read_to_string(dir_path.join("b.txt")).unwrap(), "file b");

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that unzip returns an error when the zip file does not exist.
fn test_unzip_nonexistent_zip_errors() {
    let result = unzip(Path::new("./no_such_archive_for_unzip_test.zip"), Path::new("."));
    assert!(result.is_err());
}

#[test]
// Verifies the basic happy-path: download_dataset_with writes a file and returns its path.
fn test_download_dataset_with_basic() {
    let dir = "./test_download_dataset_with_basic";
    create_dir_all(dir).unwrap();

    let result = download_dataset_with(
        dir,
        "output.txt",
        "test_dataset",
        Some(HELLO_WORLD_SHA256),
        |temp_path| {
            let dst = temp_path.join("output.txt");
            fs::write(&dst, b"hello world").unwrap();
            Ok(dst)
        },
    );

    assert!(result.is_ok());
    let out = result.unwrap();
    assert!(out.exists());
    assert_eq!(fs::read(&out).unwrap(), b"hello world");

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that download_dataset_with succeeds without SHA256 validation when no hash is provided.
fn test_download_dataset_with_no_sha256_validation() {
    let dir = "./test_download_dataset_with_no_sha256_validation";
    create_dir_all(dir).unwrap();

    let result = download_dataset_with(
        dir,
        "output.txt",
        "test_dataset",
        None,
        |temp_path| {
            let dst = temp_path.join("output.txt");
            fs::write(&dst, b"any content, no hash check").unwrap();
            Ok(dst)
        },
    );

    assert!(result.is_ok());
    assert!(result.unwrap().exists());

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that download_dataset_with returns an error when the downloaded file's hash doesn't match.
fn test_download_dataset_with_sha256_mismatch_errors() {
    let dir = "./test_download_dataset_with_sha256_mismatch_errors";
    create_dir_all(dir).unwrap();

    let result = download_dataset_with(
        dir,
        "output.txt",
        "test_dataset",
        Some(ZERO_SHA256), // wrong hash
        |temp_path| {
            let dst = temp_path.join("output.txt");
            fs::write(&dst, b"hello world").unwrap();
            Ok(dst)
        },
    );

    assert!(result.is_err());

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that the download closure is not invoked when a valid cached file already exists.
fn test_download_dataset_with_skips_download_when_cached() {
    let dir = "./test_download_dataset_with_skips_download_when_cached";
    create_dir_all(dir).unwrap();
    let dir_path = Path::new(dir);

    // Pre-populate the destination with the correct file
    fs::write(dir_path.join("output.txt"), b"hello world").unwrap();

    // The closure must NOT be called; if it is, the test panics
    let result = download_dataset_with(
        dir,
        "output.txt",
        "test_dataset",
        Some(HELLO_WORLD_SHA256),
        |_temp_path| panic!("closure must not run when file is already cached"),
    );

    assert!(result.is_ok());

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that the download closure is skipped when the file exists and no hash check is required.
fn test_download_dataset_with_no_sha256_skips_download_when_file_exists() {
    let dir = "./test_download_dataset_with_no_sha256_skips_download_when_file_exists";
    create_dir_all(dir).unwrap();
    let dir_path = Path::new(dir);

    fs::write(dir_path.join("output.txt"), b"cached content").unwrap();

    let result = download_dataset_with(
        dir,
        "output.txt",
        "test_dataset",
        None,
        |_temp_path| panic!("closure must not run when file exists and no hash is required"),
    );

    assert!(result.is_ok());
    // Original content is preserved
    assert_eq!(fs::read(dir_path.join("output.txt")).unwrap(), b"cached content");

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that a stale file with a mismatched hash is overwritten by re-downloading.
fn test_download_dataset_with_overwrites_stale_file() {
    let dir = "./test_download_dataset_with_overwrites_stale_file";
    create_dir_all(dir).unwrap();
    let dir_path = Path::new(dir);

    // Stale file whose hash doesn't match
    fs::write(dir_path.join("output.txt"), b"stale content").unwrap();

    let result = download_dataset_with(
        dir,
        "output.txt",
        "test_dataset",
        Some(HELLO_WORLD_SHA256),
        |temp_path| {
            let dst = temp_path.join("output.txt");
            fs::write(&dst, b"hello world").unwrap();
            Ok(dst)
        },
    );

    assert!(result.is_ok());
    assert_eq!(fs::read(dir_path.join("output.txt")).unwrap(), b"hello world");

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that download_dataset_with creates the destination directory if it does not exist.
fn test_download_dataset_with_creates_directory() {
    // dir does not exist yet — the function must create it
    let dir = "./test_download_dataset_with_creates_directory";

    let result = download_dataset_with(
        dir,
        "output.txt",
        "test_dataset",
        None,
        |temp_path| {
            let dst = temp_path.join("output.txt");
            fs::write(&dst, b"content").unwrap();
            Ok(dst)
        },
    );

    assert!(result.is_ok());
    assert!(Path::new(dir).exists());

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that download_dataset_with returns the correct final file path after download.
fn test_download_dataset_with_returns_correct_path() {
    let dir = "./test_download_dataset_with_returns_correct_path";
    create_dir_all(dir).unwrap();

    let result = download_dataset_with(
        dir,
        "my_data.txt",
        "test_dataset",
        None,
        |temp_path| {
            let dst = temp_path.join("my_data.txt");
            fs::write(&dst, b"data").unwrap();
            Ok(dst)
        },
    )
    .unwrap();

    assert_eq!(result, Path::new(dir).join("my_data.txt"));

    remove_dir_all(dir).unwrap();
}

#[test]
// Verifies that download_to fetches a remote file and saves it to the given directory.
fn test_download_to_downloads_file() {
    let dir = "./test_download_to_downloads_file";
    create_dir_all(dir).unwrap();

    download_to(
        "https://archive.ics.uci.edu/static/public/53/iris.zip",
        Path::new(dir),
    )
    .unwrap();

    assert!(Path::new(dir).join("iris.zip").exists());

    remove_dir_all(dir).unwrap();
}
