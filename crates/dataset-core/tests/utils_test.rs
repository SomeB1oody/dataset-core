#![cfg(feature = "utils")]

use dataset_core::utils::{acquire_dataset, download_to, gunzip, unzip};
use flate2::Compression;
use flate2::write::GzEncoder;
use std::fs::{self, File, create_dir_all, remove_dir_all};
use std::io::Write;
use std::path::Path;
use zip::ZipWriter;
use zip::write::SimpleFileOptions;

/// SHA256 of "hello world"
const HELLO_WORLD_SHA256: &str = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";

/// All-zero hash (always wrong)
const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

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

fn create_gz(gz_path: &Path, content: &[u8]) {
    let file = File::create(gz_path).unwrap();
    let mut encoder = GzEncoder::new(file, Compression::default());
    encoder.write_all(content).unwrap();
    encoder.finish().unwrap();
}

#[test]
fn test_gunzip_round_trip() {
    let dir = "./test_gunzip_round_trip";
    create_dir_all(dir).unwrap();
    let dir_path = Path::new(dir);

    let payload = b"feature_0,feature_1,label\n2596,51,5\n2590,56,5\n";
    let gz_path = dir_path.join("data.csv.gz");
    let out_path = dir_path.join("data.csv");
    create_gz(&gz_path, payload);

    gunzip(&gz_path, &out_path).unwrap();

    assert_eq!(fs::read(&out_path).unwrap(), payload);

    remove_dir_all(dir).unwrap();
}

#[test]
fn test_gunzip_nonexistent_source_errors() {
    let result = gunzip(
        Path::new("./no_such_archive_for_gunzip_test.gz"),
        Path::new("./no_such_archive_for_gunzip_test.out"),
    );
    assert!(result.is_err());
}

#[test]
fn test_gunzip_invalid_stream_errors() {
    let dir = "./test_gunzip_invalid_stream_errors";
    create_dir_all(dir).unwrap();
    let dir_path = Path::new(dir);

    // The bytes are plain text, not a gzip stream. Decompression must fail
    // here, not succeed silently.
    let bogus = dir_path.join("not_really.gz");
    fs::write(&bogus, b"this is not a gzip stream").unwrap();

    let result = gunzip(&bogus, &dir_path.join("out.bin"));
    assert!(result.is_err());

    remove_dir_all(dir).unwrap();
}

#[test]
fn test_unzip_single_file() {
    let dir = "./test_unzip_single_file";
    create_dir_all(dir).unwrap();
    let dir_path = Path::new(dir);

    let zip_path = dir_path.join("archive.zip");
    create_zip(&zip_path, &[("hello.txt", b"hello world")]);

    unzip(&zip_path, dir_path).unwrap();

    assert_eq!(
        fs::read_to_string(dir_path.join("hello.txt")).unwrap(),
        "hello world"
    );

    remove_dir_all(dir).unwrap();
}

#[test]
fn test_unzip_multiple_files() {
    let dir = "./test_unzip_multiple_files";
    create_dir_all(dir).unwrap();
    let dir_path = Path::new(dir);

    let zip_path = dir_path.join("multi.zip");
    create_zip(&zip_path, &[("a.txt", b"file a"), ("b.txt", b"file b")]);

    unzip(&zip_path, dir_path).unwrap();

    assert_eq!(
        fs::read_to_string(dir_path.join("a.txt")).unwrap(),
        "file a"
    );
    assert_eq!(
        fs::read_to_string(dir_path.join("b.txt")).unwrap(),
        "file b"
    );

    remove_dir_all(dir).unwrap();
}

#[test]
fn test_unzip_nonexistent_zip_errors() {
    let result = unzip(
        Path::new("./no_such_archive_for_unzip_test.zip"),
        Path::new("."),
    );
    assert!(result.is_err());
}

#[test]
fn test_acquire_dataset_basic() {
    let dir = "./test_acquire_dataset_basic";
    create_dir_all(dir).unwrap();

    let result = acquire_dataset(
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
fn test_acquire_dataset_no_sha256_validation() {
    let dir = "./test_acquire_dataset_no_sha256_validation";
    create_dir_all(dir).unwrap();

    let result = acquire_dataset(dir, "output.txt", "test_dataset", None, |temp_path| {
        let dst = temp_path.join("output.txt");
        fs::write(&dst, b"any content, no hash check").unwrap();
        Ok(dst)
    });

    assert!(result.is_ok());
    assert!(result.unwrap().exists());

    remove_dir_all(dir).unwrap();
}

#[test]
fn test_acquire_dataset_sha256_mismatch_errors() {
    let dir = "./test_acquire_dataset_sha256_mismatch_errors";
    create_dir_all(dir).unwrap();

    let result = acquire_dataset(
        dir,
        "output.txt",
        "test_dataset",
        Some(ZERO_SHA256),
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
fn test_acquire_dataset_skips_acquisition_when_cached() {
    let dir = "./test_acquire_dataset_skips_acquisition_when_cached";
    create_dir_all(dir).unwrap();
    let dir_path = Path::new(dir);

    // The destination already has the correct file.
    fs::write(dir_path.join("output.txt"), b"hello world").unwrap();

    let result = acquire_dataset(
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
fn test_acquire_dataset_no_sha256_skips_acquisition_when_file_exists() {
    let dir = "./test_acquire_dataset_no_sha256_skips_acquisition_when_file_exists";
    create_dir_all(dir).unwrap();
    let dir_path = Path::new(dir);

    fs::write(dir_path.join("output.txt"), b"cached content").unwrap();

    let result = acquire_dataset(dir, "output.txt", "test_dataset", None, |_temp_path| {
        panic!("closure must not run when file exists and no hash is required")
    });

    assert!(result.is_ok());
    assert_eq!(
        fs::read(dir_path.join("output.txt")).unwrap(),
        b"cached content"
    );

    remove_dir_all(dir).unwrap();
}

#[test]
fn test_acquire_dataset_overwrites_stale_file() {
    let dir = "./test_acquire_dataset_overwrites_stale_file";
    create_dir_all(dir).unwrap();
    let dir_path = Path::new(dir);

    fs::write(dir_path.join("output.txt"), b"stale content").unwrap();

    let result = acquire_dataset(
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
    assert_eq!(
        fs::read(dir_path.join("output.txt")).unwrap(),
        b"hello world"
    );

    remove_dir_all(dir).unwrap();
}

#[test]
fn test_acquire_dataset_creates_directory() {
    // dir does not exist yet. acquire_dataset must create it.
    let dir = "./test_acquire_dataset_creates_directory";

    let result = acquire_dataset(dir, "output.txt", "test_dataset", None, |temp_path| {
        let dst = temp_path.join("output.txt");
        fs::write(&dst, b"content").unwrap();
        Ok(dst)
    });

    assert!(result.is_ok());
    assert!(Path::new(dir).exists());

    remove_dir_all(dir).unwrap();
}

#[test]
fn test_acquire_dataset_returns_correct_path() {
    let dir = "./test_acquire_dataset_returns_correct_path";
    create_dir_all(dir).unwrap();

    let result = acquire_dataset(dir, "my_data.txt", "test_dataset", None, |temp_path| {
        let dst = temp_path.join("my_data.txt");
        fs::write(&dst, b"data").unwrap();
        Ok(dst)
    })
    .unwrap();

    assert_eq!(result, Path::new(dir).join("my_data.txt"));

    remove_dir_all(dir).unwrap();
}

#[test]
fn test_download_to_downloads_file() {
    let dir = "./test_download_to_downloads_file";
    create_dir_all(dir).unwrap();

    download_to(
        "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv",
        Path::new(dir),
        None,
    )
    .unwrap();

    assert!(Path::new(dir).join("iris.csv").exists());

    remove_dir_all(dir).unwrap();
}
