use rustyml_dataset::titanic::*;

#[test]
fn test_load_titanic() {
    let download_dir = "./downloads"; // the code will create the directory if it doesn't exist

    let (string_features, num_features, labels) = load_titanic(download_dir).unwrap();

    assert_eq!(string_features.nrows(), 891); // 891 samples
    assert_eq!(string_features.ncols(), 5); // 5 features
    assert_eq!(num_features.nrows(), 891); // 891 samples
    assert_eq!(num_features.ncols(), 6); // 6 features
    assert_eq!(labels.len(), 891); // 891 samples

    // clean up: remove the downloaded files
    if let Ok(entries) = std::fs::read_dir(download_dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[test]
fn test_load_titanic_owned() {
    let download_dir = "./downloads"; // the code will create the directory if it doesn't exist

    let (string_features, mut num_features, labels) = load_titanic_owned(download_dir).unwrap();

    assert_eq!(string_features.nrows(), 891); // 891 samples
    assert_eq!(string_features.ncols(), 5); // 5 features
    assert_eq!(num_features.nrows(), 891); // 891 samples
    assert_eq!(num_features.ncols(), 6); // 6 features
    assert_eq!(labels.len(), 891); // 891 samples

    // modify the data (not possible with references)
    num_features.mapv_inplace(|x| {
        if x.is_nan() { 0.0 } else { x }
    });

    // clean up: remove the downloaded files
    if let Ok(entries) = std::fs::read_dir(download_dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}