use rustyml_dataset::diabetes::*;
use std::fs::remove_dir_all;

#[test]
fn test_load_diabetes() {
    let download_dir = "./test_load_diabetes"; // the code will create the directory if it doesn't exist
    
    let (features, labels) = load_diabetes(download_dir).unwrap();
    assert_eq!(features.shape(), &[768, 8]);
    assert_eq!(labels.len(), 768);
    
    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_diabetes_owned() {
    let download_dir = "./test_load_diabetes_owned"; // the code will create the directory if it doesn't exist
    
    let (mut features, mut labels) = load_diabetes_owned(download_dir).unwrap();
    
    assert_eq!(features.shape(), &[768, 8]);
    assert_eq!(labels.len(), 768);
    
    // Example: Modify feature values (not possible with references)
    features[[0, 0]] = 10.0;
    labels[0] = 1.0;
    
    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();   
}