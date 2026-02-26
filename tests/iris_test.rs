use rustyml_dataset::iris::*;
use std::fs::remove_dir_all;

#[test]
fn test_load_iris() {
    let download_dir = "./test_load_iris"; // the code will create the directory if it doesn't exist
    
    let (features, labels) = load_iris(download_dir).unwrap();
    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);
    
    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}

#[test]
fn test_load_iris_owned() {
    let download_dir = "./test_load_iris_owned"; // the code will create the directory if it doesn't exist
    
    let (mut features, labels) = load_iris_owned(download_dir).unwrap();
    
    // You can now modify the data since these are owned copies
    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);
    
    // Example: Modify feature values (not possible with references)
    features[[0, 0]] = 5.5;
    
    // clean up: remove the downloaded files
    remove_dir_all(download_dir).unwrap();
}
