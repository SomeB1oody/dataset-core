use rustyml_dataset::iris::*;

#[test]
fn test_load_iris() {
    let download_dir = "./downloads"; // you need to create a directory manually beforehand
    
    let (features, labels) = load_iris(download_dir).unwrap();
    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);
    
    // clean up: remove the downloaded files if they exist
    if let Ok(entries) = std::fs::read_dir(download_dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[test]
fn test_load_iris_owned() {
    let download_dir = "./downloads"; // you need to create a directory manually beforehand
    
    let (mut features, labels) = load_iris_owned(download_dir).unwrap();
    
    // You can now modify the data since these are owned copies
    assert_eq!(features.shape(), &[150, 4]);
    assert_eq!(labels.len(), 150);
    
    // Example: Modify feature values (not possible with references)
    features[[0, 0]] = 5.5;
    
    // clean up: remove the downloaded files if they exist
    if let Ok(entries) = std::fs::read_dir(download_dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}
