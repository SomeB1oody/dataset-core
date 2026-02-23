use rustyml_dataset::wine_quality::*;

#[test]
fn test_load_red_wine_quality() {
    let download_dir = "./downloads"; // the code will create this directory if it doesn't exist
    
    let (features, targets) = load_red_wine_quality(download_dir).unwrap();
    
    // Use the feature matrix for machine learning
    assert_eq!(features.shape(), &[1599, 11]);  // 1599 samples, 11 features
    assert_eq!(targets.len(), 1599); // 1599 samples
    
    // clean up: remove the downloaded files if they exist
    if let Ok(entries) = std::fs::read_dir(download_dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[test]
fn test_load_red_wine_quality_owned() {
    let download_dir = "./downloads"; // the code will create this directory if it doesn't exist
    
    let (mut features, targets) = load_red_wine_quality_owned(download_dir).unwrap();
    
    // Use the feature matrix for machine learning
    assert_eq!(features.shape(), &[1599, 11]);  // 1599 samples, 11 features
    assert_eq!(targets.len(), 1599); // 1599 samples
    
    // Example: Modify feature values (not possible with references)
    features[[0, 0]] = 10.0;
    
    // clean up: remove the downloaded files if they exist
    if let Ok(entries) = std::fs::read_dir(download_dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[test]
fn test_load_white_wine_quality() {
    let download_dir = "./downloads"; // the code will create this directory if it doesn't exist
    
    let (features, targets) = load_white_wine_quality(download_dir).unwrap();
    
    // Use the feature matrix for machine learning
    assert_eq!(features.shape(), &[4898, 11]);  // 4898 samples, 11 features
    assert_eq!(targets.len(), 4898); // 4898 samples
    
    // clean up: remove the downloaded files if they exist
    if let Ok(entries) = std::fs::read_dir(download_dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[test]
fn test_load_white_wine_quality_owned() {
    let download_dir = "./downloads"; // the code will create this directory if it doesn't exist
    
    let (mut features, targets) = load_white_wine_quality_owned(download_dir).unwrap();
    
    // Use the feature matrix for machine learning
    assert_eq!(features.shape(), &[4898, 11]);  // 4898 samples, 11 features
    assert_eq!(targets.len(), 4898); // 4898 samples
    
    // Example: Modify feature values (not possible with references)
    features[[0, 0]] = 10.0;
    
    // clean up: remove the downloaded files if they exist
    if let Ok(entries) = std::fs::read_dir(download_dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}