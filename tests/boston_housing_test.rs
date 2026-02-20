use rustyml_dataset::boston_housing::*;

#[test]
fn test_load_boston_housing() {
    let download_dir = "./downloads"; // you need to create a directory manually beforehand

    let (features, targets) = load_boston_housing(download_dir).unwrap();
    assert_eq!(features.shape(), &[506, 13]);
    assert_eq!(targets.len(), 506);

    // clean up: remove the downloaded files if they exist
    if let Ok(entries) = std::fs::read_dir(download_dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[test]
fn test_load_boston_housing_owned() {
    let download_dir = "./downloads"; // you need to create a directory manually beforehand

    let (mut features, mut targets) = load_boston_housing_owned(download_dir).unwrap();

    // You can now modify the data since these are owned copies
    assert_eq!(features.shape(), &[506, 13]);
    assert_eq!(targets.len(), 506);

    // Example: Modify feature values (not possible with references)
    features[[0, 0]] = 0.1;
    targets[0] = 25.5;

    // clean up: remove the downloaded files if they exist
    if let Ok(entries) = std::fs::read_dir(download_dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}