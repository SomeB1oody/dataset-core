pub mod red_wine_quality;
pub mod white_wine_quality;

use crate::DatasetError;
use ndarray::{Array1, Array2};
use csv::ReaderBuilder;

/// Parses a single Wine Quality CSV (red or white) into `(features, targets)`.
///
/// The CSV is expected to be `;`-separated with a **header row**, followed by data rows.
/// Each data row must contain:
/// - 11 feature columns (all parseable as `f64`)
/// - 1 target column (`quality`, parseable as `f64`)
///
/// # Parameters
///
/// - `dataset_name` - Name of the dataset for error messages.
/// - `reader` - CSV file reader.
///
/// # Returns
///
/// - `Array2<f64>` - Feature matrix with shape `(n_samples, 11)`.
/// - `Array1<f64>` - Target vector with length `n_samples`.
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Any row has an unexpected number of columns
/// - Any feature/target value fails to parse as `f64`
/// - The final number of parsed values does not match the expected shape
fn parse_wine_data_to_array<R: std::io::Read>(
    dataset_name: &str,
    reader: R,
) -> Result<(Array2<f64>, Array1<f64>), DatasetError> {
    let mut rdr = ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(true)
        .from_reader(reader);

    let mut features_array = Vec::new();
    let mut target_array = Vec::new();
    let mut num_features: Option<usize> = None;

    for (idx, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| {
            DatasetError::csv_read_error(dataset_name, e)
        })?;
        let line_num = idx + 2; // +1 for 0-indexed, +1 for header

        if num_features.is_none() {
            if record.len() < 2 {
                return Err(DatasetError::invalid_column_count(
                    dataset_name,
                    2,
                    record.len(),
                    line_num,
                    &format!("{:?}", record),
                ));
            }
            num_features = Some(record.len() - 1);
        }

        let n_features = num_features.unwrap();
        if record.len() != n_features + 1 {
            return Err(DatasetError::invalid_column_count(
                dataset_name,
                n_features + 1,
                record.len(),
                line_num,
                &format!("{:?}", record),
            ));
        }

        for i in 0..n_features {
            let field = format!("feature[{i}]");
            features_array.push(
                record[i]
                    .parse::<f64>()
                    .map_err(|e| DatasetError::parse_failed(dataset_name, &field, line_num, &format!("{:?}", record), e))?,
            );
        }

        target_array.push(
            record[n_features]
                .parse::<f64>()
                .map_err(|e| DatasetError::parse_failed(dataset_name, "target", line_num, &format!("{:?}", record), e))?,
        );
    }

    let n_samples = target_array.len();
    if n_samples == 0 {
        return Err(DatasetError::empty_dataset(dataset_name));
    }

    let n_features = num_features.unwrap();
    let features_array =
        Array2::from_shape_vec((n_samples, n_features), features_array)
            .map_err(|e| DatasetError::array_shape_error(dataset_name, "features", e))?;
    let target_array = Array1::from_vec(target_array);

    Ok((features_array, target_array))
}
