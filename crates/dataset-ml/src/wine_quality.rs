//! Wine Quality dataset.
//!
//! Physicochemical measurements and quality scores for Portuguese "Vinho Verde"
//! red and white wines, commonly used for regression or ordinal classification.
//!
//! **Features (11):**
//! - `fixed acidity`
//! - `volatile acidity`
//! - `citric acid`
//! - `residual sugar`
//! - `chlorides`
//! - `free sulfur dioxide`
//! - `total sulfur dioxide`
//! - `density`
//! - `pH`
//! - `sulphates`
//! - `alcohol`
//!
//! **Target:** `quality` - quality score between `0` and `10`
//!
//! **Samples:**
//! - Red wine subset: 1599
//! - White wine subset: 4898
//!
//! **Application:** Regression / ordinal classification of wine quality
//!
//! **Source:** UCI Machine Learning Repository
//! <https://doi.org/10.24432/C56S3T>
//!
//! **Subsets:**
//! - `red_wine_quality::RedWineQuality`
//! - `white_wine_quality::WhiteWineQuality`

pub mod red_wine_quality;
pub mod white_wine_quality;

use csv::ReaderBuilder;
use dataset_core::DatasetError;
use ndarray::{Array1, Array2};
use serde::Deserialize;

/// Type alias shared by both Wine Quality subsets: (features, targets).
pub(crate) type WineData = (Array2<f64>, Array1<f64>);

/// One CSV record of a Wine Quality file (red or white): 11 `f64` feature
/// columns followed by the `quality` target.
///
/// Fields are declared in CSV column order and deserialized **positionally**
/// (the parser disables csv's header handling), so this struct is independent
/// of the exact header spelling.
#[derive(Deserialize)]
struct WineRecord {
    fixed_acidity: f64,
    volatile_acidity: f64,
    citric_acid: f64,
    residual_sugar: f64,
    chlorides: f64,
    free_sulfur_dioxide: f64,
    total_sulfur_dioxide: f64,
    density: f64,
    ph: f64,
    sulphates: f64,
    alcohol: f64,
    quality: f64,
}

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
) -> Result<WineData, DatasetError> {
    // `has_headers(false)` makes csv deserialize into the named struct
    // *positionally* (by column order) rather than by header name, keeping
    // parsing independent of the exact header spelling. We skip the header row
    // ourselves with `.skip(1)`.
    let mut rdr = ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(false)
        .from_reader(reader);

    let mut features_array = Vec::new();
    let mut target_array = Vec::new();

    for result in rdr.deserialize::<WineRecord>().skip(1) {
        let WineRecord {
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            ph,
            sulphates,
            alcohol,
            quality,
        } = result.map_err(|e| DatasetError::csv_read_error(dataset_name, e))?;

        features_array.extend_from_slice(&[
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            ph,
            sulphates,
            alcohol,
        ]);
        target_array.push(quality);
    }

    let n_samples = target_array.len();
    if n_samples == 0 {
        return Err(DatasetError::empty_dataset(dataset_name));
    }

    // Wine Quality has a fixed schema of 11 numeric features per sample.
    let features_array = Array2::from_shape_vec((n_samples, 11), features_array)
        .map_err(|e| DatasetError::array_shape_error(dataset_name, "features", e))?;
    let target_array = Array1::from_vec(target_array);

    Ok((features_array, target_array))
}
