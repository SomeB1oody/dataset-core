//! Wine Quality dataset.
//!
//! Physicochemical measurements and quality scores for Portuguese "Vinho Verde"
//! red and white wines, commonly used for regression or ordinal classification.
//!
//! **Columns (12):**
//!
//! | Name                   | Type      | Description                        |
//! |------------------------|-----------|-------------------------------------|
//! | `fixed acidity`        | `Numeric` | fixed acidity                      |
//! | `volatile acidity`     | `Numeric` | volatile acidity                   |
//! | `citric acid`          | `Numeric` | citric acid                        |
//! | `residual sugar`       | `Numeric` | residual sugar                     |
//! | `chlorides`            | `Numeric` | chlorides                          |
//! | `free sulfur dioxide`  | `Numeric` | free sulfur dioxide                |
//! | `total sulfur dioxide` | `Numeric` | total sulfur dioxide               |
//! | `density`              | `Numeric` | density                            |
//! | `pH`                   | `Numeric` | pH                                 |
//! | `sulphates`            | `Numeric` | sulphates                          |
//! | `alcohol`              | `Numeric` | alcohol                            |
//! | `quality`              | `Numeric` | quality score between `0` and `10` |
//!
//! Each subset designates the first 11 columns as the inputs
//! (`FEATURE_NAMES`) and `quality` as the label (`TARGET`).
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

use crate::table::{Column, ColumnData, Table};
use csv::ReaderBuilder;
use dataset_core::DatasetError;
use ndarray::Array1;
use serde::Deserialize;

/// Number of feature columns each Wine Quality subset holds.
const N_FEATURES: usize = 11;

/// The columns each Wine Quality subset designates as the model inputs, in
/// source order.
///
/// Each subset's struct carries its own copy of this constant.
const FEATURE_NAMES: [&str; N_FEATURES] = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
];

/// The column each Wine Quality subset designates as the label.
///
/// Each subset's struct carries its own copy of this constant.
const TARGET: &str = "quality";

/// One CSV record of a Wine Quality file (red or white): 11 `f64` feature
/// columns followed by the `quality` target.
///
/// This struct declares its fields in CSV column order. The parser
/// deserializes them **positionally** and disables csv's header handling, so
/// the struct does not depend on the header spelling.
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

/// Parses a single Wine Quality CSV (red or white) into a [`Table`].
///
/// The CSV must be `;`-separated with a **header row**, followed by data rows.
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
/// - `Table` - the 12 columns listed in the [module documentation](self).
///
/// # Errors
///
/// Returns `DatasetError` if:
/// - Any row has an unexpected number of columns
/// - Any feature/target value fails to parse as `f64`
/// - The file holds no data row
fn parse_wine_data_to_table<R: std::io::Read>(
    dataset_name: &'static str,
    reader: R,
) -> Result<Table, DatasetError> {
    let mut rdr = ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(false)
        .from_reader(reader);

    let mut fixed_acidity = Vec::new();
    let mut volatile_acidity = Vec::new();
    let mut citric_acid = Vec::new();
    let mut residual_sugar = Vec::new();
    let mut chlorides = Vec::new();
    let mut free_sulfur_dioxide = Vec::new();
    let mut total_sulfur_dioxide = Vec::new();
    let mut density = Vec::new();
    let mut ph = Vec::new();
    let mut sulphates = Vec::new();
    let mut alcohol = Vec::new();
    let mut quality = Vec::new();

    for result in rdr.deserialize::<WineRecord>().skip(1) {
        let record = result.map_err(|e| DatasetError::csv_read_error(dataset_name, e))?;

        fixed_acidity.push(record.fixed_acidity);
        volatile_acidity.push(record.volatile_acidity);
        citric_acid.push(record.citric_acid);
        residual_sugar.push(record.residual_sugar);
        chlorides.push(record.chlorides);
        free_sulfur_dioxide.push(record.free_sulfur_dioxide);
        total_sulfur_dioxide.push(record.total_sulfur_dioxide);
        density.push(record.density);
        ph.push(record.ph);
        sulphates.push(record.sulphates);
        alcohol.push(record.alcohol);
        quality.push(record.quality);
    }

    Table::new(
        dataset_name,
        vec![
            Column::new(
                FEATURE_NAMES[0],
                ColumnData::Numeric(Array1::from_vec(fixed_acidity)),
            ),
            Column::new(
                FEATURE_NAMES[1],
                ColumnData::Numeric(Array1::from_vec(volatile_acidity)),
            ),
            Column::new(
                FEATURE_NAMES[2],
                ColumnData::Numeric(Array1::from_vec(citric_acid)),
            ),
            Column::new(
                FEATURE_NAMES[3],
                ColumnData::Numeric(Array1::from_vec(residual_sugar)),
            ),
            Column::new(
                FEATURE_NAMES[4],
                ColumnData::Numeric(Array1::from_vec(chlorides)),
            ),
            Column::new(
                FEATURE_NAMES[5],
                ColumnData::Numeric(Array1::from_vec(free_sulfur_dioxide)),
            ),
            Column::new(
                FEATURE_NAMES[6],
                ColumnData::Numeric(Array1::from_vec(total_sulfur_dioxide)),
            ),
            Column::new(
                FEATURE_NAMES[7],
                ColumnData::Numeric(Array1::from_vec(density)),
            ),
            Column::new(FEATURE_NAMES[8], ColumnData::Numeric(Array1::from_vec(ph))),
            Column::new(
                FEATURE_NAMES[9],
                ColumnData::Numeric(Array1::from_vec(sulphates)),
            ),
            Column::new(
                FEATURE_NAMES[10],
                ColumnData::Numeric(Array1::from_vec(alcohol)),
            ),
            Column::new(TARGET, ColumnData::Numeric(Array1::from_vec(quality))),
        ],
    )
}
