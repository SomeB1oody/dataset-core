use zip::result::ZipError;

/// Specific kinds of data format errors that can occur during dataset parsing.
///
/// # Variants
///
/// - `CsvReadError` - Failed to read a CSV record.
/// - `InvalidColumnCount` - The row has an unexpected number of columns.
/// - `ParseFailed` - Failed to parse a field value into the target type.
/// - `InvalidValue` - The field value is syntactically valid but semantically incorrect.
/// - `LengthMismatch` - The total parsed data length doesn't match expected dimensions.
/// - `ArrayShapeError` - Failed to construct ndarray with the given shape and data.
#[derive(Debug, Clone)]
pub enum DataFormatErrorKind {
    /// Failed to read a CSV record
    CsvReadError {
        /// Dataset identifier
        dataset_name: String,
        /// The underlying CSV error message
        error: String,
    },
    /// The row has an unexpected number of columns
    InvalidColumnCount {
        /// Dataset identifier
        dataset_name: String,
        /// Expected number of columns
        expected: usize,
        /// Actual number of columns found
        actual: usize,
        /// Line number (1-based)
        line_num: usize,
        /// The original input line
        line: String,
    },
    /// Failed to parse a field value into the target type
    ParseFailed {
        /// Dataset identifier
        dataset_name: String,
        /// Field name that failed to parse
        field_name: String,
        /// Line number (1-based)
        line_num: usize,
        /// The original input line
        line: String,
        /// The underlying parse error message
        error: String,
    },
    /// The field value is syntactically valid but semantically incorrect
    InvalidValue {
        /// Dataset identifier
        dataset_name: String,
        /// Field name with invalid value
        field_name: String,
        /// The invalid value
        value: String,
        /// Line number (1-based)
        line_num: usize,
        /// The original input line
        line: String,
    },
    /// The total parsed data length doesn't match expected dimensions
    LengthMismatch {
        /// Dataset identifier
        dataset_name: String,
        /// Field name whose length is being validated
        field_name: String,
        /// Expected length
        expected: usize,
        /// Actual length
        actual: usize,
    },
    /// Failed to construct ndarray with the given shape and data
    ArrayShapeError {
        /// Dataset identifier
        dataset_name: String,
        /// Array name that failed to build
        array_name: String,
        /// The underlying shape error message
        error: String,
    },
}

impl std::fmt::Display for DataFormatErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataFormatErrorKind::CsvReadError { dataset_name, error } => {
                write!(f, "[{}] failed to read CSV record: {}", dataset_name, error)
            }
            DataFormatErrorKind::InvalidColumnCount {
                dataset_name,
                expected,
                actual,
                line_num,
                line,
            } => {
                write!(
                    f,
                    "[{}] invalid column count at line {}: expected {}, got {} (line: `{}`)",
                    dataset_name, line_num, expected, actual, line
                )
            }
            DataFormatErrorKind::ParseFailed {
                dataset_name,
                field_name,
                line_num,
                line,
                error,
            } => {
                write!(
                    f,
                    "[{}] failed to parse `{}` at line {}: {} (line: `{}`)",
                    dataset_name, field_name, line_num, error, line
                )
            }
            DataFormatErrorKind::InvalidValue {
                dataset_name,
                field_name,
                value,
                line_num,
                line,
            } => {
                write!(
                    f,
                    "[{}] invalid value for `{}` at line {}: `{}` (line: `{}`)",
                    dataset_name, field_name, line_num, value, line
                )
            }
            DataFormatErrorKind::LengthMismatch {
                dataset_name,
                field_name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "[{}] invalid `{}` length: expected {}, got {}",
                    dataset_name, field_name, expected, actual
                )
            }
            DataFormatErrorKind::ArrayShapeError {
                dataset_name,
                array_name,
                error,
            } => {
                write!(
                    f,
                    "[{}] failed to build `{}` array: {}",
                    dataset_name, array_name, error
                )
            }
        }
    }
}

/// Error type used by dataset loading utilities.
///
/// # Variants
///
/// - `DownloadError` - The download step failed (network, invalid URL, or downloader configuration).
/// - `ValidationError` - Downloaded file content failed integrity validation (SHA256 mismatch).
/// - `UnzipError` - Extracting a zip archive failed.
/// - `IoError` - A standard I/O operation failed (reading directories, opening/removing files, etc.).
/// - `DataFormatError` - The dataset content was not in the expected format.
#[derive(Debug)]
pub enum DatasetError {
    DownloadError(downloader::Error),
    ValidationError(String),
    UnzipError(ZipError),
    IoError(std::io::Error),
    DataFormatError(DataFormatErrorKind),
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::DownloadError(e) => write!(f, "Download error: {}", e),
            DatasetError::ValidationError(e) => write!(f, "Validation error: {}", e),
            DatasetError::UnzipError(e) => write!(f, "Unzip error: {}", e),
            DatasetError::IoError(e) => write!(f, "I/O error: {}", e),
            DatasetError::DataFormatError(e) => write!(f, "Data format error: {}", e),
        }
    }
}

impl std::error::Error for DatasetError {}

impl From<downloader::Error> for DatasetError {
    fn from(error: downloader::Error) -> Self {
        Self::DownloadError(error)
    }
}

impl From<ZipError> for DatasetError {
    fn from(error: ZipError) -> Self {
        Self::UnzipError(error)
    }
}

impl From<std::io::Error> for DatasetError {
    fn from(error: std::io::Error) -> Self {
        Self::IoError(error)
    }
}

impl DatasetError {
    /// Creates a standard SHA256 validation failure error message for a file.
    ///
    /// # Parameters
    ///
    /// - `dataset_name` - The dataset identifier used in the error prefix.
    /// - `file_name` - The dataset file name that failed checksum validation.
    ///
    /// # Returns
    ///
    /// - `DatasetError::ValidationError` - A variant of `DatasetError` that contains the unified SHA256 failure message.
    pub fn sha256_validation_failed(dataset_name: &str, file_name: &str) -> Self {
        Self::ValidationError(format!(
            "[{}] SHA256 validation failed for file `{}`",
            dataset_name, file_name
        ))
    }

    /// Creates a CSV read error.
    ///
    /// # Parameters
    ///
    /// - `dataset_name` - The dataset identifier.
    /// - `error` - The underlying CSV error.
    ///
    /// # Returns
    ///
    /// - `DatasetError::DataFormatError(DataFormatErrorKind::CsvReadError)` - A variant of `DatasetError` describing the CSV read error.
    pub fn csv_read_error(dataset_name: &str, error: impl std::fmt::Display) -> Self {
        Self::DataFormatError(
            DataFormatErrorKind::CsvReadError {
                dataset_name: dataset_name.to_string(),
                error: error.to_string(),
        })
    }

    /// Creates a unified invalid-column-count data format error.
    ///
    /// # Parameters
    ///
    /// - `dataset_name` - The dataset identifier used in the error prefix.
    /// - `expected` - The expected number of columns.
    /// - `actual` - The actual number of columns found.
    /// - `line_num` - The line number (1-based) where the error occurred.
    /// - `line` - The original input line that failed validation.
    ///
    /// # Returns
    ///
    /// - `DatasetError::DataFormatError(DataFormatErrorKind::InvalidColumnCount)` - A variant of `DatasetError` describing the column count mismatch.
    pub fn invalid_column_count(dataset_name: &str, expected: usize, actual: usize, line_num: usize, line: &str) -> Self {
        Self::DataFormatError(
            DataFormatErrorKind::InvalidColumnCount {
                dataset_name: dataset_name.to_string(),
                expected,
                actual,
                line_num,
                line: line.to_string(),
            }
        )
    }

    /// Creates a unified parse failure data format error.
    ///
    /// # Parameters
    ///
    /// - `dataset_name` - The dataset identifier used in the error prefix.
    /// - `field_name` - The logical field name that failed to parse.
    /// - `line_num` - The line number (1-based) where the error occurred.
    /// - `line` - The original input line where parsing failed.
    /// - `err` - The underlying parser error detail.
    ///
    /// # Returns
    ///
    /// - `DatasetError::DataFormatError(DataFormatErrorKind::ParseFailed)` - A variant of `DatasetError` describing the parse failure.
    pub fn parse_failed(
        dataset_name: &str,
        field_name: &str,
        line_num: usize,
        line: &str,
        err: impl std::fmt::Display,
    ) -> Self {
        Self::DataFormatError(
            DataFormatErrorKind::ParseFailed {
                dataset_name: dataset_name.to_string(),
                field_name: field_name.to_string(),
                line_num,
                line: line.to_string(),
                error: err.to_string(),
            }
        )
    }

    /// Creates a unified invalid-field-value data format error.
    ///
    /// # Parameters
    ///
    /// - `dataset_name` - The dataset identifier used in the error prefix.
    /// - `field_name` - The logical field name with an invalid value.
    /// - `value` - The invalid raw value.
    /// - `line_num` - The line number (1-based) where the error occurred.
    /// - `line` - The original input line where the invalid value was found.
    ///
    /// # Returns
    ///
    /// - `DatasetError::DataFormatError(DataFormatErrorKind::InvalidValue)` - A variant of `DatasetError` describing the invalid value.
    pub fn invalid_value(dataset_name: &str, field_name: &str, value: &str, line_num: usize, line: &str) -> Self {
        Self::DataFormatError(
            DataFormatErrorKind::InvalidValue {
                dataset_name: dataset_name.to_string(),
                field_name: field_name.to_string(),
                value: value.to_string(),
                line_num,
                line: line.to_string(),
            }
        )
    }

    /// Creates a unified vector/row length mismatch data format error.
    ///
    /// # Parameters
    ///
    /// - `dataset_name` - The dataset identifier used in the error prefix.
    /// - `field_name` - The logical field name whose length is being validated.
    /// - `expected` - The expected length.
    /// - `actual` - The actual length.
    ///
    /// # Returns
    ///
    /// - `DatasetError::DataFormatError(DataFormatErrorKind::LengthMismatch)` - A variant of `DatasetError` describing the length mismatch.
    pub fn length_mismatch(dataset_name: &str, field_name: &str, expected: usize, actual: usize) -> Self {
        Self::DataFormatError(
            DataFormatErrorKind::LengthMismatch {
                dataset_name: dataset_name.to_string(),
                field_name: field_name.to_string(),
                expected,
                actual,
            }
        )
    }

    /// Creates a unified ndarray shape construction data format error.
    ///
    /// # Parameters
    ///
    /// - `dataset_name` - The dataset identifier used in the error prefix.
    /// - `array_name` - The logical array name that failed to build.
    /// - `err` - The underlying ndarray shape construction error detail.
    ///
    /// # Returns
    ///
    /// - `DatasetError::DataFormatError(DataFormatErrorKind::ArrayShapeError)` - A variant of `DatasetError` describing the array shape failure.
    pub fn array_shape_error(dataset_name: &str, array_name: &str, err: impl std::fmt::Display) -> Self {
        Self::DataFormatError(
            DataFormatErrorKind::ArrayShapeError {
                dataset_name: dataset_name.to_string(),
                array_name: array_name.to_string(),
                error: err.to_string(),
            }
        )
    }
}