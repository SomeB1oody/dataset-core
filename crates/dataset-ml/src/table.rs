//! Column storage for a parsed dataset.
//!
//! Every loader in this crate parses its source into a [`Table`] and returns
//! that table. This module holds the table and the two types it is built from.
//!
//! # Contents
//!
//! - [`ColumnData`] holds the values of one column, in the type the source
//!   uses. It has one variant per storage type: [`Numeric`](ColumnData::Numeric),
//!   [`Integer`](ColumnData::Integer), [`String`](ColumnData::String), and
//!   [`Bytes`](ColumnData::Bytes).
//! - [`Column`] adds the name that the source gives those values.
//! - [`Table`] holds one [`Column`] per source column. It checks the columns
//!   when it builds them, and it finds a column by name.
//!
//! # How a loader fills a table
//!
//! A loader builds one [`Column`] for each column of its source, then passes
//! them all to [`Table::new`]. The table keeps them in source order. The loader
//! stores each value in the type the source uses, and applies no encoding. The
//! choice between an ordinal code and a one-hot code stays with the caller.
//!
//! # How a caller reads a table
//!
//! [`Table::column`] finds a single column by name, and the `as_*` methods of
//! [`Column`] read its values in their source type.
//! [`Table::numeric_matrix`] builds one `f64` matrix out of the columns the
//! caller names, in the order the caller names them.
//!
//! Each loader lists the names of its columns in associated constants, such as
//! `Iris::FEATURE_NAMES` and `Iris::TARGET`. Pass one of these constants to
//! [`Table::numeric_matrix`], or name the columns directly.
//!
//! # Guarantees
//!
//! [`Table::new`] checks these three conditions before it builds a table. Every
//! method of [`Table`] relies on them:
//!
//! - The table holds at least one column.
//! - Every column holds the same number of samples.
//! - No two columns share a name.
//!
//! # Examples
//!
//! ```rust
//! use dataset_ml::table::{Column, ColumnData, Table};
//! use ndarray::array;
//!
//! let table = Table::new(
//!     "example",
//!     vec![
//!         Column::new("width", ColumnData::Numeric(array![1.0, 2.0])),
//!         Column::new("height", ColumnData::Numeric(array![3.0, 4.0])),
//!         Column::new(
//!             "species",
//!             ColumnData::String(array!["a".to_string(), "b".to_string()]),
//!         ),
//!     ],
//! )
//! .unwrap();
//!
//! assert_eq!(table.n_samples(), 2);
//!
//! // Name the columns you want, in the order you want them.
//! let matrix = table.numeric_matrix(&["height", "width"]).unwrap();
//! assert_eq!(matrix.shape(), &[2, 2]);
//! assert_eq!(matrix.row(0).to_vec(), vec![3.0, 1.0]);
//!
//! // Reach one column by name, whatever its position.
//! let species = table.column("species").unwrap().as_string().unwrap();
//! assert_eq!(species[0], "a");
//! ```

use dataset_core::DatasetError;
use ndarray::{Array1, Array2};

/// The values of one column, in the type the source uses.
///
/// Every variant except [`ColumnData::Bytes`] holds one value per sample.
/// `Bytes` holds one **row** per sample, for a source whose columns have no
/// individual names.
///
/// # Examples
///
/// ```rust
/// use dataset_ml::table::ColumnData;
/// use ndarray::array;
///
/// let counts = ColumnData::Integer(array![3, 5, 8]);
/// assert_eq!(counts.len(), 3);
/// assert_eq!(counts.kind(), "integer");
/// assert_eq!(counts.width(), 1);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnData {
    /// A value with a fraction. A missing value is `f64::NAN`.
    Numeric(Array1<f64>),
    /// A whole number.
    Integer(Array1<i64>),
    /// Text, spelled as the source spells it. A missing value is the empty
    /// string, unless the dataset documents its own token.
    String(Array1<String>),
    /// A fixed-width row of unsigned bytes per sample, such as the pixels of an
    /// image. The columns inside the row have no individual names.
    Bytes(Array2<u8>),
}

impl ColumnData {
    /// The number of samples the column holds.
    ///
    /// # Returns
    ///
    /// - `usize` - the sample count. For [`ColumnData::Bytes`], this is the
    ///   number of rows, not the number of bytes.
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Numeric(values) => values.len(),
            ColumnData::Integer(values) => values.len(),
            ColumnData::String(values) => values.len(),
            ColumnData::Bytes(values) => values.nrows(),
        }
    }

    /// Whether the column holds no sample.
    ///
    /// # Returns
    ///
    /// - `bool` - `true` if the column holds no sample, and `false` if it holds
    ///   at least one.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The name of the variant this column holds.
    ///
    /// # Returns
    ///
    /// - `&'static str` - one of `"numeric"`, `"integer"`, `"string"`, or
    ///   `"bytes"`.
    ///
    /// # Notes
    ///
    /// [`Table::numeric_matrix`] puts this name in the error it returns for a
    /// column it cannot read as a number.
    pub fn kind(&self) -> &'static str {
        match self {
            ColumnData::Numeric(_) => "numeric",
            ColumnData::Integer(_) => "integer",
            ColumnData::String(_) => "string",
            ColumnData::Bytes(_) => "bytes",
        }
    }

    /// The number of values this column contributes to one row of a matrix.
    ///
    /// # Returns
    ///
    /// - `usize` - the row width. Every variant contributes `1`, except
    ///   [`ColumnData::Bytes`], which contributes the number of bytes in one of
    ///   its rows.
    pub fn width(&self) -> usize {
        match self {
            ColumnData::Bytes(values) => values.ncols(),
            _ => 1,
        }
    }
}

/// One named column of a [`Table`].
///
/// A column pairs the values of one source column with the name that the source
/// gives it. The name is fixed when the column is built.
///
/// # Examples
///
/// ```rust
/// use dataset_ml::table::{Column, ColumnData};
/// use ndarray::array;
///
/// let column = Column::new("petal_width", ColumnData::Numeric(array![0.2, 1.4]));
///
/// assert_eq!(column.name(), "petal_width");
/// assert_eq!(column.len(), 2);
/// assert_eq!(column.as_numeric().unwrap()[0], 0.2);
///
/// // An `as_*` method returns `None` for every other variant.
/// assert!(column.as_string().is_none());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    /// The name the source gives this column.
    name: &'static str,
    /// The values in this column.
    data: ColumnData,
}

impl Column {
    /// Build a column from its name and its values.
    ///
    /// # Parameters
    ///
    /// - `name` - The name the source gives the column.
    /// - `data` - The values in the column.
    ///
    /// # Returns
    ///
    /// - `Self` - the new column.
    pub fn new(name: &'static str, data: ColumnData) -> Self {
        Column { name, data }
    }

    /// The column's name.
    ///
    /// # Returns
    ///
    /// - `&'static str` - the name, as the source spells it.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The column's values.
    ///
    /// # Returns
    ///
    /// - `&ColumnData` - a shared reference to the values.
    pub fn data(&self) -> &ColumnData {
        &self.data
    }

    /// The column's values, for in-place editing.
    ///
    /// # Returns
    ///
    /// - `&mut ColumnData` - a mutable reference to the values.
    ///
    /// # Notes
    ///
    /// Change a value, but do not change the number of samples. A [`Table`] that
    /// holds this column relies on every column having the same length.
    pub fn data_mut(&mut self) -> &mut ColumnData {
        &mut self.data
    }

    /// The number of samples the column holds.
    ///
    /// # Returns
    ///
    /// - `usize` - the sample count.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the column holds no sample.
    ///
    /// # Returns
    ///
    /// - `bool` - `true` if the column holds no sample, and `false` if it holds
    ///   at least one.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The values, if the column is [`ColumnData::Numeric`].
    ///
    /// # Returns
    ///
    /// - `Some(&Array1<f64>)` - the values, one per sample.
    /// - `None` - if the column holds another variant.
    pub fn as_numeric(&self) -> Option<&Array1<f64>> {
        match &self.data {
            ColumnData::Numeric(values) => Some(values),
            _ => None,
        }
    }

    /// The values, if the column is [`ColumnData::Integer`].
    ///
    /// # Returns
    ///
    /// - `Some(&Array1<i64>)` - the values, one per sample.
    /// - `None` - if the column holds another variant.
    pub fn as_integer(&self) -> Option<&Array1<i64>> {
        match &self.data {
            ColumnData::Integer(values) => Some(values),
            _ => None,
        }
    }

    /// The values, if the column is [`ColumnData::String`].
    ///
    /// # Returns
    ///
    /// - `Some(&Array1<String>)` - the values, one per sample.
    /// - `None` - if the column holds another variant.
    pub fn as_string(&self) -> Option<&Array1<String>> {
        match &self.data {
            ColumnData::String(values) => Some(values),
            _ => None,
        }
    }

    /// The values, if the column is [`ColumnData::Bytes`].
    ///
    /// # Returns
    ///
    /// - `Some(&Array2<u8>)` - the rows, one per sample.
    /// - `None` - if the column holds another variant.
    pub fn as_bytes(&self) -> Option<&Array2<u8>> {
        match &self.data {
            ColumnData::Bytes(values) => Some(values),
            _ => None,
        }
    }

    /// The values as `f64`, one per sample.
    ///
    /// # Returns
    ///
    /// - `Some(Array1<f64>)` - the values. [`ColumnData::Numeric`] returns them
    ///   unchanged, and [`ColumnData::Integer`] converts each one.
    /// - `None` - if the column is [`ColumnData::String`], which has no numeric
    ///   reading, or [`ColumnData::Bytes`], which holds more than one value per
    ///   sample. For a `Bytes` column, use [`Table::numeric_matrix`].
    ///
    /// # Notes
    ///
    /// An `i64` above 2^53 loses precision as an `f64`.
    pub fn to_numeric(&self) -> Option<Array1<f64>> {
        match &self.data {
            ColumnData::Numeric(values) => Some(values.clone()),
            ColumnData::Integer(values) => Some(values.mapv(|value| value as f64)),
            _ => None,
        }
    }
}

/// A parsed dataset: named columns of equal length.
///
/// A table holds one [`Column`] per source column, in source order, together
/// with the dataset name. [`Table::new`] checks the columns, so every table a
/// caller receives meets the guarantees in the [module documentation](self).
///
/// # Examples
///
/// ```rust
/// use dataset_ml::table::{Column, ColumnData, Table};
/// use ndarray::array;
///
/// let table = Table::new(
///     "example",
///     vec![
///         Column::new("id", ColumnData::Integer(array![1, 2])),
///         Column::new("width", ColumnData::Numeric(array![1.5, 2.5])),
///     ],
/// )
/// .unwrap();
///
/// assert_eq!(table.name(), "example");
/// assert_eq!(table.n_samples(), 2);
/// assert_eq!(table.n_columns(), 2);
/// assert_eq!(table.names().collect::<Vec<_>>(), vec!["id", "width"]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    /// The dataset name. It appears in the errors this table returns.
    name: &'static str,
    /// The columns, in the order the source lists them.
    columns: Vec<Column>,
    /// The number of samples every column holds.
    n_samples: usize,
}

impl Table {
    /// Build a table and check its columns.
    ///
    /// # Parameters
    ///
    /// - `name` - The dataset name. It appears in the errors this table
    ///   returns.
    /// - `columns` - The columns, in the order the source lists them.
    ///
    /// # Returns
    ///
    /// - `Self` - the new table, if the columns pass every check.
    ///
    /// # Errors
    ///
    /// - `DatasetError::DataFormatError` with `EmptyDataset` - if `columns` is
    ///   empty, or if the columns hold no sample.
    /// - `DatasetError::DataFormatError` with `LengthMismatch` - if two columns
    ///   hold a different number of samples.
    /// - `DatasetError::DataFormatError` with `InvalidValue` - if two columns
    ///   share a name.
    pub fn new(name: &'static str, columns: Vec<Column>) -> Result<Self, DatasetError> {
        let Some(first) = columns.first() else {
            return Err(DatasetError::empty_dataset(name));
        };

        let n_samples = first.len();
        if n_samples == 0 {
            return Err(DatasetError::empty_dataset(name));
        }

        for column in &columns {
            if column.len() != n_samples {
                return Err(DatasetError::length_mismatch(
                    name,
                    column.name(),
                    n_samples,
                    column.len(),
                ));
            }
        }

        for (index, column) in columns.iter().enumerate() {
            if columns[..index]
                .iter()
                .any(|other| other.name() == column.name())
            {
                return Err(DatasetError::invalid_value(
                    name,
                    "column name",
                    column.name(),
                    index + 1,
                ));
            }
        }

        Ok(Table {
            name,
            columns,
            n_samples,
        })
    }

    /// The dataset name.
    ///
    /// # Returns
    ///
    /// - `&'static str` - the name the loader passed to [`Table::new`].
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The number of samples every column holds.
    ///
    /// # Returns
    ///
    /// - `usize` - the sample count. It is always at least `1`.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// The number of columns.
    ///
    /// # Returns
    ///
    /// - `usize` - the column count. It is always at least `1`.
    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// Every column, in source order.
    ///
    /// # Returns
    ///
    /// - `&[Column]` - a shared slice of every column.
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Every column, in source order, for in-place editing.
    ///
    /// # Returns
    ///
    /// - `&mut [Column]` - a mutable slice of every column.
    ///
    /// # Notes
    ///
    /// A change to a value keeps the table valid. Do not change the length of a
    /// column: the table's guarantees no longer hold if you do.
    pub fn columns_mut(&mut self) -> &mut [Column] {
        &mut self.columns
    }

    /// Every column name, in source order.
    ///
    /// # Returns
    ///
    /// - `impl Iterator<Item = &'static str>` - the names, in source order.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.columns.iter().map(Column::name)
    }

    /// Find one column by name.
    ///
    /// # Parameters
    ///
    /// - `name` - The name to look for. The comparison is exact.
    ///
    /// # Returns
    ///
    /// - `Some(&Column)` - the column of that name.
    /// - `None` - if the table holds no column of that name.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|column| column.name() == name)
    }

    /// Find one column by name, for in-place editing.
    ///
    /// # Parameters
    ///
    /// - `name` - The name to look for. The comparison is exact.
    ///
    /// # Returns
    ///
    /// - `Some(&mut Column)` - the column of that name.
    /// - `None` - if the table holds no column of that name.
    ///
    /// # Notes
    ///
    /// A change to a value keeps the table valid. Do not change the length of
    /// the column: the table's guarantees no longer hold if you do.
    pub fn column_mut(&mut self, name: &str) -> Option<&mut Column> {
        self.columns.iter_mut().find(|column| column.name() == name)
    }

    /// Build one `f64` matrix out of the named columns.
    ///
    /// The matrix keeps the order of `names`, which does not have to be the
    /// source order. A name may repeat, and the matrix then holds that column
    /// once per mention.
    ///
    /// # Parameters
    ///
    /// - `names` - The columns to put in the matrix, in the order you want
    ///   them. A [`ColumnData::Bytes`] column contributes its full row width,
    ///   and every other column contributes one value.
    ///
    /// # Returns
    ///
    /// - `Array2<f64>` - a matrix of [`Table::n_samples`] rows. Its width is the
    ///   sum of the [`ColumnData::width`] of every named column.
    ///
    /// # Errors
    ///
    /// - `DatasetError::DataFormatError` with `LengthMismatch` - if `names` is
    ///   empty.
    /// - `DatasetError::DataFormatError` with `UnknownColumn` - if the table
    ///   holds no column of a given name.
    /// - `DatasetError::DataFormatError` with `ColumnTypeMismatch` - if a named
    ///   column is [`ColumnData::String`], which has no numeric reading.
    ///
    /// # Performance
    ///
    /// This builds a new matrix on every call, and that matrix holds
    /// `n_samples × width` values. Call it once and keep the result.
    pub fn numeric_matrix(&self, names: &[&str]) -> Result<Array2<f64>, DatasetError> {
        if names.is_empty() {
            return Err(DatasetError::length_mismatch(
                self.name,
                "requested columns",
                1,
                0,
            ));
        }

        let mut selected = Vec::with_capacity(names.len());
        for name in names {
            let Some(column) = self.column(name) else {
                return Err(DatasetError::unknown_column(self.name, name));
            };
            selected.push(column);
        }

        let width: usize = selected.iter().map(|column| column.data().width()).sum();
        let mut values: Vec<f64> = Vec::with_capacity(self.n_samples * width);

        for row in 0..self.n_samples {
            for column in &selected {
                match column.data() {
                    ColumnData::Numeric(source) => values.push(source[row]),
                    ColumnData::Integer(source) => values.push(source[row] as f64),
                    ColumnData::Bytes(source) => {
                        for col in 0..source.ncols() {
                            values.push(f64::from(source[[row, col]]));
                        }
                    }
                    other => {
                        return Err(DatasetError::column_type_mismatch(
                            self.name,
                            column.name(),
                            "numeric",
                            other.kind(),
                        ));
                    }
                }
            }
        }

        Array2::from_shape_vec((self.n_samples, width), values)
            .map_err(|e| DatasetError::array_shape_error(self.name, "numeric matrix", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn numeric(name: &'static str, values: [f64; 3]) -> Column {
        Column::new(name, ColumnData::Numeric(Array1::from_vec(values.to_vec())))
    }

    fn sample_table() -> Table {
        Table::new(
            "sample",
            vec![
                numeric("a", [1.0, 2.0, 3.0]),
                numeric("b", [4.0, 5.0, 6.0]),
                Column::new(
                    "label",
                    ColumnData::String(array!["x".into(), "y".into(), "x".into()]),
                ),
            ],
        )
        .unwrap()
    }

    #[test]
    fn new_rejects_an_empty_column_list() {
        assert!(Table::new("t", vec![]).is_err());
    }

    #[test]
    fn new_rejects_zero_samples() {
        let column = Column::new("a", ColumnData::Numeric(Array1::zeros(0)));
        assert!(Table::new("t", vec![column]).is_err());
    }

    #[test]
    fn new_rejects_columns_of_different_lengths() {
        let short = Column::new("a", ColumnData::Numeric(array![1.0, 2.0]));
        let long = Column::new("b", ColumnData::Numeric(array![1.0, 2.0, 3.0]));
        let error = Table::new("t", vec![short, long]).unwrap_err().to_string();
        assert!(error.contains("expected 2"), "{error}");
    }

    #[test]
    fn new_rejects_a_repeated_name() {
        let one = numeric("a", [1.0, 2.0, 3.0]);
        let two = numeric("a", [4.0, 5.0, 6.0]);
        assert!(Table::new("t", vec![one, two]).is_err());
    }

    #[test]
    fn a_table_reports_its_name_shape_and_column_names() {
        let table = sample_table();
        assert_eq!(table.name(), "sample");
        assert_eq!(table.n_samples(), 3);
        assert_eq!(table.n_columns(), 3);
        assert_eq!(table.names().collect::<Vec<_>>(), vec!["a", "b", "label"]);
    }

    #[test]
    fn column_lookup_is_by_name_not_position() {
        let table = sample_table();
        assert_eq!(table.column("b").unwrap().as_numeric().unwrap()[0], 4.0);
        assert!(table.column("missing").is_none());
    }

    #[test]
    fn numeric_matrix_keeps_the_requested_order() {
        let table = sample_table();
        let matrix = table.numeric_matrix(&["b", "a"]).unwrap();
        assert_eq!(matrix.shape(), &[3, 2]);
        assert_eq!(matrix.row(0).to_vec(), vec![4.0, 1.0]);
        assert_eq!(matrix.row(2).to_vec(), vec![6.0, 3.0]);
    }

    #[test]
    fn numeric_matrix_takes_a_subset() {
        let table = sample_table();
        let matrix = table.numeric_matrix(&["a"]).unwrap();
        assert_eq!(matrix.shape(), &[3, 1]);
    }

    #[test]
    fn numeric_matrix_repeats_a_repeated_name() {
        let table = sample_table();
        let matrix = table.numeric_matrix(&["a", "a"]).unwrap();
        assert_eq!(matrix.shape(), &[3, 2]);
        assert_eq!(matrix.row(1).to_vec(), vec![2.0, 2.0]);
    }

    #[test]
    fn numeric_matrix_converts_integers() {
        let table = Table::new(
            "t",
            vec![
                Column::new("count", ColumnData::Integer(array![1, 2])),
                Column::new("when", ColumnData::Integer(array![10, 20])),
            ],
        )
        .unwrap();
        let matrix = table.numeric_matrix(&["count", "when"]).unwrap();
        assert_eq!(matrix.row(1).to_vec(), vec![2.0, 20.0]);
    }

    #[test]
    fn numeric_matrix_expands_a_bytes_column_to_its_width() {
        let pixels = Array2::from_shape_vec((2, 3), vec![1u8, 2, 3, 4, 5, 6]).unwrap();
        let table =
            Table::new("t", vec![Column::new("pixels", ColumnData::Bytes(pixels))]).unwrap();
        let matrix = table.numeric_matrix(&["pixels"]).unwrap();
        assert_eq!(matrix.shape(), &[2, 3]);
        assert_eq!(matrix.row(1).to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn numeric_matrix_rejects_an_empty_request() {
        let table = sample_table();
        let error = table.numeric_matrix(&[]).unwrap_err().to_string();
        assert!(error.contains("requested columns"), "{error}");
    }

    #[test]
    fn numeric_matrix_rejects_an_unknown_name() {
        let table = sample_table();
        let error = table.numeric_matrix(&["a", "missing"]).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("no column named `missing`"), "{message}");
    }

    #[test]
    fn numeric_matrix_rejects_a_string_column() {
        let table = sample_table();
        let message = table.numeric_matrix(&["label"]).unwrap_err().to_string();
        assert!(message.contains("`label`"), "{message}");
        assert!(message.contains("expected `numeric`"), "{message}");
    }

    #[test]
    fn numeric_matrix_names_the_string_column_wherever_it_sits() {
        let table = sample_table();
        // The offending column is neither first nor last in the request.
        let message = table
            .numeric_matrix(&["a", "label", "b"])
            .unwrap_err()
            .to_string();
        assert!(message.contains("`label`"), "{message}");
        assert!(message.contains("`string`"), "{message}");
    }

    #[test]
    fn to_numeric_reads_numbers_and_refuses_the_rest() {
        let pixels = Array2::<u8>::zeros((3, 4));
        let table = Table::new(
            "t",
            vec![
                numeric("a", [1.0, 2.0, 3.0]),
                Column::new("count", ColumnData::Integer(array![1, 2, 3])),
                Column::new(
                    "label",
                    ColumnData::String(Array1::from_vec(vec!["x".to_string(); 3])),
                ),
                Column::new("pixels", ColumnData::Bytes(pixels)),
            ],
        )
        .unwrap();
        assert_eq!(table.column("a").unwrap().to_numeric().unwrap()[0], 1.0);
        assert_eq!(table.column("count").unwrap().to_numeric().unwrap()[2], 3.0);
        assert!(table.column("label").unwrap().to_numeric().is_none());
        assert!(table.column("pixels").unwrap().to_numeric().is_none());
    }

    #[test]
    fn column_mut_edits_in_place() {
        let mut table = sample_table();
        if let Some(ColumnData::Numeric(values)) = table.column_mut("a").map(Column::data_mut) {
            values[0] = 99.0;
        }
        assert_eq!(table.column("a").unwrap().as_numeric().unwrap()[0], 99.0);
    }

    #[test]
    fn width_is_one_except_for_bytes() {
        assert_eq!(ColumnData::Numeric(array![1.0]).width(), 1);
        assert_eq!(ColumnData::Integer(array![1]).width(), 1);
        assert_eq!(ColumnData::String(array!["x".to_string()]).width(), 1);
        let pixels = Array2::<u8>::zeros((1, 5));
        assert_eq!(ColumnData::Bytes(pixels).width(), 5);
    }
}
