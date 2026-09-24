use std::error::Error;
use std::fmt::{Display, Formatter};

use half::f16;

#[derive(Clone, Debug, PartialEq)]
pub struct PackedRows {
    pub(crate) codes: Vec<i8>,
    pub(crate) inverse_norms: Vec<f16>,
    pub(crate) rows: usize,
    pub(crate) dimensions: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedRowsError;

impl Display for PackedRowsError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("packed int8 row authority differs")
    }
}

impl Error for PackedRowsError {}

impl PackedRows {
    pub fn new(
        codes: Vec<i8>,
        inverse_norms: Vec<f16>,
        rows: usize,
        dimensions: usize,
    ) -> Result<Self, PackedRowsError> {
        if rows < 1
            || dimensions < 2
            || rows.checked_mul(dimensions) != Some(codes.len())
            || inverse_norms.len() != rows
            || inverse_norms
                .iter()
                .any(|value| !value.is_finite() || value.to_f32() <= 0.0)
        {
            return Err(PackedRowsError);
        }
        Ok(Self {
            codes,
            inverse_norms,
            rows,
            dimensions,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
}

pub fn scalar_scores(
    queries: &PackedRows,
    gallery: &PackedRows,
) -> Result<Vec<f32>, PackedRowsError> {
    if queries.dimensions != gallery.dimensions {
        return Err(PackedRowsError);
    }
    let mut scores = Vec::with_capacity(queries.rows * gallery.rows);
    for query_row in 0..queries.rows {
        let query_start = query_row * queries.dimensions;
        let query = &queries.codes[query_start..query_start + queries.dimensions];
        let query_norm = queries.inverse_norms[query_row].to_f32();
        for gallery_row in 0..gallery.rows {
            let gallery_start = gallery_row * gallery.dimensions;
            let gallery_values = &gallery.codes[gallery_start..gallery_start + gallery.dimensions];
            let dot = query
                .iter()
                .zip(gallery_values)
                .map(|(&left, &right)| i64::from(left) * i64::from(right))
                .sum::<i64>();
            scores.push((dot as f32) * query_norm * gallery.inverse_norms[gallery_row].to_f32());
        }
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::{PackedRows, scalar_scores};

    fn packed(codes: &[i8], inverse_norms: &[f32], rows: usize, dimensions: usize) -> PackedRows {
        PackedRows::new(
            codes.to_vec(),
            inverse_norms.iter().copied().map(f16::from_f32).collect(),
            rows,
            dimensions,
        )
        .unwrap()
    }

    #[test]
    fn valid_rows_and_signed_extremes_score_exactly() {
        let queries = packed(&[-127, 127, 1, -1], &[0.5], 1, 4);
        let gallery = packed(&[-127, 127, 1, -1, 127, -127, -1, 1], &[0.25, 0.125], 2, 4);

        let observed = scalar_scores(&queries, &gallery).unwrap();

        let positive_dot = 2 * 127i32 * 127i32 + 2;
        let negative_dot = -positive_dot;
        assert_eq!(
            observed[0].to_bits(),
            ((positive_dot as f32) * 0.5 * 0.25).to_bits()
        );
        assert_eq!(
            observed[1].to_bits(),
            ((negative_dot as f32) * 0.5 * 0.125).to_bits()
        );
    }

    #[test]
    fn scalar_score_handles_valid_width_beyond_i32_dot_range() {
        let dimensions = 131_072;
        let codes = vec![-128; dimensions];
        let queries = packed(&codes, &[1.0], 1, dimensions);
        let gallery = packed(&codes, &[1.0], 1, dimensions);

        assert_eq!(
            scalar_scores(&queries, &gallery).unwrap(),
            vec![2_147_483_648.0]
        );
    }

    #[test]
    fn invalid_wire_lengths_and_metadata_fail_closed() {
        assert!(PackedRows::new(vec![1, 2, 3], vec![f16::ONE], 1, 4).is_err());
        assert!(PackedRows::new(vec![1, 2, 3, 4], vec![], 1, 4).is_err());
        assert!(PackedRows::new(vec![1, 2, 3, 4], vec![f16::ZERO], 1, 4).is_err());
        assert!(PackedRows::new(vec![1, 2, 3, 4], vec![f16::from_f32(f32::NAN)], 1, 4).is_err());
    }

    #[test]
    fn score_rejects_dimension_mismatch() {
        let queries = packed(&[1, 2], &[1.0], 1, 2);
        let gallery = packed(&[1, 2, 3], &[1.0], 1, 3);

        assert!(scalar_scores(&queries, &gallery).is_err());
    }
}
