use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::time::Instant;

use cuda_core::Device;
use cutile::prelude::*;

use crate::PackedRows;

const DIMENSIONS: usize = 128;
const BN: usize = 128;
const BK: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchShape {
    One,
    ThirtyTwo,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CutileScoreError {
    Authority,
    Runtime,
}

impl Display for CutileScoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Authority => formatter.write_str("cuTile score authority differs"),
            Self::Runtime => formatter.write_str("cuTile score execution failed"),
        }
    }
}

impl Error for CutileScoreError {}

#[cutile::module]
mod score_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn signed_int8_score<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        output: &mut Tensor<f32, { [BM, BN] }>,
        queries: &Tensor<i8, { [-1, K] }>,
        gallery_transposed: &Tensor<i8, { [K, -1] }>,
        query_norms: &Tensor<f32, { [-1, 1] }>,
        gallery_norms: &Tensor<f32, { [1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let query_partition = queries.partition(const_shape![BM, BK]);
        let gallery_partition = gallery_transposed.partition(const_shape![BK, BN]);
        let mut accumulator: Tile<i32, { [BM, BN] }> = constant(0i32, const_shape![BM, BN]);
        for inner in 0i32..(K / BK) {
            accumulator = mmai(
                query_partition.load([pid.0, inner]),
                gallery_partition.load([inner, pid.1]),
                accumulator,
                signedness::Signed,
                signedness::Signed,
            );
        }
        let dots: Tile<f32, { [BM, BN] }> = convert_tile(accumulator);
        let query_scale: Tile<f32, { [BM, 1] }> = query_norms
            .partition(const_shape![BM, 1])
            .load([pid.0, 0i32]);
        let gallery_scale: Tile<f32, { [1, BN] }> = gallery_norms
            .partition(const_shape![1, BN])
            .load([0i32, pid.1]);
        let scores = dots
            * query_scale.broadcast(const_shape![BM, BN])
            * gallery_scale.broadcast(const_shape![BM, BN]);
        output.store(scores);
    }
}

#[cutile::module]
mod float_score_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn float_score<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        output: &mut Tensor<f32, { [BM, BN] }>,
        queries: &Tensor<f32, { [-1, K] }>,
        gallery_transposed: &Tensor<f32, { [K, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let query_partition = queries.partition(const_shape![BM, BK]);
        let gallery_partition = gallery_transposed.partition(const_shape![BK, BN]);
        let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for inner in 0i32..(K / BK) {
            accumulator = mma(
                query_partition.load([pid.0, inner]),
                gallery_partition.load([inner, pid.1]),
                accumulator,
            );
        }
        output.store(accumulator);
    }
}

#[derive(Clone, Debug)]
pub struct ScorePlaneMeasurements {
    pub packed_compile_ns: u64,
    pub packed_samples_ns: Vec<u64>,
    pub resident_f32_compile_ns: u64,
    pub resident_f32_samples_ns: Vec<u64>,
    pub packed_persistent_bytes: u64,
    pub resident_f32_persistent_bytes: u64,
    pub score_plane_bytes: u64,
}

fn elapsed_ns(start: Instant) -> Result<u64, CutileScoreError> {
    u64::try_from(start.elapsed().as_nanos()).map_err(|_| CutileScoreError::Runtime)
}

pub fn benchmark_score_planes(
    device: &Arc<Device>,
    queries: &PackedRows,
    gallery: &PackedRows,
    batch: BatchShape,
    warmups: usize,
    samples: usize,
) -> Result<ScorePlaneMeasurements, CutileScoreError> {
    let bm = match batch {
        BatchShape::One => 1usize,
        BatchShape::ThirtyTwo => 32usize,
    };
    if queries.rows != bm
        || queries.dimensions != DIMENSIONS
        || gallery.dimensions != DIMENSIONS
        || warmups == 0
        || samples == 0
    {
        return Err(CutileScoreError::Authority);
    }
    let padded_gallery_rows = gallery.rows.div_ceil(BN) * BN;
    let mut gallery_codes = vec![0i8; DIMENSIONS * padded_gallery_rows];
    let mut resident_gallery = vec![0.0f32; DIMENSIONS * padded_gallery_rows];
    for row in 0..gallery.rows {
        let inverse_norm = gallery.inverse_norms[row].to_f32();
        for column in 0..DIMENSIONS {
            let value = gallery.codes[row * DIMENSIONS + column];
            gallery_codes[column * padded_gallery_rows + row] = value;
            resident_gallery[column * padded_gallery_rows + row] = f32::from(value) * inverse_norm;
        }
    }
    let mut query_codes = vec![0i8; bm * DIMENSIONS];
    query_codes.copy_from_slice(&queries.codes);
    let mut query_norms = vec![0.0f32; bm];
    let mut resident_queries = vec![0.0f32; bm * DIMENSIONS];
    for row in 0..bm {
        let inverse_norm = queries.inverse_norms[row].to_f32();
        query_norms[row] = inverse_norm;
        for column in 0..DIMENSIONS {
            resident_queries[row * DIMENSIONS + column] =
                f32::from(queries.codes[row * DIMENSIONS + column]) * inverse_norm;
        }
    }
    let mut gallery_norms = vec![0.0f32; padded_gallery_rows];
    for (output, input) in gallery_norms.iter_mut().zip(&gallery.inverse_norms) {
        *output = input.to_f32();
    }

    let stream = device.new_stream().map_err(|_| CutileScoreError::Runtime)?;
    let mut packed_output = cutile::api::zeros::<f32>(&[bm, padded_gallery_rows])
        .sync_on(&stream)
        .map_err(|_| CutileScoreError::Runtime)?;
    let mut float_output = cutile::api::zeros::<f32>(&[bm, padded_gallery_rows])
        .sync_on(&stream)
        .map_err(|_| CutileScoreError::Runtime)?;
    let query_tensor: Arc<Tensor<i8>> =
        cutile::api::copy_host_vec_to_device(&Arc::new(query_codes))
            .reshape(&[bm, DIMENSIONS])
            .sync_on(&stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .into();
    let gallery_tensor: Arc<Tensor<i8>> =
        cutile::api::copy_host_vec_to_device(&Arc::new(gallery_codes))
            .reshape(&[DIMENSIONS, padded_gallery_rows])
            .sync_on(&stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .into();
    let query_norm_tensor: Arc<Tensor<f32>> =
        cutile::api::copy_host_vec_to_device(&Arc::new(query_norms))
            .reshape(&[bm, 1])
            .sync_on(&stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .into();
    let gallery_norm_tensor: Arc<Tensor<f32>> =
        cutile::api::copy_host_vec_to_device(&Arc::new(gallery_norms))
            .reshape(&[1, padded_gallery_rows])
            .sync_on(&stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .into();
    let resident_query_tensor: Arc<Tensor<f32>> =
        cutile::api::copy_host_vec_to_device(&Arc::new(resident_queries))
            .reshape(&[bm, DIMENSIONS])
            .sync_on(&stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .into();
    let resident_gallery_tensor: Arc<Tensor<f32>> =
        cutile::api::copy_host_vec_to_device(&Arc::new(resident_gallery))
            .reshape(&[DIMENSIONS, padded_gallery_rows])
            .sync_on(&stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .into();
    let generics = vec![
        bm.to_string(),
        BN.to_string(),
        BK.to_string(),
        DIMENSIONS.to_string(),
    ];

    let mut launch_packed = || -> Result<(), CutileScoreError> {
        let output = (&mut packed_output).partition([bm, BN]);
        let _ = score_module::signed_int8_score(
            output,
            query_tensor.clone(),
            gallery_tensor.clone(),
            query_norm_tensor.clone(),
            gallery_norm_tensor.clone(),
        )
        .generics(generics.clone())
        .sync_on(&stream)
        .map_err(|_| CutileScoreError::Runtime)?;
        Ok(())
    };
    let packed_compile_start = Instant::now();
    launch_packed()?;
    let packed_compile_ns = elapsed_ns(packed_compile_start)?;
    for _ in 0..warmups {
        launch_packed()?;
    }
    let mut packed_samples_ns = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        launch_packed()?;
        packed_samples_ns.push(elapsed_ns(start)?);
    }

    let mut launch_float = || -> Result<(), CutileScoreError> {
        let output = (&mut float_output).partition([bm, BN]);
        let _ = float_score_module::float_score(
            output,
            resident_query_tensor.clone(),
            resident_gallery_tensor.clone(),
        )
        .generics(generics.clone())
        .sync_on(&stream)
        .map_err(|_| CutileScoreError::Runtime)?;
        Ok(())
    };
    let float_compile_start = Instant::now();
    launch_float()?;
    let resident_f32_compile_ns = elapsed_ns(float_compile_start)?;
    for _ in 0..warmups {
        launch_float()?;
    }
    let mut resident_f32_samples_ns = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        launch_float()?;
        resident_f32_samples_ns.push(elapsed_ns(start)?);
    }

    Ok(ScorePlaneMeasurements {
        packed_compile_ns,
        packed_samples_ns,
        resident_f32_compile_ns,
        resident_f32_samples_ns,
        packed_persistent_bytes: u64::try_from(gallery.rows * (DIMENSIONS + 2))
            .map_err(|_| CutileScoreError::Runtime)?,
        resident_f32_persistent_bytes: u64::try_from(gallery.rows * DIMENSIONS * 4)
            .map_err(|_| CutileScoreError::Runtime)?,
        score_plane_bytes: u64::try_from(bm * gallery.rows * 4)
            .map_err(|_| CutileScoreError::Runtime)?,
    })
}

pub fn score_cutile(
    device: &Arc<Device>,
    queries: &PackedRows,
    gallery: &PackedRows,
    batch: BatchShape,
) -> Result<Vec<f32>, CutileScoreError> {
    let bm = match batch {
        BatchShape::One => 1usize,
        BatchShape::ThirtyTwo => 32usize,
    };
    if queries.dimensions != DIMENSIONS || gallery.dimensions != DIMENSIONS || queries.rows != bm {
        return Err(CutileScoreError::Authority);
    }

    let padded_query_rows = queries.rows;
    let padded_gallery_rows = gallery.rows.div_ceil(BN) * BN;
    let mut query_codes = vec![0i8; padded_query_rows * DIMENSIONS];
    query_codes[..queries.codes.len()].copy_from_slice(&queries.codes);
    let mut gallery_transposed = vec![0i8; DIMENSIONS * padded_gallery_rows];
    for row in 0..gallery.rows {
        for column in 0..DIMENSIONS {
            gallery_transposed[column * padded_gallery_rows + row] =
                gallery.codes[row * DIMENSIONS + column];
        }
    }
    let mut query_norms = vec![0.0f32; padded_query_rows];
    for (output, input) in query_norms.iter_mut().zip(&queries.inverse_norms) {
        *output = input.to_f32();
    }
    let mut gallery_norms = vec![0.0f32; padded_gallery_rows];
    for (output, input) in gallery_norms.iter_mut().zip(&gallery.inverse_norms) {
        *output = input.to_f32();
    }

    let stream = device.new_stream().map_err(|_| CutileScoreError::Runtime)?;
    let output = cutile::api::zeros::<f32>(&[padded_query_rows, padded_gallery_rows])
        .sync_on(&stream)
        .map_err(|_| CutileScoreError::Runtime)?
        .partition([bm, BN]);
    let query_tensor: Arc<Tensor<i8>> =
        cutile::api::copy_host_vec_to_device(&Arc::new(query_codes))
            .reshape(&[padded_query_rows, DIMENSIONS])
            .sync_on(&stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .into();
    let gallery_tensor: Arc<Tensor<i8>> =
        cutile::api::copy_host_vec_to_device(&Arc::new(gallery_transposed))
            .reshape(&[DIMENSIONS, padded_gallery_rows])
            .sync_on(&stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .into();
    let query_norm_tensor: Arc<Tensor<f32>> =
        cutile::api::copy_host_vec_to_device(&Arc::new(query_norms))
            .reshape(&[padded_query_rows, 1])
            .sync_on(&stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .into();
    let gallery_norm_tensor: Arc<Tensor<f32>> =
        cutile::api::copy_host_vec_to_device(&Arc::new(gallery_norms))
            .reshape(&[1, padded_gallery_rows])
            .sync_on(&stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .into();
    let generics = vec![
        bm.to_string(),
        BN.to_string(),
        BK.to_string(),
        DIMENSIONS.to_string(),
    ];
    let (output, _, _, _, _) = score_module::signed_int8_score(
        output,
        query_tensor,
        gallery_tensor,
        query_norm_tensor,
        gallery_norm_tensor,
    )
    .generics(generics)
    .sync_on(&stream)
    .map_err(|_| CutileScoreError::Runtime)?;
    let padded_scores = output
        .unpartition()
        .to_host_vec()
        .sync_on(&stream)
        .map_err(|_| CutileScoreError::Runtime)?;
    let mut scores = Vec::with_capacity(queries.rows * gallery.rows);
    for query_row in 0..queries.rows {
        let start = query_row * padded_gallery_rows;
        scores.extend_from_slice(&padded_scores[start..start + gallery.rows]);
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use cuda_core::Device;
    use half::f16;

    use super::{BatchShape, score_cutile};
    use crate::{PackedRows, scalar_scores};

    fn top_ten(scores: &[f32]) -> Vec<usize> {
        let mut ordinals = (0..scores.len()).collect::<Vec<_>>();
        ordinals.sort_by(|&left, &right| {
            scores[right]
                .total_cmp(&scores[left])
                .then_with(|| left.cmp(&right))
        });
        ordinals.truncate(10);
        ordinals
    }

    fn fixture() -> (PackedRows, PackedRows) {
        let dimensions = 128;
        let query_codes = (0..dimensions)
            .map(|column| if column % 2 == 0 { -127 } else { 127 })
            .collect::<Vec<_>>();
        let mut gallery_codes = Vec::with_capacity(129 * dimensions);
        for row in 0..129 {
            for (column, &query_value) in query_codes.iter().enumerate() {
                let value = match row {
                    0 | 1 => query_value,
                    2 => -query_value,
                    _ => (((row * 17 + column * 13) % 255) as i16 - 127) as i8,
                };
                gallery_codes.push(value);
            }
        }
        let gallery_norms = (0..129)
            .map(|row| {
                let row = if row == 1 { 0 } else { row };
                f16::from_f32(0.001 + row as f32 * 0.000_001)
            })
            .collect();
        (
            PackedRows::new(query_codes, vec![f16::from_f32(0.002)], 1, dimensions).unwrap(),
            PackedRows::new(gallery_codes, gallery_norms, 129, dimensions).unwrap(),
        )
    }

    #[test]
    fn cutile_scores_match_scalar_bits_and_lexicographic_top_ten() {
        let device = Device::new(0).unwrap();
        let (queries, gallery) = fixture();
        let expected = scalar_scores(&queries, &gallery).unwrap();

        let observed = score_cutile(&device, &queries, &gallery, BatchShape::One).unwrap();

        assert_eq!(observed.len(), expected.len());
        assert!(
            observed
                .iter()
                .zip(&expected)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        );
        assert_eq!(top_ten(&observed), top_ten(&expected));
        assert_eq!(top_ten(&observed)[0..2], [0, 1]);
    }
}
