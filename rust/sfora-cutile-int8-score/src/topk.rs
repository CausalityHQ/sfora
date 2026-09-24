use std::sync::Arc;

use cuda_core::{Device, Stream};
use cutile::prelude::*;

use crate::{BatchShape, CutileScoreError, PackedRows};

const DIMENSIONS: usize = 128;
const SCORE_BLOCK: usize = 128;
const INNER_BLOCK: usize = 32;
const TOP_K: usize = 10;
const TOP_K_TILE: usize = 16;
const MERGE_WIDTH_BATCH_ONE: usize = 2048;
const MERGE_WIDTH_BATCH_THIRTY_TWO: usize = 512;

fn validated_topk_padded_rows(rows: usize) -> Result<usize, CutileScoreError> {
    if rows < TOP_K {
        return Err(CutileScoreError::Authority);
    }
    let padded = rows
        .checked_add(SCORE_BLOCK - 1)
        .and_then(|value| value.checked_div(SCORE_BLOCK))
        .and_then(|blocks| blocks.checked_mul(SCORE_BLOCK))
        .ok_or(CutileScoreError::Authority)?;
    // All kernel ordinals, including padded lanes, must fit below the sentinel.
    if padded > i32::MAX as usize {
        return Err(CutileScoreError::Authority);
    }
    Ok(padded)
}

fn checked_output_ordinal(value: i32, rows: usize) -> Result<u32, CutileScoreError> {
    let ordinal = usize::try_from(value).map_err(|_| CutileScoreError::Runtime)?;
    if ordinal >= rows {
        return Err(CutileScoreError::Runtime);
    }
    u32::try_from(ordinal).map_err(|_| CutileScoreError::Runtime)
}

#[derive(Clone, Debug)]
pub struct TopKResult {
    ordinals: Vec<u32>,
    scores: Vec<f32>,
}

impl TopKResult {
    pub fn len(&self) -> usize {
        self.ordinals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ordinals.is_empty()
    }

    pub fn ordinals(&self) -> &[u32] {
        &self.ordinals
    }

    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    pub fn as_pairs(&self) -> Vec<(u32, f32)> {
        self.ordinals
            .iter()
            .copied()
            .zip(self.scores.iter().copied())
            .collect()
    }
}

#[cutile::module]
mod topk_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn score_tile_for_profile<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
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
        let query_scale: Tile<f32, { [BM, 1] }> = query_norms
            .partition(const_shape![BM, 1])
            .load([pid.0, 0i32]);
        let gallery_scale: Tile<f32, { [1, BN] }> = gallery_norms
            .partition(const_shape![1, BN])
            .load([0i32, pid.1]);
        let scores: Tile<f32, { [BM, BN] }> = convert_tile(accumulator)
            * query_scale.broadcast(const_shape![BM, BN])
            * gallery_scale.broadcast(const_shape![BM, BN]);
        output.store(scores);
    }

    #[cutile::entry()]
    fn select_tile_for_profile<
        const BM: i32,
        const BN: i32,
        const TOP_TILE: i32,
        const TOP: i32,
        const GALLERY_ROWS: i32,
    >(
        output_scores: &mut Tensor<f32, { [BM, TOP_TILE] }>,
        output_ordinals: &mut Tensor<i32, { [BM, TOP_TILE] }>,
        input_scores: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let mut scores = input_scores
            .partition(const_shape![BM, BN])
            .load([pid.0, pid.1]);
        let local_ordinals: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let ordinal_offset: Tile<i32, { [BM, BN] }> = (pid.1 * BN).broadcast(const_shape![BM, BN]);
        let ordinals: Tile<i32, { [BM, BN] }> = local_ordinals
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN])
            + ordinal_offset;
        let valid = lt_tile(ordinals, GALLERY_ROWS.broadcast(const_shape![BM, BN]));
        let negative_infinity: Tile<f32, { [BM, BN] }> =
            constant(f32::NEG_INFINITY, const_shape![BM, BN]);
        let false_tile: Tile<bool, { [BM, BN] }> = constant(false, const_shape![BM, BN]);
        let maximum_ordinal: Tile<i32, { [BM, BN] }> =
            constant(2147483647i32, const_shape![BM, BN]);
        scores = select(valid, scores, negative_infinity);
        let mut selected_scores: Tile<f32, { [BM, TOP_TILE] }> =
            constant(f32::NEG_INFINITY, const_shape![BM, TOP_TILE]);
        let mut selected_ordinals: Tile<i32, { [BM, TOP_TILE] }> =
            constant(2147483647i32, const_shape![BM, TOP_TILE]);
        let rank_range: Tile<i32, { [TOP_TILE] }> = iota(const_shape![TOP_TILE]);
        let rank_lanes: Tile<i32, { [BM, TOP_TILE] }> = rank_range
            .reshape(const_shape![1, TOP_TILE])
            .broadcast(const_shape![BM, TOP_TILE]);
        for rank in 0i32..TOP {
            let best_score: Tile<f32, { [BM] }> = reduce_max(scores, 1i32);
            let best_score = best_score.reshape(const_shape![BM, 1]);
            let equal_score = eq_tile(scores, best_score.broadcast(const_shape![BM, BN]));
            let eligible = select(valid, equal_score, false_tile);
            let eligible_ordinals = select(eligible, ordinals, maximum_ordinal);
            let best_ordinal: Tile<i32, { [BM] }> = reduce_min(eligible_ordinals, 1i32);
            let best_ordinal = best_ordinal.reshape(const_shape![BM, 1]);
            let rank_mask = eq_tile(rank_lanes, rank.broadcast(const_shape![BM, TOP_TILE]));
            selected_scores = select(
                rank_mask,
                best_score.broadcast(const_shape![BM, TOP_TILE]),
                selected_scores,
            );
            selected_ordinals = select(
                rank_mask,
                best_ordinal.broadcast(const_shape![BM, TOP_TILE]),
                selected_ordinals,
            );
            scores = select(
                eq_tile(ordinals, best_ordinal.broadcast(const_shape![BM, BN])),
                negative_infinity,
                scores,
            );
        }
        output_scores.store(selected_scores);
        output_ordinals.store(selected_ordinals);
    }

    #[cutile::entry()]
    fn score_block_topk<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const K: i32,
        const TOP_TILE: i32,
        const TOP: i32,
        const GALLERY_ROWS: i32,
    >(
        output_scores: &mut Tensor<f32, { [BM, TOP_TILE] }>,
        output_ordinals: &mut Tensor<i32, { [BM, TOP_TILE] }>,
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
        let query_scale: Tile<f32, { [BM, 1] }> = query_norms
            .partition(const_shape![BM, 1])
            .load([pid.0, 0i32]);
        let gallery_scale: Tile<f32, { [1, BN] }> = gallery_norms
            .partition(const_shape![1, BN])
            .load([0i32, pid.1]);
        let mut scores: Tile<f32, { [BM, BN] }> = convert_tile(accumulator)
            * query_scale.broadcast(const_shape![BM, BN])
            * gallery_scale.broadcast(const_shape![BM, BN]);
        let local_ordinals: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let ordinal_offset: Tile<i32, { [BM, BN] }> = (pid.1 * BN).broadcast(const_shape![BM, BN]);
        let ordinals: Tile<i32, { [BM, BN] }> = local_ordinals
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN])
            + ordinal_offset;
        let valid = lt_tile(ordinals, GALLERY_ROWS.broadcast(const_shape![BM, BN]));
        let negative_infinity: Tile<f32, { [BM, BN] }> =
            constant(f32::NEG_INFINITY, const_shape![BM, BN]);
        let false_tile: Tile<bool, { [BM, BN] }> = constant(false, const_shape![BM, BN]);
        let maximum_ordinal: Tile<i32, { [BM, BN] }> =
            constant(2147483647i32, const_shape![BM, BN]);
        scores = select(valid, scores, negative_infinity);
        let mut selected_scores: Tile<f32, { [BM, TOP_TILE] }> =
            constant(f32::NEG_INFINITY, const_shape![BM, TOP_TILE]);
        let mut selected_ordinals: Tile<i32, { [BM, TOP_TILE] }> =
            constant(2147483647i32, const_shape![BM, TOP_TILE]);
        let rank_range: Tile<i32, { [TOP_TILE] }> = iota(const_shape![TOP_TILE]);
        let rank_lanes: Tile<i32, { [BM, TOP_TILE] }> = rank_range
            .reshape(const_shape![1, TOP_TILE])
            .broadcast(const_shape![BM, TOP_TILE]);
        for rank in 0i32..TOP {
            let best_score: Tile<f32, { [BM] }> = reduce_max(scores, 1i32);
            let best_score = best_score.reshape(const_shape![BM, 1]);
            let equal_score = eq_tile(scores, best_score.broadcast(const_shape![BM, BN]));
            let eligible = select(valid, equal_score, false_tile);
            let eligible_ordinals = select(eligible, ordinals, maximum_ordinal);
            let best_ordinal: Tile<i32, { [BM] }> = reduce_min(eligible_ordinals, 1i32);
            let best_ordinal = best_ordinal.reshape(const_shape![BM, 1]);
            let rank_mask = eq_tile(rank_lanes, rank.broadcast(const_shape![BM, TOP_TILE]));
            selected_scores = select(
                rank_mask,
                best_score.broadcast(const_shape![BM, TOP_TILE]),
                selected_scores,
            );
            selected_ordinals = select(
                rank_mask,
                best_ordinal.broadcast(const_shape![BM, TOP_TILE]),
                selected_ordinals,
            );
            scores = select(
                eq_tile(ordinals, best_ordinal.broadcast(const_shape![BM, BN])),
                negative_infinity,
                scores,
            );
        }
        output_scores.store(selected_scores);
        output_ordinals.store(selected_ordinals);
    }

    #[cutile::entry()]
    fn merge_topk<
        const BM: i32,
        const WIDTH: i32,
        const TOP_TILE: i32,
        const TOP: i32,
        const INPUT_ENTRIES: i32,
    >(
        output_scores: &mut Tensor<f32, { [BM, TOP_TILE] }>,
        output_ordinals: &mut Tensor<i32, { [BM, TOP_TILE] }>,
        input_scores: &Tensor<f32, { [-1, -1] }>,
        input_ordinals: &Tensor<i32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let mut scores = input_scores
            .partition(const_shape![BM, WIDTH])
            .load([pid.0, pid.1]);
        let mut ordinals = input_ordinals
            .partition(const_shape![BM, WIDTH])
            .load([pid.0, pid.1]);
        let local_positions: Tile<i32, { [WIDTH] }> = iota(const_shape![WIDTH]);
        let position_offset: Tile<i32, { [BM, WIDTH] }> =
            (pid.1 * WIDTH).broadcast(const_shape![BM, WIDTH]);
        let positions = local_positions
            .reshape(const_shape![1, WIDTH])
            .broadcast(const_shape![BM, WIDTH])
            + position_offset;
        let valid = lt_tile(positions, INPUT_ENTRIES.broadcast(const_shape![BM, WIDTH]));
        let mut selected_scores: Tile<f32, { [BM, TOP_TILE] }> =
            constant(f32::NEG_INFINITY, const_shape![BM, TOP_TILE]);
        let mut selected_ordinals: Tile<i32, { [BM, TOP_TILE] }> =
            constant(2147483647i32, const_shape![BM, TOP_TILE]);
        let rank_range: Tile<i32, { [TOP_TILE] }> = iota(const_shape![TOP_TILE]);
        let rank_lanes: Tile<i32, { [BM, TOP_TILE] }> = rank_range
            .reshape(const_shape![1, TOP_TILE])
            .broadcast(const_shape![BM, TOP_TILE]);
        let maximum_ordinal: Tile<i32, { [BM, WIDTH] }> =
            constant(2147483647i32, const_shape![BM, WIDTH]);
        let negative_infinity: Tile<f32, { [BM, WIDTH] }> =
            constant(f32::NEG_INFINITY, const_shape![BM, WIDTH]);
        scores = select(valid, scores, negative_infinity);
        ordinals = select(valid, ordinals, maximum_ordinal);
        for rank in 0i32..TOP {
            let best_score: Tile<f32, { [BM] }> = reduce_max(scores, 1i32);
            let best_score = best_score.reshape(const_shape![BM, 1]);
            let equal_score = eq_tile(scores, best_score.broadcast(const_shape![BM, WIDTH]));
            let eligible_ordinals = select(equal_score, ordinals, maximum_ordinal);
            let best_ordinal: Tile<i32, { [BM] }> = reduce_min(eligible_ordinals, 1i32);
            let best_ordinal = best_ordinal.reshape(const_shape![BM, 1]);
            let rank_mask = eq_tile(rank_lanes, rank.broadcast(const_shape![BM, TOP_TILE]));
            selected_scores = select(
                rank_mask,
                best_score.broadcast(const_shape![BM, TOP_TILE]),
                selected_scores,
            );
            selected_ordinals = select(
                rank_mask,
                best_ordinal.broadcast(const_shape![BM, TOP_TILE]),
                selected_ordinals,
            );
            scores = select(
                eq_tile(ordinals, best_ordinal.broadcast(const_shape![BM, WIDTH])),
                negative_infinity,
                scores,
            );
        }
        output_scores.store(selected_scores);
        output_ordinals.store(selected_ordinals);
    }
}

pub struct PreparedPackedGallery {
    stream: Arc<Stream>,
    codes: Arc<Tensor<i8>>,
    inverse_norms: Arc<Tensor<f32>>,
    rows: usize,
    padded_rows: usize,
}

impl PreparedPackedGallery {
    pub fn new(device: &Arc<Device>, gallery: &PackedRows) -> Result<Self, CutileScoreError> {
        if gallery.dimensions != DIMENSIONS {
            return Err(CutileScoreError::Authority);
        }
        let padded_rows = validated_topk_padded_rows(gallery.rows)?;
        let mut transposed = vec![0i8; DIMENSIONS * padded_rows];
        for row in 0..gallery.rows {
            for column in 0..DIMENSIONS {
                transposed[column * padded_rows + row] = gallery.codes[row * DIMENSIONS + column];
            }
        }
        let mut norms = vec![0.0f32; padded_rows];
        for (output, input) in norms.iter_mut().zip(&gallery.inverse_norms) {
            *output = input.to_f32();
        }
        let stream = device.new_stream().map_err(|_| CutileScoreError::Runtime)?;
        let codes = cutile::api::copy_host_vec_to_device(&Arc::new(transposed))
            .reshape(&[DIMENSIONS, padded_rows])
            .sync_on(&stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .into();
        let inverse_norms = cutile::api::copy_host_vec_to_device(&Arc::new(norms))
            .reshape(&[1, padded_rows])
            .sync_on(&stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .into();
        Ok(Self {
            stream,
            codes,
            inverse_norms,
            rows: gallery.rows,
            padded_rows,
        })
    }

    pub fn search(
        &mut self,
        queries: &PackedRows,
        batch: BatchShape,
        k: usize,
    ) -> Result<TopKResult, CutileScoreError> {
        self.search_internal(queries, batch, k, false)
    }

    /// Diagnostic split of the fused score/select kernel; never used by the public FFI.
    pub fn search_split_for_profile(
        &mut self,
        queries: &PackedRows,
        batch: BatchShape,
        k: usize,
    ) -> Result<TopKResult, CutileScoreError> {
        self.search_internal(queries, batch, k, true)
    }

    fn search_internal(
        &mut self,
        queries: &PackedRows,
        batch: BatchShape,
        k: usize,
        split_for_profile: bool,
    ) -> Result<TopKResult, CutileScoreError> {
        let bm = match batch {
            BatchShape::One => 1usize,
            BatchShape::ThirtyTwo => 32usize,
        };
        if k != TOP_K || queries.rows != bm || queries.dimensions != DIMENSIONS {
            return Err(CutileScoreError::Authority);
        }
        let query_codes: Arc<Tensor<i8>> =
            cutile::api::copy_host_vec_to_device(&Arc::new(queries.codes.clone()))
                .reshape(&[bm, DIMENSIONS])
                .sync_on(&self.stream)
                .map_err(|_| CutileScoreError::Runtime)?
                .into();
        let query_norms = queries
            .inverse_norms
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let query_norms: Arc<Tensor<f32>> =
            cutile::api::copy_host_vec_to_device(&Arc::new(query_norms))
                .reshape(&[bm, 1])
                .sync_on(&self.stream)
                .map_err(|_| CutileScoreError::Runtime)?
                .into();
        let score_blocks = self.padded_rows / SCORE_BLOCK;
        let score_entries = score_blocks * TOP_K_TILE;
        let score_output = cutile::api::full(f32::NEG_INFINITY, &[bm, score_entries])
            .sync_on(&self.stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .partition([bm, TOP_K_TILE]);
        let ordinal_output = cutile::api::full(i32::MAX, &[bm, score_entries])
            .sync_on(&self.stream)
            .map_err(|_| CutileScoreError::Runtime)?
            .partition([bm, TOP_K_TILE]);
        let generics = vec![
            bm.to_string(),
            SCORE_BLOCK.to_string(),
            INNER_BLOCK.to_string(),
            DIMENSIONS.to_string(),
            TOP_K_TILE.to_string(),
            TOP_K.to_string(),
            self.rows.to_string(),
        ];
        let (score_output, ordinal_output) = if split_for_profile {
            let score_plane = cutile::api::zeros::<f32>(&[bm, self.padded_rows])
                .sync_on(&self.stream)
                .map_err(|_| CutileScoreError::Runtime)?
                .partition([bm, SCORE_BLOCK]);
            let score_generics = vec![
                bm.to_string(),
                SCORE_BLOCK.to_string(),
                INNER_BLOCK.to_string(),
                DIMENSIONS.to_string(),
            ];
            let (score_plane, _, _, _, _) = topk_module::score_tile_for_profile(
                score_plane,
                query_codes,
                self.codes.clone(),
                query_norms,
                self.inverse_norms.clone(),
            )
            .grid((1, score_blocks as u32, 1))
            .generics(score_generics)
            .sync_on(&self.stream)
            .map_err(|_| CutileScoreError::Runtime)?;
            let selection_generics = vec![
                bm.to_string(),
                SCORE_BLOCK.to_string(),
                TOP_K_TILE.to_string(),
                TOP_K.to_string(),
                self.rows.to_string(),
            ];
            let (score_output, ordinal_output, _) = topk_module::select_tile_for_profile(
                score_output,
                ordinal_output,
                Arc::new(score_plane.unpartition()),
            )
            .grid((1, score_blocks as u32, 1))
            .generics(selection_generics)
            .sync_on(&self.stream)
            .map_err(|_| CutileScoreError::Runtime)?;
            (score_output, ordinal_output)
        } else {
            let (score_output, ordinal_output, _, _, _, _) = topk_module::score_block_topk(
                score_output,
                ordinal_output,
                query_codes,
                self.codes.clone(),
                query_norms,
                self.inverse_norms.clone(),
            )
            .grid((1, score_blocks as u32, 1))
            .generics(generics)
            .sync_on(&self.stream)
            .map_err(|error| {
                eprintln!("score_block_topk failed: {error:?}");
                CutileScoreError::Runtime
            })?;
            (score_output, ordinal_output)
        };
        let mut scores: Arc<Tensor<f32>> = Arc::new(score_output.unpartition());
        let mut ordinals: Arc<Tensor<i32>> = Arc::new(ordinal_output.unpartition());
        let mut real_entries = score_blocks * TOP_K_TILE;
        let merge_width = match batch {
            BatchShape::One => MERGE_WIDTH_BATCH_ONE,
            BatchShape::ThirtyTwo => MERGE_WIDTH_BATCH_THIRTY_TWO,
        };
        // The merge is independent for each query. A one-row tile keeps the
        // generated reduction program small even when the score stage uses
        // a 32-row MMA tile.
        let merge_bm = 1usize;
        loop {
            let groups = real_entries.div_ceil(merge_width);
            let output_real_entries = groups * TOP_K_TILE;
            let output_scores = cutile::api::full(f32::NEG_INFINITY, &[bm, output_real_entries])
                .sync_on(&self.stream)
                .map_err(|_| CutileScoreError::Runtime)?
                .partition([merge_bm, TOP_K_TILE]);
            let output_ordinals = cutile::api::full(i32::MAX, &[bm, output_real_entries])
                .sync_on(&self.stream)
                .map_err(|_| CutileScoreError::Runtime)?
                .partition([merge_bm, TOP_K_TILE]);
            let merge_generics = vec![
                merge_bm.to_string(),
                merge_width.to_string(),
                TOP_K_TILE.to_string(),
                TOP_K.to_string(),
                real_entries.to_string(),
            ];
            let (output_scores, output_ordinals, _, _) =
                topk_module::merge_topk(output_scores, output_ordinals, scores, ordinals)
                    .grid((bm as u32, groups as u32, 1))
                    .generics(merge_generics)
                    .sync_on(&self.stream)
                    .map_err(|error| {
                        eprintln!("merge_topk failed: {error:?}");
                        CutileScoreError::Runtime
                    })?;
            let score_tensor = output_scores.unpartition();
            let ordinal_tensor = output_ordinals.unpartition();
            if groups == 1 {
                let score_values = score_tensor
                    .to_host_vec()
                    .sync_on(&self.stream)
                    .map_err(|_| CutileScoreError::Runtime)?;
                let ordinal_values = ordinal_tensor
                    .to_host_vec()
                    .sync_on(&self.stream)
                    .map_err(|_| CutileScoreError::Runtime)?;
                let (ordinal_rows, ordinal_remainder) = ordinal_values.as_chunks::<TOP_K_TILE>();
                let (score_rows, score_remainder) = score_values.as_chunks::<TOP_K_TILE>();
                if !ordinal_remainder.is_empty() || !score_remainder.is_empty() {
                    return Err(CutileScoreError::Runtime);
                }
                return Ok(TopKResult {
                    ordinals: ordinal_rows
                        .iter()
                        .flat_map(|row| row[..TOP_K].iter().copied())
                        .map(|value| checked_output_ordinal(value, self.rows))
                        .collect::<Result<Vec<_>, _>>()?,
                    scores: score_rows
                        .iter()
                        .flat_map(|row| row[..TOP_K].iter().copied())
                        .collect(),
                });
            }
            scores = Arc::new(score_tensor);
            ordinals = Arc::new(ordinal_tensor);
            real_entries = output_real_entries;
        }
    }
}

#[cfg(test)]
mod tests {
    use cuda_core::Device;
    use half::f16;

    use super::{PreparedPackedGallery, checked_output_ordinal, validated_topk_padded_rows};
    use crate::{BatchShape, CutileScoreError, PackedRows, scalar_scores};

    #[test]
    fn gallery_row_limit_preserves_signed_ordinals_and_padded_shape() {
        let maximum_padded = (i32::MAX as usize / 128) * 128;
        assert_eq!(validated_topk_padded_rows(10), Ok(128));
        assert_eq!(
            validated_topk_padded_rows(maximum_padded),
            Ok(maximum_padded)
        );
        assert_eq!(
            validated_topk_padded_rows(maximum_padded + 1),
            Err(CutileScoreError::Authority)
        );
        assert_eq!(
            validated_topk_padded_rows(9),
            Err(CutileScoreError::Authority)
        );
        assert_eq!(
            validated_topk_padded_rows(usize::MAX),
            Err(CutileScoreError::Authority)
        );
    }

    #[test]
    fn output_ordinal_rejects_placeholders_and_out_of_gallery_values() {
        assert_eq!(checked_output_ordinal(0, 10), Ok(0));
        assert_eq!(checked_output_ordinal(9, 10), Ok(9));
        for invalid in [-1, 10, i32::MAX] {
            assert_eq!(
                checked_output_ordinal(invalid, 10),
                Err(CutileScoreError::Runtime)
            );
        }
    }

    fn packed(codes: Vec<i8>, rows: usize) -> PackedRows {
        let dimensions = 128;
        let norms = codes
            .chunks_exact(dimensions)
            .map(|row| {
                let squared = row
                    .iter()
                    .map(|&value| i32::from(value) * i32::from(value))
                    .sum::<i32>();
                f16::from_f32(1.0 / (squared as f32).sqrt())
            })
            .collect();
        PackedRows::new(codes, norms, rows, dimensions).unwrap()
    }

    fn fixture(query_rows: usize, gallery_rows: usize) -> (PackedRows, PackedRows) {
        let query = (0..query_rows * 128)
            .map(|index| if index % 2 == 0 { 127 } else { -127 })
            .collect::<Vec<_>>();
        let mut gallery = Vec::with_capacity(gallery_rows * 128);
        for row in 0..gallery_rows {
            for column in 0..128 {
                let value = if row == 0 || row == 128 {
                    if column % 2 == 0 { 127 } else { -127 }
                } else if row < 10 {
                    if column % 2 == 0 { -127 } else { 127 }
                } else {
                    (((row * 29 + column * 31) % 255) as i16 - 127) as i8
                };
                gallery.push(value);
            }
        }
        (packed(query, query_rows), packed(gallery, gallery_rows))
    }

    fn scalar_top_ten(queries: &PackedRows, gallery: &PackedRows) -> Vec<(u32, f32)> {
        scalar_scores(queries, gallery)
            .unwrap()
            .chunks(gallery.rows())
            .flat_map(|scores| {
                let mut ordinals = (0..scores.len()).collect::<Vec<_>>();
                ordinals.sort_by(|&left, &right| {
                    scores[right]
                        .total_cmp(&scores[left])
                        .then_with(|| left.cmp(&right))
                });
                ordinals
                    .into_iter()
                    .take(10)
                    .map(|ordinal| (ordinal as u32, scores[ordinal]))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn device_top_ten_matches_scalar_across_boundaries_batches_and_ties() {
        let device = Device::new(0).unwrap();
        for (batch, query_rows) in [(BatchShape::One, 1), (BatchShape::ThirtyTwo, 32)] {
            for gallery_rows in [10, 127, 128, 129] {
                let (queries, gallery) = fixture(query_rows, gallery_rows);
                let expected = scalar_top_ten(&queries, &gallery);
                let mut prepared = PreparedPackedGallery::new(&device, &gallery).unwrap();

                let observed = prepared.search(&queries, batch, 10).unwrap();

                assert_eq!(observed.len(), query_rows * 10);
                assert_eq!(observed.as_pairs(), expected);
                for row in observed.ordinals().as_chunks::<10>().0 {
                    assert!(row.windows(2).all(|pair| pair[0] != pair[1]));
                    assert!(row.iter().all(|&ordinal| ordinal < gallery_rows as u32));
                }
                if gallery_rows == 129 {
                    assert_eq!(&observed.ordinals()[..2], &[0, 128]);
                }
            }
        }
    }

    #[test]
    fn diagnostic_split_preserves_exact_scores_and_stable_ordinals() {
        let device = Device::new(0).unwrap();
        for (batch, query_rows) in [(BatchShape::One, 1), (BatchShape::ThirtyTwo, 32)] {
            let (queries, gallery) = fixture(query_rows, 129);
            let expected = scalar_top_ten(&queries, &gallery);
            let mut prepared = PreparedPackedGallery::new(&device, &gallery).unwrap();
            let fused = prepared.search(&queries, batch, 10).unwrap();
            let split = prepared
                .search_split_for_profile(&queries, batch, 10)
                .unwrap();

            assert_eq!(split.as_pairs(), fused.as_pairs());
            assert_eq!(split.as_pairs(), expected);
            assert_eq!(&split.ordinals()[..2], &[0, 128]);
        }
    }
}
