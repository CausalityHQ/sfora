use cuda_core::Device;
use half::f16;
use sfora_cutile_int8_score::{BatchShape, PackedRows, PreparedPackedGallery, scalar_scores};

fn packed(codes: Vec<i8>, rows: usize) -> PackedRows {
    let inverse_norms = codes
        .chunks_exact(128)
        .map(|row| {
            let squared = row
                .iter()
                .map(|&value| i32::from(value) * i32::from(value))
                .sum::<i32>();
            f16::from_f32(1.0 / (squared as f32).sqrt())
        })
        .collect();
    PackedRows::new(codes, inverse_norms, rows, 128).unwrap()
}

#[test]
fn batch_thirty_two_merge_keeps_ties_across_width_boundary() {
    let query_row = (0..128)
        .map(|column| if column % 2 == 0 { 127 } else { -127 })
        .collect::<Vec<_>>();
    let queries = packed(query_row.repeat(32), 32);
    let mut gallery_codes = Vec::with_capacity(4097 * 128);
    for row in 0..4097 {
        for column in 0..128 {
            let value = if [0, 128, 4095, 4096].contains(&row) {
                query_row[column]
            } else if row < 10 {
                -query_row[column]
            } else {
                (((row * 29 + column * 31) % 255) as i16 - 127) as i8
            };
            gallery_codes.push(value);
        }
    }
    let gallery = packed(gallery_codes, 4097);
    let expected = scalar_scores(&queries, &gallery)
        .unwrap()
        .chunks(4097)
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
        .collect::<Vec<_>>();

    let device = Device::new(0).unwrap();
    let mut prepared = PreparedPackedGallery::new(&device, &gallery).unwrap();
    let observed = prepared
        .search(&queries, BatchShape::ThirtyTwo, 10)
        .unwrap();

    assert_eq!(observed.as_pairs(), expected);
    assert_eq!(&observed.ordinals()[..4], &[0, 128, 4095, 4096]);
}

#[test]
fn batch_thirty_two_merge_preserves_distinct_query_rows() {
    let query_rows = (0..32)
        .map(|row| {
            (0..128)
                .map(|column| {
                    (((row * 47 + column * 73 + (row ^ column) * 11) % 255) as i16 - 127) as i8
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let queries = packed(query_rows.iter().flatten().copied().collect(), 32);
    let mut gallery_codes = Vec::with_capacity(4097 * 128);
    for ordinal in 0..4097 {
        let matched_row = if ordinal < 32 {
            Some(ordinal)
        } else if (128..160).contains(&ordinal) {
            Some(ordinal - 128)
        } else if (4065..=4096).contains(&ordinal) {
            Some(4096 - ordinal)
        } else {
            None
        };
        if let Some(row) = matched_row {
            gallery_codes.extend_from_slice(&query_rows[row]);
        } else {
            gallery_codes.extend(
                (0..128).map(|column| (((ordinal * 29 + column * 31) % 255) as i16 - 127) as i8),
            );
        }
    }
    let gallery = packed(gallery_codes, 4097);
    let expected = scalar_scores(&queries, &gallery)
        .unwrap()
        .chunks(4097)
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
        .collect::<Vec<_>>();

    let device = Device::new(0).unwrap();
    let mut prepared = PreparedPackedGallery::new(&device, &gallery).unwrap();
    let observed = prepared
        .search(&queries, BatchShape::ThirtyTwo, 10)
        .unwrap();

    assert_eq!(observed.as_pairs(), expected);
    for row in 0..32 {
        assert_eq!(
            &observed.ordinals()[row * 10..row * 10 + 3],
            &[row as u32, (128 + row) as u32, (4096 - row) as u32]
        );
    }
}
