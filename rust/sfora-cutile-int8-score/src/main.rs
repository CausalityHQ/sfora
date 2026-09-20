use std::error::Error;
use std::process::Command;

use cuda_core::Device;
use half::f16;
use serde::Serialize;
use sfora_cutile_int8_score::{
    BatchShape, PackedRows, benchmark_score_planes, scalar_scores, score_cutile,
};

const GALLERY_ROWS: usize = 1_000_000;
const DIMENSIONS: usize = 128;
const WARMUPS: usize = 5;
const SAMPLES: usize = 50;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct TimingSummary {
    compile_ns: u64,
    p50_ns: u64,
    p99_ns: u64,
    raw_ns: Vec<u64>,
    sample_count: usize,
}

fn nearest_rank_percentile(samples: &[u64], percentile: usize) -> u64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (percentile * sorted.len()).div_ceil(100);
    sorted[rank.saturating_sub(1)]
}

fn validate_timing_samples(
    compile_ns: u64,
    raw_ns: &[u64],
    expected_samples: usize,
) -> Result<TimingSummary, &'static str> {
    if compile_ns == 0 || raw_ns.len() != expected_samples || raw_ns.contains(&0) {
        return Err("timing authority differs");
    }
    Ok(TimingSummary {
        compile_ns,
        p50_ns: nearest_rank_percentile(raw_ns, 50),
        p99_ns: nearest_rank_percentile(raw_ns, 99),
        raw_ns: raw_ns.to_vec(),
        sample_count: raw_ns.len(),
    })
}

#[derive(Serialize)]
struct BatchReceipt {
    advance: bool,
    batch: usize,
    exact_score_bits: bool,
    exact_top_ten: bool,
    packed: TimingSummary,
    packed_persistent_bytes: u64,
    resident_f32: TimingSummary,
    resident_f32_persistent_bytes: u64,
    score_plane_bytes: u64,
}

#[derive(Serialize)]
struct Receipt {
    advance: bool,
    batches: Vec<BatchReceipt>,
    device: String,
    dimensions: usize,
    gallery_rows: usize,
    schema: &'static str,
    source_commit: String,
    tileiras: String,
    warmups: usize,
}

fn packed_rows(rows: usize, seed: usize) -> PackedRows {
    let mut codes = Vec::with_capacity(rows * DIMENSIONS);
    let mut norms = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = codes.len();
        let mut squared_norm = 0i64;
        for column in 0..DIMENSIONS {
            let mixed = row
                .wrapping_mul(1_664_525)
                .wrapping_add(column.wrapping_mul(1_013_904_223))
                .wrapping_add(seed);
            let value = ((mixed % 255) as i16 - 127) as i8;
            codes.push(value);
            squared_norm += i64::from(value) * i64::from(value);
        }
        if squared_norm == 0 {
            codes[start] = 1;
            squared_norm = 1;
        }
        norms.push(f16::from_f32(1.0 / (squared_norm as f32).sqrt()));
    }
    PackedRows::new(codes, norms, rows, DIMENSIONS).expect("deterministic packed rows")
}

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

fn exactness(
    device: &std::sync::Arc<Device>,
    batch: BatchShape,
) -> Result<(bool, bool), Box<dyn Error>> {
    let rows = match batch {
        BatchShape::One => 1,
        BatchShape::ThirtyTwo => 32,
    };
    let queries = packed_rows(rows, 17);
    let gallery = packed_rows(129, 29);
    let scalar = scalar_scores(&queries, &gallery)?;
    let observed = score_cutile(device, &queries, &gallery, batch)?;
    let bit_equal = scalar
        .iter()
        .zip(&observed)
        .all(|(left, right)| left.to_bits() == right.to_bits());
    let top_equal = scalar
        .chunks(gallery.rows())
        .zip(observed.chunks(gallery.rows()))
        .all(|(left, right)| top_ten(left) == top_ten(right));
    Ok((bit_equal, top_equal))
}

fn tool_output(program: &str, argument: &str) -> String {
    Command::new(program)
        .arg(argument)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn main() -> Result<(), Box<dyn Error>> {
    let source_commit = std::env::args().nth(1).ok_or("source commit required")?;
    let device = Device::new(0)?;
    let gallery = packed_rows(GALLERY_ROWS, 41);
    let mut batches = Vec::new();
    for batch in [BatchShape::One, BatchShape::ThirtyTwo] {
        let batch_size = match batch {
            BatchShape::One => 1,
            BatchShape::ThirtyTwo => 32,
        };
        let queries = packed_rows(batch_size, 53);
        let measured =
            benchmark_score_planes(&device, &queries, &gallery, batch, WARMUPS, SAMPLES)?;
        let packed = validate_timing_samples(
            measured.packed_compile_ns,
            &measured.packed_samples_ns,
            SAMPLES,
        )
        .map_err(std::io::Error::other)?;
        let resident_f32 = validate_timing_samples(
            measured.resident_f32_compile_ns,
            &measured.resident_f32_samples_ns,
            SAMPLES,
        )
        .map_err(std::io::Error::other)?;
        let (exact_score_bits, exact_top_ten) = exactness(&device, batch)?;
        let advance = exact_score_bits && exact_top_ten && packed.p99_ns < resident_f32.p99_ns;
        batches.push(BatchReceipt {
            advance,
            batch: batch_size,
            exact_score_bits,
            exact_top_ten,
            packed,
            packed_persistent_bytes: measured.packed_persistent_bytes,
            resident_f32,
            resident_f32_persistent_bytes: measured.resident_f32_persistent_bytes,
            score_plane_bytes: measured.score_plane_bytes,
        });
    }
    let receipt = Receipt {
        advance: batches.iter().all(|batch| batch.advance),
        batches,
        device: device.name()?,
        dimensions: DIMENSIONS,
        gallery_rows: GALLERY_ROWS,
        schema: "sfora-packed-int8-cutile-score-v1",
        source_commit,
        tileiras: std::env::var("CUTILE_TILEIRAS_PATH")
            .map(|path| tool_output(&path, "--version"))
            .unwrap_or_else(|_| "unavailable".to_owned()),
        warmups: WARMUPS,
    };
    let mut bytes = serde_json::to_vec(&receipt)?;
    bytes.push(b'\n');
    std::io::Write::write_all(&mut std::io::stdout().lock(), &bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{nearest_rank_percentile, validate_timing_samples};

    #[test]
    fn receipt_recomputes_nearest_rank_percentiles_and_excludes_compile_time() {
        let raw_ns = (1u64..=50).collect::<Vec<_>>();

        let summary = validate_timing_samples(9_999, &raw_ns, 50).unwrap();

        assert_eq!(summary.compile_ns, 9_999);
        assert_eq!(summary.sample_count, 50);
        assert_eq!(summary.p50_ns, nearest_rank_percentile(&raw_ns, 50));
        assert_eq!(summary.p99_ns, nearest_rank_percentile(&raw_ns, 99));
        assert_eq!(summary.p50_ns, 25);
        assert_eq!(summary.p99_ns, 50);
        assert!(!raw_ns.contains(&summary.compile_ns));
        assert!(validate_timing_samples(9_999, &[], 50).is_err());
        assert!(validate_timing_samples(9_999, &raw_ns[..49], 50).is_err());
    }
}
