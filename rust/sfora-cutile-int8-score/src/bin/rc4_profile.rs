//! Diagnostic-only RC4 packed top-k replay; no public FFI symbols.

use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::Instant;

use cuda_core::Device;
use half::f16;
use serde_json::json;
use sfora_cutile_int8_score::{BatchShape, PackedRows, PreparedPackedGallery, TopKResult};

const DIMENSIONS: usize = 128;
const WARMUPS: usize = 5;
const SAMPLES: usize = 50;

fn decode_norms(bytes: &[u8]) -> Result<Vec<f16>, &'static str> {
    let (chunks, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err("inverse norm bytes are not aligned to f16");
    }
    Ok(chunks
        .iter()
        .map(|bytes| f16::from_bits(u16::from_le_bytes(*bytes)))
        .collect())
}

fn read_packed(
    codes_path: &Path,
    norms_path: &Path,
    rows: usize,
) -> Result<PackedRows, Box<dyn Error>> {
    let codes = fs::read(codes_path)?;
    let norms = decode_norms(&fs::read(norms_path)?)?;
    if codes.len() != rows * DIMENSIONS || norms.len() != rows {
        return Err("packed input length differs".into());
    }
    Ok(PackedRows::new(
        codes.into_iter().map(|byte| byte as i8).collect(),
        norms,
        rows,
        DIMENSIONS,
    )?)
}

fn elapsed_ns(start: Instant) -> Result<u64, Box<dyn Error>> {
    Ok(u64::try_from(start.elapsed().as_nanos())?)
}

fn peak_rss_bytes() -> Result<u64, Box<dyn Error>> {
    let status = fs::read_to_string("/proc/self/status")?;
    let line = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .ok_or("VmHWM is absent")?;
    let kibibytes: u64 = line
        .split_whitespace()
        .nth(1)
        .ok_or("VmHWM value is absent")?
        .parse()?;
    Ok(kibibytes * 1024)
}

fn score_bits(result: &TopKResult) -> Vec<u32> {
    result
        .scores()
        .iter()
        .map(|score| score.to_bits())
        .collect()
}

fn search(
    prepared: &mut PreparedPackedGallery,
    queries: &PackedRows,
    batch: BatchShape,
    mode: &str,
) -> Result<TopKResult, Box<dyn Error>> {
    match mode {
        "fused" => Ok(prepared.search(queries, batch, 10)?),
        "split" => Ok(prepared.search_split_for_profile(queries, batch, 10)?),
        _ => Err("profile mode differs".into()),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 8 {
        return Err("usage: rc4_profile MODE GALLERY_ROWS BATCH GALLERY_CODES GALLERY_NORMS QUERY_CODES QUERY_NORMS".into());
    }
    let mode = args[1].as_str();
    if !matches!(mode, "exactness" | "fused" | "split") {
        return Err("profile mode differs".into());
    }
    let gallery_rows: usize = args[2].parse()?;
    let batch_rows: usize = args[3].parse()?;
    let batch = match batch_rows {
        1 => BatchShape::One,
        32 => BatchShape::ThirtyTwo,
        _ => return Err("profile batch differs".into()),
    };
    let gallery = read_packed(Path::new(&args[4]), Path::new(&args[5]), gallery_rows)?;
    let queries = read_packed(Path::new(&args[6]), Path::new(&args[7]), batch_rows)?;
    let device = Device::new(0)?;
    let mut prepared = PreparedPackedGallery::new(&device, &gallery)?;
    if mode == "exactness" {
        let fused = search(&mut prepared, &queries, batch, "fused")?;
        let split = search(&mut prepared, &queries, batch, "split")?;
        println!(
            "{}",
            json!({
                "mode": mode,
                "gallery_rows": gallery_rows,
                "batch": batch_rows,
                "exact_ordinals": fused.ordinals() == split.ordinals(),
                "exact_score_bits": score_bits(&fused) == score_bits(&split),
                "fused_ordinals": fused.ordinals(),
                "fused_score_bits": score_bits(&fused),
                "split_ordinals": split.ordinals(),
                "split_score_bits": score_bits(&split),
                "process_peak_rss_bytes": peak_rss_bytes()?,
            })
        );
        return Ok(());
    }
    let compile_start = Instant::now();
    let first = search(&mut prepared, &queries, batch, mode)?;
    let first_call_ns = elapsed_ns(compile_start)?;
    for _ in 0..WARMUPS {
        search(&mut prepared, &queries, batch, mode)?;
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        search(&mut prepared, &queries, batch, mode)?;
        samples.push(elapsed_ns(start)?);
    }
    println!(
        "{}",
        json!({
            "mode": mode,
            "gallery_rows": gallery_rows,
            "batch": batch_rows,
            "first_call_ns": first_call_ns,
            "warmups": WARMUPS,
            "raw_end_to_end_ns": samples,
            "ordinals": first.ordinals(),
            "score_bits": score_bits(&first),
            "process_peak_rss_bytes": peak_rss_bytes()?,
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::decode_norms;

    #[test]
    fn f16_wire_rejects_odd_bytes_and_preserves_little_endian_bits() {
        assert!(decode_norms(&[0]).is_err());
        assert_eq!(decode_norms(&[0x00, 0x3c]).unwrap()[0].to_bits(), 0x3c00);
    }
}
