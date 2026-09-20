use std::ffi::c_void;
use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;

use cuda_core::Device;
use half::f16;

use crate::{BatchShape, PackedRows, PreparedPackedGallery};

const STATUS_OK: i32 = 0;
const STATUS_INVALID: i32 = 1;
const STATUS_RUNTIME: i32 = 2;
const STATUS_PANIC: i32 = 3;
const DIMENSIONS: usize = 128;
const TOP_K: usize = 10;

struct PackedGalleryHandle {
    gallery: PreparedPackedGallery,
}

fn aligned<T>(pointer: *const T) -> bool {
    !pointer.is_null() && (pointer as usize).is_multiple_of(align_of::<T>())
}

fn mutable_aligned<T>(pointer: *mut T) -> bool {
    aligned(pointer.cast_const())
}

fn ranges_overlap<T, U>(left: *mut T, left_len: usize, right: *mut U, right_len: usize) -> bool {
    let left_start = left as usize;
    let right_start = right as usize;
    let Some(left_bytes) = left_len.checked_mul(size_of::<T>()) else {
        return true;
    };
    let Some(right_bytes) = right_len.checked_mul(size_of::<U>()) else {
        return true;
    };
    let Some(left_end) = left_start.checked_add(left_bytes) else {
        return true;
    };
    let Some(right_end) = right_start.checked_add(right_bytes) else {
        return true;
    };
    left_start < right_end && right_start < left_end
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sfora_cutile_int8_create(
    codes: *const i8,
    inverse_norm_bits: *const u16,
    rows: usize,
    dimensions: usize,
    output_handle: *mut *mut c_void,
) -> i32 {
    if dimensions != DIMENSIONS
        || rows < TOP_K
        || !aligned(codes)
        || !aligned(inverse_norm_bits)
        || !mutable_aligned(output_handle)
        || rows.checked_mul(dimensions).is_none()
    {
        return STATUS_INVALID;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        let code_len = rows * dimensions;
        let copied_codes = unsafe { slice::from_raw_parts(codes, code_len) }.to_vec();
        let copied_norms = unsafe { slice::from_raw_parts(inverse_norm_bits, rows) }
            .iter()
            .copied()
            .map(f16::from_bits)
            .collect::<Vec<_>>();
        let packed = PackedRows::new(copied_codes, copied_norms, rows, dimensions)
            .map_err(|_| STATUS_INVALID)?;
        let device = Device::new(0).map_err(|_| STATUS_RUNTIME)?;
        let gallery = PreparedPackedGallery::new(&device, &packed).map_err(|_| STATUS_RUNTIME)?;
        let handle = Box::new(PackedGalleryHandle { gallery });
        unsafe { ptr::write(output_handle, Box::into_raw(handle).cast()) };
        Ok::<(), i32>(())
    })) {
        Ok(Ok(())) => STATUS_OK,
        Ok(Err(status)) => status,
        Err(_) => STATUS_PANIC,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sfora_cutile_int8_search(
    handle: *mut c_void,
    query_codes: *const i8,
    query_inverse_norm_bits: *const u16,
    query_rows: usize,
    dimensions: usize,
    k: usize,
    output_ordinals: *mut u32,
    output_scores: *mut f32,
) -> i32 {
    let Some(code_len) = query_rows.checked_mul(dimensions) else {
        return STATUS_INVALID;
    };
    let Some(output_len) = query_rows.checked_mul(k) else {
        return STATUS_INVALID;
    };
    if dimensions != DIMENSIONS
        || !matches!(query_rows, 1 | 32)
        || k != TOP_K
        || !mutable_aligned(handle.cast::<PackedGalleryHandle>())
        || !aligned(query_codes)
        || !aligned(query_inverse_norm_bits)
        || !mutable_aligned(output_ordinals)
        || !mutable_aligned(output_scores)
        || ranges_overlap(output_ordinals, output_len, output_scores, output_len)
    {
        return STATUS_INVALID;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        let copied_codes = unsafe { slice::from_raw_parts(query_codes, code_len) }.to_vec();
        let copied_norms = unsafe { slice::from_raw_parts(query_inverse_norm_bits, query_rows) }
            .iter()
            .copied()
            .map(f16::from_bits)
            .collect::<Vec<_>>();
        let queries = PackedRows::new(copied_codes, copied_norms, query_rows, dimensions)
            .map_err(|_| STATUS_INVALID)?;
        let batch = if query_rows == 1 {
            BatchShape::One
        } else {
            BatchShape::ThirtyTwo
        };
        let gallery = unsafe { &mut *handle.cast::<PackedGalleryHandle>() };
        let result = gallery
            .gallery
            .search(&queries, batch, k)
            .map_err(|_| STATUS_RUNTIME)?;
        if result.len() != output_len {
            return Err(STATUS_RUNTIME);
        }
        unsafe {
            ptr::copy_nonoverlapping(result.ordinals().as_ptr(), output_ordinals, output_len);
            ptr::copy_nonoverlapping(result.scores().as_ptr(), output_scores, output_len);
        }
        Ok::<(), i32>(())
    })) {
        Ok(Ok(())) => STATUS_OK,
        Ok(Err(status)) => status,
        Err(_) => STATUS_PANIC,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sfora_cutile_int8_destroy(handle: *mut c_void) -> i32 {
    if !mutable_aligned(handle.cast::<PackedGalleryHandle>()) {
        return STATUS_INVALID;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(handle.cast::<PackedGalleryHandle>()) });
    })) {
        Ok(()) => STATUS_OK,
        Err(_) => STATUS_PANIC,
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::ptr;

    use half::f16;

    use super::{sfora_cutile_int8_create, sfora_cutile_int8_destroy, sfora_cutile_int8_search};

    fn fixture(rows: usize, seed: usize) -> (Vec<i8>, Vec<u16>) {
        let mut codes = Vec::with_capacity(rows * 128);
        let mut norms = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = codes.len();
            let mut squared = 0i64;
            for column in 0..128 {
                let value = (((row * 29 + column * 31 + seed) % 255) as i16 - 127) as i8;
                codes.push(value);
                squared += i64::from(value) * i64::from(value);
            }
            if squared == 0 {
                codes[start] = 1;
                squared = 1;
            }
            norms.push(f16::from_f32(1.0 / (squared as f32).sqrt()).to_bits());
        }
        (codes, norms)
    }

    #[test]
    fn ffi_fails_closed_and_round_trips_exact_top_ten() {
        unsafe {
            let mut handle: *mut c_void = ptr::null_mut();
            assert_ne!(
                sfora_cutile_int8_create(ptr::null(), ptr::null(), 10, 128, &mut handle),
                0
            );
            let (gallery_codes, gallery_norms) = fixture(129, 7);
            assert_eq!(
                sfora_cutile_int8_create(
                    gallery_codes.as_ptr(),
                    gallery_norms.as_ptr(),
                    129,
                    128,
                    &mut handle,
                ),
                0
            );
            assert!(!handle.is_null());
            let (query_codes, query_norms) = fixture(1, 11);
            let mut ordinals = [u32::MAX; 10];
            let mut scores = [f32::NAN; 10];
            assert_ne!(
                sfora_cutile_int8_search(
                    handle,
                    query_codes.as_ptr(),
                    query_norms.as_ptr(),
                    1,
                    127,
                    10,
                    ordinals.as_mut_ptr(),
                    scores.as_mut_ptr(),
                ),
                0
            );
            assert_eq!(ordinals, [u32::MAX; 10]);
            assert!(scores.iter().all(|value| value.is_nan()));
            assert_eq!(
                sfora_cutile_int8_search(
                    handle,
                    query_codes.as_ptr(),
                    query_norms.as_ptr(),
                    1,
                    128,
                    10,
                    ordinals.as_mut_ptr(),
                    scores.as_mut_ptr(),
                ),
                0
            );
            assert!(ordinals.iter().all(|&ordinal| ordinal < 129));
            assert!(ordinals.windows(2).all(|pair| pair[0] != pair[1]));
            assert!(scores.iter().all(|value| value.is_finite()));
            assert_eq!(sfora_cutile_int8_destroy(handle), 0);
            assert_ne!(sfora_cutile_int8_destroy(ptr::null_mut()), 0);
        }
    }
}
