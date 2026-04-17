mod kernels;
mod poly_ops;

use core::sync::atomic::{AtomicUsize, Ordering};

pub use kernels::{barrett_reduce_vec, fqmul_vec};
pub use poly_ops::{
    poly_add, poly_add_assign, poly_basemul, poly_compress_coeffs, poly_decompress_coeffs,
    poly_mul_scalar_montgomery, poly_reduce, poly_sub, poly_to_montgomery,
};

/// SIMD lane width for `i16` operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum LaneWidth {
    W128Bit  = 128,
    W256Bit  = 256,
    W512Bit  = 512,
    W1024Bit = 1024,
}

static LANE_WIDTH: AtomicUsize = AtomicUsize::new(LaneWidth::W256Bit as usize);

/// Set the global SIMD lane width used by all polynomial / NTT operations.
pub fn set_lane_width(w: LaneWidth) {
    LANE_WIDTH.store(w as usize, Ordering::Relaxed);
}

#[must_use]
pub fn get_lane_width() -> LaneWidth {
    match LANE_WIDTH.load(Ordering::Relaxed) {
        v if v == LaneWidth::W1024Bit as usize => LaneWidth::W1024Bit,
        v if v == LaneWidth::W512Bit as usize => LaneWidth::W512Bit,
        v if v == LaneWidth::W256Bit as usize => LaneWidth::W256Bit,
        v if v == LaneWidth::W128Bit as usize => LaneWidth::W128Bit,
        _ => unreachable!(),
    }
}

/// Dispatch a generic `fn<const L: usize>(...)` over the runtime lane width.
macro_rules! dispatch_lanes {
    ($fn:ident ( $($arg:expr),* $(,)? )) => {
        match $crate::simd::get_lane_width() {
            $crate::simd::LaneWidth::W128Bit  => $fn::<8>($($arg),*),
            $crate::simd::LaneWidth::W256Bit => $fn::<16>($($arg),*),
            $crate::simd::LaneWidth::W512Bit => $fn::<32>($($arg),*),
            $crate::simd::LaneWidth::W1024Bit => $fn::<64>($($arg),*),
        }
    };
}
pub(crate) use dispatch_lanes;

#[cfg(test)]
mod tests {
    extern crate alloc;

    use super::*;

    const ALL_WIDTHS: [LaneWidth; 4] = [
        LaneWidth::W128Bit,
        LaneWidth::W256Bit,
        LaneWidth::W512Bit,
        LaneWidth::W1024Bit,
    ];

    #[test]
    fn set_get_roundtrip() {
        for &w in &ALL_WIDTHS {
            set_lane_width(w);
            assert_eq!(get_lane_width(), w);
        }
        set_lane_width(LaneWidth::W256Bit);
    }
}
