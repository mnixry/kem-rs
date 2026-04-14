mod kernels;
mod poly_ops;

use core::{
    fmt::Display,
    sync::atomic::{AtomicU8, Ordering},
};

pub use kernels::{barrett_reduce_vec, fqmul_vec};
pub use poly_ops::{
    poly_add, poly_add_assign, poly_basemul, poly_compress_coeffs, poly_decompress_coeffs,
    poly_mul_scalar_montgomery, poly_reduce, poly_sub, poly_to_montgomery,
};

/// SIMD lane width for `i16` operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LaneWidth {
    L8  = 8,
    L16 = 16,
    L32 = 32,
    L64 = 64,
}

impl Display for LaneWidth {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::L8 => write!(f, "8"),
            Self::L16 => write!(f, "16"),
            Self::L32 => write!(f, "32"),
            Self::L64 => write!(f, "64"),
        }
    }
}

impl From<u8> for LaneWidth {
    fn from(value: u8) -> Self {
        match value {
            8 => Self::L8,
            16 => Self::L16,
            32 => Self::L32,
            64 => Self::L64,
            _ => unreachable!(),
        }
    }
}

static LANE_WIDTH: AtomicU8 = AtomicU8::new(LaneWidth::L16 as u8);

/// Set the global SIMD lane width used by all polynomial / NTT operations.
pub fn set_lane_width(w: LaneWidth) {
    LANE_WIDTH.store(w as u8, Ordering::Relaxed);
}

#[must_use]
pub fn get_lane_width() -> LaneWidth {
    LANE_WIDTH.load(Ordering::Relaxed).into()
}

/// Dispatch a generic `fn<const L: usize>(...)` over the runtime lane width.
macro_rules! dispatch_lanes {
    ($fn:ident ( $($arg:expr),* $(,)? )) => {
        match $crate::simd::get_lane_width() {
            $crate::simd::LaneWidth::L8  => $fn::<8>($($arg),*),
            $crate::simd::LaneWidth::L16 => $fn::<16>($($arg),*),
            $crate::simd::LaneWidth::L32 => $fn::<32>($($arg),*),
            $crate::simd::LaneWidth::L64 => $fn::<64>($($arg),*),
        }
    };
}
pub(crate) use dispatch_lanes;

#[cfg(test)]
mod tests {
    extern crate alloc;

    use super::*;

    const ALL_WIDTHS: [LaneWidth; 4] = [
        LaneWidth::L8,
        LaneWidth::L16,
        LaneWidth::L32,
        LaneWidth::L64,
    ];

    #[test]
    fn set_get_roundtrip() {
        for &w in &ALL_WIDTHS {
            set_lane_width(w);
            assert_eq!(get_lane_width(), w);
        }
        set_lane_width(LaneWidth::L16);
    }

    #[test]
    fn display() {
        use alloc::string::ToString;
        assert_eq!(LaneWidth::L8.to_string(), "8");
        assert_eq!(LaneWidth::L16.to_string(), "16");
        assert_eq!(LaneWidth::L32.to_string(), "32");
        assert_eq!(LaneWidth::L64.to_string(), "64");
    }

    #[test]
    fn from_u8() {
        for &w in &ALL_WIDTHS {
            assert_eq!(LaneWidth::from(w as u8), w);
        }
    }
}
