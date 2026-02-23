mod kernels;
mod poly_ops;

use core::{
    fmt::Display,
    sync::atomic::{AtomicU8, Ordering},
};

pub use kernels::{barrett_reduce_vec, fqmul_vec};
pub use poly_ops::{
    poly_add, poly_add_assign, poly_basemul, poly_mul_scalar_montgomery, poly_reduce, poly_sub,
    poly_to_montgomery,
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
