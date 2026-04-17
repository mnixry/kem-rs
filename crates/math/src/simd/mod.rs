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

/// Pick the widest vector width the target guarantees at compile time.
///
/// Most ISAs expose fixed-width SIMD directly. RISC-V RVV instead exposes
/// minimum vector lengths via `zvl*`; we map the widest guaranteed minimum onto
/// the closest supported lane width. Targets with scalable vectors that do not
/// expose a usable minimum length (for example SVE) fall back to the
/// historical 256-bit default.
pub const fn default_lane_width() -> LaneWidth {
    cfg_select! {
        any(
            all(target_arch = "hexagon", target_feature = "hvx-length128b"),
            all(any(target_arch = "riscv32", target_arch = "riscv64"), target_feature = "zvl1024b"),
        ) => LaneWidth::W1024Bit,
        any(
            all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx512bw"),
            all(target_arch = "hexagon", target_feature = "hvx"),
            all(any(target_arch = "riscv32", target_arch = "riscv64"), target_feature = "zvl512b"),
        ) => LaneWidth::W512Bit,
        any(
            all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"),
            all(target_arch = "loongarch64", target_feature = "lasx"),
            all(any(target_arch = "riscv32", target_arch = "riscv64"), target_feature = "zvl256b"),
        ) => LaneWidth::W256Bit,
        any(
            all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"),
            all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"),
            all(target_family = "wasm", target_feature = "simd128"),
            all(any(target_arch = "powerpc", target_arch = "powerpc64"), any(target_feature = "altivec", target_feature = "vsx")),
            all(target_arch = "loongarch64", target_feature = "lsx"),
            all(any(target_arch = "mips", target_arch = "mips64"), target_feature = "msa"),
            all(target_arch = "s390x", target_feature = "vector"),
            all(
                any(target_arch = "riscv32", target_arch = "riscv64"),
                any(
                    target_feature = "v",
                    target_feature = "zve32x",
                    target_feature = "zve32f",
                    target_feature = "zve64x",
                    target_feature = "zve64f",
                    target_feature = "zve64d",
                    target_feature = "zvl128b",
                ),
            ),
        ) => LaneWidth::W128Bit,
        _ => LaneWidth::W256Bit,
    }
}

static LANE_WIDTH: AtomicUsize = AtomicUsize::new(default_lane_width() as usize);

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
        set_lane_width(default_lane_width());
    }
}
