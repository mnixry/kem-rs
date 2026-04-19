//! Number-Theoretic Transform and base multiplication in `Z_q[X]/(X^2 - zeta)`.

use crate::{
    N,
    simd::{
        Q64, centred, ntt_layer_fwd, ntt_layer_fwd_packed, ntt_layer_inv_nored,
        ntt_layer_inv_nored_packed, pow_mod,
    },
};

macro_rules! ntt_layer {
    (fwd, $r:ident, $k:ident, $len:literal, $sw:literal) => {
        ntt_layer_fwd::<$len, $sw>($r, &mut $k);
    };
    (inv_nored, $r:ident, $k:ident, $len:literal, $sw:literal) => {
        ntt_layer_inv_nored::<$len, $sw>($r, &mut $k);
    };
}

/// Packed NTT butterfly layer for small layers where `$len < $lanes`.
///
/// Packs `$lanes / $len` adjacent butterfly groups into full-width
/// `Simd<i16, $lanes>` vectors using block-deinterleave shuffles.
/// This achieves full SIMD utilization for the small NTT layers
/// (len = 2, 4, ...) that would otherwise waste register lanes.
///
/// On `AArch64`, `simd_swizzle!` compiles to `uzp1/uzp2` (deinterleave)
/// and `zip1/zip2` (interleave) at the appropriate element granularity.
macro_rules! ntt_layer_packed {
    (fwd, $r:ident, $k:ident, $len:literal, $lanes:literal) => {
        ntt_layer_fwd_packed::<$len, $lanes>($r, &mut $k);
    };
    (inv_nored, $r:ident, $k:ident, $len:literal, $lanes:literal) => {
        ntt_layer_inv_nored_packed::<$len, $lanes>($r, &mut $k);
    };
}

/// Dispatch one layer with the most efficient strategy:
///
/// - When `len >= lanes`: use the regular `ntt_layer!` with `sw = min(len,
///   lanes)`.
/// - When `len < lanes`: use `ntt_layer_packed!` to pack multiple butterfly
///   groups into full-width SIMD vectors.
macro_rules! ntt_dispatch_layer {
    // len=128: always exceeds max supported lanes (64) → regular.
    ($dir:ident, $r:ident, $k:ident, 128, $lanes:tt) => {
        ntt_layer!($dir, $r, $k, 128, $lanes);
    };

    // len=64: equals max lanes (64), never less → regular with sw = min(64, lanes).
    ($dir:ident, $r:ident, $k:ident, 64, 8) => {
        ntt_layer!($dir, $r, $k, 64, 8);
    };
    ($dir:ident, $r:ident, $k:ident, 64, 16) => {
        ntt_layer!($dir, $r, $k, 64, 16);
    };
    ($dir:ident, $r:ident, $k:ident, 64, 32) => {
        ntt_layer!($dir, $r, $k, 64, 32);
    };
    ($dir:ident, $r:ident, $k:ident, 64, $lanes:tt) => {
        ntt_layer!($dir, $r, $k, 64, 64);
    };

    // len=32: regular when lanes ≤ 32, packed when lanes > 32.
    ($dir:ident, $r:ident, $k:ident, 32, 8) => {
        ntt_layer!($dir, $r, $k, 32, 8);
    };
    ($dir:ident, $r:ident, $k:ident, 32, 16) => {
        ntt_layer!($dir, $r, $k, 32, 16);
    };
    ($dir:ident, $r:ident, $k:ident, 32, 32) => {
        ntt_layer!($dir, $r, $k, 32, 32);
    };
    ($dir:ident, $r:ident, $k:ident, 32, $lanes:tt) => {
        ntt_layer_packed!($dir, $r, $k, 32, $lanes);
    };

    // len=16: regular when lanes ≤ 16, packed when lanes > 16.
    ($dir:ident, $r:ident, $k:ident, 16, 8) => {
        ntt_layer!($dir, $r, $k, 16, 8);
    };
    ($dir:ident, $r:ident, $k:ident, 16, 16) => {
        ntt_layer!($dir, $r, $k, 16, 16);
    };
    ($dir:ident, $r:ident, $k:ident, 16, $lanes:tt) => {
        ntt_layer_packed!($dir, $r, $k, 16, $lanes);
    };

    // len=8: regular when lanes = 8, packed when lanes > 8.
    ($dir:ident, $r:ident, $k:ident, 8, 8) => {
        ntt_layer!($dir, $r, $k, 8, 8);
    };
    ($dir:ident, $r:ident, $k:ident, 8, $lanes:tt) => {
        ntt_layer_packed!($dir, $r, $k, 8, $lanes);
    };

    // len=4: always less than min lanes (8) → packed.
    ($dir:ident, $r:ident, $k:ident, 4, $lanes:tt) => {
        ntt_layer_packed!($dir, $r, $k, 4, $lanes);
    };

    // len=2: always less than min lanes (8) → packed.
    ($dir:ident, $r:ident, $k:ident, 2, $lanes:tt) => {
        ntt_layer_packed!($dir, $r, $k, 2, $lanes);
    };
}

/// Forward NTT (in-place). Standard order in, bit-reversed order out.
pub fn forward_ntt(r: &mut [i16; N]) {
    macro_rules! body {
        ($lanes:tt) => {{
            let mut k = 1usize;
            ntt_dispatch_layer!(fwd, r, k, 128, $lanes);
            ntt_dispatch_layer!(fwd, r, k, 64, $lanes);
            ntt_dispatch_layer!(fwd, r, k, 32, $lanes);
            ntt_dispatch_layer!(fwd, r, k, 16, $lanes);
            ntt_dispatch_layer!(fwd, r, k, 8, $lanes);
            ntt_dispatch_layer!(fwd, r, k, 4, $lanes);
            ntt_dispatch_layer!(fwd, r, k, 2, $lanes);
        }};
    }
    match crate::simd::get_lane_width() {
        crate::simd::LaneWidth::W128Bit => body!(8),
        crate::simd::LaneWidth::W256Bit => body!(16),
        crate::simd::LaneWidth::W512Bit => body!(32),
        crate::simd::LaneWidth::W1024Bit => body!(64),
    }
}

/// Inverse NTT (in-place). Bit-reversed in, standard order out,
/// each coefficient scaled by Montgomery factor `R = 2^{16}`.
///
/// Uses lazy Barrett reduction: an initial `poly_reduce` brings all
/// coefficients into `[-q/2, q/2]`, then layers 1–4 (len 2..16) run without
/// per-butterfly Barrett reduction because `|a + b|` stays within i16 range (≤
/// 27 312). A second `poly_reduce` after layer 4 restores the invariant for
/// layers 5–7.
pub fn inverse_ntt(r: &mut [i16; N]) {
    const F: i16 = centred(pow_mod(2, 32, Q64) * pow_mod(128, Q64 - 2, Q64) % Q64);
    macro_rules! body {
        ($lanes:tt) => {{
            let mut k = 127usize;
            // Reduce to [-q/2, q/2] so layers 1–4 cannot overflow i16.
            crate::simd::poly_reduce(r);
            ntt_dispatch_layer!(inv_nored, r, k, 2, $lanes);
            ntt_dispatch_layer!(inv_nored, r, k, 4, $lanes);
            ntt_dispatch_layer!(inv_nored, r, k, 8, $lanes);
            ntt_dispatch_layer!(inv_nored, r, k, 16, $lanes);
            // Reduce again: max value after 4 lazy layers ≈ 27 312.
            crate::simd::poly_reduce(r);
            ntt_dispatch_layer!(inv_nored, r, k, 32, $lanes);
            ntt_dispatch_layer!(inv_nored, r, k, 64, $lanes);
            ntt_dispatch_layer!(inv_nored, r, k, 128, $lanes);
        }};
    }
    match crate::simd::get_lane_width() {
        crate::simd::LaneWidth::W128Bit => body!(8),
        crate::simd::LaneWidth::W256Bit => body!(16),
        crate::simd::LaneWidth::W512Bit => body!(32),
        crate::simd::LaneWidth::W1024Bit => body!(64),
    }
    crate::simd::poly_mul_scalar_montgomery(r, F);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduce::{barrett_reduce, fqmul};

    #[test]
    fn ntt_inverse_ntt_roundtrip() {
        let mut a = [0i16; N];
        for (i, c) in a.iter_mut().enumerate() {
            *c = (i % 13) as i16;
        }
        let original = a;
        forward_ntt(&mut a);
        assert_ne!(a, original, "NTT should change coefficients");
        inverse_ntt(&mut a);

        for c in &mut a {
            *c = barrett_reduce(fqmul(*c, 1));
        }
        for i in 0..N {
            assert_eq!(a[i], original[i], "mismatch at index {i}");
        }
    }

    fn schoolbook_mul(a: &[i16; N], b: &[i16; N]) -> [i16; N] {
        let mut c = [0i32; N];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                let prod = (ai as i32) * (bj as i32);
                if i + j < N {
                    c[i + j] += prod;
                } else {
                    c[i + j - N] -= prod;
                }
            }
        }
        let q = crate::Q as i32;
        let mut result = [0i16; N];
        for (r, &ci) in result.iter_mut().zip(c.iter()) {
            *r = (ci % q) as i16;
            if *r < 0 {
                *r += q as i16;
            }
        }
        result
    }

    fn normalise(c: i16) -> i16 {
        let mut v = barrett_reduce(c);
        if v < 0 {
            v += crate::Q;
        }
        v
    }

    #[test]
    fn ntt_basemul_matches_schoolbook() {
        use crate::poly::Polynomial;

        let mut a = Polynomial::zero();
        let mut b = Polynomial::zero();
        for i in 0..N {
            a.0[i] = ((i * 7 + 3) % 100) as i16;
            b.0[i] = ((i * 13 + 1) % 100) as i16;
        }
        let expected = schoolbook_mul(&a.0, &b.0);

        let a_ntt = a.ntt();
        let b_ntt = b.ntt();

        let c_ntt = a_ntt.basemul(&b_ntt);
        let c = c_ntt.ntt_inverse();

        for (i, (&got_raw, &exp)) in c.coeffs().iter().zip(expected.iter()).enumerate() {
            let got = normalise(got_raw);
            assert_eq!(got, exp, "mismatch at {i}: got {got}, expected {exp}");
        }
    }

    #[test]
    fn ntt_roundtrip_all_lane_widths() {
        use crate::simd::{LaneWidth, default_lane_width, set_lane_width};

        let mut a = [0i16; N];
        for (i, c) in a.iter_mut().enumerate() {
            *c = (i % 13) as i16;
        }
        let original = a;

        for width in [
            LaneWidth::W128Bit,
            LaneWidth::W256Bit,
            LaneWidth::W512Bit,
            LaneWidth::W1024Bit,
        ] {
            let mut r = original;
            set_lane_width(width);
            forward_ntt(&mut r);
            assert_ne!(r, original, "NTT no-op at width {width:?}");
            inverse_ntt(&mut r);
            for c in &mut r {
                *c = barrett_reduce(fqmul(*c, 1));
            }
            assert_eq!(r, original, "roundtrip failed at width {width:?}");
        }
        set_lane_width(default_lane_width());
    }

    #[test]
    fn ntt_basemul_all_lane_widths() {
        use crate::{
            poly::Polynomial,
            simd::{LaneWidth, default_lane_width, set_lane_width},
        };

        let mut a = Polynomial::zero();
        let mut b = Polynomial::zero();
        for i in 0..N {
            a.0[i] = ((i * 7 + 3) % 100) as i16;
            b.0[i] = ((i * 13 + 1) % 100) as i16;
        }
        let expected = schoolbook_mul(&a.0, &b.0);

        for width in [
            LaneWidth::W128Bit,
            LaneWidth::W256Bit,
            LaneWidth::W512Bit,
            LaneWidth::W1024Bit,
        ] {
            set_lane_width(width);
            let a_ntt = a.ntt();
            let b_ntt = b.ntt();
            let c = a_ntt.basemul(&b_ntt).ntt_inverse();
            for (i, (&got_raw, &exp)) in c.coeffs().iter().zip(expected.iter()).enumerate() {
                let got = normalise(got_raw);
                assert_eq!(got, exp, "mismatch at {i} (width {width:?})");
            }
        }
        set_lane_width(default_lane_width());
    }
}
