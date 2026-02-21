//! Number-Theoretic Transform and base multiplication in `Z_q[X]/(X^2 - zeta)`.

use core::simd::Simd;

use crate::{
    N,
    simd::{barrett_reduce_vec, fqmul_vec},
};

const Q64: i64 = crate::Q as i64;

const fn pow_mod(mut base: i64, mut exp: i64, modulus: i64) -> i64 {
    let mut result: i64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % modulus;
        }
        exp >>= 1;
        base = base * base % modulus;
    }
    result
}

const fn bitrev7(x: usize) -> usize {
    ((x >> 6) & 1)
        | (((x >> 5) & 1) << 1)
        | (((x >> 4) & 1) << 2)
        | (((x >> 3) & 1) << 3)
        | (((x >> 2) & 1) << 4)
        | (((x >> 1) & 1) << 5)
        | ((x & 1) << 6)
}

/// Centred representative of `val mod q` in `[-(q-1)/2, (q-1)/2]`.
const fn centred(val: i64) -> i16 {
    if val > Q64 / 2 {
        (val - Q64) as i16
    } else {
        val as i16
    }
}

/// Twiddle factors in Montgomery form, from primitive 512th root \zeta = 17,
/// bit-reversed indexing.
///
/// `ZETAS[i] = \zeta^{BitRev_7(i)} * 2^{16}  (mod q)`, centred to signed.
///
/// Values match Appendix A of FIPS 203 (scaled into Montgomery domain).
pub const ZETAS: [i16; 128] = {
    const ZETA: i64 = 17;
    const MONT: i64 = 1 << 16;

    let mut zetas = [0i16; 128];
    let mut i = 0;
    while i < 128 {
        let val = pow_mod(ZETA, bitrev7(i) as i64, Q64) * MONT % Q64;
        zetas[i] = centred(val);
        i += 1;
    }
    zetas
};

/// Single NTT butterfly layer. `$sw` (SIMD width) must divide `$len`.
/// When `$sw == $len` the inner loop executes exactly once (no tail).
macro_rules! ntt_layer {
    (fwd, $r:ident, $k:ident, $len:literal, $sw:literal) => {{
        let mut start = 0usize;
        while start < N {
            let zeta = ZETAS[$k];
            $k += 1;
            let (lo, hi) = $r[start..start + 2 * $len].split_at_mut($len);
            let z = Simd::<i16, $sw>::splat(zeta);
            let mut i = 0usize;
            while i < $len {
                let a = Simd::<i16, $sw>::from_slice(&lo[i..]);
                let b = Simd::<i16, $sw>::from_slice(&hi[i..]);
                let t = fqmul_vec(z, b);
                (a + t).copy_to_slice(&mut lo[i..i + $sw]);
                (a - t).copy_to_slice(&mut hi[i..i + $sw]);
                i += $sw;
            }
            start += 2 * $len;
        }
    }};
    (inv, $r:ident, $k:ident, $len:literal, $sw:literal) => {{
        let mut start = 0usize;
        while start < N {
            let zeta = ZETAS[$k];
            $k = $k.wrapping_sub(1);
            let (lo, hi) = $r[start..start + 2 * $len].split_at_mut($len);
            let z = Simd::<i16, $sw>::splat(zeta);
            let mut i = 0usize;
            while i < $len {
                let a = Simd::<i16, $sw>::from_slice(&lo[i..]);
                let b = Simd::<i16, $sw>::from_slice(&hi[i..]);
                barrett_reduce_vec(a + b).copy_to_slice(&mut lo[i..i + $sw]);
                fqmul_vec(z, b - a).copy_to_slice(&mut hi[i..i + $sw]);
                i += $sw;
            }
            start += 2 * $len;
        }
    }};
}

/// Dispatch one layer with `simd_width = min(len, lanes)`.
///
/// Pattern-matching computes the minimum at macro-expansion time:
/// - `len = 128` always exceeds max supported lanes (64) -> use lanes.
/// - `len in {2, 4, 8}` always fits in min supported lanes (8) -> use len.
/// - Remaining lengths match specific lane values for the crossover.
macro_rules! ntt_dispatch_layer {
    ($dir:ident, $r:ident, $k:ident, 128, $lanes:literal) => {
        ntt_layer!($dir, $r, $k, 128, $lanes);
    };
    ($dir:ident, $r:ident, $k:ident, 2, $lanes:literal) => {
        ntt_layer!($dir, $r, $k, 2, 2);
    };
    ($dir:ident, $r:ident, $k:ident, 4, $lanes:literal) => {
        ntt_layer!($dir, $r, $k, 4, 4);
    };
    ($dir:ident, $r:ident, $k:ident, 8, $lanes:literal) => {
        ntt_layer!($dir, $r, $k, 8, 8);
    };
    // len = 16: min(16, lanes)
    ($dir:ident, $r:ident, $k:ident, 16, 8) => {
        ntt_layer!($dir, $r, $k, 16, 8);
    };
    ($dir:ident, $r:ident, $k:ident, 16, $lanes:literal) => {
        ntt_layer!($dir, $r, $k, 16, 16);
    };
    // len = 32: min(32, lanes)
    ($dir:ident, $r:ident, $k:ident, 32, 8) => {
        ntt_layer!($dir, $r, $k, 32, 8);
    };
    ($dir:ident, $r:ident, $k:ident, 32, 16) => {
        ntt_layer!($dir, $r, $k, 32, 16);
    };
    ($dir:ident, $r:ident, $k:ident, 32, $lanes:literal) => {
        ntt_layer!($dir, $r, $k, 32, 32);
    };
    // len = 64: min(64, lanes)
    ($dir:ident, $r:ident, $k:ident, 64, 8) => {
        ntt_layer!($dir, $r, $k, 64, 8);
    };
    ($dir:ident, $r:ident, $k:ident, 64, 16) => {
        ntt_layer!($dir, $r, $k, 64, 16);
    };
    ($dir:ident, $r:ident, $k:ident, 64, 32) => {
        ntt_layer!($dir, $r, $k, 64, 32);
    };
    ($dir:ident, $r:ident, $k:ident, 64, $lanes:literal) => {
        ntt_layer!($dir, $r, $k, 64, 64);
    };
}

/// Forward NTT (in-place). Standard order in, bit-reversed order out.
pub fn forward_ntt(r: &mut [i16; N]) {
    macro_rules! body {
        ($lanes:literal) => {{
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
        crate::simd::LaneWidth::L8 => body!(8),
        crate::simd::LaneWidth::L16 => body!(16),
        crate::simd::LaneWidth::L32 => body!(32),
        crate::simd::LaneWidth::L64 => body!(64),
    }
}

/// Inverse NTT (in-place). Bit-reversed in, standard order out,
/// each coefficient scaled by Montgomery factor `R = 2^{16}`.
pub fn inverse_ntt(r: &mut [i16; N]) {
    const F: i16 = centred(pow_mod(2, 32, Q64) * pow_mod(128, Q64 - 2, Q64) % Q64);
    macro_rules! body {
        ($lanes:literal) => {{
            let mut k = 127usize;
            ntt_dispatch_layer!(inv, r, k, 2, $lanes);
            ntt_dispatch_layer!(inv, r, k, 4, $lanes);
            ntt_dispatch_layer!(inv, r, k, 8, $lanes);
            ntt_dispatch_layer!(inv, r, k, 16, $lanes);
            ntt_dispatch_layer!(inv, r, k, 32, $lanes);
            ntt_dispatch_layer!(inv, r, k, 64, $lanes);
            ntt_dispatch_layer!(inv, r, k, 128, $lanes);
        }};
    }
    match crate::simd::get_lane_width() {
        crate::simd::LaneWidth::L8 => body!(8),
        crate::simd::LaneWidth::L16 => body!(16),
        crate::simd::LaneWidth::L32 => body!(32),
        crate::simd::LaneWidth::L64 => body!(64),
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
}
