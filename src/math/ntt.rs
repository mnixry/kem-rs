//! Number-Theoretic Transform and base multiplication in `Z_q[X]/(X^2 - zeta)`.
//!
//! - `ntt`: forward NTT, standard order -> bit-reversed order.
//! - `invntt`: inverse NTT with Montgomery scaling.
//! - `basemul`: degree-1 multiplication in the NTT domain.

use super::reduce::fqmul;
use crate::params::N;

/// Twiddle factors in Montgomery form, from primitive 512th root zeta=17,
/// bit-reversed indexing.
pub static ZETAS: [i16; 128] = [
    -1044, -758, -359, -1517, 1493, 1422, 287, 202, -171, 622, 1577, 182, 962, -1202, -1474,
    1468, 573, -1325, 264, 383, -829, 1458, -1602, -130, -681, 1017, 732, 608, -1542, 411, -205,
    -1571, 1223, 652, -552, 1015, -1293, 1491, -282, -1544, 516, -8, -320, -666, -1618, -1162,
    126, 1469, -853, -90, -271, 830, 107, -1421, -247, -951, -398, 961, -1508, -725, 448, -1065,
    677, -1275, -1103, 430, 555, 843, -1251, 871, 1550, 105, 422, 587, 177, -235, -291, -460,
    1574, 1653, -246, 778, 1159, -147, -777, 1483, -602, 1119, -1590, 644, -872, 349, 418, 329,
    -156, -75, 817, 1097, 603, 610, 1322, -1285, -1465, 384, -1215, -136, 1218, -1335, -874, 220,
    -1187, -1659, -1185, -1530, -1278, 794, -1510, -854, -870, 478, -108, -308, 996, 991, 958,
    -1460, 1522, 1628,
];

/// Forward NTT (in-place). Standard order in, bit-reversed order out.
pub fn ntt(r: &mut [i16; N]) {
    let mut k: usize = 1;
    let mut len = 128;
    while len >= 2 {
        let mut start = 0;
        while start < N {
            let zeta = ZETAS[k];
            k += 1;
            let (lo, hi) = r[start..start + 2 * len].split_at_mut(len);
            crate::simd::butterfly_fwd(lo, hi, zeta);
            start += 2 * len;
        }
        len >>= 1;
    }
}

/// Inverse NTT (in-place). Bit-reversed in, standard order out,
/// each coefficient scaled by Montgomery factor `R = 2^{16}`.
pub fn invntt(r: &mut [i16; N]) {
    const F: i16 = 1441; // mont^2 * 128^{-1} mod q
    let mut k: usize = 127;
    let mut len = 2;
    while len <= 128 {
        let mut start = 0;
        while start < N {
            let zeta = ZETAS[k];
            k = k.wrapping_sub(1);
            let (lo, hi) = r[start..start + 2 * len].split_at_mut(len);
            crate::simd::butterfly_inv(lo, hi, zeta);
            start += 2 * len;
        }
        len <<= 1;
    }
    crate::simd::poly_fqmul_scalar(r, F);
}

/// Base multiply two degree-1 polys in `Z_q[X]/(X^2 - zeta)`.
/// `r = a * b mod (X^2 - zeta)`.
#[inline]
pub fn basemul(r: &mut [i16; 2], a: &[i16; 2], b: &[i16; 2], zeta: i16) {
    r[0] = fqmul(a[1], b[1]);
    r[0] = fqmul(r[0], zeta);
    r[0] += fqmul(a[0], b[0]);
    r[1] = fqmul(a[0], b[1]);
    r[1] += fqmul(a[1], b[0]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::reduce::barrett_reduce;

    #[test]
    fn ntt_invntt_roundtrip() {
        let mut a = [0i16; N];
        for (i, c) in a.iter_mut().enumerate() {
            *c = (i % 13) as i16;
        }
        let original = a;
        ntt(&mut a);
        assert_ne!(a, original, "NTT should change coefficients");
        invntt(&mut a);

        // invntt(ntt(a))[i] = a[i] * R mod q; undo with fqmul(c, 1) = c * R^{-1}
        for c in a.iter_mut() {
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
                if i + j < N { c[i + j] += prod; } else { c[i + j - N] -= prod; }
            }
        }
        let q = crate::params::Q as i32;
        let mut result = [0i16; N];
        for (r, &ci) in result.iter_mut().zip(c.iter()) {
            *r = (ci % q) as i16;
            if *r < 0 { *r += q as i16; }
        }
        result
    }

    fn normalise(c: i16) -> i16 {
        let mut v = barrett_reduce(c);
        if v < 0 { v += crate::params::Q; }
        v
    }

    #[test]
    fn ntt_basemul_matches_schoolbook() {
        use super::super::poly::Poly;

        let mut a = Poly::zero();
        let mut b = Poly::zero();
        for i in 0..N {
            a.coeffs[i] = ((i * 7 + 3) % 100) as i16;
            b.coeffs[i] = ((i * 13 + 1) % 100) as i16;
        }
        let expected = schoolbook_mul(&a.coeffs, &b.coeffs);

        let mut a_ntt = a;
        let mut b_ntt = b;
        a_ntt.ntt();
        b_ntt.ntt();

        let mut c_ntt = Poly::zero();
        c_ntt.basemul_montgomery(&a_ntt, &b_ntt);
        c_ntt.invntt_tomont();

        for (i, (&got_raw, &exp)) in c_ntt.coeffs.iter().zip(expected.iter()).enumerate() {
            let got = normalise(got_raw);
            assert_eq!(got, exp, "mismatch at {i}: got {got}, expected {exp}");
        }
    }
}
