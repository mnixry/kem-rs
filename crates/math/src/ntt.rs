//! Number-Theoretic Transform and base multiplication in `Z_q[X]/(X^2 - zeta)`.
//!
//! - `forward_ntt`: forward NTT, standard order -> bit-reversed order.
//! - `inverse_ntt`: inverse NTT with Montgomery scaling.
//! - `basemul`: degree-1 multiplication in the NTT domain.

use crate::N;

/// Twiddle factors in Montgomery form, from primitive 512th root zeta=17,
/// bit-reversed indexing.
pub static ZETAS: [i16; 128] = [
    -1044, -758, -359, -1517, 1493, 1422, 287, 202, -171, 622, 1577, 182, 962, -1202, -1474, 1468,
    573, -1325, 264, 383, -829, 1458, -1602, -130, -681, 1017, 732, 608, -1542, 411, -205, -1571,
    1223, 652, -552, 1015, -1293, 1491, -282, -1544, 516, -8, -320, -666, -1618, -1162, 126, 1469,
    -853, -90, -271, 830, 107, -1421, -247, -951, -398, 961, -1508, -725, 448, -1065, 677, -1275,
    -1103, 430, 555, 843, -1251, 871, 1550, 105, 422, 587, 177, -235, -291, -460, 1574, 1653, -246,
    778, 1159, -147, -777, 1483, -602, 1119, -1590, 644, -872, 349, 418, 329, -156, -75, 817, 1097,
    603, 610, 1322, -1285, -1465, 384, -1215, -136, 1218, -1335, -874, 220, -1187, -1659, -1185,
    -1530, -1278, 794, -1510, -854, -870, 478, -108, -308, 996, 991, 958, -1460, 1522, 1628,
];

/// Forward NTT (in-place). Standard order in, bit-reversed order out.
pub fn forward_ntt(r: &mut [i16; N]) {
    let mut k: usize = 1;
    let mut len = 128;
    while len >= 2 {
        let mut start = 0;
        while start < N {
            let zeta = ZETAS[k];
            k += 1;
            let (lo, hi) = r[start..start + 2 * len].split_at_mut(len);
            crate::simd::butterfly_forward(lo, hi, zeta);
            start += 2 * len;
        }
        len >>= 1;
    }
}

/// Inverse NTT (in-place). Bit-reversed in, standard order out,
/// each coefficient scaled by Montgomery factor `R = 2^{16}`.
pub fn inverse_ntt(r: &mut [i16; N]) {
    const F: i16 = 1441; // mont^2 * 128^{-1} mod q
    let mut k: usize = 127;
    let mut len = 2;
    while len <= 128 {
        let mut start = 0;
        while start < N {
            let zeta = ZETAS[k];
            k = k.wrapping_sub(1);
            let (lo, hi) = r[start..start + 2 * len].split_at_mut(len);
            crate::simd::butterfly_inverse(lo, hi, zeta);
            start += 2 * len;
        }
        len <<= 1;
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
