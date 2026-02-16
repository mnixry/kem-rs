//! Number-Theoretic Transform and base multiplication in Zq\[X\]/(X² − ζ).
//!
//! - [`ntt`]: forward NTT, standard order → bit-reversed order.
//! - [`invntt`]: inverse NTT with Montgomery scaling factor.
//! - [`basemul`]: degree-1 multiplication in the NTT domain.
//! - [`ZETAS`]: precomputed twiddle factors (128 entries, Montgomery form).

use super::reduce::{barrett_reduce, fqmul};
use crate::params::N;

/// Precomputed twiddle factors in Montgomery form.
///
/// Generated from the primitive 512-th root of unity (ζ = 17)
/// via bit-reversed indexing, then scaled into Montgomery domain.
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

/// Forward NTT (in-place).
///
/// Input in standard coefficient order, output in bit-reversed order.
/// All arithmetic modulo q = 3329.
pub fn ntt(r: &mut [i16; N]) {
    let mut k: usize = 1;
    let mut len = 128;
    while len >= 2 {
        let mut start = 0;
        while start < N {
            let zeta = ZETAS[k];
            k += 1;
            for j in start..(start + len) {
                let t = fqmul(zeta, r[j + len]);
                r[j + len] = r[j] - t;
                r[j] = r[j] + t;
            }
            start += 2 * len;
        }
        len >>= 1;
    }
}

/// Inverse NTT (in-place).
///
/// Input in bit-reversed order, output in standard order with
/// each coefficient multiplied by Montgomery factor 2¹⁶.
pub fn invntt(r: &mut [i16; N]) {
    const F: i16 = 1441; // mont² / 128

    let mut k: usize = 127;
    let mut len = 2;
    while len <= 128 {
        let mut start = 0;
        while start < N {
            let zeta = ZETAS[k];
            k = k.wrapping_sub(1);
            for j in start..(start + len) {
                let t = r[j];
                r[j] = barrett_reduce(t + r[j + len]);
                r[j + len] = fqmul(zeta, r[j + len] - t);
            }
            start += 2 * len;
        }
        len <<= 1;
    }
    for coeff in r.iter_mut() {
        *coeff = fqmul(*coeff, F);
    }
}

/// Base multiplication of two degree-1 polynomials in Zq\[X\]/(X² − ζ).
///
/// Computes `r = a · b mod (X² − ζ)` where `a`, `b`, `r` are pairs
/// of NTT-domain coefficients.
#[inline]
pub fn basemul(r: &mut [i16; 2], a: &[i16; 2], b: &[i16; 2], zeta: i16) {
    r[0] = fqmul(a[1], b[1]);
    r[0] = fqmul(r[0], zeta);
    r[0] += fqmul(a[0], b[0]);
    r[1] = fqmul(a[0], b[1]);
    r[1] += fqmul(a[1], b[0]);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntt_invntt_roundtrip() {
        let mut a = [0i16; N];
        for i in 0..N {
            a[i] = (i % 13) as i16;
        }
        let original = a;

        ntt(&mut a);
        assert_ne!(a, original, "NTT should change coefficients");

        invntt(&mut a);

        // invntt(ntt(a))[i] ≡ a[i] · R (mod q).
        // Undo the Montgomery factor by multiplying each coefficient by R⁻¹.
        for coeff in a.iter_mut() {
            *coeff = fqmul(*coeff, 1); // fqmul(x, 1) = x · R⁻¹ mod q
            *coeff = barrett_reduce(*coeff);
        }

        for i in 0..N {
            assert_eq!(a[i], original[i], "mismatch at index {i}");
        }
    }

    /// Schoolbook polynomial multiplication in Z_q[x]/(x^256 + 1).
    fn schoolbook_mul(a: &[i16; N], b: &[i16; N]) -> [i16; N] {
        let mut c = [0i32; N];
        for i in 0..N {
            for j in 0..N {
                if i + j < N {
                    c[i + j] += (a[i] as i32) * (b[j] as i32);
                } else {
                    c[i + j - N] -= (a[i] as i32) * (b[j] as i32);
                }
            }
        }
        let q = crate::params::Q as i32;
        let mut result = [0i16; N];
        for i in 0..N {
            result[i] = (c[i] % q) as i16;
            if result[i] < 0 {
                result[i] += q as i16;
            }
        }
        result
    }

    /// Normalise a coefficient to [0, q).
    fn normalise(c: i16) -> i16 {
        let q = crate::params::Q;
        let mut v = barrett_reduce(c);
        if v < 0 {
            v += q;
        }
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

        // The result is a*b*R^{-1} in Montgomery domain after invntt_tomont.
        // To get standard coefficients: fqmul(c, 1) = c * R^{-1}.
        // But basemul introduces one R^{-1} and invntt_tomont handles another.
        // The net scaling is: c = a*b (no extra factor? or with R factor?)
        //
        // Actually: NTT introduces zeta scaling, basemul multiplies + R^{-1},
        // invntt undoes NTT and applies f=1441=R^2/128 via fqmul (another R^{-1}).
        // Net for basemul(NTT(a), NTT(b)) + invntt:
        //   a*b * R^{-1} (from basemul) * R (from invntt_tomont's handling)
        //   = a*b, but the exact factor needs to be verified empirically.
        for i in 0..N {
            let got = normalise(c_ntt.coeffs[i]);
            assert_eq!(
                got, expected[i],
                "mismatch at index {i}: got {got}, expected {}",
                expected[i]
            );
        }
    }
}
