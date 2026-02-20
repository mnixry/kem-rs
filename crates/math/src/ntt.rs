//! Number-Theoretic Transform and base multiplication in `Z_q[X]/(X^2 - zeta)`.
//!
//! - `forward_ntt`: forward NTT, standard order -> bit-reversed order.
//! - `inverse_ntt`: inverse NTT with Montgomery scaling.
//! - `basemul`: degree-1 multiplication in the NTT domain.

use crate::N;

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

/// Centred representative of `val mod q` in `[−(q−1)/2, (q−1)/2]`.
const fn centred(val: i64) -> i16 {
    if val > Q64 / 2 {
        (val - Q64) as i16
    } else {
        val as i16
    }
}

/// Twiddle factors in Montgomery form, from primitive 512th root ζ = 17,
/// bit-reversed indexing.
///
/// `ZETAS[i] = ζ^{BitRev₇(i)} · 2¹⁶  (mod q)`, centred to signed.
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
    // R² · 128⁻¹ mod q, where R = 2¹⁶
    const F: i16 = centred(pow_mod(2, 32, Q64) * pow_mod(128, Q64 - 2, Q64) % Q64);
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
