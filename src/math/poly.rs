//! Polynomial type and coefficient-level operations.
//!
//! `Poly` wraps `[i16; N]` (N = 256) and provides arithmetic, NTT transforms,
//! compression, serialisation, and noise sampling.

use crate::hash;
use crate::params::{N, Q, SYMBYTES};
use super::{ntt, pack, reduce, sample};

/// Polynomial in Rq = Zq[X]/(X^n + 1), stored as N = 256 coefficients.
#[derive(Clone, Copy)]
pub struct Poly {
    pub(crate) coeffs: [i16; N],
}

impl Poly {
    /// The zero polynomial.
    #[inline]
    pub const fn zero() -> Self {
        Poly { coeffs: [0i16; N] }
    }

    // ---- Arithmetic -------------------------------------------------------

    /// Coefficient-wise addition: `self = a + b`.
    #[inline]
    pub fn add(&mut self, a: &Poly, b: &Poly) {
        for i in 0..N {
            self.coeffs[i] = a.coeffs[i] + b.coeffs[i];
        }
    }

    /// Coefficient-wise subtraction: `self = a − b`.
    #[inline]
    pub fn sub(&mut self, a: &Poly, b: &Poly) {
        for i in 0..N {
            self.coeffs[i] = a.coeffs[i] - b.coeffs[i];
        }
    }

    /// In-place addition: `self += other`.
    #[inline]
    pub fn add_assign(&mut self, other: &Poly) {
        for i in 0..N {
            self.coeffs[i] += other.coeffs[i];
        }
    }

    /// Barrett-reduce every coefficient to the centered range.
    #[inline]
    pub fn reduce(&mut self) {
        for c in self.coeffs.iter_mut() {
            *c = reduce::barrett_reduce(*c);
        }
    }

    // ---- NTT / inverse NTT -----------------------------------------------

    /// Forward NTT (in-place).
    #[inline]
    pub fn ntt(&mut self) {
        ntt::ntt(&mut self.coeffs);
    }

    /// Inverse NTT (in-place), result in Montgomery domain.
    #[inline]
    pub fn invntt_tomont(&mut self) {
        ntt::invntt(&mut self.coeffs);
    }

    /// Convert all coefficients to Montgomery representation.
    pub fn tomont(&mut self) {
        const F: i32 = ((1u64 << 32) % (Q as u64)) as i32; // R² mod q = 1353
        for c in self.coeffs.iter_mut() {
            *c = reduce::montgomery_reduce((*c as i32) * F);
        }
    }

    /// Pointwise Montgomery multiplication: `self = a · b` (NTT domain).
    ///
    /// Uses pairs of basemul calls (128 pairs of degree-1 multiplications).
    pub fn basemul_montgomery(&mut self, a: &Poly, b: &Poly) {
        for i in 0..N / 4 {
            let zeta_idx = 64 + i;
            ntt::basemul(
                (&mut self.coeffs[4 * i..4 * i + 2]).try_into().unwrap(),
                (&a.coeffs[4 * i..4 * i + 2]).try_into().unwrap(),
                (&b.coeffs[4 * i..4 * i + 2]).try_into().unwrap(),
                ntt::ZETAS[zeta_idx],
            );
            ntt::basemul(
                (&mut self.coeffs[4 * i + 2..4 * i + 4]).try_into().unwrap(),
                (&a.coeffs[4 * i + 2..4 * i + 4]).try_into().unwrap(),
                (&b.coeffs[4 * i + 2..4 * i + 4]).try_into().unwrap(),
                -ntt::ZETAS[zeta_idx],
            );
        }
    }

    // ---- Serialisation ----------------------------------------------------

    /// Serialize to bytes (12-bit encoding → 384 bytes).
    pub fn tobytes(&self, r: &mut [u8]) {
        pack::poly_tobytes(r, &self.coeffs);
    }

    /// Deserialize from bytes (12-bit decoding).
    pub fn frombytes(a: &[u8]) -> Self {
        let mut p = Poly::zero();
        pack::poly_frombytes(&mut p.coeffs, a);
        p
    }

    // ---- Message encoding -------------------------------------------------

    /// Decode a 32-byte message to polynomial (1-bit per coefficient).
    pub fn frommsg(msg: &[u8; SYMBYTES]) -> Self {
        let mut p = Poly::zero();
        pack::poly_frommsg(&mut p.coeffs, msg);
        p
    }

    /// Encode polynomial to 32-byte message.
    pub fn tomsg(&self) -> [u8; SYMBYTES] {
        let mut msg = [0u8; SYMBYTES];
        pack::poly_tomsg(&mut msg, &self.coeffs);
        msg
    }

    // ---- Compression (d = 4 or 5, scalar ciphertext component) -----------

    /// Compress to `d` bits and write to buffer.
    pub fn compress(&self, r: &mut [u8], d: u32) {
        match d {
            4 => pack::poly_compress_d4(r, &self.coeffs),
            5 => pack::poly_compress_d5(r, &self.coeffs),
            _ => panic!("unsupported compression parameter d={d}"),
        }
    }

    /// Decompress from buffer with `d` bits.
    pub fn decompress(a: &[u8], d: u32) -> Self {
        let mut p = Poly::zero();
        match d {
            4 => pack::poly_decompress_d4(&mut p.coeffs, a),
            5 => pack::poly_decompress_d5(&mut p.coeffs, a),
            _ => panic!("unsupported compression parameter d={d}"),
        }
        p
    }

    // ---- Noise sampling ---------------------------------------------------

    /// Sample noise polynomial with η = 1 (eta1) from PRF(seed, nonce).
    pub fn getnoise_eta(eta: usize, seed: &[u8; SYMBYTES], nonce: u8) -> Self {
        let mut p = Poly::zero();
        match eta {
            2 => {
                let mut buf = [0u8; 2 * N / 4]; // 128 bytes
                hash::prf(seed, nonce, &mut buf);
                sample::cbd2(&mut p.coeffs, &buf);
            }
            3 => {
                let mut buf = [0u8; 3 * N / 4]; // 192 bytes
                hash::prf(seed, nonce, &mut buf);
                sample::cbd3(&mut p.coeffs, &buf);
            }
            _ => panic!("unsupported eta={eta}"),
        }
        p
    }
}

impl Default for Poly {
    #[inline]
    fn default() -> Self {
        Poly::zero()
    }
}

impl core::fmt::Debug for Poly {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Poly")
            .field("coeffs[..4]", &&self.coeffs[..4])
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::POLYBYTES;

    #[test]
    fn add_sub_inverse() {
        let mut a = Poly::zero();
        let mut b = Poly::zero();
        for i in 0..N {
            a.coeffs[i] = (i as i16) % Q;
            b.coeffs[i] = ((N - i) as i16) % Q;
        }
        let mut sum = Poly::zero();
        sum.add(&a, &b);

        let mut recovered = Poly::zero();
        recovered.sub(&sum, &b);
        assert_eq!(a.coeffs, recovered.coeffs);
    }

    #[test]
    fn tobytes_frombytes_roundtrip() {
        let mut p = Poly::zero();
        for i in 0..N {
            p.coeffs[i] = (i as i16 * 13) % (Q - 1);
        }
        let mut buf = [0u8; POLYBYTES];
        p.tobytes(&mut buf);

        let q = Poly::frombytes(&buf);
        assert_eq!(p.coeffs, q.coeffs);
    }

    #[test]
    fn msg_roundtrip() {
        let msg: [u8; SYMBYTES] = core::array::from_fn(|i| (i * 37) as u8);
        let p = Poly::frommsg(&msg);
        let recovered = p.tomsg();
        assert_eq!(msg, recovered);
    }

    #[test]
    fn getnoise_eta2_bounded() {
        let seed = [0u8; SYMBYTES];
        let p = Poly::getnoise_eta(2, &seed, 0);
        for &c in &p.coeffs {
            assert!((-2..=2).contains(&c));
        }
    }

    #[test]
    fn getnoise_eta3_bounded() {
        let seed = [1u8; SYMBYTES];
        let p = Poly::getnoise_eta(3, &seed, 0);
        for &c in &p.coeffs {
            assert!((-3..=3).contains(&c));
        }
    }
}
