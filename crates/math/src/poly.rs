//! Polynomial type and coefficient-level operations.
//!
//! `Poly` wraps `[i16; N]` (N=256) and provides arithmetic, NTT transforms,
//! compression, and serialisation.

use core::ops;

use super::{ntt, pack, sample};
use crate::{N, SYMBYTES};

/// Polynomial in `R_q = Z_q[X]/(X^n + 1)`, `N = 256` coefficients.
#[derive(Clone, Copy)]
pub struct Poly {
    pub coeffs: [i16; N],
}

impl Poly {
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self { coeffs: [0i16; N] }
    }

    /// Barrett-reduce every coefficient to the centered range.
    #[inline]
    pub fn reduce(&mut self) {
        crate::simd::poly_reduce(&mut self.coeffs);
    }

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
        crate::simd::poly_tomont(&mut self.coeffs);
    }

    /// Pointwise Montgomery multiply: `self = a * b` (NTT domain, 128 basemul
    /// pairs).
    ///
    /// # Panics
    ///
    /// Cannot panic — slice-to-array conversions are infallible for the fixed
    /// layout of `N = 256`.
    pub fn basemul_montgomery(&mut self, a: &Self, b: &Self) {
        for i in 0..N / 4 {
            let zi = 64 + i;
            ntt::basemul(
                (&mut self.coeffs[4 * i..4 * i + 2]).try_into().unwrap(),
                (&a.coeffs[4 * i..4 * i + 2]).try_into().unwrap(),
                (&b.coeffs[4 * i..4 * i + 2]).try_into().unwrap(),
                ntt::ZETAS[zi],
            );
            ntt::basemul(
                (&mut self.coeffs[4 * i + 2..4 * i + 4]).try_into().unwrap(),
                (&a.coeffs[4 * i + 2..4 * i + 4]).try_into().unwrap(),
                (&b.coeffs[4 * i + 2..4 * i + 4]).try_into().unwrap(),
                -ntt::ZETAS[zi],
            );
        }
    }

    /// Serialize to bytes (12-bit packing, 384 bytes).
    pub fn tobytes(&self, r: &mut [u8]) {
        pack::poly_tobytes(r, &self.coeffs);
    }

    /// Deserialize from bytes (12-bit unpacking).
    #[must_use]
    pub fn frombytes(a: &[u8]) -> Self {
        let mut p = Self::zero();
        pack::poly_frombytes(&mut p.coeffs, a);
        p
    }

    /// Decode a 32-byte message to polynomial (1 bit per coefficient).
    #[must_use]
    pub fn frommsg(msg: &[u8; SYMBYTES]) -> Self {
        let mut p = Self::zero();
        pack::poly_frommsg(&mut p.coeffs, msg);
        p
    }

    /// Encode polynomial to 32-byte message.
    #[must_use]
    pub fn tomsg(&self) -> [u8; SYMBYTES] {
        let mut msg = [0u8; SYMBYTES];
        pack::poly_tomsg(&mut msg, &self.coeffs);
        msg
    }

    /// Compress to `d` bits and write to buffer.
    pub fn compress(&self, r: &mut [u8], d: u32) {
        match d {
            4 => pack::poly_compress_d4(r, &self.coeffs),
            5 => pack::poly_compress_d5(r, &self.coeffs),
            _ => unreachable!(),
        }
    }

    /// Decompress from buffer with `d` bits.
    #[must_use]
    pub fn decompress(a: &[u8], d: u32) -> Self {
        let mut p = Self::zero();
        match d {
            4 => pack::poly_decompress_d4(&mut p.coeffs, a),
            5 => pack::poly_decompress_d5(&mut p.coeffs, a),
            _ => unreachable!(),
        }
        p
    }

    /// Sample noise polynomial via CBD(eta=2) from a pre-filled buffer.
    #[must_use]
    pub fn from_cbd2(buf: &[u8]) -> Self {
        let mut p = Self::zero();
        sample::cbd2(&mut p.coeffs, buf);
        p
    }

    /// Sample noise polynomial via CBD(eta=3) from a pre-filled buffer.
    #[must_use]
    pub fn from_cbd3(buf: &[u8]) -> Self {
        let mut p = Self::zero();
        sample::cbd3(&mut p.coeffs, buf);
        p
    }
}

// -- conversions & traits ----------------------------------------------------

impl From<[i16; N]> for Poly {
    #[inline]
    fn from(coeffs: [i16; N]) -> Self {
        Self { coeffs }
    }
}

impl From<Poly> for [i16; N] {
    #[inline]
    fn from(p: Poly) -> Self {
        p.coeffs
    }
}

impl Default for Poly {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl core::fmt::Debug for Poly {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Poly")
            .field("coeffs[..4]", &&self.coeffs[..4])
            .finish_non_exhaustive()
    }
}

// -- operator overloading (SIMD-backed) --------------------------------------

impl<'b> ops::Add<&'b Poly> for &Poly {
    type Output = Poly;
    #[inline]
    fn add(self, rhs: &'b Poly) -> Poly {
        let mut r = Poly::zero();
        crate::simd::poly_add(&mut r.coeffs, &self.coeffs, &rhs.coeffs);
        r
    }
}

impl<'b> ops::Sub<&'b Poly> for &Poly {
    type Output = Poly;
    #[inline]
    fn sub(self, rhs: &'b Poly) -> Poly {
        let mut r = Poly::zero();
        crate::simd::poly_sub(&mut r.coeffs, &self.coeffs, &rhs.coeffs);
        r
    }
}

impl ops::AddAssign<&Self> for Poly {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        crate::simd::poly_add_assign(&mut self.coeffs, &rhs.coeffs);
    }
}

impl ops::SubAssign<&Self> for Poly {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        for i in 0..N {
            self.coeffs[i] -= rhs.coeffs[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{POLYBYTES, Q};

    #[test]
    fn add_sub_inverse() {
        let mut a = Poly::zero();
        let mut b = Poly::zero();
        for i in 0..N {
            a.coeffs[i] = (i as i16) % Q;
            b.coeffs[i] = ((N - i) as i16) % Q;
        }
        let sum = &a + &b;
        let recovered = &sum - &b;
        assert_eq!(a.coeffs, recovered.coeffs);
    }

    #[test]
    fn add_assign_works() {
        let mut a = Poly::zero();
        let b = Poly::from([1i16; N]);
        a += &b;
        assert!(a.coeffs.iter().all(|&c| c == 1));
        a += &b;
        assert!(a.coeffs.iter().all(|&c| c == 2));
    }

    #[test]
    fn from_array_roundtrip() {
        let arr: [i16; N] = core::array::from_fn(|i| i as i16);
        let p = Poly::from(arr);
        let arr2: [i16; N] = p.into();
        assert_eq!(arr, arr2);
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
}
