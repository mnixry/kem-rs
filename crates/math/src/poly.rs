//! Domain-separated polynomial types.
//!
//! [`Polynomial`] represents a polynomial in standard (coefficient) form.
//! [`NttPolynomial`] represents a polynomial in NTT (bit-reversed) form.
//! The NTT transform is a consuming operation that produces the other type,
//! preventing accidental misuse of domain-mismatched polynomials.

use core::ops;

use crate::{N, SYMBYTES, compress::CompressWidth, encode, ntt, reduce::fqmul, sample::CbdWidth};

/// Polynomial in standard (coefficient) form over `R_q = Z_q[X]/(X^{256}+1)`.
#[derive(Clone, Copy)]
pub struct Polynomial(pub(crate) [i16; N]);

/// Polynomial in NTT (bit-reversed) domain.
#[derive(Clone, Copy)]
pub struct NttPolynomial(pub(crate) [i16; N]);

// -- Polynomial (standard form) ----------------------------------------------

impl Polynomial {
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self([0i16; N])
    }

    /// Consuming forward NTT transform.
    #[must_use]
    pub fn ntt(mut self) -> NttPolynomial {
        ntt::forward_ntt(&mut self.0);
        NttPolynomial(self.0)
    }

    pub fn reduce(&mut self) {
        crate::simd::poly_reduce(&mut self.0);
    }

    /// Compress to `D` bits and write to buffer.
    pub fn compress<D: CompressWidth>(&self, r: &mut [u8]) {
        D::compress_poly(r, &self.0);
    }

    /// Decompress from buffer with `D` bits.
    #[must_use]
    pub fn decompress<D: CompressWidth>(a: &[u8]) -> Self {
        let mut p = Self::zero();
        D::decompress_poly(&mut p.0, a);
        p
    }

    /// Decode a 32-byte message into a polynomial.
    #[must_use]
    pub fn from_message(msg: &[u8; SYMBYTES]) -> Self {
        let mut p = Self::zero();
        encode::message_to_coeffs(&mut p.0, msg);
        p
    }

    /// Encode polynomial to 32-byte message.
    #[must_use]
    pub fn to_message(&self) -> [u8; SYMBYTES] {
        let mut msg = [0u8; SYMBYTES];
        encode::coeffs_to_message(&mut msg, &self.0);
        msg
    }

    /// Sample noise polynomial via the sealed [`CbdWidth`] trait.
    #[must_use]
    pub fn sample_cbd<Eta: CbdWidth>(buf: &[u8]) -> Self {
        let mut p = Self::zero();
        Eta::sample(&mut p.0, buf);
        p
    }

    #[must_use]
    pub const fn coeffs(&self) -> &[i16; N] {
        &self.0
    }

    #[must_use]
    pub const fn coeffs_mut(&mut self) -> &mut [i16; N] {
        &mut self.0
    }
}

// -- NttPolynomial (NTT domain) ----------------------------------------------

impl NttPolynomial {
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self([0i16; N])
    }

    /// Consuming inverse NTT transform (result in Montgomery domain).
    #[must_use]
    pub fn ntt_inverse(mut self) -> Polynomial {
        ntt::inverse_ntt(&mut self.0);
        Polynomial(self.0)
    }

    pub fn reduce(&mut self) {
        crate::simd::poly_reduce(&mut self.0);
    }

    /// Convert all coefficients to Montgomery representation.
    pub fn to_mont(&mut self) {
        crate::simd::poly_to_montgomery(&mut self.0);
    }

    /// Pointwise Montgomery multiply (128 basemul pairs in NTT domain).
    #[must_use]
    pub fn basemul(&self, other: &Self) -> Self {
        let mut r = Self::zero();
        for i in 0..N / 4 {
            let zi = 64 + i;
            let base = 4 * i;
            let zeta = ntt::ZETAS[zi];

            r.0[base] = fqmul(self.0[base + 1], other.0[base + 1]);
            r.0[base] = fqmul(r.0[base], zeta);
            r.0[base] += fqmul(self.0[base], other.0[base]);
            r.0[base + 1] = fqmul(self.0[base], other.0[base + 1]);
            r.0[base + 1] += fqmul(self.0[base + 1], other.0[base]);

            r.0[base + 2] = fqmul(self.0[base + 3], other.0[base + 3]);
            r.0[base + 2] = fqmul(r.0[base + 2], -zeta);
            r.0[base + 2] += fqmul(self.0[base + 2], other.0[base + 2]);
            r.0[base + 3] = fqmul(self.0[base + 2], other.0[base + 3]);
            r.0[base + 3] += fqmul(self.0[base + 3], other.0[base + 2]);
        }
        r
    }

    /// Serialize to bytes (12-bit packing, 384 bytes).
    pub fn to_bytes(&self, r: &mut [u8]) {
        encode::coeffs_to_bytes(r, &self.0);
    }

    /// Deserialize from bytes (12-bit unpacking).
    #[must_use]
    pub fn from_bytes(a: &[u8]) -> Self {
        let mut p = Self::zero();
        encode::bytes_to_coeffs(&mut p.0, a);
        p
    }

    #[must_use]
    pub const fn coeffs(&self) -> &[i16; N] {
        &self.0
    }

    #[must_use]
    pub const fn coeffs_mut(&mut self) -> &mut [i16; N] {
        &mut self.0
    }
}

// -- Conversions & traits for Polynomial -------------------------------------

impl From<[i16; N]> for Polynomial {
    #[inline]
    fn from(coeffs: [i16; N]) -> Self {
        Self(coeffs)
    }
}

impl Default for Polynomial {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl core::fmt::Debug for Polynomial {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Polynomial")
            .field("coeffs[..4]", &&self.0[..4])
            .finish_non_exhaustive()
    }
}

impl<'b> ops::Add<&'b Polynomial> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn add(self, rhs: &'b Polynomial) -> Polynomial {
        let mut r = Polynomial::zero();
        crate::simd::poly_add(&mut r.0, &self.0, &rhs.0);
        r
    }
}

impl<'b> ops::Sub<&'b Polynomial> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn sub(self, rhs: &'b Polynomial) -> Polynomial {
        let mut r = Polynomial::zero();
        crate::simd::poly_sub(&mut r.0, &self.0, &rhs.0);
        r
    }
}

impl ops::AddAssign<&Self> for Polynomial {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        crate::simd::poly_add_assign(&mut self.0, &rhs.0);
    }
}

impl ops::SubAssign<&Self> for Polynomial {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        for i in 0..N {
            self.0[i] -= rhs.0[i];
        }
    }
}

// -- Conversions & traits for NttPolynomial ----------------------------------

impl Default for NttPolynomial {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl core::fmt::Debug for NttPolynomial {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NttPolynomial")
            .field("coeffs[..4]", &&self.0[..4])
            .finish_non_exhaustive()
    }
}

impl<'b> ops::Add<&'b NttPolynomial> for &NttPolynomial {
    type Output = NttPolynomial;
    #[inline]
    fn add(self, rhs: &'b NttPolynomial) -> NttPolynomial {
        let mut r = NttPolynomial::zero();
        crate::simd::poly_add(&mut r.0, &self.0, &rhs.0);
        r
    }
}

impl ops::AddAssign<&Self> for NttPolynomial {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        crate::simd::poly_add_assign(&mut self.0, &rhs.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{POLYBYTES, Q, reduce::barrett_reduce};

    #[test]
    fn add_sub_inverse() {
        let mut a = Polynomial::zero();
        let mut b = Polynomial::zero();
        for i in 0..N {
            a.0[i] = (i as i16) % Q;
            b.0[i] = ((N - i) as i16) % Q;
        }
        let sum = &a + &b;
        let recovered = &sum - &b;
        assert_eq!(a.0, recovered.0);
    }

    #[test]
    fn ntt_roundtrip() {
        let mut p = Polynomial::zero();
        for (i, c) in p.0.iter_mut().enumerate() {
            *c = (i % 13) as i16;
        }
        let original = p.0;

        let ntt_p = p.ntt();
        assert_ne!(ntt_p.0, original);

        let recovered = ntt_p.ntt_inverse();
        let mut coeffs = recovered.0;
        for c in &mut coeffs {
            *c = barrett_reduce(fqmul(*c, 1));
        }
        assert_eq!(coeffs, original);
    }

    #[test]
    fn tobytes_frombytes_roundtrip() {
        let mut p = NttPolynomial::zero();
        for i in 0..N {
            p.0[i] = (i as i16 * 13) % (Q - 1);
        }
        let mut buf = [0u8; POLYBYTES];
        p.to_bytes(&mut buf);
        let q = NttPolynomial::from_bytes(&buf);
        assert_eq!(p.0, q.0);
    }

    #[test]
    fn msg_roundtrip() {
        let msg: [u8; SYMBYTES] = core::array::from_fn(|i| (i * 37) as u8);
        let p = Polynomial::from_message(&msg);
        let recovered = p.to_message();
        assert_eq!(msg, recovered);
    }
}
