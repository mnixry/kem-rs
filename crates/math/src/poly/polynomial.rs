use core::ops;

use super::NttPolynomial;
use crate::{N, SYMBYTES, compress::CompressWidth, encode, ntt, sample::CbdWidth};

/// Polynomial in standard (coefficient) form over `R_q = Z_q[X]/(X^{256}+1)`.
#[derive(Clone, Copy)]
pub struct Polynomial(pub(crate) [i16; N]);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Q, SYMBYTES};

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
    fn msg_roundtrip() {
        let msg: [u8; SYMBYTES] = core::array::from_fn(|i| (i * 37) as u8);
        let p = Polynomial::from_message(&msg);
        let recovered = p.to_message();
        assert_eq!(msg, recovered);
    }
}
