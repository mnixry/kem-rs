use core::ops;

use super::Polynomial;
use crate::{N, encode, ntt};

/// Polynomial in NTT (bit-reversed) domain.
#[derive(Clone, Copy)]
pub struct NttPolynomial(pub(crate) [i16; N]);

impl NttPolynomial {
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self([0i16; N])
    }

    /// Inverse NTT; result is in Montgomery domain.
    #[must_use]
    pub fn ntt_inverse(mut self) -> Polynomial {
        ntt::inverse_ntt(&mut self.0);
        Polynomial(self.0)
    }

    pub fn reduce(&mut self) {
        crate::simd::poly_reduce(&mut self.0);
    }

    pub fn to_mont(&mut self) {
        crate::simd::poly_to_montgomery(&mut self.0);
    }

    /// Pointwise basemul: 128 degree-1 multiplications in NTT domain.
    #[must_use]
    pub fn basemul(&self, other: &Self) -> Self {
        let result = crate::simd::poly_basemul(&self.0, &other.0);
        Self(result)
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
        NttPolynomial(crate::simd::poly_add(&self.0, &rhs.0))
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
    use crate::{
        POLYBYTES, Q,
        reduce::{barrett_reduce, fqmul},
    };

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
}
