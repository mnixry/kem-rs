use core::ops;

use super::Polynomial;
use crate::{N, encode, ntt, simd::poly_ops};

/// Polynomial in NTT (bit-reversed) domain.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct NttPolynomial(pub(crate) [i16; N]);

impl NttPolynomial {
    #[inline(always)]
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
        poly_ops::reduce(&mut self.0);
    }

    pub fn to_mont(&mut self) {
        poly_ops::to_montgomery(&mut self.0);
    }

    /// Pointwise basemul: 128 degree-1 multiplications in NTT domain.
    #[must_use]
    pub fn basemul(&self, other: &Self) -> Self {
        let result = poly_ops::basemul(&self.0, &other.0);
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

#[cfg_attr(coverage_nightly, coverage(off))]
impl Default for NttPolynomial {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
impl core::fmt::Debug for NttPolynomial {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NttPolynomial")
            .field("coeffs[..4]", &&self.0[..4])
            .finish_non_exhaustive()
    }
}

impl<'b> ops::Add<&'b NttPolynomial> for &NttPolynomial {
    type Output = NttPolynomial;
    #[inline(always)]
    fn add(self, rhs: &'b NttPolynomial) -> NttPolynomial {
        NttPolynomial(poly_ops::add(&self.0, &rhs.0))
    }
}

impl ops::AddAssign<&Self> for NttPolynomial {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &Self) {
        poly_ops::add_assign(&mut self.0, &rhs.0);
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

    #[test]
    fn add_produces_sum() {
        let mut a = NttPolynomial::zero();
        let mut b = NttPolynomial::zero();
        for i in 0..N {
            a.0[i] = 100;
            b.0[i] = 200;
        }
        let c = &a + &b;
        for &coeff in c.coeffs() {
            assert_eq!(coeff, 300);
        }
    }

    #[test]
    fn add_assign_matches_add() {
        let mut a = NttPolynomial::zero();
        let b = {
            let mut p = NttPolynomial::zero();
            for i in 0..N {
                p.0[i] = (i as i16) % 50;
            }
            p
        };
        for i in 0..N {
            a.0[i] = (i as i16) % 100;
        }
        let expected = &a + &b;
        a += &b;
        assert_eq!(a.0, expected.0);
    }

    #[test]
    fn reduce_normalises() {
        let mut p = NttPolynomial::zero();
        for i in 0..N {
            p.0[i] = Q + (i as i16 % 100);
        }
        p.reduce();
        for (i, &c) in p.0.iter().enumerate() {
            assert!(
                c.unsigned_abs() <= Q as u16,
                "coefficient {i} not reduced: {c}"
            );
        }
    }

    #[test]
    fn to_mont_changes_nonzero() {
        let mut p = NttPolynomial::zero();
        for i in 0..N {
            p.0[i] = ((i as i16) % (Q - 1)) + 1;
        }
        let before = p.0;
        p.to_mont();
        assert_ne!(p.0, before);
    }
}
