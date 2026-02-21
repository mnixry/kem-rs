use core::ops;

use super::{NttPolynomial, Polynomial};
use crate::{POLYBYTES, compress::CompressWidth, encode};

/// A vector of `K` polynomials in standard (coefficient) form.
#[derive(Clone)]
pub struct Vector<const K: usize> {
    pub(crate) polys: [Polynomial; K],
}

/// A vector of `K` polynomials in NTT domain.
#[derive(Clone)]
pub struct NttVector<const K: usize> {
    pub(crate) polys: [NttPolynomial; K],
}

impl<const K: usize> Vector<K> {
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            polys: [Polynomial::zero(); K],
        }
    }

    #[must_use]
    pub fn ntt(self) -> NttVector<K> {
        let polys = self.polys.map(Polynomial::ntt);
        NttVector { polys }
    }

    pub fn reduce(&mut self) {
        for p in &mut self.polys {
            p.reduce();
        }
    }

    pub fn compress<D: CompressWidth>(&self, r: &mut [u8]) {
        for (i, p) in self.polys.iter().enumerate() {
            p.compress::<D>(&mut r[i * D::POLY_BYTES..(i + 1) * D::POLY_BYTES]);
        }
    }

    #[must_use]
    pub fn decompress<D: CompressWidth>(a: &[u8]) -> Self {
        let mut v = Self::zero();
        for (i, p) in v.polys.iter_mut().enumerate() {
            *p = Polynomial::decompress::<D>(&a[i * D::POLY_BYTES..(i + 1) * D::POLY_BYTES]);
        }
        v
    }

    #[must_use]
    pub const fn polys(&self) -> &[Polynomial; K] {
        &self.polys
    }

    #[must_use]
    pub const fn polys_mut(&mut self) -> &mut [Polynomial; K] {
        &mut self.polys
    }
}

impl<const K: usize> NttVector<K> {
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            polys: [NttPolynomial::zero(); K],
        }
    }

    #[must_use]
    pub fn ntt_inverse(self) -> Vector<K> {
        let polys = self.polys.map(NttPolynomial::ntt_inverse);
        Vector { polys }
    }

    pub fn reduce(&mut self) {
        for p in &mut self.polys {
            p.reduce();
        }
    }

    /// `sum_i(self[i] * other[i])` in NTT domain.
    #[must_use]
    pub fn inner_product(&self, other: &Self) -> NttPolynomial {
        let mut acc = self.polys[0].basemul(&other.polys[0]);
        for i in 1..K {
            acc += &self.polys[i].basemul(&other.polys[i]);
        }
        acc.reduce();
        acc
    }

    /// Serialize to `K * 384` bytes (12-bit packing).
    pub fn to_bytes(&self, r: &mut [u8]) {
        for (i, p) in self.polys.iter().enumerate() {
            encode::coeffs_to_bytes(&mut r[i * POLYBYTES..(i + 1) * POLYBYTES], p.coeffs());
        }
    }

    #[must_use]
    pub fn from_bytes(a: &[u8]) -> Self {
        let mut v = Self::zero();
        for (i, p) in v.polys.iter_mut().enumerate() {
            *p = NttPolynomial::from_bytes(&a[i * POLYBYTES..(i + 1) * POLYBYTES]);
        }
        v
    }

    #[must_use]
    pub const fn polys(&self) -> &[NttPolynomial; K] {
        &self.polys
    }

    #[must_use]
    pub const fn polys_mut(&mut self) -> &mut [NttPolynomial; K] {
        &mut self.polys
    }
}

/// K x K matrix of NTT-domain polynomials (public matrix A).
pub struct NttMatrix<const K: usize> {
    pub(crate) rows: [NttVector<K>; K],
}

impl<const K: usize> NttMatrix<K> {
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Self {
            rows: core::array::from_fn(|_| NttVector::zero()),
        }
    }

    /// `A * v` with Montgomery conversion on each result row.
    #[must_use]
    pub fn mul_vec_tomont(&self, v: &NttVector<K>) -> NttVector<K> {
        let mut result = NttVector::zero();
        for (r_poly, a_row) in result.polys.iter_mut().zip(self.rows.iter()) {
            *r_poly = a_row.inner_product(v);
            r_poly.to_mont();
        }
        result
    }

    /// `A * v` without Montgomery conversion.
    #[must_use]
    pub fn mul_vec(&self, v: &NttVector<K>) -> NttVector<K> {
        let mut result = NttVector::zero();
        for (r_poly, a_row) in result.polys.iter_mut().zip(self.rows.iter()) {
            *r_poly = a_row.inner_product(v);
        }
        result
    }

    #[must_use]
    pub const fn rows_mut(&mut self) -> &mut [NttVector<K>; K] {
        &mut self.rows
    }
}

impl<'b, const K: usize> ops::Add<&'b Vector<K>> for &Vector<K> {
    type Output = Vector<K>;
    fn add(self, rhs: &'b Vector<K>) -> Vector<K> {
        let mut r = Vector::zero();
        for i in 0..K {
            r.polys[i] = &self.polys[i] + &rhs.polys[i];
        }
        r
    }
}

impl<const K: usize> ops::AddAssign<&Self> for Vector<K> {
    fn add_assign(&mut self, rhs: &Self) {
        for i in 0..K {
            self.polys[i] += &rhs.polys[i];
        }
    }
}

impl<const K: usize> ops::AddAssign<&Self> for NttVector<K> {
    fn add_assign(&mut self, rhs: &Self) {
        for i in 0..K {
            self.polys[i] += &rhs.polys[i];
        }
    }
}

impl<const K: usize> Default for Vector<K> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const K: usize> Default for NttVector<K> {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{N, Q};

    #[test]
    fn tobytes_frombytes_roundtrip() {
        let mut v = NttVector::<3>::zero();
        for k in 0..3 {
            for i in 0..N {
                v.polys[k].0[i] = ((k * N + i) as i16 * 7) % (Q - 1);
            }
        }
        let mut buf = [0u8; 3 * POLYBYTES];
        v.to_bytes(&mut buf);
        let v2 = NttVector::<3>::from_bytes(&buf);
        for k in 0..3 {
            assert_eq!(v.polys[k].0, v2.polys[k].0, "poly {k} mismatch");
        }
    }

    #[test]
    fn add_zero_identity() {
        let mut v = Vector::<2>::zero();
        v.polys[0].0[0] = 42;
        v.polys[1].0[255] = 100;
        let zero = Vector::<2>::zero();
        let result = &v + &zero;
        assert_eq!(result.polys[0].0[0], 42);
        assert_eq!(result.polys[1].0[255], 100);
    }
}
