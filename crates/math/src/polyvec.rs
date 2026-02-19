//! Vector of polynomials and associated operations.
//!
//! `PolyVec<K>` holds `K` polynomials and provides NTT, inner product,
//! compression, and byte serialisation parameterised by const-generic rank `K`.

use core::ops;

use super::{pack, poly::Poly};
use crate::{N, POLYBYTES};

/// A vector of `K` polynomials (K = 2, 3, or 4 in ML-KEM).
#[derive(Clone)]
pub struct PolyVec<const K: usize> {
    pub polys: [Poly; K],
}

impl<const K: usize> PolyVec<K> {
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            polys: [Poly::zero(); K],
        }
    }

    /// Forward NTT on every polynomial.
    pub fn ntt(&mut self) {
        for p in &mut self.polys {
            p.ntt();
        }
    }

    /// Inverse NTT on every polynomial (result in Montgomery domain).
    pub fn invntt_tomont(&mut self) {
        for p in &mut self.polys {
            p.invntt_tomont();
        }
    }

    /// Barrett-reduce all coefficients in every polynomial.
    pub fn reduce(&mut self) {
        for p in &mut self.polys {
            p.reduce();
        }
    }

    /// Inner product with accumulation: `r = sum_i(a[i] * b[i])` (NTT domain).
    pub fn basemul_acc_montgomery(r: &mut Poly, a: &Self, b: &Self) {
        let mut tmp = Poly::zero();
        r.basemul_montgomery(&a.polys[0], &b.polys[0]);
        for i in 1..K {
            tmp.basemul_montgomery(&a.polys[i], &b.polys[i]);
            *r += &tmp;
        }
        r.reduce();
    }

    /// Serialize to `K * 384` bytes (12-bit packing).
    pub fn tobytes(&self, r: &mut [u8]) {
        for (i, p) in self.polys.iter().enumerate() {
            pack::poly_tobytes(&mut r[i * POLYBYTES..(i + 1) * POLYBYTES], &p.coeffs);
        }
    }

    /// Deserialize from bytes.
    #[must_use]
    pub fn frombytes(a: &[u8]) -> Self {
        let mut pv = Self::zero();
        for (i, p) in pv.polys.iter_mut().enumerate() {
            pack::poly_frombytes(&mut p.coeffs, &a[i * POLYBYTES..(i + 1) * POLYBYTES]);
        }
        pv
    }

    /// Compress vector with `d_u` bits per coefficient.
    pub fn compress(&self, r: &mut [u8], d_u: u32) {
        let bpp = N * d_u as usize / 8;
        for (i, p) in self.polys.iter().enumerate() {
            let s = &mut r[i * bpp..(i + 1) * bpp];
            match d_u {
                10 => pack::poly_compress_d10(s, &p.coeffs),
                11 => pack::poly_compress_d11(s, &p.coeffs),
                _ => unreachable!(),
            }
        }
    }

    /// Decompress vector with `d_u` bits per coefficient.
    #[must_use]
    pub fn decompress(a: &[u8], d_u: u32) -> Self {
        let bpp = N * d_u as usize / 8;
        let mut pv = Self::zero();
        for (i, p) in pv.polys.iter_mut().enumerate() {
            let s = &a[i * bpp..(i + 1) * bpp];
            match d_u {
                10 => pack::poly_decompress_d10(&mut p.coeffs, s),
                11 => pack::poly_decompress_d11(&mut p.coeffs, s),
                _ => unreachable!(),
            }
        }
        pv
    }
}

impl<const K: usize> Default for PolyVec<K> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<'b, const K: usize> ops::Add<&'b PolyVec<K>> for &PolyVec<K> {
    type Output = PolyVec<K>;
    fn add(self, rhs: &'b PolyVec<K>) -> PolyVec<K> {
        let mut r = PolyVec::zero();
        for i in 0..K {
            r.polys[i] = &self.polys[i] + &rhs.polys[i];
        }
        r
    }
}

impl<const K: usize> ops::AddAssign<&Self> for PolyVec<K> {
    fn add_assign(&mut self, rhs: &Self) {
        for i in 0..K {
            self.polys[i] += &rhs.polys[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Q;

    #[test]
    fn tobytes_frombytes_roundtrip() {
        let mut pv = PolyVec::<3>::zero();
        for k in 0..3 {
            for i in 0..N {
                pv.polys[k].coeffs[i] = ((k * N + i) as i16 * 7) % (Q - 1);
            }
        }
        let mut buf = [0u8; 3 * POLYBYTES];
        pv.tobytes(&mut buf);
        let pv2 = PolyVec::<3>::frombytes(&buf);
        for k in 0..3 {
            assert_eq!(pv.polys[k].coeffs, pv2.polys[k].coeffs, "poly {k} mismatch");
        }
    }

    #[test]
    fn add_zero_identity() {
        let mut pv = PolyVec::<2>::zero();
        pv.polys[0].coeffs[0] = 42;
        pv.polys[1].coeffs[255] = 100;
        let zero = PolyVec::<2>::zero();
        let result = &pv + &zero;
        assert_eq!(result.polys[0].coeffs[0], 42);
        assert_eq!(result.polys[1].coeffs[255], 100);
    }

    #[test]
    fn add_assign_works() {
        let mut a = PolyVec::<2>::zero();
        a.polys[0].coeffs[0] = 10;
        let mut b = PolyVec::<2>::zero();
        b.polys[0].coeffs[0] = 5;
        a += &b;
        assert_eq!(a.polys[0].coeffs[0], 15);
    }
}
