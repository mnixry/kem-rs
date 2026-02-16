//! Vector of polynomials and associated operations.
//!
//! `PolyVec<K>` holds `K` polynomials and provides NTT, inner product,
//! compression, and byte serialisation — all parameterised by the
//! const-generic rank `K`.

use crate::params::{N, POLYBYTES};
use super::{pack, poly::Poly};

/// A vector of `K` polynomials (K = 2, 3, or 4 in ML-KEM).
#[derive(Clone)]
pub struct PolyVec<const K: usize> {
    pub(crate) polys: [Poly; K],
}

impl<const K: usize> PolyVec<K> {
    /// Zero-initialised polynomial vector.
    #[inline]
    pub fn zero() -> Self {
        PolyVec {
            polys: [Poly::zero(); K],
        }
    }

    // ---- NTT / inverse NTT -----------------------------------------------

    /// Forward NTT on every polynomial in the vector.
    pub fn ntt(&mut self) {
        for p in self.polys.iter_mut() {
            p.ntt();
        }
    }

    /// Inverse NTT on every polynomial (result in Montgomery domain).
    pub fn invntt_tomont(&mut self) {
        for p in self.polys.iter_mut() {
            p.invntt_tomont();
        }
    }

    // ---- Arithmetic -------------------------------------------------------

    /// Barrett-reduce all coefficients in every polynomial.
    pub fn reduce(&mut self) {
        for p in self.polys.iter_mut() {
            p.reduce();
        }
    }

    /// Pointwise add: `self = a + b`.
    pub fn add(&mut self, a: &PolyVec<K>, b: &PolyVec<K>) {
        for i in 0..K {
            self.polys[i].add(&a.polys[i], &b.polys[i]);
        }
    }

    /// In-place addition: `self += other`.
    pub fn add_assign(&mut self, other: &PolyVec<K>) {
        for i in 0..K {
            self.polys[i].add_assign(&other.polys[i]);
        }
    }

    /// Pointwise Montgomery inner product with accumulation:
    /// `r = sum_i(a[i] * b[i])` (all in NTT domain).
    pub fn basemul_acc_montgomery(r: &mut Poly, a: &PolyVec<K>, b: &PolyVec<K>) {
        let mut tmp = Poly::zero();
        r.basemul_montgomery(&a.polys[0], &b.polys[0]);
        for i in 1..K {
            tmp.basemul_montgomery(&a.polys[i], &b.polys[i]);
            for j in 0..N {
                r.coeffs[j] += tmp.coeffs[j];
            }
        }
        r.reduce();
    }

    // ---- 12-bit byte serialisation ----------------------------------------

    /// Serialize to bytes: `K × 384` bytes.
    pub fn tobytes(&self, r: &mut [u8]) {
        for i in 0..K {
            pack::poly_tobytes(&mut r[i * POLYBYTES..(i + 1) * POLYBYTES], &self.polys[i].coeffs);
        }
    }

    /// Deserialize from bytes.
    pub fn frombytes(a: &[u8]) -> Self {
        let mut pv = PolyVec::zero();
        for i in 0..K {
            pack::poly_frombytes(
                &mut pv.polys[i].coeffs,
                &a[i * POLYBYTES..(i + 1) * POLYBYTES],
            );
        }
        pv
    }

    // ---- Compression for ciphertext u component ---------------------------

    /// Compress vector with `d_u` bits per coefficient.
    pub fn compress(&self, r: &mut [u8], d_u: u32) {
        let bytes_per_poly = N * d_u as usize / 8;
        for i in 0..K {
            let slice = &mut r[i * bytes_per_poly..(i + 1) * bytes_per_poly];
            match d_u {
                10 => pack::poly_compress_d10(slice, &self.polys[i].coeffs),
                11 => pack::poly_compress_d11(slice, &self.polys[i].coeffs),
                _ => panic!("unsupported d_u={d_u}"),
            }
        }
    }

    /// Decompress vector with `d_u` bits per coefficient.
    pub fn decompress(a: &[u8], d_u: u32) -> Self {
        let bytes_per_poly = N * d_u as usize / 8;
        let mut pv = PolyVec::zero();
        for i in 0..K {
            let slice = &a[i * bytes_per_poly..(i + 1) * bytes_per_poly];
            match d_u {
                10 => pack::poly_decompress_d10(&mut pv.polys[i].coeffs, slice),
                11 => pack::poly_decompress_d11(&mut pv.polys[i].coeffs, slice),
                _ => panic!("unsupported d_u={d_u}"),
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Q;

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
        let mut result = PolyVec::<2>::zero();
        result.add(&pv, &zero);

        assert_eq!(result.polys[0].coeffs[0], 42);
        assert_eq!(result.polys[1].coeffs[255], 100);
    }
}
