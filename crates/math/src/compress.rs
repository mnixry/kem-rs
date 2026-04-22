//! Sealed compression-width traits and compress/decompress operations.
//!
//! Each compression width (D=1,4,5,10,11) is a zero-sized marker type
//! implementing [`CompressWidth`], eliminating runtime dispatch.
//!
//! D4/D5/D10/D11 use Barrett reciprocal multiplication in SIMD to replace
//! per-coefficient scalar u32 division by Q.

use crate::{N, Q, SYMBYTES, simd::poly_ops};

mod sealed {
    pub trait Sealed {}
}

pub trait CompressWidthParams: sealed::Sealed {
    const D: u32;
    const POLY_BYTES: usize;
}

pub trait CompressWidth: CompressWidthParams {
    fn compress_poly(r: &mut [u8], coeffs: &[i16; N]);
    fn decompress_poly(coeffs: &mut [i16; N], a: &[u8]);
}

macro_rules! compress_width {
    ($($name:ident: $d:expr, $poly_bytes:expr),*) => {
        $(
            pub struct $name;
            impl sealed::Sealed for $name {}
            impl CompressWidthParams for $name {
                const D: u32 = $d;
                const POLY_BYTES: usize = $poly_bytes;
            }
        )*
    };
}

compress_width!(
    D1: 1, SYMBYTES,
    D4: 4, 128,
    D5: 5, 160,
    D10: 10, 320,
    D11: 11, 352
);

#[inline]
#[must_use]
pub const fn csubq(a: i16) -> u16 {
    let mut t = a as u16;
    t = t.wrapping_add(((a >> 15) as u16) & (Q as u16));
    t
}

#[cfg(test)]
#[inline]
const fn compress_coeff(x: u16, d: u32) -> u16 {
    let t = ((x as u32) << d).wrapping_add((Q as u32) / 2) / (Q as u32);
    (t & ((1u32 << d) - 1)) as u16
}

#[cfg(test)]
#[inline]
const fn decompress_coeff(y: u16, d: u32) -> u16 {
    (((y as u32) * (Q as u32) + (1u32 << (d - 1))) >> d) as u16
}

impl CompressWidth for D1 {
    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        let t = poly_ops::compress_coeffs(a, 1);
        let (chunks, _) = t.as_chunks::<8>();
        for (byte, chunk) in r.iter_mut().zip(chunks) {
            let mut b = 0u8;
            unroll!(j in (..8), {
                b |= (chunk[j] as u8) << j;
            });
            *byte = b;
        }
    }

    fn decompress_poly(r: &mut [i16; N], msg: &[u8]) {
        let (chunks, _) = r.as_chunks_mut::<8>();
        for (chunk, &byte) in chunks.iter_mut().zip(msg) {
            *chunk = unroll!(j in [..8], {
                let mask = -(((byte >> j) & 1) as i16);
                mask & ((Q + 1) / 2)
            });
        }
    }
}

impl CompressWidth for D4 {
    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        let t = poly_ops::compress_coeffs(a, 4);
        let (chunks, _) = t.as_chunks::<2>();
        for (byte, &[lo, hi]) in r.iter_mut().zip(chunks) {
            *byte = (lo as u8) | ((hi as u8) << 4);
        }
    }

    fn decompress_poly(r: &mut [i16; N], a: &[u8]) {
        let mut t = [0i16; N];
        let (chunks, _) = t.as_chunks_mut::<2>();
        for (chunk, &byte) in chunks.iter_mut().zip(a) {
            *chunk = [byte & 0x0F, byte >> 4].map(|x| x as i16);
        }
        *r = poly_ops::decompress_coeffs(&t, 4);
    }
}

impl CompressWidth for D5 {
    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        let t = poly_ops::compress_coeffs(a, 5);
        let (in_chunks, _) = t.as_chunks::<8>();
        let (out_chunks, _) = r.as_chunks_mut::<5>();
        for (o, chunk) in out_chunks.iter_mut().zip(in_chunks) {
            let mut packed = 0u64;
            unroll!(i in (..8), {
                packed |= (chunk[i] as u16 as u64) << (i * 5);
            });
            o.copy_from_slice(&packed.to_le_bytes()[..5]);
        }
    }

    fn decompress_poly(r: &mut [i16; N], a: &[u8]) {
        let mut t = [0i16; N];
        let (in_chunks, _) = a.as_chunks::<5>();
        let (out_chunks, _) = t.as_chunks_mut::<8>();
        for (o, b) in out_chunks.iter_mut().zip(in_chunks) {
            let mut buf = [0u8; 8];
            buf[..5].copy_from_slice(b);
            let packed = u64::from_le_bytes(buf);
            *o = unroll!(i in [..8], {
                ((packed >> (i * 5)) & 0x1F) as i16
            });
        }
        *r = poly_ops::decompress_coeffs(&t, 5);
    }
}

impl CompressWidth for D10 {
    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        let t = poly_ops::compress_coeffs(a, 10);
        let (in_chunks, _) = t.as_chunks::<4>();
        let (out_chunks, _) = r.as_chunks_mut::<5>();
        for (o, chunk) in out_chunks.iter_mut().zip(in_chunks) {
            let mut packed = 0u64;
            unroll!(i in (..4), {
                packed |= (chunk[i] as u16 as u64) << (i * 10);
            });
            o.copy_from_slice(&packed.to_le_bytes()[..5]);
        }
    }

    fn decompress_poly(r: &mut [i16; N], a: &[u8]) {
        let mut t = [0i16; N];
        let (in_chunks, _) = a.as_chunks::<5>();
        let (out_chunks, _) = t.as_chunks_mut::<4>();
        for (o, chunk) in out_chunks.iter_mut().zip(in_chunks) {
            let mut buf = [0u8; 8];
            buf[..5].copy_from_slice(chunk);
            let packed = u64::from_le_bytes(buf);
            *o = unroll!(i in [..4], ((packed >> (i * 10)) & 0x3FF) as i16);
        }
        *r = poly_ops::decompress_coeffs(&t, 10);
    }
}

impl CompressWidth for D11 {
    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        let t = poly_ops::compress_coeffs(a, 11);
        let (in_chunks, _) = t.as_chunks::<8>();
        let (out_chunks, _) = r.as_chunks_mut::<11>();
        for (o, chunk) in out_chunks.iter_mut().zip(in_chunks) {
            let mut packed = 0u128;
            unroll!(i in (..8), {
                packed |= (chunk[i] as u16 as u128) << (i * 11);
            });
            o.copy_from_slice(&packed.to_le_bytes()[..11]);
        }
    }

    fn decompress_poly(r: &mut [i16; N], a: &[u8]) {
        let mut t = [0i16; N];
        let (in_chunks, _) = a.as_chunks::<11>();
        let (out_chunks, _) = t.as_chunks_mut::<8>();
        for (o, chunk) in out_chunks.iter_mut().zip(in_chunks) {
            let mut buf = [0u8; 16];
            buf[..11].copy_from_slice(chunk);
            let packed = u128::from_le_bytes(buf);
            *o = unroll!(i in [..8], {
                ((packed >> (i * 11)) & 0x7FF) as i16
            });
        }
        *r = poly_ops::decompress_coeffs(&t, 11);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_check<D: CompressWidth>(multiplier: i16) {
        let mut a = [0i16; N];
        for (i, c) in a.iter_mut().enumerate() {
            *c = (i as i16 * multiplier) % Q;
        }
        let mut buf = [0u8; 384]; // oversized; only D::POLY_BYTES used
        D::compress_poly(&mut buf[..D::POLY_BYTES], &a);

        let mut b = [0i16; N];
        D::decompress_poly(&mut b, &buf[..D::POLY_BYTES]);

        let max_err = (Q as i32) / (1i32 << D::D) + 1;
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            let original = csubq(ai) as i32;
            let recovered = bi as i32;
            let diff = (original - recovered + Q as i32) % Q as i32;
            let diff = diff.min(Q as i32 - diff);
            assert!(
                diff <= max_err,
                "D={}: excessive error at index {i}: orig={original} rec={recovered} diff={diff}",
                D::D
            );
        }
    }

    #[test]
    fn compress_decompress_d4_roundtrip() {
        roundtrip_check::<D4>(7);
    }

    #[test]
    fn compress_decompress_d5_roundtrip() {
        roundtrip_check::<D5>(13);
    }

    #[test]
    fn compress_decompress_d10_roundtrip() {
        roundtrip_check::<D10>(11);
    }

    #[test]
    fn compress_decompress_d11_roundtrip() {
        roundtrip_check::<D11>(17);
    }

    #[test]
    fn simd_compress_matches_scalar_reference() {
        for d in [4u32, 5, 10, 11] {
            let mut coeffs = [0i16; N];
            for (i, c) in coeffs.iter_mut().enumerate() {
                // mix of positive, negative, and boundary values
                *c = ((i as i16 * 31) - 1500) % Q;
            }

            let simd_out = poly_ops::compress_coeffs(&coeffs, d);

            for (i, &c) in coeffs.iter().enumerate() {
                let expected = compress_coeff(csubq(c), d) as i16;
                assert_eq!(
                    simd_out[i], expected,
                    "compress mismatch at d={d} i={i} coeff={c}"
                );
            }
        }
    }

    #[test]
    fn simd_decompress_matches_scalar_reference() {
        for d in [4u32, 5, 10, 11] {
            let max_val = (1i16 << d) - 1;
            let mut compressed = [0i16; N];
            for (i, c) in compressed.iter_mut().enumerate() {
                *c = (i as i16) % (max_val + 1);
            }

            let simd_out = poly_ops::decompress_coeffs(&compressed, d);

            for (i, &y) in compressed.iter().enumerate() {
                let expected = decompress_coeff(y as u16, d) as i16;
                assert_eq!(
                    simd_out[i], expected,
                    "decompress mismatch at d={d} i={i} y={y}"
                );
            }
        }
    }
}
