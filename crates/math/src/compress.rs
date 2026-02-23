//! Sealed compression-width traits and compress/decompress operations.
//!
//! Each compression width (D=1,4,5,10,11) is a zero-sized marker type
//! implementing [`CompressWidth`], eliminating runtime dispatch.
//!
//! D4/D5/D10/D11 use Barrett reciprocal multiplication in SIMD to replace
//! per-coefficient scalar u32 division by Q.

use crate::{N, Q, SYMBYTES};

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
        let t = crate::simd::poly_compress_coeffs(a, 1);
        for i in 0..N / 8 {
            let mut byte = 0u8;
            for j in 0..8 {
                byte |= (t[8 * i + j] as u8) << j;
            }
            r[i] = byte;
        }
    }

    fn decompress_poly(r: &mut [i16; N], msg: &[u8]) {
        for i in 0..N / 8 {
            for j in 0..8u32 {
                let mask = -(((msg[i] >> j) & 1) as i16);
                r[8 * i + j as usize] = mask & ((Q + 1) / 2);
            }
        }
    }
}

impl CompressWidth for D4 {
    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        let t = crate::simd::poly_compress_coeffs(a, 4);
        for i in 0..N / 2 {
            r[i] = (t[2 * i] as u8) | ((t[2 * i + 1] as u8) << 4);
        }
    }

    fn decompress_poly(r: &mut [i16; N], a: &[u8]) {
        let mut t = [0i16; N];
        for i in 0..N / 2 {
            t[2 * i] = (a[i] & 0x0F) as i16;
            t[2 * i + 1] = (a[i] >> 4) as i16;
        }
        *r = crate::simd::poly_decompress_coeffs(&t, 4);
    }
}

impl CompressWidth for D5 {
    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        let t = crate::simd::poly_compress_coeffs(a, 5);
        for i in 0..N / 8 {
            let s = &t[8 * i..];
            let (s0, s1, s2, s3) = (s[0] as u8, s[1] as u8, s[2] as u8, s[3] as u8);
            let (s4, s5, s6, s7) = (s[4] as u8, s[5] as u8, s[6] as u8, s[7] as u8);
            r[5 * i] = s0 | (s1 << 5);
            r[5 * i + 1] = (s1 >> 3) | (s2 << 2) | (s3 << 7);
            r[5 * i + 2] = (s3 >> 1) | (s4 << 4);
            r[5 * i + 3] = (s4 >> 4) | (s5 << 1) | (s6 << 6);
            r[5 * i + 4] = (s6 >> 2) | (s7 << 3);
        }
    }

    fn decompress_poly(r: &mut [i16; N], a: &[u8]) {
        let mut t = [0i16; N];
        for i in 0..N / 8 {
            let b = &a[5 * i..];
            t[8 * i] = (b[0] & 0x1F) as i16;
            t[8 * i + 1] = ((b[0] >> 5) | ((b[1] & 0x03) << 3)) as i16;
            t[8 * i + 2] = ((b[1] >> 2) & 0x1F) as i16;
            t[8 * i + 3] = ((b[1] >> 7) | ((b[2] & 0x0F) << 1)) as i16;
            t[8 * i + 4] = ((b[2] >> 4) | ((b[3] & 0x01) << 4)) as i16;
            t[8 * i + 5] = ((b[3] >> 1) & 0x1F) as i16;
            t[8 * i + 6] = ((b[3] >> 6) | ((b[4] & 0x07) << 2)) as i16;
            t[8 * i + 7] = (b[4] >> 3) as i16;
        }
        *r = crate::simd::poly_decompress_coeffs(&t, 5);
    }
}

impl CompressWidth for D10 {
    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        let t = crate::simd::poly_compress_coeffs(a, 10);
        for i in 0..N / 4 {
            let s = &t[4 * i..];
            let (s0, s1, s2, s3) = (s[0] as u16, s[1] as u16, s[2] as u16, s[3] as u16);
            r[5 * i] = s0 as u8;
            r[5 * i + 1] = ((s0 >> 8) | (s1 << 2)) as u8;
            r[5 * i + 2] = ((s1 >> 6) | (s2 << 4)) as u8;
            r[5 * i + 3] = ((s2 >> 4) | (s3 << 6)) as u8;
            r[5 * i + 4] = (s3 >> 2) as u8;
        }
    }

    fn decompress_poly(r: &mut [i16; N], a: &[u8]) {
        let mut t = [0i16; N];
        for i in 0..N / 4 {
            let b = &a[5 * i..];
            t[4 * i] = ((b[0] as u16) | (((b[1] as u16) & 0x03) << 8)) as i16;
            t[4 * i + 1] = (((b[1] as u16) >> 2) | (((b[2] as u16) & 0x0F) << 6)) as i16;
            t[4 * i + 2] = (((b[2] as u16) >> 4) | (((b[3] as u16) & 0x3F) << 4)) as i16;
            t[4 * i + 3] = (((b[3] as u16) >> 6) | ((b[4] as u16) << 2)) as i16;
        }
        *r = crate::simd::poly_decompress_coeffs(&t, 10);
    }
}

impl CompressWidth for D11 {
    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        let t = crate::simd::poly_compress_coeffs(a, 11);
        for i in 0..N / 8 {
            let s = &t[8 * i..];
            let (s0, s1, s2, s3) = (s[0] as u16, s[1] as u16, s[2] as u16, s[3] as u16);
            let (s4, s5, s6, s7) = (s[4] as u16, s[5] as u16, s[6] as u16, s[7] as u16);
            r[11 * i] = s0 as u8;
            r[11 * i + 1] = ((s0 >> 8) | (s1 << 3)) as u8;
            r[11 * i + 2] = ((s1 >> 5) | (s2 << 6)) as u8;
            r[11 * i + 3] = (s2 >> 2) as u8;
            r[11 * i + 4] = ((s2 >> 10) | (s3 << 1)) as u8;
            r[11 * i + 5] = ((s3 >> 7) | (s4 << 4)) as u8;
            r[11 * i + 6] = ((s4 >> 4) | (s5 << 7)) as u8;
            r[11 * i + 7] = (s5 >> 1) as u8;
            r[11 * i + 8] = ((s5 >> 9) | (s6 << 2)) as u8;
            r[11 * i + 9] = ((s6 >> 6) | (s7 << 5)) as u8;
            r[11 * i + 10] = (s7 >> 3) as u8;
        }
    }

    fn decompress_poly(r: &mut [i16; N], a: &[u8]) {
        let mut t = [0i16; N];
        for i in 0..N / 8 {
            let b = &a[11 * i..];
            t[8 * i] = ((b[0] as u16) | (((b[1] as u16) & 0x07) << 8)) as i16;
            t[8 * i + 1] = (((b[1] as u16) >> 3) | (((b[2] as u16) & 0x3F) << 5)) as i16;
            t[8 * i + 2] = (((b[2] as u16) >> 6)
                | ((b[3] as u16) << 2)
                | (((b[4] as u16) & 0x01) << 10)) as i16;
            t[8 * i + 3] = (((b[4] as u16) >> 1) | (((b[5] as u16) & 0x0F) << 7)) as i16;
            t[8 * i + 4] = (((b[5] as u16) >> 4) | (((b[6] as u16) & 0x7F) << 4)) as i16;
            t[8 * i + 5] = (((b[6] as u16) >> 7)
                | ((b[7] as u16) << 1)
                | (((b[8] as u16) & 0x03) << 9)) as i16;
            t[8 * i + 6] = (((b[8] as u16) >> 2) | (((b[9] as u16) & 0x1F) << 6)) as i16;
            t[8 * i + 7] = (((b[9] as u16) >> 5) | ((b[10] as u16) << 3)) as i16;
        }
        *r = crate::simd::poly_decompress_coeffs(&t, 11);
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

            let simd_out = crate::simd::poly_compress_coeffs(&coeffs, d);

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

            let simd_out = crate::simd::poly_decompress_coeffs(&compressed, d);

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
