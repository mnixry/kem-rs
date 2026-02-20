//! Sealed compression-width traits and compress/decompress operations.
//!
//! Each ML-KEM compression width (D=1, 4, 5, 10, 11, 12) is a zero-sized marker
//! type implementing [`CompressWidth`]. This eliminates runtime `match d { ...
//! }` dispatch and the associated `unreachable!()` branches.

use crate::{N, Q, SYMBYTES};

mod sealed {
    pub trait Sealed {}
}

pub trait CompressWidth: sealed::Sealed {
    const D: u32;
    const POLY_BYTES: usize;

    fn compress_poly(r: &mut [u8], coeffs: &[i16; N]);
    fn decompress_poly(coeffs: &mut [i16; N], a: &[u8]);
}

pub struct D1;
pub struct D4;
pub struct D5;
pub struct D10;
pub struct D11;

impl sealed::Sealed for D1 {}
impl sealed::Sealed for D4 {}
impl sealed::Sealed for D5 {}
impl sealed::Sealed for D10 {}
impl sealed::Sealed for D11 {}

#[inline]
pub const fn csubq(a: i16) -> u16 {
    let mut t = a as u16;
    t = t.wrapping_add(((a >> 15) as u16) & (Q as u16));
    t
}

#[inline]
const fn compress_coeff(x: u16, d: u32) -> u16 {
    let t = ((x as u32) << d).wrapping_add((Q as u32) / 2) / (Q as u32);
    (t & ((1u32 << d) - 1)) as u16
}

#[inline]
const fn decompress_coeff(y: u16, d: u32) -> u16 {
    (((y as u32) * (Q as u32) + (1u32 << (d - 1))) >> d) as u16
}

// -- D1: message encode/decode -----------------------------------------------

impl CompressWidth for D1 {
    const D: u32 = 1;
    const POLY_BYTES: usize = SYMBYTES;

    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        for i in 0..N / 8 {
            r[i] = 0;
            for j in 0..8u32 {
                let mut t = a[8 * i + j as usize] as u32;
                t = t.wrapping_add(((t as i16 >> 15) as u32) & (Q as u32));
                t = (((t << 1) + (Q as u32) / 2) / (Q as u32)) & 1;
                r[i] |= (t as u8) << j;
            }
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

// -- D4: ML-KEM-512/768 Dv --------------------------------------------------

impl CompressWidth for D4 {
    const D: u32 = 4;
    const POLY_BYTES: usize = 128;

    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        for i in 0..N / 2 {
            let t0 = compress_coeff(csubq(a[2 * i]), 4) as u8;
            let t1 = compress_coeff(csubq(a[2 * i + 1]), 4) as u8;
            r[i] = t0 | (t1 << 4);
        }
    }

    fn decompress_poly(r: &mut [i16; N], a: &[u8]) {
        for i in 0..N / 2 {
            r[2 * i] = decompress_coeff((a[i] & 0x0F) as u16, 4) as i16;
            r[2 * i + 1] = decompress_coeff((a[i] >> 4) as u16, 4) as i16;
        }
    }
}

// -- D5: ML-KEM-1024 Dv -----------------------------------------------------

impl CompressWidth for D5 {
    const D: u32 = 5;
    const POLY_BYTES: usize = 160;

    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        for i in 0..N / 8 {
            let t: [u8; 8] = core::array::from_fn(|j| compress_coeff(csubq(a[8 * i + j]), 5) as u8);
            r[5 * i] = t[0] | (t[1] << 5);
            r[5 * i + 1] = (t[1] >> 3) | (t[2] << 2) | (t[3] << 7);
            r[5 * i + 2] = (t[3] >> 1) | (t[4] << 4);
            r[5 * i + 3] = (t[4] >> 4) | (t[5] << 1) | (t[6] << 6);
            r[5 * i + 4] = (t[6] >> 2) | (t[7] << 3);
        }
    }

    fn decompress_poly(r: &mut [i16; N], a: &[u8]) {
        for i in 0..N / 8 {
            let b = &a[5 * i..];
            r[8 * i] = decompress_coeff((b[0] & 0x1F) as u16, 5) as i16;
            r[8 * i + 1] = decompress_coeff(((b[0] >> 5) | ((b[1] & 0x03) << 3)) as u16, 5) as i16;
            r[8 * i + 2] = decompress_coeff(((b[1] >> 2) & 0x1F) as u16, 5) as i16;
            r[8 * i + 3] = decompress_coeff(((b[1] >> 7) | ((b[2] & 0x0F) << 1)) as u16, 5) as i16;
            r[8 * i + 4] = decompress_coeff(((b[2] >> 4) | ((b[3] & 0x01) << 4)) as u16, 5) as i16;
            r[8 * i + 5] = decompress_coeff(((b[3] >> 1) & 0x1F) as u16, 5) as i16;
            r[8 * i + 6] = decompress_coeff(((b[3] >> 6) | ((b[4] & 0x07) << 2)) as u16, 5) as i16;
            r[8 * i + 7] = decompress_coeff((b[4] >> 3) as u16, 5) as i16;
        }
    }
}

// -- D10: ML-KEM-512/768 Du -------------------------------------------------

impl CompressWidth for D10 {
    const D: u32 = 10;
    const POLY_BYTES: usize = 320;

    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        for i in 0..N / 4 {
            let t: [u16; 4] = core::array::from_fn(|j| compress_coeff(csubq(a[4 * i + j]), 10));
            r[5 * i] = t[0] as u8;
            r[5 * i + 1] = ((t[0] >> 8) | (t[1] << 2)) as u8;
            r[5 * i + 2] = ((t[1] >> 6) | (t[2] << 4)) as u8;
            r[5 * i + 3] = ((t[2] >> 4) | (t[3] << 6)) as u8;
            r[5 * i + 4] = (t[3] >> 2) as u8;
        }
    }

    fn decompress_poly(r: &mut [i16; N], a: &[u8]) {
        for i in 0..N / 4 {
            let b = &a[5 * i..];
            r[4 * i] = decompress_coeff((b[0] as u16) | (((b[1] as u16) & 0x03) << 8), 10) as i16;
            r[4 * i + 1] =
                decompress_coeff(((b[1] as u16) >> 2) | (((b[2] as u16) & 0x0F) << 6), 10) as i16;
            r[4 * i + 2] =
                decompress_coeff(((b[2] as u16) >> 4) | (((b[3] as u16) & 0x3F) << 4), 10) as i16;
            r[4 * i + 3] = decompress_coeff(((b[3] as u16) >> 6) | ((b[4] as u16) << 2), 10) as i16;
        }
    }
}

// -- D11: ML-KEM-1024 Du ----------------------------------------------------

impl CompressWidth for D11 {
    const D: u32 = 11;
    const POLY_BYTES: usize = 352;

    fn compress_poly(r: &mut [u8], a: &[i16; N]) {
        for i in 0..N / 8 {
            let t: [u16; 8] = core::array::from_fn(|j| compress_coeff(csubq(a[8 * i + j]), 11));
            r[11 * i] = t[0] as u8;
            r[11 * i + 1] = ((t[0] >> 8) | (t[1] << 3)) as u8;
            r[11 * i + 2] = ((t[1] >> 5) | (t[2] << 6)) as u8;
            r[11 * i + 3] = (t[2] >> 2) as u8;
            r[11 * i + 4] = ((t[2] >> 10) | (t[3] << 1)) as u8;
            r[11 * i + 5] = ((t[3] >> 7) | (t[4] << 4)) as u8;
            r[11 * i + 6] = ((t[4] >> 4) | (t[5] << 7)) as u8;
            r[11 * i + 7] = (t[5] >> 1) as u8;
            r[11 * i + 8] = ((t[5] >> 9) | (t[6] << 2)) as u8;
            r[11 * i + 9] = ((t[6] >> 6) | (t[7] << 5)) as u8;
            r[11 * i + 10] = (t[7] >> 3) as u8;
        }
    }

    fn decompress_poly(r: &mut [i16; N], a: &[u8]) {
        for i in 0..N / 8 {
            let b = &a[11 * i..];
            r[8 * i] = decompress_coeff((b[0] as u16) | (((b[1] as u16) & 0x07) << 8), 11) as i16;
            r[8 * i + 1] =
                decompress_coeff(((b[1] as u16) >> 3) | (((b[2] as u16) & 0x3F) << 5), 11) as i16;
            r[8 * i + 2] = decompress_coeff(
                ((b[2] as u16) >> 6) | ((b[3] as u16) << 2) | (((b[4] as u16) & 0x01) << 10),
                11,
            ) as i16;
            r[8 * i + 3] =
                decompress_coeff(((b[4] as u16) >> 1) | (((b[5] as u16) & 0x0F) << 7), 11) as i16;
            r[8 * i + 4] =
                decompress_coeff(((b[5] as u16) >> 4) | (((b[6] as u16) & 0x7F) << 4), 11) as i16;
            r[8 * i + 5] = decompress_coeff(
                ((b[6] as u16) >> 7) | ((b[7] as u16) << 1) | (((b[8] as u16) & 0x03) << 9),
                11,
            ) as i16;
            r[8 * i + 6] =
                decompress_coeff(((b[8] as u16) >> 2) | (((b[9] as u16) & 0x1F) << 6), 11) as i16;
            r[8 * i + 7] =
                decompress_coeff(((b[9] as u16) >> 5) | ((b[10] as u16) << 3), 11) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_decompress_d4_roundtrip() {
        let mut a = [0i16; N];
        for (i, c) in a.iter_mut().enumerate() {
            *c = (i as i16 * 7) % Q;
        }
        let mut buf = [0u8; D4::POLY_BYTES];
        D4::compress_poly(&mut buf, &a);

        let mut b = [0i16; N];
        D4::decompress_poly(&mut b, &buf);

        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            let original = csubq(ai) as i32;
            let recovered = bi as i32;
            let diff = (original - recovered + Q as i32) % Q as i32;
            let diff = diff.min(Q as i32 - diff);
            assert!(
                diff <= (Q as i32) / (1 << 4),
                "excessive error at index {i}"
            );
        }
    }

    #[test]
    fn compress_decompress_d10_roundtrip() {
        let mut a = [0i16; N];
        for (i, c) in a.iter_mut().enumerate() {
            *c = (i as i16 * 11) % Q;
        }
        let mut buf = [0u8; D10::POLY_BYTES];
        D10::compress_poly(&mut buf, &a);

        let mut b = [0i16; N];
        D10::decompress_poly(&mut b, &buf);

        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            let original = csubq(ai) as i32;
            let recovered = bi as i32;
            let diff = (original - recovered + Q as i32) % Q as i32;
            let diff = diff.min(Q as i32 - diff);
            assert!(
                diff <= (Q as i32) / (1 << 10) + 1,
                "excessive error at index {i}"
            );
        }
    }
}
