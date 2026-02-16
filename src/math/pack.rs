//! Byte-level packing, unpacking, compression, and decompression.
//!
//! All functions operate on raw coefficient slices (`&[i16; N]`) and byte
//! buffers, keeping this module independent of the `Poly` wrapper.

use crate::params::{N, Q, POLYBYTES, SYMBYTES};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Conditionally add q to make a coefficient non-negative.
#[inline]
pub(crate) fn csubq(a: i16) -> u16 {
    let mut t = a as u16;
    t = t.wrapping_add(((a >> 15) as u16) & (Q as u16));
    t
}

/// Compress coefficient to `d` bits: `round((2^d / q) · x) mod 2^d`.
///
/// Input `x` must be in `[0, q)`.
#[inline]
fn compress(x: u16, d: u32) -> u16 {
    let t = ((x as u32) << d).wrapping_add((Q as u32) / 2) / (Q as u32);
    (t & ((1u32 << d) - 1)) as u16
}

/// Decompress `d`-bit value: `round((q / 2^d) · y)`.
#[inline]
fn decompress(y: u16, d: u32) -> u16 {
    (((y as u32) * (Q as u32) + (1u32 << (d - 1))) >> d) as u16
}

// ---------------------------------------------------------------------------
// 12-bit serialisation (POLYBYTES = 384)
// ---------------------------------------------------------------------------

/// Serialize coefficients to bytes (12-bit encoding, 2 coefficients → 3 bytes).
pub fn poly_tobytes(r: &mut [u8], a: &[i16; N]) {
    debug_assert!(r.len() >= POLYBYTES);
    for i in 0..N / 2 {
        let t0 = csubq(a[2 * i]);
        let t1 = csubq(a[2 * i + 1]);
        r[3 * i] = t0 as u8;
        r[3 * i + 1] = ((t0 >> 8) | (t1 << 4)) as u8;
        r[3 * i + 2] = (t1 >> 4) as u8;
    }
}

/// Deserialize bytes to coefficients (12-bit decoding).
pub fn poly_frombytes(r: &mut [i16; N], a: &[u8]) {
    debug_assert!(a.len() >= POLYBYTES);
    for i in 0..N / 2 {
        r[2 * i] = ((a[3 * i] as u16) | (((a[3 * i + 1] as u16) & 0x0F) << 8)) as i16;
        r[2 * i + 1] = (((a[3 * i + 1] as u16) >> 4) | ((a[3 * i + 2] as u16) << 4)) as i16;
    }
}

// ---------------------------------------------------------------------------
// Message encoding (1-bit per coefficient)
// ---------------------------------------------------------------------------

/// Decode a 32-byte message into polynomial coefficients.
///
/// Each bit maps to `0` or `⌈q/2⌉ = 1665`.
pub fn poly_frommsg(r: &mut [i16; N], msg: &[u8; SYMBYTES]) {
    for i in 0..N / 8 {
        for j in 0..8u32 {
            let mask = -(((msg[i] >> j) & 1) as i16);
            r[8 * i + j as usize] = mask & ((Q + 1) / 2);
        }
    }
}

/// Encode polynomial to 32-byte message (compress to 1 bit per coefficient).
pub fn poly_tomsg(msg: &mut [u8; SYMBYTES], a: &[i16; N]) {
    for i in 0..N / 8 {
        msg[i] = 0;
        for j in 0..8u32 {
            let mut t = a[8 * i + j as usize] as u32;
            // Make non-negative
            t = t.wrapping_add(((t as i16 >> 15) as u32) & (Q as u32));
            // Compress to 1 bit
            t = (((t << 1) + (Q as u32) / 2) / (Q as u32)) & 1;
            msg[i] |= (t as u8) << j;
        }
    }
}

// ---------------------------------------------------------------------------
// Polynomial compression (d = 4 or 5, for ciphertext v component)
// ---------------------------------------------------------------------------

/// Compress with d = 4 (ML-KEM-512 / 768): 2 coefficients → 1 byte, 128 bytes total.
pub fn poly_compress_d4(r: &mut [u8], a: &[i16; N]) {
    debug_assert!(r.len() >= 128);
    for i in 0..N / 2 {
        let t0 = compress(csubq(a[2 * i]), 4) as u8;
        let t1 = compress(csubq(a[2 * i + 1]), 4) as u8;
        r[i] = t0 | (t1 << 4);
    }
}

/// Decompress with d = 4.
pub fn poly_decompress_d4(r: &mut [i16; N], a: &[u8]) {
    debug_assert!(a.len() >= 128);
    for i in 0..N / 2 {
        r[2 * i] = decompress((a[i] & 0x0F) as u16, 4) as i16;
        r[2 * i + 1] = decompress((a[i] >> 4) as u16, 4) as i16;
    }
}

/// Compress with d = 5 (ML-KEM-1024): 8 coefficients → 5 bytes, 160 bytes total.
pub fn poly_compress_d5(r: &mut [u8], a: &[i16; N]) {
    debug_assert!(r.len() >= 160);
    for i in 0..N / 8 {
        let t: [u8; 8] = core::array::from_fn(|j| compress(csubq(a[8 * i + j]), 5) as u8);
        r[5 * i] = t[0] | (t[1] << 5);
        r[5 * i + 1] = (t[1] >> 3) | (t[2] << 2) | (t[3] << 7);
        r[5 * i + 2] = (t[3] >> 1) | (t[4] << 4);
        r[5 * i + 3] = (t[4] >> 4) | (t[5] << 1) | (t[6] << 6);
        r[5 * i + 4] = (t[6] >> 2) | (t[7] << 3);
    }
}

/// Decompress with d = 5.
pub fn poly_decompress_d5(r: &mut [i16; N], a: &[u8]) {
    debug_assert!(a.len() >= 160);
    for i in 0..N / 8 {
        let b = &a[5 * i..];
        r[8 * i] = decompress((b[0] & 0x1F) as u16, 5) as i16;
        r[8 * i + 1] = decompress(((b[0] >> 5) | ((b[1] & 0x03) << 3)) as u16, 5) as i16;
        r[8 * i + 2] = decompress(((b[1] >> 2) & 0x1F) as u16, 5) as i16;
        r[8 * i + 3] = decompress(((b[1] >> 7) | ((b[2] & 0x0F) << 1)) as u16, 5) as i16;
        r[8 * i + 4] = decompress(((b[2] >> 4) | ((b[3] & 0x01) << 4)) as u16, 5) as i16;
        r[8 * i + 5] = decompress(((b[3] >> 1) & 0x1F) as u16, 5) as i16;
        r[8 * i + 6] = decompress(((b[3] >> 6) | ((b[4] & 0x07) << 2)) as u16, 5) as i16;
        r[8 * i + 7] = decompress((b[4] >> 3) as u16, 5) as i16;
    }
}

// ---------------------------------------------------------------------------
// Polynomial-vector element compression (d = 10 or 11, for ciphertext u)
// ---------------------------------------------------------------------------

/// Compress one polynomial with d = 10 (ML-KEM-512/768): 4 coefficients → 5 bytes, 320 bytes.
pub fn poly_compress_d10(r: &mut [u8], a: &[i16; N]) {
    debug_assert!(r.len() >= 320);
    for i in 0..N / 4 {
        let t: [u16; 4] = core::array::from_fn(|j| compress(csubq(a[4 * i + j]), 10));
        r[5 * i] = t[0] as u8;
        r[5 * i + 1] = ((t[0] >> 8) | (t[1] << 2)) as u8;
        r[5 * i + 2] = ((t[1] >> 6) | (t[2] << 4)) as u8;
        r[5 * i + 3] = ((t[2] >> 4) | (t[3] << 6)) as u8;
        r[5 * i + 4] = (t[3] >> 2) as u8;
    }
}

/// Decompress one polynomial with d = 10.
pub fn poly_decompress_d10(r: &mut [i16; N], a: &[u8]) {
    debug_assert!(a.len() >= 320);
    for i in 0..N / 4 {
        let b = &a[5 * i..];
        r[4 * i] = decompress((b[0] as u16) | (((b[1] as u16) & 0x03) << 8), 10) as i16;
        r[4 * i + 1] =
            decompress(((b[1] as u16) >> 2) | (((b[2] as u16) & 0x0F) << 6), 10) as i16;
        r[4 * i + 2] =
            decompress(((b[2] as u16) >> 4) | (((b[3] as u16) & 0x3F) << 4), 10) as i16;
        r[4 * i + 3] = decompress(((b[3] as u16) >> 6) | ((b[4] as u16) << 2), 10) as i16;
    }
}

/// Compress one polynomial with d = 11 (ML-KEM-1024): 8 coefficients → 11 bytes, 352 bytes.
pub fn poly_compress_d11(r: &mut [u8], a: &[i16; N]) {
    debug_assert!(r.len() >= 352);
    for i in 0..N / 8 {
        let t: [u16; 8] = core::array::from_fn(|j| compress(csubq(a[8 * i + j]), 11));
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

/// Decompress one polynomial with d = 11.
pub fn poly_decompress_d11(r: &mut [i16; N], a: &[u8]) {
    debug_assert!(a.len() >= 352);
    for i in 0..N / 8 {
        let b = &a[11 * i..];
        r[8 * i] = decompress((b[0] as u16) | (((b[1] as u16) & 0x07) << 8), 11) as i16;
        r[8 * i + 1] =
            decompress(((b[1] as u16) >> 3) | (((b[2] as u16) & 0x3F) << 5), 11) as i16;
        r[8 * i + 2] = decompress(
            ((b[2] as u16) >> 6) | ((b[3] as u16) << 2) | (((b[4] as u16) & 0x01) << 10),
            11,
        ) as i16;
        r[8 * i + 3] =
            decompress(((b[4] as u16) >> 1) | (((b[5] as u16) & 0x0F) << 7), 11) as i16;
        r[8 * i + 4] =
            decompress(((b[5] as u16) >> 4) | (((b[6] as u16) & 0x7F) << 4), 11) as i16;
        r[8 * i + 5] = decompress(
            ((b[6] as u16) >> 7) | ((b[7] as u16) << 1) | (((b[8] as u16) & 0x03) << 9),
            11,
        ) as i16;
        r[8 * i + 6] =
            decompress(((b[8] as u16) >> 2) | (((b[9] as u16) & 0x1F) << 6), 11) as i16;
        r[8 * i + 7] =
            decompress(((b[9] as u16) >> 5) | ((b[10] as u16) << 3), 11) as i16;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tobytes_frombytes_roundtrip() {
        let mut a = [0i16; N];
        for i in 0..N {
            a[i] = (i as i16 * 13) % (Q - 1); // values in [0, q-1)
        }
        let mut buf = [0u8; POLYBYTES];
        poly_tobytes(&mut buf, &a);

        let mut b = [0i16; N];
        poly_frombytes(&mut b, &buf);
        assert_eq!(a, b);
    }

    #[test]
    fn frommsg_tomsg_roundtrip() {
        let msg: [u8; SYMBYTES] = core::array::from_fn(|i| (i * 37) as u8);
        let mut poly = [0i16; N];
        poly_frommsg(&mut poly, &msg);

        let mut recovered = [0u8; SYMBYTES];
        poly_tomsg(&mut recovered, &poly);
        assert_eq!(msg, recovered);
    }

    #[test]
    fn compress_decompress_d4_roundtrip() {
        let mut a = [0i16; N];
        for i in 0..N {
            a[i] = (i as i16 * 7) % Q;
        }
        let mut buf = [0u8; 128];
        poly_compress_d4(&mut buf, &a);

        let mut b = [0i16; N];
        poly_decompress_d4(&mut b, &buf);

        // Compression is lossy; check round-trip error is bounded
        for i in 0..N {
            let original = csubq(a[i]) as i32;
            let recovered = b[i] as i32;
            let diff = (original - recovered + Q as i32) % Q as i32;
            let diff = diff.min(Q as i32 - diff);
            assert!(diff <= (Q as i32) / (1 << 4), "excessive error at index {i}");
        }
    }

    #[test]
    fn compress_decompress_d10_roundtrip() {
        let mut a = [0i16; N];
        for i in 0..N {
            a[i] = (i as i16 * 11) % Q;
        }
        let mut buf = [0u8; 320];
        poly_compress_d10(&mut buf, &a);

        let mut b = [0i16; N];
        poly_decompress_d10(&mut b, &buf);

        for i in 0..N {
            let original = csubq(a[i]) as i32;
            let recovered = b[i] as i32;
            let diff = (original - recovered + Q as i32) % Q as i32;
            let diff = diff.min(Q as i32 - diff);
            assert!(diff <= (Q as i32) / (1 << 10) + 1, "excessive error at index {i}");
        }
    }
}
