//! Polynomial byte-level encoding (12-bit packing) and message encode/decode.

use crate::{
    N, POLYBYTES, SYMBYTES,
    compress::{CompressWidth, D1, csubq},
};

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

pub fn poly_frombytes(r: &mut [i16; N], a: &[u8]) {
    debug_assert!(a.len() >= POLYBYTES);
    for i in 0..N / 2 {
        r[2 * i] = ((a[3 * i] as u16) | (((a[3 * i + 1] as u16) & 0x0F) << 8)) as i16;
        r[2 * i + 1] = (((a[3 * i + 1] as u16) >> 4) | ((a[3 * i + 2] as u16) << 4)) as i16;
    }
}

pub fn poly_frommsg(r: &mut [i16; N], msg: &[u8; SYMBYTES]) {
    D1::decompress_poly(r, msg);
}

pub fn poly_tomsg(msg: &mut [u8; SYMBYTES], a: &[i16; N]) {
    D1::compress_poly(msg, a);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Q;

    #[test]
    fn tobytes_frombytes_roundtrip() {
        let mut a = [0i16; N];
        for (i, c) in a.iter_mut().enumerate() {
            *c = (i as i16 * 13) % (Q - 1);
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
}
