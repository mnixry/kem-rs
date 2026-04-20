//! Polynomial byte-level encoding (12-bit packing) and message encode/decode.

use core::simd::Simd;

use crate::{
    N, POLYBYTES, SYMBYTES,
    compress::{CompressWidth, D1, csubq},
};

pub fn coeffs_to_bytes(r: &mut [u8], a: &[i16; N]) {
    debug_assert!(r.len() >= POLYBYTES);
    let (out_triples, _) = r[..POLYBYTES].as_chunks_mut::<3>();
    let (in_pairs, _) = a.as_chunks::<2>();
    for (triple, &[a0, a1]) in out_triples.iter_mut().zip(in_pairs) {
        let t0 = csubq(a0);
        let t1 = csubq(a1);
        triple[0] = t0 as u8;
        triple[1] = ((t0 >> 8) | (t1 << 4)) as u8;
        triple[2] = (t1 >> 4) as u8;
    }
}

pub fn bytes_to_coeffs(r: &mut [i16; N], a: &[u8]) {
    debug_assert!(a.len() >= POLYBYTES);
    crate::simd::dispatch_lanes!(bytes_to_coeffs_lanes(r, a));
}

/// SIMD 12-bit unpacking: deinterleave overlapping u16 words, mask/shift to
/// extract 12-bit coefficients, then re-interleave into natural order.
#[inline]
fn bytes_to_coeffs_lanes<const L: usize>(r: &mut [i16; N], a: &[u8]) {
    let zero = Simd::<i16, L>::splat(0);
    let mask = Simd::<i16, L>::splat(0x0FFF);

    for (chunk, bytes) in r
        .as_chunks_mut::<L>()
        .0
        .iter_mut()
        .zip(a[..POLYBYTES].chunks_exact(3 * L / 2))
    {
        // Build alternating [lo_word, hi_word] pairs from byte triples.
        // lo_word = b0 | (b1 << 8)  contains even coefficient in low 12 bits
        // hi_word = b1 | (b2 << 8)  contains odd coefficient after >> 4
        let mut raw = [0i16; L];
        {
            let (pairs, _) = raw.as_chunks_mut::<2>();
            let (triples, _) = bytes.as_chunks::<3>();
            for ([lo, hi], &[b0, b1, b2]) in pairs.iter_mut().zip(triples) {
                *lo = (b0 as i16) | ((b1 as i16) << 8);
                *hi = (b1 as i16) | ((b2 as i16) << 8);
            }
        }

        let v = Simd::from_array(raw);
        // Separate lo_words (even indices) and hi_words (odd indices)
        let (lo_words, hi_words) = v.deinterleave(zero);
        let even = lo_words & mask;
        let odd = (hi_words >> Simd::splat(4)) & mask;
        // Re-interleave: [even0, odd0, even1, odd1, ...]
        let (result, _) = even.interleave(odd);
        *chunk = result.into();
    }
}

pub fn coeffs_to_message(msg: &mut [u8; SYMBYTES], a: &[i16; N]) {
    D1::compress_poly(msg, a);
}

pub fn message_to_coeffs(r: &mut [i16; N], msg: &[u8; SYMBYTES]) {
    D1::decompress_poly(r, msg);
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
        coeffs_to_bytes(&mut buf, &a);

        let mut b = [0i16; N];
        bytes_to_coeffs(&mut b, &buf);
        assert_eq!(a, b);
    }

    #[test]
    fn frommsg_tomsg_roundtrip() {
        let msg: [u8; SYMBYTES] = core::array::from_fn(|i| (i * 37) as u8);
        let mut poly = [0i16; N];
        message_to_coeffs(&mut poly, &msg);

        let mut recovered = [0u8; SYMBYTES];
        coeffs_to_message(&mut recovered, &poly);
        assert_eq!(msg, recovered);
    }
}
