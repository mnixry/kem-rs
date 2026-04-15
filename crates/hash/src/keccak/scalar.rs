//! Scalar (single-lane) Keccak sponge for fixed-output hashes.

use core::simd::Simd;

use super::{PLEN, f1600};
use crate::{SHA3_256_RATE, SHA3_512_RATE, SHA3_PAD, SHAKE_PAD, SHAKE256_RATE};

type Lane = Simd<u64, 1>;

#[inline]
fn absorb_block<const R: usize>(state: &mut [Lane; PLEN], block: &[u8; R]) {
    const { assert!(R.is_multiple_of(8)) }
    for (s, b) in state.iter_mut().zip(block.as_chunks::<8>().0) {
        *s ^= Lane::splat(u64::from_le_bytes(*b));
    }
    f1600(state);
}

fn absorb_padded<const R: usize>(state: &mut [Lane; PLEN], input: &[u8], pad: u8) {
    let (blocks, remainder) = input.as_chunks::<R>();
    for block in blocks {
        absorb_block(state, block);
    }
    let mut last = [0u8; R];
    let remaining = remainder.len();
    last[..remaining].copy_from_slice(remainder);
    last[remaining] = pad;
    last[R - 1] |= 0x80;
    absorb_block(state, &last);
}

/// Streaming absorb: feeds `src` byte-by-byte into rate-sized blocks,
/// permuting whenever a block fills. Used by [`rkprf`] for multi-piece
/// input without allocation.
#[inline]
fn consume<const R: usize>(
    state: &mut [Lane; PLEN], block: &mut [u8; R], block_pos: &mut usize, src: &[u8],
) {
    let mut i = 0;
    while i < src.len() {
        let space = R - *block_pos;
        let n = space.min(src.len() - i);
        block[*block_pos..*block_pos + n].copy_from_slice(&src[i..i + n]);
        *block_pos += n;
        i += n;
        if *block_pos == R {
            absorb_block(state, block);
            block.fill(0);
            *block_pos = 0;
        }
    }
}

fn squeeze_into<const R: usize>(state: &mut [Lane; PLEN], output: &mut [u8]) {
    let mut offset = 0;
    while offset < output.len() {
        let n = (output.len() - offset).min(R);
        let (chunks, tail) = output[offset..offset + n].as_chunks_mut::<8>();
        for (chunk, word) in chunks.iter_mut().zip(state.iter()) {
            *chunk = word.to_array()[0].to_le_bytes();
        }
        if !tail.is_empty() {
            tail.copy_from_slice(&state[chunks.len()].to_array()[0].to_le_bytes()[..tail.len()]);
        }
        offset += n;
        if offset < output.len() {
            f1600(state);
        }
    }
}

fn sponge<const R: usize>(pad: u8, input: &[u8], output: &mut [u8]) {
    let mut state = [Lane::splat(0); PLEN];
    absorb_padded::<R>(&mut state, input, pad);
    squeeze_into::<R>(&mut state, output);
}

/// H(input) = SHA3-256(input) -> 32 bytes.
#[inline]
pub fn hash_h(input: impl AsRef<[u8]>) -> [u8; 32] {
    let mut out = [0u8; 32];
    sponge::<SHA3_256_RATE>(SHA3_PAD, input.as_ref(), &mut out);
    out
}

/// G(input) = SHA3-512(input) -> 64 bytes.
#[inline]
pub fn hash_g(input: impl AsRef<[u8]>) -> [u8; 64] {
    let mut out = [0u8; 64];
    sponge::<SHA3_512_RATE>(SHA3_PAD, input.as_ref(), &mut out);
    out
}

/// SHAKE-128(input) with arbitrary output length.
pub fn shake128(input: impl AsRef<[u8]>, output: &mut [u8]) {
    sponge::<{ crate::SHAKE128_RATE }>(SHAKE_PAD, input.as_ref(), output);
}

/// SHAKE-256(input) with arbitrary output length.
pub fn shake256(input: impl AsRef<[u8]>, output: &mut [u8]) {
    sponge::<SHAKE256_RATE>(SHAKE_PAD, input.as_ref(), output);
}

/// J(key, ct) = SHAKE-256(key || ct) -> 32 bytes (implicit-reject PRF).
///
/// Uses streaming absorb to avoid concatenating `key` and `ct`.
pub fn rkprf(key: impl AsRef<[u8]>, ct: impl AsRef<[u8]>) -> [u8; 32] {
    let mut state = [Lane::splat(0); PLEN];
    let mut block = [0u8; SHAKE256_RATE];
    let mut block_pos = 0;

    consume(&mut state, &mut block, &mut block_pos, key.as_ref());
    consume(&mut state, &mut block, &mut block_pos, ct.as_ref());

    block[block_pos] = SHAKE_PAD;
    block[block.len() - 1] |= 0x80;
    absorb_block(&mut state, &block);

    let mut out = [0u8; 32];
    squeeze_into::<SHAKE256_RATE>(&mut state, &mut out);
    out
}
