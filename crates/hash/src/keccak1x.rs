//! Scalar Keccak sponge for fixed-output hashes.
//!
//! Only used for `hash_h` (SHA3-256), `hash_g` (SHA3-512), and `rkprf`
//! (SHAKE-256 implicit-reject) which have variable-length inputs and no
//! natural batching opportunity.

use crate::{SHA3_256_RATE, SHA3_512_RATE, SHA3_PAD, SHAKE_PAD, SHAKE256_RATE};

const PLEN: usize = 25;

#[inline]
fn absorb_block<const R: usize>(state: &mut [u64; PLEN], block: &[u8; R]) {
    const { assert!(R.is_multiple_of(8)) }
    for (s, b) in state.iter_mut().zip(block.as_chunks().0) {
        *s ^= u64::from_le_bytes(*b);
    }
    keccak::f1600(state);
}

fn absorb_padded<const R: usize>(state: &mut [u64; PLEN], input: &[u8], pad: u8) {
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

#[inline]
fn squeeze<const N: usize>(state: &[u64; PLEN]) -> [u8; N] {
    const { assert!(N.is_multiple_of(8)) }
    let mut out = [0u8; N];
    for (chunk, word) in out.as_chunks_mut().0.iter_mut().zip(state) {
        *chunk = word.to_le_bytes();
    }
    out
}

/// H(input) = SHA3-256(input) -> 32 bytes.
#[inline]
pub fn hash_h(input: impl AsRef<[u8]>) -> [u8; 32] {
    let mut state = [0u64; PLEN];
    absorb_padded::<SHA3_256_RATE>(&mut state, input.as_ref(), SHA3_PAD);
    squeeze(&state)
}

/// G(input) = SHA3-512(input) -> 64 bytes.
#[inline]
pub fn hash_g(input: impl AsRef<[u8]>) -> [u8; 64] {
    let mut state = [0u64; PLEN];
    absorb_padded::<SHA3_512_RATE>(&mut state, input.as_ref(), SHA3_PAD);
    squeeze(&state)
}

#[inline]
fn consume<const R: usize>(
    state: &mut [u64; PLEN], block: &mut [u8; R], block_pos: &mut usize, src: &[u8],
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

/// J(key, ct) = SHAKE-256(key || ct) -> 32 bytes (implicit-reject PRF).
pub fn rkprf(key: impl AsRef<[u8]>, ct: impl AsRef<[u8]>) -> [u8; 32] {
    let mut state = [0u64; PLEN];

    let key = key.as_ref();
    let ct = ct.as_ref();
    let mut block = [0u8; SHAKE256_RATE];
    let mut block_pos = 0;

    consume(&mut state, &mut block, &mut block_pos, key);
    consume(&mut state, &mut block, &mut block_pos, ct);

    block[block_pos] = SHAKE_PAD;
    block[block.len() - 1] |= 0x80;
    absorb_block(&mut state, &block);

    squeeze(&state)
}
