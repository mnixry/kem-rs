//! Scalar Keccak sponge for fixed-output hashes.
//!
//! Only used for `hash_h` (SHA3-256), `hash_g` (SHA3-512), and `rkprf`
//! (SHAKE-256 implicit-reject) which have variable-length inputs and no
//! natural batching opportunity.

use crate::{SHA3_256_RATE, SHA3_512_RATE, SHA3_PAD, SHAKE_PAD, SHAKE256_RATE};

const PLEN: usize = 25;

#[inline]
fn absorb_block(state: &mut [u64; PLEN], block: &[u8]) {
    debug_assert!(block.len().is_multiple_of(8));
    for (b, s) in block.chunks_exact(8).zip(state.iter_mut()) {
        *s ^= u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
    }
    keccak::f1600(state);
}

fn absorb_padded(state: &mut [u64; PLEN], input: &[u8], rate: usize, pad: u8) {
    let mut offset = 0;
    while offset + rate <= input.len() {
        absorb_block(state, &input[offset..offset + rate]);
        offset += rate;
    }
    let mut last = [0u8; 200];
    let remaining = input.len() - offset;
    last[..remaining].copy_from_slice(&input[offset..]);
    last[remaining] = pad;
    last[rate - 1] |= 0x80;
    absorb_block(state, &last[..rate]);
}

#[inline]
fn squeeze(state: &[u64; PLEN], out: &mut [u8]) {
    let mut written = 0;
    for &word in state {
        if written >= out.len() {
            break;
        }
        let bytes = word.to_le_bytes();
        let remaining = out.len() - written;
        let n = remaining.min(8);
        out[written..written + n].copy_from_slice(&bytes[..n]);
        written += n;
    }
}

/// H(input) = SHA3-256(input) -> 32 bytes.
#[inline]
pub fn hash_h(input: impl AsRef<[u8]>) -> [u8; 32] {
    let mut state = [0u64; PLEN];
    absorb_padded(&mut state, input.as_ref(), SHA3_256_RATE, SHA3_PAD);
    let mut out = [0u8; 32];
    squeeze(&state, &mut out);
    out
}

/// G(input) = SHA3-512(input) -> 64 bytes.
#[inline]
pub fn hash_g(input: impl AsRef<[u8]>) -> [u8; 64] {
    let mut state = [0u64; PLEN];
    absorb_padded(&mut state, input.as_ref(), SHA3_512_RATE, SHA3_PAD);
    let mut out = [0u8; 64];
    squeeze(&state, &mut out);
    out
}

/// J(key, ct) = SHAKE-256(key || ct) -> 32 bytes (implicit-reject PRF).
pub fn rkprf(key: impl AsRef<[u8]>, ct: impl AsRef<[u8]>) -> [u8; 32] {
    let mut state = [0u64; PLEN];

    let key = key.as_ref();
    let ct = ct.as_ref();
    let rate = SHAKE256_RATE;

    let mut block = [0u8; 200];
    let mut block_pos = 0;

    let consume =
        |state: &mut [u64; PLEN], block: &mut [u8; 200], block_pos: &mut usize, src: &[u8]| {
            let mut i = 0;
            while i < src.len() {
                let space = rate - *block_pos;
                let n = space.min(src.len() - i);
                block[*block_pos..*block_pos + n].copy_from_slice(&src[i..i + n]);
                *block_pos += n;
                i += n;
                if *block_pos == rate {
                    absorb_block(state, &block[..rate]);
                    block[..rate].fill(0);
                    *block_pos = 0;
                }
            }
        };

    consume(&mut state, &mut block, &mut block_pos, key);
    consume(&mut state, &mut block, &mut block_pos, ct);

    block[block_pos] = SHAKE_PAD;
    block[rate - 1] |= 0x80;
    absorb_block(&mut state, &block[..rate]);

    let mut out = [0u8; 32];
    squeeze(&state, &mut out);
    out
}
