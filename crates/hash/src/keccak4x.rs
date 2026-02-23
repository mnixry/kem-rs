//! 4-way SIMD Keccak sponge via `keccak::simd::f1600x4`.
//!
//! All XOF (SHAKE-128) and PRF (SHAKE-256) sampling goes through these
//! functions. When fewer than 4 lanes are needed, callers pad with dummy
//! inputs and ignore the extra output.

use core::simd::u64x4;

use kem_math::{ByteArray, CbdWidth};

use crate::{SHAKE_PAD, SHAKE128_RATE, SHAKE256_RATE};

const PLEN: usize = 25;
const SHAKE128_RATE_WORDS: usize = SHAKE128_RATE / 8;

fn absorb_seed(state: &mut [u64x4; PLEN], seed: &[u8; 32]) {
    for (i, s) in state.iter_mut().enumerate().take(4) {
        let off = i * 8;
        let word = u64::from_le_bytes([
            seed[off],
            seed[off + 1],
            seed[off + 2],
            seed[off + 3],
            seed[off + 4],
            seed[off + 5],
            seed[off + 6],
            seed[off + 7],
        ]);
        *s = u64x4::splat(word);
    }
}

/// 4-way parallel SHAKE-128 XOF reader.
///
/// Created by [`xof_absorb_x4`]. Each call to
/// [`squeeze_blocks`](Self::squeeze_blocks) produces one 168-byte block per
/// lane and advances all four states with a single SIMD permutation.
pub struct Shake128x4Reader {
    state: [u64x4; PLEN],
}

impl Shake128x4Reader {
    /// Squeeze one SHAKE-128 rate block (168 bytes) from each of the 4 lanes.
    #[inline]
    pub fn squeeze_blocks(&mut self, out: &mut [[u8; SHAKE128_RATE]; 4]) {
        for (i, s) in self.state.iter().enumerate().take(SHAKE128_RATE_WORDS) {
            let lanes = s.to_array();
            let b0 = lanes[0].to_le_bytes();
            let b1 = lanes[1].to_le_bytes();
            let b2 = lanes[2].to_le_bytes();
            let b3 = lanes[3].to_le_bytes();
            let off = i * 8;
            out[0][off..off + 8].copy_from_slice(&b0);
            out[1][off..off + 8].copy_from_slice(&b1);
            out[2][off..off + 8].copy_from_slice(&b2);
            out[3][off..off + 8].copy_from_slice(&b3);
        }
        keccak::simd::f1600x4(&mut self.state);
    }
}

/// Absorb `seed || x || y` into 4 parallel SHAKE-128 states and return a
/// reader. Each lane gets a different `(x, y)` pair.
///
/// For remainder batches (fewer than 4 real elements), pad `indices` with
/// dummy `(0, 0)` pairs and discard the corresponding output lanes.
#[must_use]
pub fn xof_absorb_x4(seed: &[u8; 32], indices: [(u8, u8); 4]) -> Shake128x4Reader {
    let mut state = [u64x4::splat(0); PLEN];
    absorb_seed(&mut state, seed);

    let mut w4 = [0u64; 4];
    for (lane, &(x, y)) in indices.iter().enumerate() {
        w4[lane] = u64::from(x) | (u64::from(y) << 8) | (u64::from(SHAKE_PAD) << 16);
    }
    state[4] = u64x4::from_array(w4);

    // End-of-rate padding: byte 167 = word 20, byte offset 7.
    state[20] = u64x4::splat(0x80_u64 << 56);

    keccak::simd::f1600x4(&mut state);

    Shake128x4Reader { state }
}

/// 4-way parallel SHAKE-256 PRF: absorb `seed || nonce` into 4 states and
/// squeeze `out_len` bytes per lane.
///
/// All four output slices **must** have the same length.
/// For remainder batches, pad `nonces` with `0` and ignore extra output.
#[must_use]
pub fn prf_x4<Eta: CbdWidth>(seed: &[u8; 32], nonces: [u8; 4]) -> [Eta::Buffer; 4] {
    let mut outputs: [_; 4] = core::array::from_fn(|_| Eta::Buffer::zeroed());

    let mut state = [u64x4::splat(0); PLEN];
    absorb_seed(&mut state, seed);

    let mut w4 = [0u64; 4];
    for (lane, &n) in nonces.iter().enumerate() {
        w4[lane] = u64::from(n) | (u64::from(SHAKE_PAD) << 8);
    }
    state[4] = u64x4::from_array(w4);

    // End-of-rate padding: byte 135 = word 16, byte offset 7.
    state[16] = u64x4::splat(0x80_u64 << 56);

    keccak::simd::f1600x4(&mut state);

    let mut written = 0;
    while written < Eta::BUF_BYTES {
        let chunk = (Eta::BUF_BYTES - written).min(SHAKE256_RATE);
        let full_words = chunk / 8;
        let tail_bytes = chunk % 8;

        for (i, s) in state.iter().enumerate().take(full_words) {
            let lanes = s.to_array();
            let off = written + i * 8;
            outputs[0].as_mut()[off..off + 8].copy_from_slice(&lanes[0].to_le_bytes());
            outputs[1].as_mut()[off..off + 8].copy_from_slice(&lanes[1].to_le_bytes());
            outputs[2].as_mut()[off..off + 8].copy_from_slice(&lanes[2].to_le_bytes());
            outputs[3].as_mut()[off..off + 8].copy_from_slice(&lanes[3].to_le_bytes());
        }
        if tail_bytes > 0 {
            let lanes = state[full_words].to_array();
            let off = written + full_words * 8;
            let b0 = lanes[0].to_le_bytes();
            let b1 = lanes[1].to_le_bytes();
            let b2 = lanes[2].to_le_bytes();
            let b3 = lanes[3].to_le_bytes();
            outputs[0].as_mut()[off..off + tail_bytes].copy_from_slice(&b0[..tail_bytes]);
            outputs[1].as_mut()[off..off + tail_bytes].copy_from_slice(&b1[..tail_bytes]);
            outputs[2].as_mut()[off..off + tail_bytes].copy_from_slice(&b2[..tail_bytes]);
            outputs[3].as_mut()[off..off + tail_bytes].copy_from_slice(&b3[..tail_bytes]);
        }

        written += chunk;
        if written < Eta::BUF_BYTES {
            keccak::simd::f1600x4(&mut state);
        }
    }

    outputs
}
