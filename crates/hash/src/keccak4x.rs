//! 4-way SIMD Keccak sponge via `keccak::simd::f1600x4`.
//!
//! All XOF (SHAKE-128) and PRF (SHAKE-256) sampling goes through these
//! functions. When fewer than 4 lanes are needed, callers pad with dummy
//! inputs and ignore the extra output.

use core::simd::u64x4;

use kem_math::{ByteArray, CbdWidth, unroll};

use crate::{SHAKE_PAD, SHAKE128_RATE, SHAKE256_RATE, SYMBYTES};

const PLEN: usize = 25;

fn absorb_seed(state: &mut [u64x4; PLEN], seed: &[u8; SYMBYTES]) {
    let chunks = seed.as_chunks().0;
    unroll!(i, [0, 1, 2, 3], {
        state[i] = u64x4::splat(u64::from_le_bytes(chunks[i]));
    });
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
    pub fn squeeze_blocks(&mut self) -> [[u8; SHAKE128_RATE]; 4] {
        let mut outs = [[0u8; SHAKE128_RATE]; 4];
        for (i, state) in self.state.iter().enumerate().take(SHAKE128_RATE / 8) {
            let words = state.to_array().map(u64::to_le_bytes);
            unroll!(j, [0, 1, 2, 3], {
                outs[j][i * 8..(i + 1) * 8].copy_from_slice(&words[j]);
            });
        }
        keccak::simd::f1600x4(&mut self.state);
        outs
    }
}

/// Absorb `seed || x || y` into 4 parallel SHAKE-128 states and return a
/// reader. Each lane gets a different `(x, y)` pair.
///
/// For remainder batches (fewer than 4 real elements), pad `indices` with
/// dummy `(0, 0)` pairs and discard the corresponding output lanes.
#[must_use]
pub fn xof_absorb_x4(seed: &[u8; SYMBYTES], indices: [(u8, u8); 4]) -> Shake128x4Reader {
    let mut state = [u64x4::splat(0); PLEN];
    absorb_seed(&mut state, seed);

    state[4] = u64x4::from_array(unroll!(lane, [0, 1, 2, 3], {
        let (x, y) = indices[lane];
        u64::from(x) | (u64::from(y) << 8) | (u64::from(SHAKE_PAD) << 16)
    }));

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
pub fn prf_x4<Eta: CbdWidth>(seed: &[u8; SYMBYTES], nonces: [u8; 4]) -> [Eta::Buffer; 4] {
    let mut outputs: [_; 4] = core::array::from_fn(|_| Eta::Buffer::zeroed());

    let mut state = [u64x4::splat(0); PLEN];
    absorb_seed(&mut state, seed);

    state[4] = u64x4::from_array(unroll!(lane, [0, 1, 2, 3], {
        u64::from(nonces[lane]) | (u64::from(SHAKE_PAD) << 8)
    }));

    // End-of-rate padding: byte 135 = word 16, byte offset 7.
    state[16] = u64x4::splat(0x80_u64 << 56);

    keccak::simd::f1600x4(&mut state);

    let mut written = 0;
    while written < Eta::BUF_BYTES {
        let chunk = (Eta::BUF_BYTES - written).min(SHAKE256_RATE);
        let full_words = chunk / 8;
        let tail_bytes = chunk % 8;

        for (i, s) in state.iter().enumerate().take(full_words) {
            let off = written + i * 8;
            let lanes = s.to_array().map(u64::to_le_bytes);
            unroll!(j, [0, 1, 2, 3], {
                outputs[j].as_mut()[off..off + 8].copy_from_slice(&lanes[j]);
            });
        }
        if tail_bytes > 0 {
            let off = written + full_words * 8;
            let lanes = state[full_words].to_array().map(u64::to_le_bytes);
            unroll!(j, [0, 1, 2, 3], {
                outputs[j].as_mut()[off..off + tail_bytes].copy_from_slice(&lanes[j][..tail_bytes]);
            });
        }

        written += chunk;
        if written < Eta::BUF_BYTES {
            keccak::simd::f1600x4(&mut state);
        }
    }

    outputs
}
