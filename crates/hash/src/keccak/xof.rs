//! SHAKE-128 XOF reader, generic over parallel lane count.

use core::simd::Simd;

use kem_math::SYMBYTES;

use super::{PLEN, absorb_seed, f1600};
use crate::{SHAKE_PAD, SHAKE128_RATE};

/// Parallel SHAKE-128 XOF reader over `L` lanes.
///
/// Created by [`xof_absorb`]. Each call to
/// [`squeeze_blocks`](Self::squeeze_blocks) produces one 168-byte block per
/// lane and advances all states with a single SIMD permutation.
pub struct Shake128Reader<const L: usize> {
    state: [Simd<u64, L>; PLEN],
}

impl<const L: usize> Shake128Reader<L> {
    /// Squeeze one SHAKE-128 rate block (168 bytes) from each of the `L` lanes.
    #[inline]
    pub fn squeeze_blocks(&mut self) -> [[u8; SHAKE128_RATE]; L] {
        let mut outs = [[0u8; SHAKE128_RATE]; L];
        for (i, word) in self.state.iter().enumerate().take(SHAKE128_RATE / 8) {
            let lanes = word.to_array();
            for (j, val) in lanes.iter().enumerate() {
                outs[j][i * 8..(i + 1) * 8].copy_from_slice(&val.to_le_bytes());
            }
        }
        f1600(&mut self.state);
        outs
    }
}

/// Absorb `seed || x || y` into `L` parallel SHAKE-128 states.
///
/// Each lane receives a different `(x, y)` pair from `indices`.
#[must_use]
pub fn xof_absorb<const L: usize>(
    seed: &[u8; SYMBYTES], indices: [(u8, u8); L],
) -> Shake128Reader<L> {
    let mut state = [Simd::splat(0u64); PLEN];
    absorb_seed(&mut state, seed);

    state[4] = Simd::from_array(core::array::from_fn(|lane| {
        let (x, y) = indices[lane];
        u64::from(x) | (u64::from(y) << 8) | (u64::from(SHAKE_PAD) << 16)
    }));

    // End-of-rate padding: byte 167 = word 20, byte offset 7.
    state[20] = Simd::splat(0x80_u64 << 56);

    f1600(&mut state);

    Shake128Reader { state }
}
