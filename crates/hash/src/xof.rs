//! SHAKE-128 XOF reader, generic over parallel lane count.

use core::simd::{Simd, num::SimdUint};

use kem_math::{SYMBYTES, unroll};

use super::keccak::{PLEN, absorb_seed, f1600};
use crate::{SHAKE_PAD, SHAKE128_RATE};

/// Parallel SHAKE-128 XOF reader over `L` lanes.
///
/// Created by [`xof_absorb`]. Each call to
/// [`squeeze_blocks`](Self::squeeze_blocks) produces one 168-byte block per
/// lane and advances all states with a single SIMD permutation.
#[repr(C, align(32))]
pub struct Shake128Reader<const L: usize> {
    state: [Simd<u64, L>; PLEN],
}

impl<const L: usize> Shake128Reader<L> {
    /// Squeeze one SHAKE-128 rate block (168 bytes) from each of the `L` lanes.
    ///
    /// Returns bytes in position-major order: `out[byte_pos][lane]`.
    /// This keeps each SIMD word's bytes contiguous in memory, avoiding
    /// 168-byte strides between lane writes.
    pub fn squeeze_blocks(&mut self) -> [[u8; L]; SHAKE128_RATE] {
        let mut outs = [[0u8; L]; SHAKE128_RATE];
        f1600(&mut self.state);
        for (word, out_chunk) in self.state.iter().zip(outs.as_chunks_mut::<8>().0) {
            *out_chunk = unroll!(i, [0, 1, 2, 3, 4, 5, 6, 7], {
                *(word >> (i * 8)).cast::<u8>().as_array()
            });
        }
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

    Shake128Reader { state }
}
