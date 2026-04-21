//! SHAKE-128 XOF reader, generic over parallel lane count.

use core::simd::Simd;

use kem_math::SYMBYTES;

use super::keccak::{PLEN, absorb_seed, f1600};
use crate::{SHAKE_PAD, SHAKE128_RATE};

/// Parallel SHAKE-128 XOF reader over `L` lanes.
///
/// Created by [`xof_absorb`]. Each call to
/// [`squeeze_words`](Self::squeeze_words) produces one rate block (21 `u64`
/// words per lane) and advances all states with a single SIMD permutation.
#[repr(C, align(32))]
pub struct Shake128Reader<const L: usize> {
    state: [Simd<u64, L>; PLEN],
}

impl<const L: usize> Shake128Reader<L> {
    /// Squeeze one SHAKE-128 rate block and expose it as little-endian words.
    #[must_use]
    pub fn squeeze_words(&mut self) -> [[u64; L]; SHAKE128_RATE / 8] {
        f1600(&mut self.state);
        core::array::from_fn(|i| self.state[i].to_array())
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
