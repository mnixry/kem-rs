//! SHAKE-128 XOF reader, generic over parallel lane count.

use core::simd::Simd;

use kem_math::SYMBYTES;

use super::keccak::{PLEN, absorb_seed, f1600};
use crate::{SHAKE_PAD, SHAKE128_RATE};

mod sealed {
    pub trait Sealed {}
}

pub trait SqueezeWords<const L: usize>: sealed::Sealed {
    fn squeeze_words(&mut self) -> [[u64; L]; SHAKE128_RATE / 8];
}

/// Parallel SHAKE-128 XOF reader over `L` lanes.
///
/// Created by [`xof_absorb`]. Each call to
/// [`squeeze_words`](Self::squeeze_words) produces one rate block (21 `u64`
/// words per lane) and advances all states with a single SIMD permutation.
#[repr(C, align(32))]
pub struct VectorShake128Reader<const L: usize>([Simd<u64, L>; PLEN]);
impl<const L: usize> sealed::Sealed for VectorShake128Reader<L> {}
impl<const L: usize> SqueezeWords<L> for VectorShake128Reader<L> {
    /// Squeeze one SHAKE-128 rate block and expose it as little-endian words.
    fn squeeze_words(&mut self) -> [[u64; L]; SHAKE128_RATE / 8] {
        f1600(&mut self.0);
        core::array::from_fn(|i| self.0[i].to_array())
    }
}

pub struct ScalarShake128Reader([u64; PLEN]);
impl sealed::Sealed for ScalarShake128Reader {}
impl SqueezeWords<1> for ScalarShake128Reader {
    fn squeeze_words(&mut self) -> [[u64; 1]; SHAKE128_RATE / 8] {
        f1600(&mut self.0);
        core::array::from_fn(|i| [self.0[i]])
    }
}

/// Absorb `seed || x || y` into `L` parallel SHAKE-128 states.
///
/// Each lane receives a different `(x, y)` pair from `indices`.
#[inline]
fn xof_absorb_vec<const L: usize>(
    seed: &[u8; SYMBYTES], indices: [(u8, u8); L],
) -> impl SqueezeWords<L> {
    let mut state = [Simd::splat(0u64); PLEN];
    absorb_seed(&mut state, seed);

    state[4] = Simd::from_array(core::array::from_fn(|lane| {
        let (x, y) = indices[lane];
        u64::from(x) | (u64::from(y) << 8) | (u64::from(SHAKE_PAD) << 16)
    }));

    // End-of-rate padding: byte 167 = word 20, byte offset 7.
    state[20] = Simd::splat(0x80_u64 << 56);

    VectorShake128Reader(state)
}

#[inline]
fn xof_absorb_scalar(seed: &[u8; SYMBYTES], indices: [(u8, u8); 1]) -> impl SqueezeWords<1> {
    let mut state = [0u64; PLEN];
    absorb_seed(&mut state, seed);
    state[4] =
        u64::from(indices[0].0) | (u64::from(indices[0].1) << 8) | (u64::from(SHAKE_PAD) << 16);
    state[20] = 0x80_u64 << 56;
    ScalarShake128Reader(state)
}

pub struct XofAbsorb<const L: usize>;

pub trait XofAbsorbLanes<const L: usize>: sealed::Sealed {
    fn xof_absorb(seed: &[u8; SYMBYTES], indices: [(u8, u8); L]) -> impl SqueezeWords<L>;
}

impl sealed::Sealed for XofAbsorb<1> {}
impl XofAbsorbLanes<1> for XofAbsorb<1> {
    fn xof_absorb(seed: &[u8; SYMBYTES], indices: [(u8, u8); 1]) -> impl SqueezeWords<1> {
        xof_absorb_scalar(seed, indices)
    }
}

macro_rules! impl_xof_absorb_vec {
    ($L:expr) => {
        impl sealed::Sealed for XofAbsorb<$L> {}
        impl XofAbsorbLanes<$L> for XofAbsorb<$L> {
            fn xof_absorb(seed: &[u8; SYMBYTES], indices: [(u8, u8); $L]) -> impl SqueezeWords<$L> {
                xof_absorb_vec(seed, indices)
            }
        }
    };
}

impl_xof_absorb_vec!(2);
impl_xof_absorb_vec!(4);
impl_xof_absorb_vec!(8);
impl_xof_absorb_vec!(16);
