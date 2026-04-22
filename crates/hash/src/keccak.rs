//! Lane-count-generic Keccak-f[1600] permutation.
//!
//! All sponge constructions in this crate — scalar hashes, parallel XOF,
//! and parallel PRF — share this single permutation, instantiated at
//! different lane counts: 1 for scalar, K for PRF, K*K for XOF.

use core::simd::Simd;

use kem_math::unroll;

pub const PLEN: usize = 25;

const RHO: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];

const PI: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

const RC: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808A,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808B,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008A,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000A,
    0x0000_0000_8000_808B,
    0x8000_0000_0000_008B,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800A,
    0x8000_0000_8000_000A,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

pub trait Arithmetics:
    Copy
    + Default
    + core::ops::BitXorAssign
    + core::ops::BitXor<Output = Self>
    + core::ops::Not<Output = Self>
    + core::ops::BitAnd<Output = Self> {
    fn rotate_left(self, n: u32) -> Self;
    fn load_u64(value: u64) -> Self;
}

impl Arithmetics for u64 {
    #[inline]
    fn rotate_left(self, n: u32) -> Self {
        self.rotate_left(n)
    }

    #[inline]
    fn load_u64(value: u64) -> Self {
        value
    }
}

impl<const L: usize> Arithmetics for Simd<u64, L> {
    #[inline]
    fn rotate_left(self, n: u32) -> Self {
        (self << Self::splat(u64::from(n))) | (self >> Self::splat(u64::from(64 - n)))
    }

    #[inline]
    fn load_u64(value: u64) -> Self {
        Self::splat(value)
    }
}

/// Keccak-f[1600] permutation over `L` parallel states.
#[allow(unused_assignments)]
pub fn f1600<T: Arithmetics>(state: &mut [T; PLEN]) {
    for &rc in &RC {
        let mut array = [T::default(); 5];

        // Theta
        unroll!(x in (..5), {
            unroll!(y in (..5), {
                array[x] ^= state[5 * y + x];
            });
        });
        unroll!(x in (..5), {
            let t1 = array[(x + 4) % 5];
            let t2 = array[(x + 1) % 5].rotate_left(1);
            unroll!(y in (..5), {
                state[5 * y + x] ^= t1 ^ t2;
            });
        });

        // Rho and Pi
        let mut last = state[1];
        unroll!(i in (..24), {
            array[0] = state[PI[i]];
            state[PI[i]] = last.rotate_left(RHO[i]);
            last = array[0];
        });

        // Chi
        unroll!(y_step in (..5), {
            let y = 5 * y_step;
            array.copy_from_slice(&state[y..y + 5]);
            unroll!(x in (..5), {
                state[y + x] = array[x] ^ (!array[(x + 1) % 5] & array[(x + 2) % 5]);
            });
        });

        // Iota
        state[0] ^= T::load_u64(rc);
    }
}

/// Splat a 32-byte seed into the first 4 words of all lanes.
#[inline]
pub fn absorb_seed<T: Arithmetics>(state: &mut [T; PLEN], seed: &[u8; 32]) {
    let (chunks, _) = seed.as_chunks();
    unroll!(i in (..4), {
        state[i] = T::load_u64(u64::from_le_bytes(chunks[i]));
    });
}
