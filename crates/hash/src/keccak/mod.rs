//! Lane-count-generic Keccak-f[1600] permutation.
//!
//! All sponge constructions in this crate — scalar hashes, parallel XOF,
//! and parallel PRF — share this single permutation, instantiated at
//! different lane counts: 1 for scalar, K for PRF, K*K for XOF.

pub mod prf;
pub mod scalar;
pub mod xof;

use core::simd::Simd;

const PLEN: usize = 25;

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

macro_rules! unroll {
    (5, $var:ident, $body:expr) => {
        kem_math::unroll!($var, [0, 1, 2, 3, 4], $body);
    };
    (24, $var:ident, $body:expr) => {
        kem_math::unroll!(
            $var,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23
            ],
            $body
        );
    };
}

macro_rules! rotl {
    ($x:expr, $n:expr) => {
        ($x << Simd::splat($n as u64)) | ($x >> Simd::splat((64 - $n) as u64))
    };
}

/// Keccak-f[1600] permutation over `L` parallel states.
#[allow(unused_assignments, clippy::cast_lossless)]
pub fn f1600<const L: usize>(state: &mut [Simd<u64, L>; PLEN]) {
    for &rc in &RC {
        let mut array = [Simd::splat(0u64); 5];

        // Theta
        unroll!(5, x, {
            unroll!(5, y, {
                array[x] ^= state[5 * y + x];
            });
        });
        unroll!(5, x, {
            let t1 = array[(x + 4) % 5];
            let t2 = rotl!(array[(x + 1) % 5], 1);
            unroll!(5, y, {
                state[5 * y + x] ^= t1 ^ t2;
            });
        });

        // Rho and Pi
        let mut last = state[1];
        unroll!(24, i, {
            array[0] = state[PI[i]];
            state[PI[i]] = rotl!(last, RHO[i]);
            last = array[0];
        });

        // Chi
        unroll!(5, y_step, {
            let y = 5 * y_step;
            array.copy_from_slice(&state[y..y + 5]);
            unroll!(5, x, {
                state[y + x] = array[x] ^ (!array[(x + 1) % 5] & array[(x + 2) % 5]);
            });
        });

        // Iota
        state[0] ^= Simd::splat(rc);
    }
}

/// Splat a 32-byte seed into the first 4 words of all lanes.
pub fn absorb_seed<const L: usize>(state: &mut [Simd<u64, L>; PLEN], seed: &[u8; 32]) {
    for (s, &c) in state.iter_mut().zip(seed.as_chunks::<8>().0) {
        *s = Simd::splat(u64::from_le_bytes(c));
    }
}
