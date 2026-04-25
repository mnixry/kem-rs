//! Lane-count-generic Keccak-f[1600] permutation.
//!
//! All sponge constructions in this crate — scalar hashes, parallel XOF,
//! and parallel PRF — share this single permutation, instantiated at
//! different lane counts: 1 for scalar, K for PRF, K*K for XOF.

use core::simd::Simd;

use kem_math::unroll;

pub const PLEN: usize = 25;

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

/// Single Keccak-f[1600] round with fused theta+rho+pi+chi+iota.
///
/// Reads from `a` (25 lanes) and writes the result to `e` (25 lanes).
/// The round constant `rc` is XOR-ed into lane 0.
///
/// This follows the Ronny Van Keer / SUPERCOP implementation strategy
/// used by mlkem-native: fusing all five steps into one pass avoids
/// redundant loads/stores between steps and lets the compiler keep more
/// state in registers.
#[inline]
fn keccak_round<T: Arithmetics>(a: &[T; PLEN], e: &mut [T; PLEN], rc: u64) {
    // Theta: column parities
    let bc0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
    let bc1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
    let bc2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
    let bc3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
    let bc4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];

    let d0 = bc4 ^ bc1.rotate_left(1);
    let d1 = bc0 ^ bc2.rotate_left(1);
    let d2 = bc1 ^ bc3.rotate_left(1);
    let d3 = bc2 ^ bc4.rotate_left(1);
    let d4 = bc3 ^ bc0.rotate_left(1);

    // Theta + Rho + Pi (fused): apply d, rotate, place at pi-permuted position.
    // The lane selection encodes the pi permutation implicitly.
    // Row 0
    let t0 = a[0] ^ d0;
    let t1 = (a[6] ^ d1).rotate_left(44);
    let t2 = (a[12] ^ d2).rotate_left(43);
    let t3 = (a[18] ^ d3).rotate_left(21);
    let t4 = (a[24] ^ d4).rotate_left(14);
    // Chi + Iota (row 0)
    e[0] = t0 ^ (!t1 & t2) ^ T::load_u64(rc);
    e[1] = t1 ^ (!t2 & t3);
    e[2] = t2 ^ (!t3 & t4);
    e[3] = t3 ^ (!t4 & t0);
    e[4] = t4 ^ (!t0 & t1);

    // Row 1
    let t0 = (a[3] ^ d3).rotate_left(28);
    let t1 = (a[9] ^ d4).rotate_left(20);
    let t2 = (a[10] ^ d0).rotate_left(3);
    let t3 = (a[16] ^ d1).rotate_left(45);
    let t4 = (a[22] ^ d2).rotate_left(61);
    e[5] = t0 ^ (!t1 & t2);
    e[6] = t1 ^ (!t2 & t3);
    e[7] = t2 ^ (!t3 & t4);
    e[8] = t3 ^ (!t4 & t0);
    e[9] = t4 ^ (!t0 & t1);

    // Row 2
    let t0 = (a[1] ^ d1).rotate_left(1);
    let t1 = (a[7] ^ d2).rotate_left(6);
    let t2 = (a[13] ^ d3).rotate_left(25);
    let t3 = (a[19] ^ d4).rotate_left(8);
    let t4 = (a[20] ^ d0).rotate_left(18);
    e[10] = t0 ^ (!t1 & t2);
    e[11] = t1 ^ (!t2 & t3);
    e[12] = t2 ^ (!t3 & t4);
    e[13] = t3 ^ (!t4 & t0);
    e[14] = t4 ^ (!t0 & t1);

    // Row 3
    let t0 = (a[4] ^ d4).rotate_left(27);
    let t1 = (a[5] ^ d0).rotate_left(36);
    let t2 = (a[11] ^ d1).rotate_left(10);
    let t3 = (a[17] ^ d2).rotate_left(15);
    let t4 = (a[23] ^ d3).rotate_left(56);
    e[15] = t0 ^ (!t1 & t2);
    e[16] = t1 ^ (!t2 & t3);
    e[17] = t2 ^ (!t3 & t4);
    e[18] = t3 ^ (!t4 & t0);
    e[19] = t4 ^ (!t0 & t1);

    // Row 4
    let t0 = (a[2] ^ d2).rotate_left(62);
    let t1 = (a[8] ^ d3).rotate_left(55);
    let t2 = (a[14] ^ d4).rotate_left(39);
    let t3 = (a[15] ^ d0).rotate_left(41);
    let t4 = (a[21] ^ d1).rotate_left(2);
    e[20] = t0 ^ (!t1 & t2);
    e[21] = t1 ^ (!t2 & t3);
    e[22] = t2 ^ (!t3 & t4);
    e[23] = t3 ^ (!t4 & t0);
    e[24] = t4 ^ (!t0 & t1);
}

/// Keccak-f[1600] permutation over `L` parallel states.
///
/// Uses two-round unrolling: each loop iteration performs two rounds,
/// ping-ponging between the original state `a` and a temporary `e`.
/// This eliminates the state copy that a naive single-round loop needs
/// and lets the compiler keep both halves in registers.
///
/// Based on the Van Keer / SUPERCOP implementation strategy, also used
/// by mlkem-native's C Keccak.
pub fn f1600<T: Arithmetics>(state: &mut [T; PLEN]) {
    let mut tmp = [T::default(); PLEN];
    for &[r1, r2] in RC.as_chunks().0 {
        keccak_round(state, &mut tmp, r1);
        keccak_round(&tmp, state, r2);
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
