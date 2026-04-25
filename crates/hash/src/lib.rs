//! SIMD-accelerated Keccak/SHA-3 primitives for ML-KEM.
//!
//! Every hash primitive — scalar fixed-output (H, G, J), parallel XOF, and
//! parallel PRF — shares a single lane-count-generic Keccak-f[1600]
//! permutation. The lane count is 1 for scalar, K for PRF, and K*K for
//! XOF matrix sampling.

#![no_std]
#![feature(portable_simd)]

mod keccak;
pub mod prf;
pub mod scalar;
pub mod xof;

pub const SHAKE128_RATE: usize = 168;
pub const SHAKE256_RATE: usize = 136;
pub const SHA3_256_RATE: usize = 136;
pub const SHA3_512_RATE: usize = 72;

const SHAKE_PAD: u8 = 0x1F;
const SHA3_PAD: u8 = 0x06;

use kem_math::{ByteArray, CbdWidth, SYMBYTES};

/// Single-lane SHAKE-256 PRF via scalar sponge.
#[must_use]
#[inline(always)]
pub fn prf<Eta: CbdWidth>(seed: &[u8; SYMBYTES], nonce: u8) -> Eta::Buffer {
    let mut input = [0u8; SYMBYTES + 1];
    input[..SYMBYTES].copy_from_slice(seed);
    input[SYMBYTES] = nonce;
    let mut buf = Eta::Buffer::zeroed();
    scalar::shake256(input, buf.as_mut());
    buf
}
