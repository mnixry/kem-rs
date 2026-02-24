//! SIMD-accelerated Keccak/SHA-3 primitives for ML-KEM.
//!
//! All XOF and PRF sampling uses the 4-way parallel path
//! (`keccak::simd::f1600x4`). Scalar sponge is only used for fixed-output
//! hashes (`hash_h`, `hash_g`, `rkprf`) that have variable-length inputs and no
//! natural batching.

#![no_std]
#![feature(portable_simd)]

mod keccak1x;
mod keccak4x;

pub const SHAKE128_RATE: usize = 168;
pub const SHAKE256_RATE: usize = 136;
pub const SHA3_256_RATE: usize = 136;
pub const SHA3_512_RATE: usize = 72;

const SHAKE_PAD: u8 = 0x1F;
const SHA3_PAD: u8 = 0x06;

pub use keccak1x::{hash_g, hash_h, rkprf, shake128, shake256};
pub use keccak4x::{Shake128x4Reader, prf_x4, xof_absorb_x4};
use kem_math::{CbdWidth, SYMBYTES};

/// Single-lane SHAKE-256 PRF via `prf_x4` with 3 dummy lanes.
#[must_use]
pub fn prf<Eta: CbdWidth>(seed: &[u8; SYMBYTES], nonce: u8) -> Eta::Buffer {
    let [result, ..] = prf_x4::<Eta>(seed, [nonce, 0, 0, 0]);
    result
}
