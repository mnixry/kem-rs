//! `kem-math` — Core mathematical primitives for ML-KEM.
//!
//! `no_std`, zero-allocation polynomial arithmetic over `Z_q[X]/(X^{256}+1)`.
//! Sub-modules cover modular reduction, the Number-Theoretic Transform,
//! polynomial and polynomial-vector arithmetic, byte-level packing and
//! compression, deterministic sampling, and portable-SIMD kernels.

#![no_std]
#![feature(portable_simd)]
#![deny(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

pub mod ntt;
pub mod pack;
pub mod poly;
pub mod polyvec;
pub mod reduce;
pub mod sample;
pub mod simd;

/// Polynomial ring degree.
pub const N: usize = 256;

/// Field modulus.
pub const Q: i16 = 3329;

/// Size in bytes of hashes, seeds, and shared secrets.
pub const SYMBYTES: usize = 32;

/// Size in bytes of a serialised polynomial (12 bits * 256 / 8).
pub const POLYBYTES: usize = 384;
