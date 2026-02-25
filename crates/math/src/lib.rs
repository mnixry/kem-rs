//! `kem-math` -- Core mathematical primitives for ML-KEM.
//!
//! `no_std`, zero-allocation polynomial arithmetic over `Z_q[X]/(X^{256}+1)`.
//! Sub-modules cover modular reduction, the Number-Theoretic Transform,
//! domain-separated polynomial and polynomial-vector types, sealed compression
//! traits, deterministic sampling, and portable-SIMD kernels.

#![no_std]
#![feature(portable_simd)]
#![deny(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

mod compress;
mod encode;
mod ntt;
mod poly;
mod reduce;
mod sample;
mod simd;

pub use compress::{CompressWidth, D4, D5, D10, D11};
pub use poly::{NttMatrix, NttPolynomial, NttVector, Polynomial, Vector};
pub use sample::{CbdWidth, CbdWidthParams, Eta2, Eta3, reject_uniform};
pub use simd::{LaneWidth, get_lane_width, set_lane_width};
use zeroize::Zeroize;

pub trait ByteArray:
    AsRef<[u8]> + AsMut<[u8]> + Clone + core::fmt::Debug + Zeroize + Send + Sync + 'static {
    const LEN: usize;
    fn zeroed() -> Self;
}

impl<const SIZE: usize> ByteArray for [u8; SIZE] {
    const LEN: usize = SIZE;

    #[inline]
    fn zeroed() -> Self {
        [0u8; SIZE]
    }
}

/// Polynomial ring degree.
pub const N: usize = 256;

/// Field modulus.
pub const Q: i16 = 3329;

/// Size in bytes of hashes, seeds, and shared secrets.
pub const SYMBYTES: usize = 32;

/// Size in bytes of a serialized polynomial (12 bits * 256 / 8).
pub const POLYBYTES: usize = 384;
