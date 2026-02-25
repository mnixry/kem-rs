//! `kem-rs` -- A safe, portable-SIMD ML-KEM implementation.
//!
//! Implements ML-KEM (FIPS 203) for all three parameter sets:
//! ML-KEM-512, ML-KEM-768, and ML-KEM-1024.

#![deny(unsafe_code)]

pub mod kem;
pub mod params;
pub mod pke;
pub mod types;

pub use kem::{decapsulate, encapsulate, encapsulate_derand, keypair, keypair_derand};
pub use kem_math as math;
pub use params::{MlKem512, MlKem768, MlKem1024, ParameterSet};
pub use types::{Ciphertext, PublicKey, SecretKey, SharedSecret};

/// Errors returned when constructing keys or ciphertexts from byte slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Input byte slice has wrong length for this parameter set.
    InvalidLength {
        /// Expected byte count.
        expected: usize,
        /// Actual byte count received.
        actual: usize,
    },
    /// Key bytes failed validation (e.g. FIPS 203 §7.2 modulus check).
    InvalidKey,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidLength { expected, actual } => {
                write!(f, "invalid length: expected {expected}, got {actual}")
            }
            Self::InvalidKey => f.write_str("invalid key"),
        }
    }
}

impl core::error::Error for Error {}
