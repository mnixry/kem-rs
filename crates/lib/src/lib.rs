//! `kem-rs` — A safe, portable-SIMD ML-KEM implementation.
//!
//! Implements ML-KEM (FIPS 203) for all three parameter sets:
//! ML-KEM-512, ML-KEM-768, and ML-KEM-1024.
//!
//! # Design principles
//!
//! - **No `unsafe`** — enforced by `#![deny(unsafe_code)]`.
//! - **Nightly `portable_simd`** for vectorized arithmetic with scalar
//!   fallback.
//! - **RAII zeroization** of secret material via the `zeroize` crate.
//! - **Constant-time** operations for secret-dependent comparisons and moves.

#![deny(unsafe_code)]

pub mod hash;
pub mod kem;
pub mod params;
pub mod pke;
pub mod types;

// Re-export the public API surface.
pub use kem::{decapsulate, encapsulate, encapsulate_derand, keypair, keypair_derand};
pub use kem_math as math;
pub use params::{MlKem512, MlKem768, MlKem1024, MlKemParams};
pub use types::{Ciphertext, PublicKey, SecretKey, SharedSecret};
