//! Core mathematical primitives for ML-KEM.
//!
//! Sub-modules cover modular reduction, the Number-Theoretic Transform,
//! polynomial and polynomial-vector arithmetic, byte-level packing and
//! compression, and deterministic sampling.

pub mod ntt;
pub mod pack;
pub mod poly;
pub mod polyvec;
pub mod reduce;
pub mod sample;
