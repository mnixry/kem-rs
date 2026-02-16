//! Portable SIMD acceleration kernels using `std::simd`.
//!
//! Provides lane-generic fast paths for modular reduction, NTT butterfly,
//! and polynomial arithmetic, with automatic scalar-tail handling.
//!
//! Lane width `L` is const-generic and selectable per target:
//!
//! | Preset | Lanes | Target class |
//! |--------|-------|-------------|
//! | `L=16` | 16    | Baseline / NEON |
//! | `L=32` | 32    | AVX2-class |
//! | `L=64` | 64    | AVX-512-class (optional) |

#[allow(unused_imports)]
use std::simd::prelude::*;
