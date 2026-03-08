# kem-rs

`kem-rs` is a Rust workspace for implementing and studying ML-KEM (FIPS 203), the standardized form of CRYSTALS-Kyber. The project combines correctness, benchmarking, and low-level optimization work for modern CPUs, with a particular interest in portable SIMD, polynomial arithmetic, and practical performance analysis.

This repository is the implementation and experimental codebase for a Bachelor's Degree Graduation Project. The academic topic began as a study of optimization through the Number Theoretic Transform (NTT), but the work evolved into a broader investigation of how the full ML-KEM pipeline behaves in practice.

## Purpose

The main purpose of this project is to build a correct and efficient ML-KEM implementation in Rust while using that implementation as a vehicle for research. In particular, the project aims to:

- implement ML-KEM-512, ML-KEM-768, and ML-KEM-1024 in safe Rust;
- study how lattice-based cryptographic kernels map onto modern CPU architectures;
- explore optimization opportunities in modular arithmetic, polynomial operations, memory layout, and portable SIMD;
- validate correctness with standard test vectors and cross-implementation checks;
- benchmark both full KEM operations and internal hot paths.

## Background

Post-quantum cryptography has become increasingly important because large-scale quantum computers would threaten classical public-key systems such as RSA and ECC. ML-KEM, standardized by NIST in FIPS 203, is one of the first major post-quantum key encapsulation mechanisms intended for real deployment.

Efficient implementations matter in practical settings such as TLS, VPNs, and embedded systems, where cryptographic latency, throughput, and energy cost all affect real-world usability.

## Research Focus

The original plan behind this project placed strong emphasis on the Number Theoretic Transform. That focus is natural: NTT is a core building block in lattice-based cryptography and is often treated as one of the most important optimization targets.

During development, however, profiling led to a more general conclusion. In this implementation, NTT is important, but it does not appear to consume as much of the total end-to-end KEM time as initially expected. Overall performance is also shaped by other parts of the pipeline, including:

- matrix generation and sampling;
- Keccak, SHAKE, and related hashing work;
- polynomial and polynomial-vector operations outside the raw NTT steps;
- encoding, decoding, and serialization;
- data layout, cache behavior, and SIMD utilization.

For that reason, this repository should not be read only as an "NTT optimization project". It is more accurately a broader practical study of ML-KEM implementation, benchmarking, and optimization trade-offs.

## Repository Layout

- `crates/lib` - high-level ML-KEM API and parameter-set-specific logic;
- `crates/math` - modular reduction, polynomial arithmetic, NTT, compression, sampling, and portable SIMD kernels;
- `crates/hash` - Keccak, SHA-3, and SHAKE primitives used by ML-KEM with portable SIMD optimizations;

## Current Characteristics

- safe Rust implementation with `#![deny(unsafe_code)]`;
- portable SIMD based experimentation;
- support for `ML-KEM-512`, `ML-KEM-768`, and `ML-KEM-1024`;
- correctness validation with NIST-style KATs and ACVP vectors;
- performance comparison against the [`ml-kem`](https://github.com/RustCrypto/KEMs/tree/master/ml-kem) crate;
- hot-path benchmarks for internal operations such as matrix generation, noise sampling, and inner products.

## Development Notes

This is a research and educational codebase rather than a production-audited cryptographic library. The goal is to understand the algorithm, validate correctness, and measure implementation trade-offs in a realistic setting.

The workspace currently targets nightly Rust because it relies on `portable_simd`.

## Basic Commands

```bash
cargo test
cargo bench # or cargo criterion
```

If you use Nix, you can also enter the development environment with:

```bash
nix develop
```
