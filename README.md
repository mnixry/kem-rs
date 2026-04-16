# kem-rs

A Rust workspace for implementing and studying ML-KEM (FIPS 203), the standardized form of CRYSTALS-Kyber. Combines correctness testing, benchmarking, and low-level optimization work targeting modern CPUs, with a focus on portable SIMD, polynomial arithmetic, and practical performance analysis.

This is the implementation and experimental codebase for a Bachelor's Degree Graduation Project. The academic topic started as a study of NTT optimization, but the work grew into a broader investigation of how the full ML-KEM pipeline behaves in practice.

## Purpose

Build a correct, efficient ML-KEM in Rust and use that implementation as a vehicle for research:

- ML-KEM-512, ML-KEM-768, and ML-KEM-1024 in safe, `no_std` Rust;
- study how lattice-based cryptographic kernels map onto modern CPU architectures;
- explore optimization in modular arithmetic, polynomial operations, memory layout, and portable SIMD;
- validate correctness against multiple test-vector suites and reference implementations;
- benchmark full KEM operations, internal hot paths, and SIMD lane-width effects.

## Research Focus

The original plan placed strong emphasis on the Number Theoretic Transform. NTT is a core building block in lattice-based cryptography and often treated as the primary optimization target.

Profiling told a different story. NTT matters, but it doesn't dominate end-to-end KEM time the way you might expect. Performance is shaped just as much by:

- matrix generation and sampling;
- Keccak / SHAKE hashing;
- polynomial and polynomial-vector operations outside the raw NTT steps;
- encoding, decoding, and serialization;
- data layout, cache behavior, and SIMD utilization.

So this isn't just an "NTT optimization project." It's a broader practical study of ML-KEM implementation trade-offs.

## Repository Layout

- `crates/lib` — ML-KEM API: key generation, encapsulation, decapsulation, and parameter-set definitions;
- `crates/math` — modular reduction, polynomial arithmetic, NTT, compression, sampling, and portable SIMD kernels;
- `crates/hash` — Keccak, SHA-3, SHAKE, and XOF primitives with portable SIMD optimizations;
- `crates/utils` — shared benchmarking utilities (Criterion config, profiling helpers);
- `bindings/mlkem-native-rs` — Rust FFI bindings to [mlkem-native](https://github.com/pq-code-package/mlkem-native) (C/asm reference);
- `bindings/pqmagic-rs` — Rust FFI bindings to [PQMagic](https://github.com/pqcrypto-cn/PQMagic) (C, SHAKE mode);

The bindings exist for cross-implementation correctness checks and head-to-head benchmarks; they aren't part of the library itself.

## Characteristics

- `#![no_std]`, safe Rust (`#![deny(unsafe_code)]`);
- portable SIMD via nightly `std::simd`;
- constant-time comparisons and conditional assignment (`ctutils`), secret zeroing (`zeroize`);
- all three ML-KEM parameter sets (512, 768, 1024);
- correctness validation against NIST KATs, ACVP vectors, Wycheproof test cases, and parse-validation / negative tests;
- byte-for-byte cross-checks against [RustCrypto `ml-kem`](https://github.com/RustCrypto/KEMs/tree/master/ml-kem), mlkem-native, and PQMagic;
- benchmarks: four-way performance comparison, SIMD lane-width sweep, internal hot-path isolation (matrix generation, noise sampling, inner products), Keccak primitives, and math-crate primitives.

## Development Notes

This is a research and educational codebase, not a production-audited cryptographic library. The goal is to understand the algorithm, validate correctness, and measure implementation trade-offs in a realistic setting.

Requires nightly Rust (edition 2024, `portable_simd` feature gate).

## Commands

```bash
cargo test              # run all correctness tests
cargo bench             # run all benchmarks (or cargo criterion)
```

If you use Nix:

```bash
nix develop             # dev shell with nightly Rust, cargo tools, perf, gnuplot, etc.
nix flake check         # clippy + nextest
nix build .#coverage    # llvm-cov coverage report
nix build .#benchmark   # criterion benchmarks with PGO
```
