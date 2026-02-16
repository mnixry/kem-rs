//! ML-KEM benchmarks (keygen / encaps / decaps across parameter sets).

use criterion::{criterion_group, criterion_main};

fn placeholder(_c: &mut criterion::Criterion) {
    // Benchmarks will be added once the KEM implementation is complete.
}

criterion_group!(benches, placeholder);
criterion_main!(benches);
