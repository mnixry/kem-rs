//! Keccak / SHA-3 benchmarks for perf profiling.
//!
//! Covers scalar hashes (`hash_g`, `hash_h`, `rkprf`) and the parallel
//! SIMD paths (`xof_absorb`, `squeeze_words`, `prf_batch`) at various
//! lane widths.

use core::hint::black_box;

use kem_math::{Eta2, Eta3};
use kem_utils::criterion::{BenchmarkId, criterion_group, criterion_main};

fn bench_scalar_hash(c: &mut kem_utils::CriterionConfig) {
    let mut g = c.benchmark_group("hash");

    let input_32 = [0x42u8; 32];
    let input_64 = [0x42u8; 64];

    g.bench_function("hash_h", |b| {
        b.iter(|| black_box(kem_hash::scalar::hash_h(black_box(&input_32))));
    });

    g.bench_function("hash_g", |b| {
        b.iter(|| black_box(kem_hash::scalar::hash_g(black_box(&input_64))));
    });

    g.finish();
}

fn bench_rkprf(c: &mut kem_utils::CriterionConfig) {
    let mut g = c.benchmark_group("rkprf");
    let key = [0x42u8; 32];

    for &ct_len in &[768usize, 1088, 1568] {
        let ct = vec![0xABu8; ct_len];
        g.bench_function(BenchmarkId::from_parameter(ct_len), |b| {
            b.iter(|| {
                black_box(kem_hash::scalar::rkprf(
                    black_box(&key),
                    black_box(ct.as_slice()),
                ))
            });
        });
    }

    g.finish();
}

fn bench_xof(c: &mut kem_utils::CriterionConfig) {
    use kem_hash::xof::{SqueezeWords, XofAbsorbLanes};

    let mut g = c.benchmark_group("xof");

    let seed = [0x42u8; 32];

    g.bench_function("xof_absorb_4", |b| {
        let indices = [(0, 0), (0, 1), (1, 0), (1, 1)];
        b.iter(|| {
            black_box(kem_hash::xof::XofAbsorb::xof_absorb(
                black_box(&seed),
                black_box(indices),
            ))
        });
    });

    g.bench_function("squeeze_words_4", |b| {
        let indices = [(0, 0), (0, 1), (1, 0), (1, 1)];
        b.iter_batched(
            || kem_hash::xof::XofAbsorb::xof_absorb(&seed, indices),
            |mut reader| black_box(reader.squeeze_words()),
            kem_utils::criterion::BatchSize::SmallInput,
        );
    });

    g.bench_function("absorb_then_3_squeezes_4", |b| {
        let indices = [(0, 0), (0, 1), (1, 0), (1, 1)];
        b.iter(|| {
            let mut reader =
                kem_hash::xof::XofAbsorb::xof_absorb(black_box(&seed), black_box(indices));
            black_box(reader.squeeze_words());
            black_box(reader.squeeze_words());
            black_box(reader.squeeze_words());
        });
    });

    g.bench_function("xof_absorb_2", |b| {
        let indices = [(0, 0), (0, 1)];
        b.iter(|| {
            black_box(kem_hash::xof::XofAbsorb::xof_absorb(
                black_box(&seed),
                black_box(indices),
            ))
        });
    });

    g.finish();
}

fn bench_prf(c: &mut kem_utils::CriterionConfig) {
    let mut g = c.benchmark_group("prf");

    let seed = [0x42u8; 32];

    g.bench_function("prf_batch_4_eta2", |b| {
        let nonces = [0u8, 1, 2, 3];
        b.iter(|| {
            black_box(kem_hash::prf::prf_batch::<Eta2, 4>(
                black_box(&seed),
                black_box(nonces),
            ))
        });
    });

    g.bench_function("prf_batch_4_eta3", |b| {
        let nonces = [0u8, 1, 2, 3];
        b.iter(|| {
            black_box(kem_hash::prf::prf_batch::<Eta3, 4>(
                black_box(&seed),
                black_box(nonces),
            ))
        });
    });

    g.bench_function("prf_batch_2_eta2", |b| {
        let nonces = [0u8, 1];
        b.iter(|| {
            black_box(kem_hash::prf::prf_batch::<Eta2, 2>(
                black_box(&seed),
                black_box(nonces),
            ))
        });
    });

    g.bench_function("prf_batch_3_eta2", |b| {
        let nonces = [0u8, 1, 2];
        b.iter(|| {
            black_box(kem_hash::prf::prf_batch::<Eta2, 3>(
                black_box(&seed),
                black_box(nonces),
            ))
        });
    });

    g.finish();
}

fn keccak_benches(c: &mut kem_utils::CriterionConfig) {
    bench_scalar_hash(c);
    bench_rkprf(c);
    bench_xof(c);
    bench_prf(c);
}

criterion_group! {
    name = benches;
    config = kem_utils::criterion_config();
    targets = keccak_benches
}
criterion_main!(benches);
