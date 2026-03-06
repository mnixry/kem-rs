//! Keccak / SHA-3 benchmarks for perf profiling.
//!
//! Covers scalar hashes (`hash_g`, `hash_h`, `rkprf`) and the 4-way parallel
//! SIMD paths (`xof_absorb_x4`, `squeeze_blocks`, `prf_x4`).

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kem_math::{Eta2, Eta3};

fn pin_core() {
    let id = core_affinity::get_core_ids()
        .and_then(|ids| ids.first().copied())
        .expect("no core ids");
    core_affinity::set_for_current(id);
}

fn bench_scalar_hash(c: &mut Criterion) {
    let mut g = c.benchmark_group("hash");

    let input_32 = [0x42u8; 32];
    let input_64 = [0x42u8; 64];

    g.bench_function("hash_h", |b| {
        b.iter(|| black_box(kem_hash::hash_h(black_box(&input_32))));
    });

    g.bench_function("hash_g", |b| {
        b.iter(|| black_box(kem_hash::hash_g(black_box(&input_64))));
    });

    g.finish();
}

fn bench_rkprf(c: &mut Criterion) {
    let mut g = c.benchmark_group("rkprf");
    let key = [0x42u8; 32];

    for &ct_len in &[768usize, 1088, 1568] {
        let ct = vec![0xABu8; ct_len];
        g.bench_function(BenchmarkId::from_parameter(ct_len), |b| {
            b.iter(|| black_box(kem_hash::rkprf(black_box(&key), black_box(ct.as_slice()))));
        });
    }

    g.finish();
}

fn bench_xof(c: &mut Criterion) {
    let mut g = c.benchmark_group("xof");

    let seed = [0x42u8; 32];
    let indices = [(0, 0), (0, 1), (1, 0), (1, 1)];

    g.bench_function("xof_absorb_x4", |b| {
        b.iter(|| {
            black_box(kem_hash::xof_absorb_x4(
                black_box(&seed),
                black_box(indices),
            ))
        });
    });

    g.bench_function("squeeze_blocks", |b| {
        b.iter_batched(
            || kem_hash::xof_absorb_x4(&seed, indices),
            |mut reader| black_box(reader.squeeze_blocks()),
            criterion::BatchSize::SmallInput,
        );
    });

    g.bench_function("absorb_then_3_squeezes", |b| {
        b.iter(|| {
            let mut reader = kem_hash::xof_absorb_x4(black_box(&seed), black_box(indices));
            black_box(reader.squeeze_blocks());
            black_box(reader.squeeze_blocks());
            black_box(reader.squeeze_blocks());
        });
    });

    g.finish();
}

fn bench_prf(c: &mut Criterion) {
    let mut g = c.benchmark_group("prf");

    let seed = [0x42u8; 32];
    let nonces = [0u8, 1, 2, 3];

    g.bench_function("prf_x4_eta2", |b| {
        b.iter(|| {
            black_box(kem_hash::prf_x4::<Eta2>(
                black_box(&seed),
                black_box(nonces),
            ))
        });
    });

    g.bench_function("prf_x4_eta3", |b| {
        b.iter(|| {
            black_box(kem_hash::prf_x4::<Eta3>(
                black_box(&seed),
                black_box(nonces),
            ))
        });
    });

    g.finish();
}

fn keccak_benches(c: &mut Criterion) {
    pin_core();
    bench_scalar_hash(c);
    bench_rkprf(c);
    bench_xof(c);
    bench_prf(c);
}

criterion_group!(benches, keccak_benches);
criterion_main!(benches);
