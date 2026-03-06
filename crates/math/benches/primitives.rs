//! Low-level math primitive benchmarks for perf profiling.
//!
//! Covers NTT forward/inverse, basemul, reduce, and compress/decompress
//! at the single-polynomial level.

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kem_math::{D4, D5, D10, D11, NttPolynomial, Polynomial, Q};

#[allow(clippy::cast_possible_wrap)]
fn test_poly(seed: i16) -> Polynomial {
    Polynomial::from(core::array::from_fn(|i| {
        ((i as i16).wrapping_mul(seed)) % Q
    }))
}

fn pin_core() {
    let id = core_affinity::get_core_ids()
        .and_then(|ids| ids.first().copied())
        .expect("no core ids");
    core_affinity::set_for_current(id);
}

fn bench_ntt(c: &mut Criterion) {
    let mut g = c.benchmark_group("ntt");

    g.bench_function("forward", |b| {
        let p = test_poly(31);
        b.iter(|| black_box(black_box(p).ntt()));
    });

    g.bench_function("inverse", |b| {
        let p = test_poly(31).ntt();
        b.iter(|| black_box(black_box(p).ntt_inverse()));
    });

    g.bench_function("roundtrip", |b| {
        let p = test_poly(31);
        b.iter(|| black_box(black_box(p).ntt().ntt_inverse()));
    });

    g.finish();
}

fn bench_basemul(c: &mut Criterion) {
    let mut g = c.benchmark_group("basemul");

    let a = test_poly(31).ntt();
    let b = test_poly(13).ntt();

    g.bench_function("poly_basemul", |b_iter| {
        b_iter.iter(|| black_box(black_box(&a).basemul(black_box(&b))));
    });

    g.finish();
}

fn bench_reduce(c: &mut Criterion) {
    let mut g = c.benchmark_group("reduce");

    g.bench_function("poly_reduce", |b| {
        let mut p = test_poly(31);
        b.iter(|| {
            black_box(&mut p).reduce();
        });
    });

    g.bench_function("ntt_reduce", |b| {
        let mut p = test_poly(31).ntt();
        b.iter(|| {
            black_box(&mut p).reduce();
        });
    });

    g.finish();
}

fn bench_compress(c: &mut Criterion) {
    let mut g = c.benchmark_group("compress");
    let p = test_poly(31);

    macro_rules! bench_width {
        ($d:ty, $bytes:expr, $label:expr) => {{
            let mut buf = [0u8; $bytes];
            g.bench_function(BenchmarkId::new("compress", $label), |b| {
                b.iter(|| black_box(&p).compress::<$d>(&mut buf));
            });
            p.compress::<$d>(&mut buf);
            g.bench_function(BenchmarkId::new("decompress", $label), |b| {
                b.iter(|| black_box(Polynomial::decompress::<$d>(black_box(&buf))));
            });
        }};
    }

    bench_width!(D4, 128, "D4");
    bench_width!(D5, 160, "D5");
    bench_width!(D10, 320, "D10");
    bench_width!(D11, 352, "D11");

    g.finish();
}

fn bench_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("encode");

    let p = test_poly(31).ntt();
    let mut buf = [0u8; 384];
    p.to_bytes(&mut buf);

    g.bench_function("to_bytes", |b| {
        b.iter(|| {
            black_box(&p).to_bytes(&mut buf);
            black_box(&buf);
        });
    });

    g.bench_function("from_bytes", |b| {
        b.iter(|| black_box(NttPolynomial::from_bytes(black_box(&buf))));
    });

    g.finish();
}

fn primitives_benches(c: &mut Criterion) {
    pin_core();
    bench_ntt(c);
    bench_basemul(c);
    bench_reduce(c);
    bench_compress(c);
    bench_encode(c);
}

criterion_group!(benches, primitives_benches);
criterion_main!(benches);
