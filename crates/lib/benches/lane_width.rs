//! Benchmark the effect of SIMD lane width on ML-KEM operations.
//!
//! Runs keypair / encapsulate / decapsulate for each lane width (8, 16, 32, 64)
//! across all three parameter sets.

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kem_math::LaneWidth;
use pprof::criterion::{Output, PProfProfiler};

const LANE_WIDTHS: [LaneWidth; 4] = [
    LaneWidth::L8,
    LaneWidth::L16,
    LaneWidth::L32,
    LaneWidth::L64,
];

fn deterministic_coins() -> ([u8; 64], [u8; 32]) {
    let keygen: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(37));
    let enc: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(53));
    (keygen, enc)
}

#[allow(clippy::significant_drop_tightening)]
fn bench_lane_widths_for<P: kem_rs::ParameterSet>(c: &mut Criterion) {
    let mut g = c.benchmark_group(format!(
        "lanes/{}",
        std::any::type_name::<P>()
            .split("::")
            .last()
            .expect("type name not found")
    ));

    let (keygen_coins, enc_coins) = deterministic_coins();

    for &width in &LANE_WIDTHS {
        g.bench_function(BenchmarkId::new("keypair", width), |b| {
            b.iter_batched(
                || kem_math::set_lane_width(width),
                |()| black_box(kem_rs::keypair_derand::<P>(black_box(&keygen_coins))),
                criterion::BatchSize::SmallInput,
            );
        });

        g.bench_function(BenchmarkId::new("encapsulate", width), |b| {
            kem_math::set_lane_width(width);
            let (pk, _sk) = kem_rs::keypair_derand::<P>(&keygen_coins);
            b.iter(|| {
                black_box(kem_rs::encapsulate_derand::<P>(
                    black_box(&pk),
                    black_box(&enc_coins),
                ))
            });
        });

        g.bench_function(BenchmarkId::new("decapsulate", width), |b| {
            kem_math::set_lane_width(width);
            let (pk, sk) = kem_rs::keypair_derand::<P>(&keygen_coins);
            let (ct, _) = kem_rs::encapsulate_derand::<P>(&pk, &enc_coins);
            b.iter(|| {
                black_box(kem_rs::decapsulate::<P>(black_box(&ct), black_box(&sk)));
            });
        });

        g.bench_function(BenchmarkId::new("roundtrip", width), |b| {
            kem_math::set_lane_width(width);
            let (pk, sk) = kem_rs::keypair_derand::<P>(&keygen_coins);
            b.iter(|| {
                let (ct, _) =
                    kem_rs::encapsulate_derand::<P>(black_box(&pk), black_box(&enc_coins));
                black_box(kem_rs::decapsulate::<P>(black_box(&ct), black_box(&sk)));
            });
        });
    }

    g.finish();
    kem_math::set_lane_width(LaneWidth::L16);
}

fn lane_width_benches(c: &mut Criterion) {
    let core_id = core_affinity::get_core_ids()
        .and_then(|ids| ids.first().copied())
        .expect("no core ids found");
    core_affinity::set_for_current(core_id);
    println!("Running lane-width benchmarks on core {core_id:?}");

    bench_lane_widths_for::<kem_rs::MlKem512>(c);
    bench_lane_widths_for::<kem_rs::MlKem768>(c);
    bench_lane_widths_for::<kem_rs::MlKem1024>(c);
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(1000, Output::Flamegraph(None)));
    targets = lane_width_benches
}
criterion_main!(benches);
