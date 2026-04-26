//! Benchmark the effect of SIMD lane width on ML-KEM operations.
//!
//! Runs keypair / encapsulate / decapsulate for each lane width (8, 16, 32, 64)
//! across all three parameter sets.

use core::hint::black_box;

use kem_math::LaneWidth;
use kem_utils::criterion::{BenchmarkId, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};

const LANE_WIDTHS: &[LaneWidth] = &[
    LaneWidth::W128Bit,
    LaneWidth::W256Bit,
    LaneWidth::W512Bit,
    LaneWidth::W1024Bit,
];

const LANE_BENCH_KEYGEN_SEED: u64 = 0x4B45_4D5F_4C4E_4B47; // "KEM_L" + "NKG"
const LANE_BENCH_ENC_SEED: u64 = 0x4B45_4D5F_4C4E_454E; // "KEM" + "LNEN"

fn rng_coins() -> ([u8; 64], [u8; 32]) {
    let mut krng = StdRng::seed_from_u64(LANE_BENCH_KEYGEN_SEED);
    let mut keygen = [0u8; 64];
    krng.fill_bytes(&mut keygen);
    let mut erng = StdRng::seed_from_u64(LANE_BENCH_ENC_SEED);
    let mut enc = [0u8; 32];
    erng.fill_bytes(&mut enc);
    (keygen, enc)
}

#[allow(clippy::significant_drop_tightening)]
fn bench_lane_widths_for<P: kem_rs::ParameterSet>(c: &mut kem_utils::CriterionConfig) {
    let mut g = c.benchmark_group(format!(
        "lanes/{}",
        std::any::type_name::<P>()
            .split("::")
            .last()
            .expect("type name not found")
    ));

    let (keygen_coins, enc_coins) = rng_coins();

    for &width in LANE_WIDTHS {
        g.bench_function(BenchmarkId::new("keypair", width as usize), |b| {
            b.iter_batched(
                || kem_math::set_lane_width(width),
                |()| black_box(kem_rs::keypair_derand::<P>(black_box(&keygen_coins))),
                kem_utils::criterion::BatchSize::SmallInput,
            );
        });

        g.bench_function(BenchmarkId::new("encapsulate", width as usize), |b| {
            kem_math::set_lane_width(width);
            let (pk, _sk) = kem_rs::keypair_derand::<P>(&keygen_coins);
            b.iter(|| {
                black_box(kem_rs::encapsulate_derand::<P>(
                    black_box(&pk),
                    black_box(&enc_coins),
                ))
            });
        });

        g.bench_function(BenchmarkId::new("decapsulate", width as usize), |b| {
            kem_math::set_lane_width(width);
            let (pk, sk) = kem_rs::keypair_derand::<P>(&keygen_coins);
            let (ct, _) = kem_rs::encapsulate_derand::<P>(&pk, &enc_coins);
            b.iter(|| {
                black_box(kem_rs::decapsulate::<P>(black_box(&ct), black_box(&sk)));
            });
        });

        g.bench_function(BenchmarkId::new("e2e", width as usize), |b| {
            kem_math::set_lane_width(width);
            b.iter(|| {
                let (pk, sk) = kem_rs::keypair_derand::<P>(black_box(&keygen_coins));
                let (ct, _) =
                    kem_rs::encapsulate_derand::<P>(black_box(&pk), black_box(&enc_coins));
                black_box(kem_rs::decapsulate::<P>(black_box(&ct), black_box(&sk)));
            });
        });
    }

    g.finish();
    kem_math::set_lane_width(LaneWidth::W256Bit);
}

fn lane_width_benches(c: &mut kem_utils::CriterionConfig) {
    bench_lane_widths_for::<kem_rs::MlKem512>(c);
    bench_lane_widths_for::<kem_rs::MlKem768>(c);
    bench_lane_widths_for::<kem_rs::MlKem1024>(c);
}

criterion_group! {
    name = benches;
    config = kem_utils::criterion_config();
    targets = lane_width_benches
}
criterion_main!(benches);
