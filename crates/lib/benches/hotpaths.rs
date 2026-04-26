//! Benchmarks for ML-KEM internal hot paths: matrix generation, noise
//! sampling, and inner-product operations.
//!
//! These isolate the mid-level kernels that dominate cycle share inside
//! keypair / encapsulate / decapsulate, making them better perf-stat
//! targets than the full end-to-end API benchmarks.

use core::hint::black_box;

use kem_rs::ParameterSet;
use kem_utils::criterion::{criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};

const HOTPATH_SEED: u64 = 0x4B_45_4D_5F_48_4F; // "KEM_HO" + pad

fn fill_seed(rng: &mut StdRng) -> [u8; 32] {
    let mut seed = [0u8; 32];
    rng.fill_bytes(&mut seed);
    seed
}

#[allow(clippy::many_single_char_names)]
fn bench_hotpaths<P: ParameterSet>(crit: &mut kem_utils::CriterionConfig, label: &str) {
    let mut group = crit.benchmark_group(format!("hotpath/{label}"));
    let mut rng = StdRng::seed_from_u64(HOTPATH_SEED);
    let seed = fill_seed(&mut rng);
    let noise_seed = fill_seed(&mut rng);

    group.bench_function("gen_matrix", |b| {
        b.iter(|| black_box(P::gen_matrix(black_box(&seed), false)));
    });

    group.bench_function("gen_matrix_transposed", |b| {
        b.iter(|| black_box(P::gen_matrix(black_box(&seed), true)));
    });

    group.bench_function("sample_noise_eta1_ntt", |b| {
        b.iter(|| {
            let mut nonce = 0u8;
            black_box(P::sample_noise_eta1(black_box(&seed), &mut nonce))
        });
    });

    group.bench_function("sample_noise_eta1_std", |b| {
        b.iter(|| {
            let mut nonce = 0u8;
            black_box(P::sample_noise_eta1_std(black_box(&seed), &mut nonce))
        });
    });

    group.bench_function("sample_noise_eta2", |b| {
        b.iter(|| {
            let mut nonce = 0u8;
            black_box(P::sample_noise_eta2(black_box(&seed), &mut nonce))
        });
    });

    group.bench_function("mat_mul_vec_tomont", |b| {
        let mat = P::gen_matrix(&seed, false);
        let mut nonce = 0u8;
        let vec = P::sample_noise_eta1(&noise_seed, &mut nonce);
        b.iter(|| black_box(P::mat_mul_vec_tomont(black_box(&mat), black_box(&vec))));
    });

    group.bench_function("inner_product", |b| {
        let mut nonce = 0u8;
        let lhs = P::sample_noise_eta1(&seed, &mut nonce);
        nonce = 0;
        let rhs = P::sample_noise_eta1(&seed, &mut nonce);
        b.iter(|| black_box(P::inner_product(black_box(&lhs), black_box(&rhs))));
    });

    group.finish();
}

fn hotpath_benches(c: &mut kem_utils::CriterionConfig) {
    bench_hotpaths::<kem_rs::MlKem512>(c, "MlKem512");
    bench_hotpaths::<kem_rs::MlKem768>(c, "MlKem768");
    bench_hotpaths::<kem_rs::MlKem1024>(c, "MlKem1024");
}

criterion_group! {
    name = benches;
    config = kem_utils::criterion_config();
    targets = hotpath_benches
}
criterion_main!(benches);
