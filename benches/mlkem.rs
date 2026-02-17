//! ML-KEM benchmarks across all parameter sets.

use core::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use kem_rs::{
    MlKem512, MlKem768, MlKem1024, MlKemParams, decapsulate, encapsulate_derand, keypair_derand,
};

fn fixed_keygen_coins(tag: u8) -> [u8; 64] {
    core::array::from_fn(|i| (i as u8).wrapping_add(tag.wrapping_mul(37)))
}

fn fixed_enc_coins(tag: u8) -> [u8; 32] {
    core::array::from_fn(|i| (i as u8).wrapping_add(tag.wrapping_mul(53)))
}

fn bench_param_set<P: MlKemParams>(c: &mut Criterion, label: &str, tag: u8) {
    let keygen_coins = fixed_keygen_coins(tag);
    let enc_coins = fixed_enc_coins(tag);
    let (pk, sk) = keypair_derand::<P>(&keygen_coins);
    let (ct, _) = encapsulate_derand::<P>(&pk, &enc_coins);

    c.bench_function(&format!("{label}/keypair_derand"), |b| {
        b.iter(|| {
            let out = keypair_derand::<P>(black_box(&keygen_coins));
            black_box(out);
        });
    });

    c.bench_function(&format!("{label}/encapsulate_derand"), |b| {
        b.iter(|| {
            let out = encapsulate_derand::<P>(black_box(&pk), black_box(&enc_coins));
            black_box(out);
        });
    });

    c.bench_function(&format!("{label}/decapsulate"), |b| {
        b.iter(|| {
            let out = decapsulate::<P>(black_box(&ct), black_box(&sk));
            black_box(out);
        });
    });
}

fn mlkem_benches(c: &mut Criterion) {
    bench_param_set::<MlKem512>(c, "mlkem512", 1);
    bench_param_set::<MlKem768>(c, "mlkem768", 2);
    bench_param_set::<MlKem1024>(c, "mlkem1024", 3);
}

criterion_group!(benches, mlkem_benches);
criterion_main!(benches);
