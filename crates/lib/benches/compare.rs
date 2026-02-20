//! Side-by-side performance comparison: kem-rs vs `RustCrypto` ml-kem.

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ml_kem::{EncapsulateDeterministic, KemCore, kem::Decapsulate};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn keygen_coins(tag: u8) -> ([u8; 32], [u8; 32], [u8; 64]) {
    let full: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_add(tag.wrapping_mul(37)));
    (
        full[..32].try_into().unwrap(),
        full[32..].try_into().unwrap(),
        full,
    )
}

fn enc_coins(tag: u8) -> [u8; 32] {
    core::array::from_fn(|i| (i as u8).wrapping_add(tag.wrapping_mul(53)))
}

#[allow(clippy::many_single_char_names, clippy::similar_names)]
fn bench_param_set<P: kem_rs::ParameterSet, RC: KemCore>(c: &mut Criterion, tag: u8) {
    let mut g = c.benchmark_group(std::any::type_name::<P>());
    let (d, z, full) = keygen_coins(tag);
    let m = enc_coins(tag);

    let (our_pk, our_sk) = kem_rs::keypair_derand::<P>(&full);
    let (our_ct, _) = kem_rs::encapsulate_derand::<P>(&our_pk, &m);

    let d_b: ml_kem::B32 = d.into();
    let z_b: ml_kem::B32 = z.into();
    let m_b: ml_kem::B32 = m.into();
    let (rc_dk, rc_ek) = RC::generate_deterministic(&d_b, &z_b);
    let (rc_ct, _) = rc_ek.encapsulate_deterministic(&m_b).unwrap();

    g.bench_function(BenchmarkId::new("keypair", "kem-rs"), |b| {
        b.iter(|| black_box(kem_rs::keypair_derand::<P>(black_box(&full))));
    });
    g.bench_function(BenchmarkId::new("keypair", "rustcrypto"), |b| {
        b.iter(|| black_box(RC::generate_deterministic(black_box(&d_b), black_box(&z_b))));
    });

    g.bench_function(BenchmarkId::new("encapsulate", "kem-rs"), |b| {
        b.iter(|| {
            black_box(kem_rs::encapsulate_derand::<P>(
                black_box(&our_pk),
                black_box(&m),
            ))
        });
    });
    g.bench_function(BenchmarkId::new("encapsulate", "rustcrypto"), |b| {
        b.iter(|| black_box(rc_ek.encapsulate_deterministic(black_box(&m_b)).unwrap()));
    });

    g.bench_function(BenchmarkId::new("decapsulate", "kem-rs"), |b| {
        b.iter(|| {
            black_box(kem_rs::decapsulate::<P>(
                black_box(&our_ct),
                black_box(&our_sk),
            ));
        });
    });
    g.bench_function(BenchmarkId::new("decapsulate", "rustcrypto"), |b| {
        b.iter(|| black_box(rc_dk.decapsulate(black_box(&rc_ct)).unwrap()));
    });

    g.bench_function(BenchmarkId::new("roundtrip", "kem-rs"), |b| {
        b.iter(|| {
            let (ct, _ss_enc) = kem_rs::encapsulate_derand::<P>(black_box(&our_pk), black_box(&m));
            black_box(kem_rs::decapsulate::<P>(black_box(&ct), black_box(&our_sk)));
        });
    });
    g.bench_function(BenchmarkId::new("roundtrip", "rustcrypto"), |b| {
        b.iter(|| {
            let (ct, _ss_enc) = rc_ek.encapsulate_deterministic(black_box(&m_b)).unwrap();
            black_box(rc_dk.decapsulate(black_box(&ct)).unwrap());
        });
    });

    g.finish();
}

fn compare_benches(c: &mut Criterion) {
    let core_id = core_affinity::get_core_ids()
        .and_then(|ids| ids.first().copied())
        .expect("no core ids found");
    core_affinity::set_for_current(core_id);
    println!("Running benchmarks on core {core_id:?}");
    bench_param_set::<kem_rs::MlKem512, ml_kem::MlKem512>(c, 1);
    bench_param_set::<kem_rs::MlKem768, ml_kem::MlKem768>(c, 2);
    bench_param_set::<kem_rs::MlKem1024, ml_kem::MlKem1024>(c, 3);
}

criterion_group!(benches, compare_benches);
criterion_main!(benches);
