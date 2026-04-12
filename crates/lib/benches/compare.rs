//! Side-by-side performance comparison: kem-rs vs `RustCrypto` ml-kem vs
//! mlkem-native (C/asm) vs `PQMagic` (C, SHAKE mode).

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ml_kem::{EncapsulateDeterministic, KemCore, kem::Decapsulate};
use pprof::criterion::{Output, PProfProfiler};

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

macro_rules! dispatch_binding_function {
    ($mod:ident, $ps:ty, $fn:ident, $($args:expr),*) => {
        match std::any::TypeId::of::<$ps>() {
            id if id == std::any::TypeId::of::<kem_rs::MlKem512>() => {
                let result = $mod::mlkem512::$fn($(black_box($args)),*);
                black_box(result).unwrap();
            }
            id if id == std::any::TypeId::of::<kem_rs::MlKem768>() => {
                let result = $mod::mlkem768::$fn($(black_box($args)),*);
                black_box(result).unwrap();
            }
            id if id == std::any::TypeId::of::<kem_rs::MlKem1024>() => {
                let result = $mod::mlkem1024::$fn($(black_box($args)),*);
                black_box(result).unwrap();
            }
            _ => unreachable!(),
        }
    };
}

#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]
fn bench_param_set<P: kem_rs::ParameterSet, RC: KemCore>(c: &mut Criterion, tag: u8) {
    let mut g = c.benchmark_group(format!(
        "compare/{}",
        std::any::type_name::<P>()
            .split("::")
            .last()
            .expect("type name not found")
    ));
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
    g.bench_function(BenchmarkId::new("keypair", "mlkem-native"), |b| {
        b.iter(|| dispatch_binding_function!(mlkem_native_rs, P, keypair_derand, &full));
    });
    g.bench_function(BenchmarkId::new("keypair", "pqmagic"), |b| {
        b.iter(|| dispatch_binding_function!(pqmagic_rs, P, keypair_derand, &full));
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
    g.bench_function(BenchmarkId::new("encapsulate", "mlkem-native"), |b| {
        b.iter(|| {
            dispatch_binding_function!(
                mlkem_native_rs,
                P,
                enc_derand,
                our_pk.as_ref().try_into().unwrap(),
                &m
            );
        });
    });
    g.bench_function(BenchmarkId::new("encapsulate", "pqmagic"), |b| {
        b.iter(|| {
            dispatch_binding_function!(
                pqmagic_rs,
                P,
                enc_derand,
                our_pk.as_ref().try_into().unwrap(),
                &m
            );
        });
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
    g.bench_function(BenchmarkId::new("decapsulate", "mlkem-native"), |b| {
        b.iter(|| {
            dispatch_binding_function!(
                mlkem_native_rs,
                P,
                dec,
                our_ct.as_ref().try_into().unwrap(),
                our_sk.as_ref().try_into().unwrap()
            );
        });
    });
    g.bench_function(BenchmarkId::new("decapsulate", "pqmagic"), |b| {
        b.iter(|| {
            dispatch_binding_function!(
                pqmagic_rs,
                P,
                dec,
                our_ct.as_ref().try_into().unwrap(),
                our_sk.as_ref().try_into().unwrap()
            );
        });
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

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(1000, Output::Flamegraph(None)));
    targets = compare_benches
}
criterion_main!(benches);
