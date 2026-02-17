//! Deterministic vector tests for ML-KEM correctness.
//!
//! Covers:
//! - KEM roundtrip (keygen → encaps → decaps ⇒ identical shared secret)
//! - Deterministic reproducibility (_derand variants)
//! - Implicit rejection (tampered ciphertext → different shared secret)
//! - Decapsulation with wrong secret key
//! - Key size consistency with parameter definitions

use kem_rs::{
    MlKem512, MlKem768, MlKem1024, MlKemParams,
    keypair_derand, encapsulate_derand, decapsulate,
    keypair, encapsulate,
    Ciphertext,
};
use rand_core::UnwrapErr;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Fixed 64-byte coins for deterministic keygen tests.
fn fixed_keygen_coins(variant: u8) -> [u8; 64] {
    core::array::from_fn(|i| (i as u8).wrapping_add(variant.wrapping_mul(37)))
}

/// Fixed 32-byte coins for deterministic encapsulation tests.
fn fixed_enc_coins(variant: u8) -> [u8; 32] {
    core::array::from_fn(|i| (i as u8).wrapping_add(variant.wrapping_mul(53)))
}

// ---------------------------------------------------------------------------
// KEM roundtrip — all parameter sets
// ---------------------------------------------------------------------------

fn kem_roundtrip_derand<P: MlKemParams>() {
    let kp_coins = fixed_keygen_coins(0);
    let enc_coins = fixed_enc_coins(0);

    let (pk, sk) = keypair_derand::<P>(&kp_coins);
    let (ct, ss_enc) = encapsulate_derand::<P>(&pk, &enc_coins);
    let ss_dec = decapsulate::<P>(&ct, &sk);

    assert_eq!(
        ss_enc.as_bytes(),
        ss_dec.as_bytes(),
        "Roundtrip: shared secrets must match"
    );
}

#[test]
fn roundtrip_mlkem512() {
    kem_roundtrip_derand::<MlKem512>();
}

#[test]
fn roundtrip_mlkem768() {
    kem_roundtrip_derand::<MlKem768>();
}

#[test]
fn roundtrip_mlkem1024() {
    kem_roundtrip_derand::<MlKem1024>();
}

// ---------------------------------------------------------------------------
// Deterministic reproducibility
// ---------------------------------------------------------------------------

fn determinism_check<P: MlKemParams>() {
    let kp_coins = fixed_keygen_coins(1);
    let enc_coins = fixed_enc_coins(1);

    let (pk1, sk1) = keypair_derand::<P>(&kp_coins);
    let (pk2, sk2) = keypair_derand::<P>(&kp_coins);

    assert_eq!(pk1.as_bytes(), pk2.as_bytes(), "Deterministic keypair: pk mismatch");
    assert_eq!(sk1.as_bytes(), sk2.as_bytes(), "Deterministic keypair: sk mismatch");

    let (ct1, ss1) = encapsulate_derand::<P>(&pk1, &enc_coins);
    let (ct2, ss2) = encapsulate_derand::<P>(&pk2, &enc_coins);

    assert_eq!(ct1.as_bytes(), ct2.as_bytes(), "Deterministic encaps: ct mismatch");
    assert_eq!(ss1.as_bytes(), ss2.as_bytes(), "Deterministic encaps: ss mismatch");

    let ss_dec1 = decapsulate::<P>(&ct1, &sk1);
    let ss_dec2 = decapsulate::<P>(&ct2, &sk2);

    assert_eq!(ss_dec1.as_bytes(), ss_dec2.as_bytes(), "Deterministic decaps: ss mismatch");
    assert_eq!(ss1.as_bytes(), ss_dec1.as_bytes(), "Deterministic: enc/dec ss must match");
}

#[test]
fn determinism_mlkem512() {
    determinism_check::<MlKem512>();
}

#[test]
fn determinism_mlkem768() {
    determinism_check::<MlKem768>();
}

#[test]
fn determinism_mlkem1024() {
    determinism_check::<MlKem1024>();
}

// ---------------------------------------------------------------------------
// Implicit rejection — tampered ciphertext yields different shared secret
// ---------------------------------------------------------------------------

fn implicit_rejection_check<P: MlKemParams>() {
    let kp_coins = fixed_keygen_coins(2);
    let enc_coins = fixed_enc_coins(2);

    let (pk, sk) = keypair_derand::<P>(&kp_coins);
    let (ct, ss_good) = encapsulate_derand::<P>(&pk, &enc_coins);

    // Tamper with the ciphertext (flip one byte)
    let mut bad_ct_bytes = ct.into_bytes();
    bad_ct_bytes.as_mut()[0] ^= 0xFF;
    let bad_ct = Ciphertext::<P>::from_bytes(bad_ct_bytes);

    let ss_bad = decapsulate::<P>(&bad_ct, &sk);

    // The shared secret must differ (implicit rejection)
    assert_ne!(
        ss_good.as_bytes(),
        ss_bad.as_bytes(),
        "Implicit rejection: tampered ct must produce different ss"
    );

    // The rejection ss must be deterministic (same ct + sk → same rejection ss)
    let ss_bad2 = decapsulate::<P>(&bad_ct, &sk);
    assert_eq!(
        ss_bad.as_bytes(),
        ss_bad2.as_bytes(),
        "Implicit rejection: rejection ss must be deterministic"
    );
}

#[test]
fn implicit_rejection_mlkem512() {
    implicit_rejection_check::<MlKem512>();
}

#[test]
fn implicit_rejection_mlkem768() {
    implicit_rejection_check::<MlKem768>();
}

#[test]
fn implicit_rejection_mlkem1024() {
    implicit_rejection_check::<MlKem1024>();
}

// ---------------------------------------------------------------------------
// Wrong secret key — decapsulation with unrelated sk
// ---------------------------------------------------------------------------

fn wrong_sk_check<P: MlKemParams>() {
    let (pk, _sk) = keypair_derand::<P>(&fixed_keygen_coins(3));
    let (_pk2, wrong_sk) = keypair_derand::<P>(&fixed_keygen_coins(4));

    let (ct, ss_enc) = encapsulate_derand::<P>(&pk, &fixed_enc_coins(3));

    let ss_wrong = decapsulate::<P>(&ct, &wrong_sk);

    assert_ne!(
        ss_enc.as_bytes(),
        ss_wrong.as_bytes(),
        "Wrong SK: must produce different shared secret"
    );
}

#[test]
fn wrong_sk_mlkem512() {
    wrong_sk_check::<MlKem512>();
}

#[test]
fn wrong_sk_mlkem768() {
    wrong_sk_check::<MlKem768>();
}

#[test]
fn wrong_sk_mlkem1024() {
    wrong_sk_check::<MlKem1024>();
}

// ---------------------------------------------------------------------------
// Key/ciphertext size consistency
// ---------------------------------------------------------------------------

fn size_check<P: MlKemParams>() {
    let (pk, sk) = keypair_derand::<P>(&fixed_keygen_coins(5));
    let (ct, _ss) = encapsulate_derand::<P>(&pk, &fixed_enc_coins(5));

    assert_eq!(pk.as_bytes().len(), P::PK_BYTES, "PK size mismatch");
    assert_eq!(sk.as_bytes().len(), P::SK_BYTES, "SK size mismatch");
    assert_eq!(ct.as_bytes().len(), P::CT_BYTES, "CT size mismatch");
}

#[test]
fn sizes_mlkem512() {
    size_check::<MlKem512>();
}

#[test]
fn sizes_mlkem768() {
    size_check::<MlKem768>();
}

#[test]
fn sizes_mlkem1024() {
    size_check::<MlKem1024>();
}

// ---------------------------------------------------------------------------
// Randomized roundtrip (exercises the `keypair`/`encapsulate` API paths)
// ---------------------------------------------------------------------------

fn randomized_roundtrip<P: MlKemParams>() {
    let mut rng = UnwrapErr(getrandom::SysRng);

    let (pk, sk) = keypair::<P>(&mut rng);
    let (ct, ss_enc) = encapsulate::<P>(&pk, &mut rng);
    let ss_dec = decapsulate::<P>(&ct, &sk);

    assert_eq!(
        ss_enc.as_bytes(),
        ss_dec.as_bytes(),
        "Randomized roundtrip: shared secrets must match"
    );
}

#[test]
fn randomized_roundtrip_mlkem512() {
    randomized_roundtrip::<MlKem512>();
}

#[test]
fn randomized_roundtrip_mlkem768() {
    randomized_roundtrip::<MlKem768>();
}

#[test]
fn randomized_roundtrip_mlkem1024() {
    randomized_roundtrip::<MlKem1024>();
}

// ---------------------------------------------------------------------------
// Multiple encapsulations with same pk yield different shared secrets
// ---------------------------------------------------------------------------

fn different_encapsulations<P: MlKemParams>() {
    let (pk, sk) = keypair_derand::<P>(&fixed_keygen_coins(6));

    let (ct1, ss1) = encapsulate_derand::<P>(&pk, &fixed_enc_coins(10));
    let (ct2, ss2) = encapsulate_derand::<P>(&pk, &fixed_enc_coins(11));

    assert_ne!(
        ct1.as_bytes(),
        ct2.as_bytes(),
        "Different coins must produce different ciphertexts"
    );
    assert_ne!(
        ss1.as_bytes(),
        ss2.as_bytes(),
        "Different coins must produce different shared secrets"
    );

    // But both must decapsulate correctly
    let ss1_dec = decapsulate::<P>(&ct1, &sk);
    let ss2_dec = decapsulate::<P>(&ct2, &sk);

    assert_eq!(ss1.as_bytes(), ss1_dec.as_bytes());
    assert_eq!(ss2.as_bytes(), ss2_dec.as_bytes());
}

#[test]
fn different_encapsulations_mlkem512() {
    different_encapsulations::<MlKem512>();
}

#[test]
fn different_encapsulations_mlkem768() {
    different_encapsulations::<MlKem768>();
}

#[test]
fn different_encapsulations_mlkem1024() {
    different_encapsulations::<MlKem1024>();
}
