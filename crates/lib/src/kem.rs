//! ML-KEM (IND-CCA2 Key Encapsulation Mechanism) -- FIPS 203.
//!
//! All public functions are panic-free: no `.unwrap()`, no `unreachable!()`.
//! Output sizes are derived from [`ParameterSet`] associated types.

use ctutils::{CtAssign, CtEq};
use kem_math::ByteArray;
use rand_core::CryptoRng;

use crate::{
    params::{ParameterSet, SSBYTES, SYMBYTES},
    pke,
    types::{Ciphertext, PublicKey, SecretKey, SharedSecret},
};

/// Deterministic key generation from a 64-byte seed `(d || z)`.
///
/// # Panics
///
/// Cannot panic: all internal array splits operate on compile-time-known sizes.
#[must_use]
pub fn keypair_derand<P: ParameterSet>(coins: &[u8; 2 * SYMBYTES]) -> (PublicKey<P>, SecretKey<P>) {
    let (d, z) = coins.split_at(SYMBYTES);
    let d: &[u8; SYMBYTES] = d.first_chunk().expect("infallible: 64-byte array split");
    let z: &[u8; SYMBYTES] = z.first_chunk().expect("infallible: 64-byte array split");

    let (pk_arr, indcpa_sk) = pke::indcpa_keypair_derand::<P>(d);

    let mut sk = P::SkArray::zeroed();
    let sk_mut = sk.as_mut();

    sk_mut[..P::INDCPA_SK_BYTES].copy_from_slice(&indcpa_sk.as_ref()[..P::INDCPA_SK_BYTES]);
    sk_mut[P::INDCPA_SK_BYTES..P::INDCPA_SK_BYTES + P::PK_BYTES]
        .copy_from_slice(&pk_arr.as_ref()[..P::PK_BYTES]);

    let h_pk = kem_hash::hash_h(&pk_arr.as_ref()[..P::PK_BYTES]);
    sk_mut[P::SK_BYTES - 2 * SYMBYTES..P::SK_BYTES - SYMBYTES].copy_from_slice(&h_pk);
    sk_mut[P::SK_BYTES - SYMBYTES..P::SK_BYTES].copy_from_slice(z);

    (PublicKey { bytes: pk_arr }, SecretKey { bytes: sk })
}

/// Randomized key generation.
pub fn keypair<P: ParameterSet>(rng: &mut impl CryptoRng) -> (PublicKey<P>, SecretKey<P>) {
    let mut coins = [0u8; 2 * SYMBYTES];
    rng.fill_bytes(&mut coins);
    keypair_derand::<P>(&coins)
}

/// Deterministic encapsulation with explicit 32-byte randomness.
///
/// # Panics
///
/// Cannot panic: all internal array splits operate on compile-time-known sizes.
#[must_use]
pub fn encapsulate_derand<P: ParameterSet>(
    pk: &PublicKey<P>, coins: &[u8; SYMBYTES],
) -> (Ciphertext<P>, SharedSecret) {
    let mut buf = [0u8; 2 * SYMBYTES];
    buf[..SYMBYTES].copy_from_slice(coins);
    let h_pk = kem_hash::hash_h(pk.as_ref());
    buf[SYMBYTES..].copy_from_slice(&h_pk);

    let kr = kem_hash::hash_g(buf);
    let (k_half, r_half) = kr.split_at(SYMBYTES);
    let k: &[u8; SYMBYTES] = k_half
        .first_chunk()
        .expect("infallible: 64-byte hash split");
    let r: &[u8; SYMBYTES] = r_half
        .first_chunk()
        .expect("infallible: 64-byte hash split");

    let ct = pke::indcpa_enc::<P>(pk.as_ref(), coins, r);

    (Ciphertext { bytes: ct }, SharedSecret { bytes: *k })
}

/// Randomized encapsulation.
pub fn encapsulate<P: ParameterSet>(
    pk: &PublicKey<P>, rng: &mut impl CryptoRng,
) -> (Ciphertext<P>, SharedSecret) {
    let mut coins = [0u8; SYMBYTES];
    rng.fill_bytes(&mut coins);
    encapsulate_derand::<P>(pk, &coins)
}

/// Decapsulation with implicit rejection.
///
/// # Panics
///
/// Cannot panic: all internal array splits operate on compile-time-known sizes.
#[must_use]
pub fn decapsulate<P: ParameterSet>(ct: &Ciphertext<P>, sk: &SecretKey<P>) -> SharedSecret {
    let sk_ref = sk.as_ref();
    let indcpa_sk = &sk_ref[..P::INDCPA_SK_BYTES];
    let pk = &sk_ref[P::INDCPA_SK_BYTES..P::INDCPA_SK_BYTES + P::PK_BYTES];
    let h: &[u8; SYMBYTES] = sk_ref[P::SK_BYTES - 2 * SYMBYTES..P::SK_BYTES - SYMBYTES]
        .first_chunk()
        .expect("infallible: sk layout guarantees SYMBYTES");
    let z: &[u8; SYMBYTES] = sk_ref[P::SK_BYTES - SYMBYTES..P::SK_BYTES]
        .first_chunk()
        .expect("infallible: sk layout guarantees SYMBYTES");

    let m_prime = pke::indcpa_dec::<P>(ct.as_ref(), indcpa_sk);

    let mut buf = [0u8; 2 * SYMBYTES];
    buf[..SYMBYTES].copy_from_slice(&m_prime);
    buf[SYMBYTES..].copy_from_slice(h);
    let kr = kem_hash::hash_g(buf);
    let (k_half, r_half) = kr.split_at(SYMBYTES);
    let k: &[u8; SYMBYTES] = k_half
        .first_chunk()
        .expect("infallible: 64-byte hash split");
    let r: &[u8; SYMBYTES] = r_half
        .first_chunk()
        .expect("infallible: 64-byte hash split");

    let ct_prime = pke::indcpa_enc::<P>(pk, &m_prime, r);

    let ok = ct.as_ref().ct_eq(ct_prime.as_ref());
    let rejection = kem_hash::rkprf(z, ct.as_ref());

    let mut ss = [0u8; SSBYTES];
    ss.copy_from_slice(k);
    ss.ct_assign(&rejection, !ok);

    SharedSecret { bytes: ss }
}
