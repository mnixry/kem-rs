//! ML-KEM (IND-CCA2 Key Encapsulation Mechanism) -- FIPS 203.
//!
//! Output sizes are derived from [`ParameterSet`] associated types.

use ctutils::{CtAssign, CtEq};
use rand_core::CryptoRng;
use zerocopy::transmute_ref;

use crate::{
    params::{ParameterSet, SSBYTES, SYMBYTES},
    pke,
    types::{Ciphertext, PublicKey, SecretKey, SharedSecret, Sym2},
};

/// Deterministic key generation from a 64-byte seed `(d || z)`.
#[must_use]
pub fn keypair_derand<P: ParameterSet>(coins: &[u8; 2 * SYMBYTES]) -> (PublicKey<P>, SecretKey<P>) {
    let Sym2(d, z) = transmute_ref!(coins);

    let (pk, indcpa_sk) = pke::indcpa_keypair_derand::<P>(d);

    let h = kem_hash::hash_h(pk.as_ref());

    let sk = SecretKey {
        indcpa_sk,
        pk_polyvec: pk.polyvec.clone(),
        pk_rho: pk.rho,
        h,
        z: *z,
    };

    (pk, sk)
}

/// Randomized key generation.
pub fn keypair<P: ParameterSet>(rng: &mut impl CryptoRng) -> (PublicKey<P>, SecretKey<P>) {
    let mut coins = [0u8; 2 * SYMBYTES];
    rng.fill_bytes(&mut coins);
    keypair_derand::<P>(&coins)
}

/// Deterministic encapsulation with explicit 32-byte randomness.
#[must_use]
pub fn encapsulate_derand<P: ParameterSet>(
    pk: &PublicKey<P>, coins: &[u8; SYMBYTES],
) -> (Ciphertext<P>, SharedSecret) {
    let mut buf = [0u8; 2 * SYMBYTES];
    buf[..SYMBYTES].copy_from_slice(coins);
    let h_pk = kem_hash::hash_h(pk.as_ref());
    buf[SYMBYTES..].copy_from_slice(&h_pk);

    let kr = kem_hash::hash_g(buf);
    let Sym2(k, r) = transmute_ref!(&kr);

    let ct = pke::indcpa_enc::<P>(pk, coins, r);

    (ct, k.into())
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
#[must_use]
pub fn decapsulate<P: ParameterSet>(ct: &Ciphertext<P>, sk: &SecretKey<P>) -> SharedSecret {
    let pk = sk.pk();

    let m_prime = pke::indcpa_dec::<P>(ct, &sk.indcpa_sk);

    let mut buf = [0u8; 2 * SYMBYTES];
    buf[..SYMBYTES].copy_from_slice(&m_prime);
    buf[SYMBYTES..].copy_from_slice(&sk.h);
    let kr = kem_hash::hash_g(buf);
    let Sym2(k, r) = transmute_ref!(&kr);

    let ct_prime = pke::indcpa_enc::<P>(&pk, &m_prime, r);

    let ok = ct.as_ref().ct_eq(ct_prime.as_ref());
    let rejection = kem_hash::rkprf(sk.z, ct.as_ref());

    let mut ss = [0u8; SSBYTES];
    ss.copy_from_slice(k);
    ss.ct_assign(&rejection, !ok);

    SharedSecret::from(&ss)
}
