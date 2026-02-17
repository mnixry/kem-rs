//! IND-CCA2 Key Encapsulation - ML-KEM (FIPS 203). Keygen, encapsulate, decapsulate.

use crate::ct;
use crate::hash;
use crate::params::{ByteArray, MlKemParams, SYMBYTES};
use crate::pke;
use crate::types::{Ciphertext, PublicKey, SecretKey, SharedSecret};

/// Deterministic key generation from 64 bytes of randomness. coins = (d || z): d seeds IND-CPA keypair, z for implicit reject.
pub fn keypair_derand<P: MlKemParams>(coins: &[u8; 2 * SYMBYTES]) -> (PublicKey<P>, SecretKey<P>) {
    let mut pk_arr = P::PkArray::zeroed();
    let mut sk_arr = P::SkArray::zeroed();

    let pk = pk_arr.as_mut();
    let sk = sk_arr.as_mut();

    // IND-CPA keypair from first 32 bytes
    pke::indcpa_keypair_derand::<P>(
        &mut pk[..P::INDCPA_PK_BYTES],
        &mut sk[..P::INDCPA_SK_BYTES],
        coins[..SYMBYTES].try_into().unwrap(),
    );

    // sk = (indcpa_sk || pk || H(pk) || z)
    sk[P::INDCPA_SK_BYTES..P::INDCPA_SK_BYTES + P::PK_BYTES]
        .copy_from_slice(&pk[..P::PK_BYTES]);

    let h_pk = hash::hash_h(&pk[..P::PK_BYTES]);
    sk[P::SK_BYTES - 2 * SYMBYTES..P::SK_BYTES - SYMBYTES].copy_from_slice(&h_pk);

    sk[P::SK_BYTES - SYMBYTES..P::SK_BYTES].copy_from_slice(&coins[SYMBYTES..]);

    (
        PublicKey::from_bytes(pk_arr),
        SecretKey::from_bytes(sk_arr),
    )
}

/// Key generation with system randomness.
pub fn keypair<P: MlKemParams>(rng: &mut impl rand_core::CryptoRng) -> (PublicKey<P>, SecretKey<P>) {
    let mut coins = [0u8; 2 * SYMBYTES];
    rng.fill_bytes(&mut coins);
    keypair_derand::<P>(&coins)
}

/// Deterministic encapsulation from 32 bytes of randomness. Produces ciphertext and shared secret.
pub fn encapsulate_derand<P: MlKemParams>(
    pk: &PublicKey<P>,
    coins: &[u8; SYMBYTES],
) -> (Ciphertext<P>, SharedSecret) {
    let mut ct_arr = P::CtArray::zeroed();

    // buf = m || H(pk)
    let mut buf = [0u8; 2 * SYMBYTES];
    buf[..SYMBYTES].copy_from_slice(coins);
    let h_pk = hash::hash_h(pk.as_bytes());
    buf[SYMBYTES..].copy_from_slice(&h_pk);

    // kr = G(buf) = (K || r)
    let kr = hash::hash_g(&buf);

    // IND-CPA encrypt: ct = Enc(pk, m; r)
    pke::indcpa_enc::<P>(
        ct_arr.as_mut(),
        coins,
        pk.as_bytes(),
        kr[SYMBYTES..].try_into().unwrap(),
    );

    // ss = K
    let mut ss = [0u8; SYMBYTES];
    ss.copy_from_slice(&kr[..SYMBYTES]);

    (Ciphertext::from_bytes(ct_arr), SharedSecret::from_bytes(ss))
}

/// Encapsulation with system randomness.
pub fn encapsulate<P: MlKemParams>(
    pk: &PublicKey<P>,
    rng: &mut impl rand_core::CryptoRng,
) -> (Ciphertext<P>, SharedSecret) {
    let mut coins = [0u8; SYMBYTES];
    rng.fill_bytes(&mut coins);
    encapsulate_derand::<P>(pk, &coins)
}

/// Decapsulate: recover shared secret. Uses implicit rejection on failure (pseudorandom ss from sk, ct).
pub fn decapsulate<P: MlKemParams>(ct: &Ciphertext<P>, sk: &SecretKey<P>) -> SharedSecret {
    let sk_bytes = sk.as_bytes();
    let ct_bytes = ct.as_bytes();

    // Parse the secret key: (indcpa_sk || pk || H(pk) || z)
    let indcpa_sk = &sk_bytes[..P::INDCPA_SK_BYTES];
    let pk_bytes = &sk_bytes[P::INDCPA_SK_BYTES..P::INDCPA_SK_BYTES + P::PK_BYTES];
    let h_pk = &sk_bytes[P::SK_BYTES - 2 * SYMBYTES..P::SK_BYTES - SYMBYTES];
    let z = &sk_bytes[P::SK_BYTES - SYMBYTES..P::SK_BYTES];

    // m' = Dec(indcpa_sk, ct)
    let mut m_prime = [0u8; SYMBYTES];
    pke::indcpa_dec::<P>(&mut m_prime, ct_bytes, indcpa_sk);

    // buf = m' || H(pk)
    let mut buf = [0u8; 2 * SYMBYTES];
    buf[..SYMBYTES].copy_from_slice(&m_prime);
    buf[SYMBYTES..].copy_from_slice(h_pk);

    // kr = G(buf) = (K' || r')
    let kr = hash::hash_g(&buf);

    // Re-encrypt: ct' = Enc(pk, m'; r')
    // Use a stack buffer large enough for any parameter set.
    const MAX_CT: usize = 1568;
    let mut cmp = [0u8; MAX_CT];
    pke::indcpa_enc::<P>(
        &mut cmp[..P::CT_BYTES],
        &m_prime,
        pk_bytes,
        kr[SYMBYTES..].try_into().unwrap(),
    );

    // Constant-time comparison: fail = (ct != ct')
    let fail = ct::ct_verify(ct_bytes, &cmp[..P::CT_BYTES]);

    // Implicit rejection: ss = J(z, ct) if fail else K'
    let z: &[u8; SYMBYTES] = z.try_into().unwrap();
    let rejection_ss = hash::rkprf(z, ct_bytes);

    let mut ss = rejection_ss;
    ct::ct_cmov(&mut ss, &kr[..SYMBYTES], 1 - fail);

    SharedSecret::from_bytes(ss)
}
