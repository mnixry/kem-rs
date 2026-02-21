//! Hash and extendable-output function (XOF) adapters.
//!
//! Wraps the SHA-3 family as used by ML-KEM (FIPS 203):

use kem_math::SYMBYTES;
pub use sha3::digest::XofReader;
use sha3::{
    Digest, Sha3_256, Sha3_512, Shake128, Shake256,
    digest::{ExtendableOutput, Update},
};

use crate::params::SSBYTES;

/// H(input) = SHA3-256(input) -> 32 bytes.
#[inline]
pub fn hash_h(input: impl AsRef<[u8]>) -> [u8; 32] {
    Sha3_256::digest(input).into()
}

/// G(input) = SHA3-512(input) -> 64 bytes.
#[inline]
pub fn hash_g(input: impl AsRef<[u8]>) -> [u8; 64] {
    Sha3_512::digest(input).into()
}

/// `PRF(seed, nonce) = SHAKE-256(seed || nonce)`, squeezed into `output`.
pub fn prf(seed: &[u8; SYMBYTES], nonce: u8, output: &mut [u8]) {
    let mut buf = [0; SYMBYTES + 1];
    buf[..SYMBYTES].copy_from_slice(seed);
    buf[SYMBYTES] = nonce;
    Shake256::digest_xof(buf, output);
}

/// SHAKE-128 XOF: absorbs `seed || x || y`, returns reader.
#[must_use]
pub fn xof_absorb(seed: &[u8; SYMBYTES], x: u8, y: u8) -> impl XofReader {
    let mut buf = [0; SYMBYTES + 2];
    buf[..SYMBYTES].copy_from_slice(seed);
    buf[SYMBYTES..].copy_from_slice(&[x, y]);
    Shake128::default().chain(buf).finalize_xof()
}

/// J(key, ct) = SHAKE-256(key || ct) -> 32 bytes (implicit-reject PRF).
pub fn rkprf(key: impl AsRef<[u8]>, ct: impl AsRef<[u8]>) -> [u8; SSBYTES] {
    let mut out = [0u8; SSBYTES];
    Shake256::default()
        .chain(key)
        .chain(ct)
        .finalize_xof()
        .read(&mut out);
    out
}
