//! Hash and extendable-output function (XOF) adapters.
//!
//! Wraps the SHA-3 family as used by ML-KEM (FIPS 203):
//!
//! | ML-KEM name | Primitive   | Function |
//! |-------------|-------------|----------|
//! | **H**       | SHA3-256    | [`hash_h`] |
//! | **G**       | SHA3-512    | [`hash_g`] |
//! | **PRF**     | SHAKE-256   | [`prf`] |
//! | **XOF**     | SHAKE-128   | [`xof_absorb`] |
//! | **J**       | SHAKE-256   | [`rkprf`] |

use sha3::{
    Digest, Sha3_256, Sha3_512, Shake128, Shake256,
    digest::{ExtendableOutput, Update, XofReader},
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
pub fn prf(seed: impl AsRef<[u8]>, nonce: u8, output: &mut [u8]) {
    Shake256::default()
        .chain(seed)
        .chain([nonce])
        .finalize_xof()
        .read(output);
}

/// SHAKE-128 XOF: absorbs `seed || x || y`, returns reader.
pub fn xof_absorb(seed: impl AsRef<[u8]>, x: u8, y: u8) -> impl XofReader {
    Shake128::default().chain(seed).chain([x, y]).finalize_xof()
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
