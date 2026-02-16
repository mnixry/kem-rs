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

use crate::params::{SSBYTES, SYMBYTES};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::{Digest, Sha3_256, Sha3_512, Shake128, Shake256};

/// H(input) = SHA3-256(input) → 32 bytes.
#[inline]
pub fn hash_h(input: &[u8]) -> [u8; 32] {
    let mut h = Sha3_256::new();
    Digest::update(&mut h, input);
    h.finalize().into()
}

/// G(input) = SHA3-512(input) → 64 bytes.
#[inline]
pub fn hash_g(input: &[u8]) -> [u8; 64] {
    let mut h = Sha3_512::new();
    Digest::update(&mut h, input);
    h.finalize().into()
}

/// PRFη(seed, nonce) = SHAKE-256(seed ‖ nonce), squeezed to fill `output`.
pub fn prf(seed: &[u8; SYMBYTES], nonce: u8, output: &mut [u8]) {
    let mut h = Shake256::default();
    Update::update(&mut h, seed);
    Update::update(&mut h, &[nonce]);
    let mut reader = h.finalize_xof();
    reader.read(output);
}

/// Create a SHAKE-128 XOF absorber for matrix sampling.
///
/// Absorbs `seed ‖ x ‖ y` and returns a reader from which uniform
/// bytes can be squeezed.
pub fn xof_absorb(seed: &[u8; SYMBYTES], x: u8, y: u8) -> impl XofReader {
    let mut h = Shake128::default();
    Update::update(&mut h, seed);
    Update::update(&mut h, &[x, y]);
    h.finalize_xof()
}

/// J(key, ct) = SHAKE-256(key ‖ ct) → 32 bytes.
///
/// Used as the rejection-key PRF in decapsulation (implicit reject).
pub fn rkprf(key: &[u8; SYMBYTES], ct: &[u8]) -> [u8; SSBYTES] {
    let mut h = Shake256::default();
    Update::update(&mut h, key);
    Update::update(&mut h, ct);
    let mut reader = h.finalize_xof();
    let mut out = [0u8; SSBYTES];
    reader.read(&mut out);
    out
}
