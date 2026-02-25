//! ML-KEM key and ciphertext types.
//!
//! All types are `#[repr(C)]` structs with named sub-component fields,
//! parameterised by [`ParameterSet`].  Secret types implement `ZeroizeOnDrop`
//! for automatic memory clearing.
//!
//! Each key/ciphertext type provides a `try_from_slice` constructor that
//! validates the input.  For [`PublicKey`] this includes the FIPS 203 §7.2
//! modulus check; for [`SecretKey`] it verifies the embedded public-key hash.
//!
//! Byte (de)serialisation is zero-cost via `zerocopy` derives on `#[repr(C)]`
//! layouts.

use kem_math::{ByteArray, SYMBYTES};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{
    Error,
    params::{ParameterSet, SSBYTES},
};

/// Two contiguous [`SYMBYTES`]-sized halves (e.g. hash outputs, coin arrays).
///
/// [`zerocopy::transmute_ref!`] guarantees at compile time that the source and
/// target types have identical size and compatible alignment, eliminating the
/// need for fallible `.expect()` / `.unwrap()` calls.
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct Sym2(pub [u8; SYMBYTES], pub [u8; SYMBYTES]);

/// ML-KEM encapsulation key (public key).
///
/// Layout: `polyvec || rho` where `polyvec` is the NTT-encoded vector and
/// `rho` is the 32-byte public seed.
#[derive(Clone, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct PublicKey<P: ParameterSet> {
    pub(crate) polyvec: P::PolyVecArray,
    pub(crate) rho: [u8; SYMBYTES],
}

impl<P: ParameterSet> TryInto<PublicKey<P>> for &[u8] {
    type Error = Error;

    /// Construct a [`PublicKey`] from a byte slice.
    ///
    /// Validates:
    /// 1. Length -- must be exactly `P::PK_BYTES`.
    /// 2. FIPS 203 §7.2 modulus check -- the NTT-vector portion is decoded and
    ///    re-encoded; if the result differs, the key contains coefficients
    ///    outside `[0, q)` and is rejected.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLength`] or [`Error::InvalidKey`].
    fn try_into(self) -> Result<PublicKey<P>, Self::Error> {
        let pk: PublicKey<P> =
            PublicKey::read_from_bytes(self).map_err(|_| Error::InvalidLength {
                expected: P::PK_BYTES,
                actual: self.len(),
            })?;

        let t_hat = P::ntt_vec_from_bytes(pk.polyvec.as_ref());
        let mut check = P::PolyVecArray::zeroed();
        P::ntt_vec_to_bytes(&t_hat, check.as_mut());
        if check.as_ref() != pk.polyvec.as_ref() {
            return Err(Error::InvalidKey);
        }

        Ok(pk)
    }
}

impl<P: ParameterSet> AsRef<[u8]> for PublicKey<P> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        <Self as IntoBytes>::as_bytes(self)
    }
}

impl<P: ParameterSet> core::fmt::Debug for PublicKey<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PublicKey")
            .field("len", &self.as_ref().len())
            .finish_non_exhaustive()
    }
}

/// ML-KEM decapsulation key (secret key).
///
/// Layout: `indcpa_sk || pk_polyvec || pk_rho || h || z`
#[derive(Clone, FromBytes, IntoBytes, KnownLayout, Immutable, Zeroize, ZeroizeOnDrop)]
#[repr(C)]
pub struct SecretKey<P: ParameterSet> {
    pub(crate) indcpa_sk: P::PolyVecArray,
    pub(crate) pk_polyvec: P::PolyVecArray,
    pub(crate) pk_rho: [u8; SYMBYTES],
    pub(crate) h: [u8; SYMBYTES],
    pub(crate) z: [u8; SYMBYTES],
}

impl<P: ParameterSet> SecretKey<P> {
    /// Reconstruct the embedded [`PublicKey`] from this secret key.
    pub(crate) fn pk(&self) -> PublicKey<P> {
        PublicKey {
            polyvec: self.pk_polyvec.clone(),
            rho: self.pk_rho,
        }
    }
}

impl<P: ParameterSet> TryInto<SecretKey<P>> for &[u8] {
    type Error = Error;

    /// Construct a [`SecretKey`] from a byte slice.
    ///
    /// Validates:
    /// 1. Length -- must be exactly `P::SK_BYTES`.
    /// 2. Hash consistency -- the embedded `h = H(pk)` field must match
    ///    `hash_h` computed over the embedded public-key portion.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLength`] or [`Error::InvalidKey`].
    fn try_into(self) -> Result<SecretKey<P>, Self::Error> {
        let sk: SecretKey<P> =
            SecretKey::read_from_bytes(self).map_err(|_| Error::InvalidLength {
                expected: P::SK_BYTES,
                actual: self.len(),
            })?;

        let pk_ref = sk.pk();
        let h_computed = kem_hash::hash_h(pk_ref.as_ref());
        if h_computed != sk.h {
            return Err(Error::InvalidKey);
        }

        Ok(sk)
    }
}

impl<P: ParameterSet> AsRef<[u8]> for SecretKey<P> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        <Self as IntoBytes>::as_bytes(self)
    }
}

impl<P: ParameterSet> core::fmt::Debug for SecretKey<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SecretKey")
            .field("len", &self.as_ref().len())
            .finish_non_exhaustive()
    }
}

/// ML-KEM ciphertext.
///
/// Layout: `u_compressed || v_compressed`
#[derive(Clone, FromBytes, IntoBytes, KnownLayout, Immutable, Zeroize, ZeroizeOnDrop)]
#[repr(C)]
pub struct Ciphertext<P: ParameterSet> {
    pub(crate) u_compressed: P::PolyVecCompArray,
    pub(crate) v_compressed: P::PolyCompArray,
}

impl<P: ParameterSet> TryInto<Ciphertext<P>> for &[u8] {
    type Error = Error;
    fn try_into(self) -> Result<Ciphertext<P>, Self::Error> {
        Ciphertext::read_from_bytes(self).map_err(|_| Error::InvalidLength {
            expected: P::CT_BYTES,
            actual: self.len(),
        })
    }
}

impl<P: ParameterSet> AsRef<[u8]> for Ciphertext<P> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        <Self as IntoBytes>::as_bytes(self)
    }
}

impl<P: ParameterSet> core::fmt::Debug for Ciphertext<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ciphertext").finish_non_exhaustive()
    }
}

/// ML-KEM shared secret (32 bytes).
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct SharedSecret {
    bytes: [u8; SSBYTES],
}

impl From<&[u8; SSBYTES]> for SharedSecret {
    #[inline]
    fn from(arr: &[u8; SSBYTES]) -> Self {
        Self { bytes: *arr }
    }
}

impl AsRef<[u8]> for SharedSecret {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl core::fmt::Debug for SharedSecret {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SharedSecret").finish_non_exhaustive()
    }
}
