//! ML-KEM key and ciphertext types.
//!
//! All types wrap typed byte arrays parameterised by [`ParameterSet`].
//! Secret types implement `ZeroizeOnDrop` for automatic memory clearing.

use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::params::{ParameterSet, SSBYTES};

/// ML-KEM encapsulation key (public key).
pub struct PublicKey<P: ParameterSet> {
    pub(crate) bytes: P::PkArray,
}

impl<P: ParameterSet> PublicKey<P> {
    pub fn from_bytes(bytes: &P::PkArray) -> Self {
        Self {
            bytes: bytes.clone(),
        }
    }
}

impl<P: ParameterSet> From<&P::PkArray> for PublicKey<P> {
    #[inline]
    fn from(arr: &P::PkArray) -> Self {
        Self { bytes: arr.clone() }
    }
}

impl<P: ParameterSet> AsRef<[u8]> for PublicKey<P> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.bytes.as_ref()
    }
}

impl<P: ParameterSet> Clone for PublicKey<P> {
    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes.clone(),
        }
    }
}

impl<P: ParameterSet> core::fmt::Debug for PublicKey<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PublicKey")
            .field("len", &self.bytes.as_ref().len())
            .finish_non_exhaustive()
    }
}

/// ML-KEM decapsulation key (secret key).
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SecretKey<P: ParameterSet> {
    pub(crate) bytes: P::SkArray,
}

impl<P: ParameterSet> SecretKey<P> {
    pub fn from_bytes(bytes: &P::SkArray) -> Self {
        Self {
            bytes: bytes.clone(),
        }
    }
}

impl<P: ParameterSet> From<&P::SkArray> for SecretKey<P> {
    #[inline]
    fn from(arr: &P::SkArray) -> Self {
        Self { bytes: arr.clone() }
    }
}

impl<P: ParameterSet> AsRef<[u8]> for SecretKey<P> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.bytes.as_ref()
    }
}

impl<P: ParameterSet> Clone for SecretKey<P> {
    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes.clone(),
        }
    }
}

impl<P: ParameterSet> core::fmt::Debug for SecretKey<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SecretKey")
            .field("len", &self.bytes.as_ref().len())
            .finish_non_exhaustive()
    }
}

/// ML-KEM ciphertext.
pub struct Ciphertext<P: ParameterSet> {
    pub(crate) bytes: P::CtArray,
}

impl<P: ParameterSet> Ciphertext<P> {
    pub fn from_bytes(bytes: &P::CtArray) -> Self {
        Self {
            bytes: bytes.clone(),
        }
    }
}

impl<P: ParameterSet> From<&P::CtArray> for Ciphertext<P> {
    #[inline]
    fn from(arr: &P::CtArray) -> Self {
        Self { bytes: arr.clone() }
    }
}

impl<P: ParameterSet> AsRef<[u8]> for Ciphertext<P> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.bytes.as_ref()
    }
}

impl<P: ParameterSet> Clone for Ciphertext<P> {
    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes.clone(),
        }
    }
}

impl<P: ParameterSet> core::fmt::Debug for Ciphertext<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ciphertext")
            .field("len", &self.bytes.as_ref().len())
            .finish_non_exhaustive()
    }
}

/// ML-KEM shared secret (32 bytes).
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct SharedSecret {
    pub(crate) bytes: [u8; SSBYTES],
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
