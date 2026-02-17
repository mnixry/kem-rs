//! Key, ciphertext, and shared-secret newtypes with RAII zeroization. Secret types zeroize on drop.

use crate::params::{MlKemParams, SSBYTES};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// ML-KEM public (encapsulation) key.
pub struct PublicKey<P: MlKemParams> {
    pub(crate) bytes: P::PkArray,
}

impl<P: MlKemParams> PublicKey<P> {
    /// Wrap an existing byte array as a public key.
    #[inline]
    pub fn from_bytes(bytes: P::PkArray) -> Self {
        Self { bytes }
    }

    /// View the key as a byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    /// Consume the wrapper and return the inner byte array.
    #[inline]
    pub fn into_bytes(self) -> P::PkArray {
        self.bytes
    }
}

impl<P: MlKemParams> AsRef<[u8]> for PublicKey<P> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.bytes.as_ref()
    }
}

impl<P: MlKemParams> Clone for PublicKey<P> {
    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes.clone(),
        }
    }
}

impl<P: MlKemParams> core::fmt::Debug for PublicKey<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PublicKey")
            .field("len", &P::PK_BYTES)
            .finish_non_exhaustive()
    }
}

/// ML-KEM secret (decapsulation) key. Zeroized on drop.
pub struct SecretKey<P: MlKemParams> {
    pub(crate) bytes: P::SkArray,
}

impl<P: MlKemParams> SecretKey<P> {
    /// Wrap an existing byte array as a secret key.
    #[inline]
    pub fn from_bytes(bytes: P::SkArray) -> Self {
        Self { bytes }
    }

    /// View the key as a byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }
}

impl<P: MlKemParams> AsRef<[u8]> for SecretKey<P> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.bytes.as_ref()
    }
}

impl<P: MlKemParams> Clone for SecretKey<P> {
    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes.clone(),
        }
    }
}

impl<P: MlKemParams> Zeroize for SecretKey<P> {
    fn zeroize(&mut self) {
        self.bytes.zeroize();
    }
}

impl<P: MlKemParams> Drop for SecretKey<P> {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl<P: MlKemParams> core::fmt::Debug for SecretKey<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("SecretKey([REDACTED])")
    }
}

/// ML-KEM ciphertext.
pub struct Ciphertext<P: MlKemParams> {
    pub(crate) bytes: P::CtArray,
}

impl<P: MlKemParams> Ciphertext<P> {
    /// Wrap an existing byte array as a ciphertext.
    #[inline]
    pub fn from_bytes(bytes: P::CtArray) -> Self {
        Self { bytes }
    }

    /// View the ciphertext as a byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    /// Consume the wrapper and return the inner byte array.
    #[inline]
    pub fn into_bytes(self) -> P::CtArray {
        self.bytes
    }
}

impl<P: MlKemParams> AsRef<[u8]> for Ciphertext<P> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.bytes.as_ref()
    }
}

impl<P: MlKemParams> Clone for Ciphertext<P> {
    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes.clone(),
        }
    }
}

impl<P: MlKemParams> core::fmt::Debug for Ciphertext<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ciphertext")
            .field("len", &P::CT_BYTES)
            .finish_non_exhaustive()
    }
}

/// Shared secret (always 32 bytes). Zeroized on drop.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct SharedSecret {
    pub(crate) bytes: [u8; SSBYTES],
}

impl SharedSecret {
    /// Wrap a raw 32-byte array as a shared secret.
    #[inline]
    pub fn from_bytes(bytes: [u8; SSBYTES]) -> Self {
        Self { bytes }
    }

    /// View the secret as a byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
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
        f.write_str("SharedSecret([REDACTED])")
    }
}
