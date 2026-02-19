//! Key, ciphertext, and shared-secret newtypes with RAII zeroization. Secret
//! types zeroize on drop.

use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::params::{MlKemParams, SSBYTES};

/// ML-KEM public (encapsulation) key.
pub struct PublicKey<P: MlKemParams> {
    pub(crate) bytes: P::PkArray,
}

impl<P, const SIZE: usize> From<[u8; SIZE]> for PublicKey<P>
where
    P: MlKemParams<PkArray = [u8; SIZE]>,
{
    #[inline]
    fn from(bytes: [u8; SIZE]) -> Self {
        Self { bytes }
    }
}

impl<P: MlKemParams> From<&P::PkArray> for PublicKey<P> {
    #[inline]
    fn from(bytes: &P::PkArray) -> Self {
        Self {
            bytes: bytes.clone(),
        }
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
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SecretKey<P: MlKemParams> {
    pub(crate) bytes: P::SkArray,
}

impl<P, const SIZE: usize> From<[u8; SIZE]> for SecretKey<P>
where
    P: MlKemParams<SkArray = [u8; SIZE]>,
{
    #[inline]
    fn from(bytes: [u8; SIZE]) -> Self {
        Self { bytes }
    }
}

impl<P: MlKemParams> From<&P::SkArray> for SecretKey<P> {
    #[inline]
    fn from(bytes: &P::SkArray) -> Self {
        Self {
            bytes: bytes.clone(),
        }
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

impl<P: MlKemParams> core::fmt::Debug for SecretKey<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("SecretKey([REDACTED])")
    }
}

/// ML-KEM ciphertext.
pub struct Ciphertext<P: MlKemParams> {
    pub(crate) bytes: P::CtArray,
}

impl<P, const SIZE: usize> From<[u8; SIZE]> for Ciphertext<P>
where
    P: MlKemParams<CtArray = [u8; SIZE]>,
{
    #[inline]
    fn from(bytes: [u8; SIZE]) -> Self {
        Self { bytes }
    }
}

impl<P: MlKemParams> From<&P::CtArray> for Ciphertext<P> {
    #[inline]
    fn from(bytes: &P::CtArray) -> Self {
        Self {
            bytes: bytes.clone(),
        }
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

impl From<[u8; SSBYTES]> for SharedSecret {
    #[inline]
    fn from(bytes: [u8; SSBYTES]) -> Self {
        Self { bytes }
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
