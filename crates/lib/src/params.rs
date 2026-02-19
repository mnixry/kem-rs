//! ML-KEM parameter definitions. MlKemParams trait and marker types MlKem512,
//! MlKem768, MlKem1024.

pub use kem_math::{N, POLYBYTES, Q, SYMBYTES};
use zeroize::Zeroize;

/// Size in bytes of the shared-secret output.
pub const SSBYTES: usize = 32;

/// Fixed-size byte buffer usable as an ML-KEM key/ciphertext backing store.
pub trait ByteArray:
    AsRef<[u8]> + AsMut<[u8]> + Clone + core::fmt::Debug + Zeroize + Send + Sync + 'static {
    /// Array length in bytes.
    const LEN: usize;

    /// Return a zero-filled instance.
    fn zeroed() -> Self;
}

impl<const SIZE: usize> ByteArray for [u8; SIZE] {
    const LEN: usize = SIZE;

    #[inline]
    fn zeroed() -> Self {
        [0u8; SIZE]
    }
}

/// ML-KEM parameter set implemented by MlKem512, MlKem768, MlKem1024.
pub trait MlKemParams: 'static {
    /// Module rank (k = 2, 3, or 4).
    const K: usize;
    /// CBD noise parameter for keygen secret polynomials.
    const ETA1: usize;
    /// CBD noise parameter for encryption noise polynomials.
    const ETA2: usize;
    /// Compression bits for polynomial-vector ciphertext component.
    const D_U: u32;
    /// Compression bits for scalar polynomial ciphertext component.
    const D_V: u32;

    /// K * POLYBYTES - serialised polynomial vector.
    const POLYVEC_BYTES: usize;
    /// Compressed bytes for one polynomial (N * D_V / 8).
    const POLY_COMPRESSED_BYTES: usize;
    /// Compressed bytes for the polynomial vector (K * N * D_U / 8).
    const POLYVEC_COMPRESSED_BYTES: usize;
    /// IND-CPA message bytes (= [`SYMBYTES`]).
    const INDCPA_MSG_BYTES: usize = SYMBYTES;
    /// IND-CPA public key bytes (POLYVEC_BYTES + SYMBYTES).
    const INDCPA_PK_BYTES: usize;
    /// IND-CPA secret key bytes (`POLYVEC_BYTES`).
    const INDCPA_SK_BYTES: usize;
    /// IND-CPA ciphertext bytes.
    const INDCPA_BYTES: usize;
    /// ML-KEM public key bytes.
    const PK_BYTES: usize;
    /// ML-KEM secret key bytes.
    const SK_BYTES: usize;
    /// ML-KEM ciphertext bytes.
    const CT_BYTES: usize;

    /// Backing array for public keys.
    type PkArray: ByteArray;
    /// Backing array for secret keys.
    type SkArray: ByteArray;
    /// Backing array for ciphertexts.
    type CtArray: ByteArray;
}

/// ML-KEM-512 parameter set (k = 2, NIST security level 1).
#[derive(Debug, Clone, Copy)]
pub struct MlKem512;

impl MlKemParams for MlKem512 {
    const K: usize = 2;
    const ETA1: usize = 3;
    const ETA2: usize = 2;
    const D_U: u32 = 10;
    const D_V: u32 = 4;

    const POLYVEC_BYTES: usize = 768; // 2 * 384
    const POLY_COMPRESSED_BYTES: usize = 128; // 256 * 4 / 8
    const POLYVEC_COMPRESSED_BYTES: usize = 640; // 2 * 256 * 10 / 8
    const INDCPA_PK_BYTES: usize = 800; // 768 + 32
    const INDCPA_SK_BYTES: usize = 768;
    const INDCPA_BYTES: usize = 768; // 640 + 128
    const PK_BYTES: usize = 800;
    const SK_BYTES: usize = 1632; // 768 + 800 + 64
    const CT_BYTES: usize = 768;

    type PkArray = [u8; 800];
    type SkArray = [u8; 1632];
    type CtArray = [u8; 768];
}

/// ML-KEM-768 parameter set (k = 3, NIST security level 3).
#[derive(Debug, Clone, Copy)]
pub struct MlKem768;

impl MlKemParams for MlKem768 {
    const K: usize = 3;
    const ETA1: usize = 2;
    const ETA2: usize = 2;
    const D_U: u32 = 10;
    const D_V: u32 = 4;

    const POLYVEC_BYTES: usize = 1152; // 3 * 384
    const POLY_COMPRESSED_BYTES: usize = 128; // 256 * 4 / 8
    const POLYVEC_COMPRESSED_BYTES: usize = 960; // 3 * 256 * 10 / 8
    const INDCPA_PK_BYTES: usize = 1184; // 1152 + 32
    const INDCPA_SK_BYTES: usize = 1152;
    const INDCPA_BYTES: usize = 1088; // 960 + 128
    const PK_BYTES: usize = 1184;
    const SK_BYTES: usize = 2400; // 1152 + 1184 + 64
    const CT_BYTES: usize = 1088;

    type PkArray = [u8; 1184];
    type SkArray = [u8; 2400];
    type CtArray = [u8; 1088];
}

/// ML-KEM-1024 parameter set (k = 4, NIST security level 5).
#[derive(Debug, Clone, Copy)]
pub struct MlKem1024;

impl MlKemParams for MlKem1024 {
    const K: usize = 4;
    const ETA1: usize = 2;
    const ETA2: usize = 2;
    const D_U: u32 = 11;
    const D_V: u32 = 5;

    const POLYVEC_BYTES: usize = 1536; // 4 * 384
    const POLY_COMPRESSED_BYTES: usize = 160; // 256 * 5 / 8
    const POLYVEC_COMPRESSED_BYTES: usize = 1408; // 4 * 256 * 11 / 8
    const INDCPA_PK_BYTES: usize = 1568; // 1536 + 32
    const INDCPA_SK_BYTES: usize = 1536;
    const INDCPA_BYTES: usize = 1568; // 1408 + 160
    const PK_BYTES: usize = 1568;
    const SK_BYTES: usize = 3168; // 1536 + 1568 + 64
    const CT_BYTES: usize = 1568;

    type PkArray = [u8; 1568];
    type SkArray = [u8; 3168];
    type CtArray = [u8; 1568];
}

const _: () = {
    // --- Structural invariants ---
    macro_rules! check_params {
        ($t:ty) => {
            assert!(<$t>::POLYVEC_BYTES == <$t>::K * POLYBYTES);
            assert!(<$t>::POLY_COMPRESSED_BYTES == N * <$t>::D_V as usize / 8);
            assert!(<$t>::POLYVEC_COMPRESSED_BYTES == <$t>::K * N * <$t>::D_U as usize / 8);
            assert!(<$t>::INDCPA_PK_BYTES == <$t>::POLYVEC_BYTES + SYMBYTES);
            assert!(<$t>::INDCPA_SK_BYTES == <$t>::POLYVEC_BYTES);
            assert!(
                <$t>::INDCPA_BYTES == <$t>::POLYVEC_COMPRESSED_BYTES + <$t>::POLY_COMPRESSED_BYTES
            );
            assert!(<$t>::PK_BYTES == <$t>::INDCPA_PK_BYTES);
            assert!(<$t>::SK_BYTES == <$t>::INDCPA_SK_BYTES + <$t>::PK_BYTES + 2 * SYMBYTES);
            assert!(<$t>::CT_BYTES == <$t>::INDCPA_BYTES);
        };
    }
    check_params!(MlKem512);
    check_params!(MlKem768);
    check_params!(MlKem1024);

    // --- Cross-check against PQClean api.h reference values ---
    assert!(MlKem512::PK_BYTES == 800);
    assert!(MlKem512::SK_BYTES == 1632);
    assert!(MlKem512::CT_BYTES == 768);

    assert!(MlKem768::PK_BYTES == 1184);
    assert!(MlKem768::SK_BYTES == 2400);
    assert!(MlKem768::CT_BYTES == 1088);

    assert!(MlKem1024::PK_BYTES == 1568);
    assert!(MlKem1024::SK_BYTES == 3168);
    assert!(MlKem1024::CT_BYTES == 1568);
};
