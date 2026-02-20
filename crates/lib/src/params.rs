//! ML-KEM parameter definitions.
//!
//! The sealed [`ParameterSet`] trait encodes all ML-KEM parameters at the type
//! level, including associated types for K-dependent algebra (vectors,
//! matrices). This eliminates runtime `match P::K { ... _ => unreachable!() }`
//! dispatch.

pub use kem_math::{N, POLYBYTES, Q, SYMBYTES};
use kem_math::{
    compress::{self, CompressWidth},
    poly::{NttPolynomial, Polynomial},
    polyvec::{NttMatrix, NttVector, Vector},
    sample::{self, CbdWidth},
};
use zeroize::Zeroize;

pub const SSBYTES: usize = 32;

pub trait ByteArray:
    AsRef<[u8]> + AsMut<[u8]> + Clone + core::fmt::Debug + Zeroize + Send + Sync + 'static {
    const LEN: usize;
    fn zeroed() -> Self;
}

impl<const SIZE: usize> ByteArray for [u8; SIZE] {
    const LEN: usize = SIZE;

    #[inline]
    fn zeroed() -> Self {
        [0u8; SIZE]
    }
}

mod sealed {
    pub trait Sealed {}
}

/// ML-KEM parameter set. Sealed -- only implemented for [`MlKem512`],
/// [`MlKem768`], [`MlKem1024`].
pub trait ParameterSet: sealed::Sealed + 'static {
    const K: usize;

    type Eta1: CbdWidth;
    type Eta2: CbdWidth;
    type Du: CompressWidth;
    type Dv: CompressWidth;

    const POLYVEC_BYTES: usize;
    const POLY_COMPRESSED_BYTES: usize;
    const POLYVEC_COMPRESSED_BYTES: usize;
    const INDCPA_PK_BYTES: usize;
    const INDCPA_SK_BYTES: usize;
    const INDCPA_BYTES: usize;
    const PK_BYTES: usize;
    const SK_BYTES: usize;
    const CT_BYTES: usize;

    type PkArray: ByteArray;
    type SkArray: ByteArray;
    type CtArray: ByteArray;

    // -- K-dependent algebra operations (monomorphized per parameter set) -----

    fn gen_matrix(seed: &[u8; SYMBYTES], transposed: bool) -> Self::Matrix;
    fn sample_noise_eta1(seed: &[u8; SYMBYTES], nonce: &mut u8) -> Self::NttVec;
    fn sample_noise_eta1_std(seed: &[u8; SYMBYTES], nonce: &mut u8) -> Self::Vec;
    fn sample_noise_eta2(seed: &[u8; SYMBYTES], nonce: &mut u8) -> Self::Vec;
    fn ntt_vec_from_bytes(bytes: &[u8]) -> Self::NttVec;
    fn ntt_vec_to_bytes(v: &Self::NttVec, out: &mut [u8]);
    fn vec_compress(v: &Self::Vec, out: &mut [u8]);
    fn vec_decompress(bytes: &[u8]) -> Self::Vec;
    fn mat_mul_vec_tomont(a: &Self::Matrix, v: &Self::NttVec) -> Self::NttVec;
    fn mat_mul_vec(a: &Self::Matrix, v: &Self::NttVec) -> Self::NttVec;
    fn inner_product(a: &Self::NttVec, b: &Self::NttVec) -> NttPolynomial;
    fn ntt_vec(v: Self::Vec) -> Self::NttVec;
    fn inv_ntt_vec(v: Self::NttVec) -> Self::Vec;
    fn add_ntt_vecs(a: &Self::NttVec, b: &Self::NttVec) -> Self::NttVec;
    fn add_vecs(a: &Self::Vec, b: &Self::Vec) -> Self::Vec;
    fn reduce_ntt_vec(v: &mut Self::NttVec);
    fn reduce_vec(v: &mut Self::Vec);

    type NttVec: Clone;
    type Vec: Clone;
    type Matrix;
}

// -- Macro to implement ParameterSet for each K ------------------------------

macro_rules! impl_parameter_set {
    (
        $name:ident, K = $K:literal,
        Eta1 = $Eta1:ty, Eta2 = $Eta2:ty,
        Du = $Du:ty, Dv = $Dv:ty,
        PkArray = $PkArray:ty, SkArray = $SkArray:ty, CtArray = $CtArray:ty,
        POLYVEC_BYTES = $pvb:literal,
        POLY_COMPRESSED_BYTES = $pcb:literal,
        POLYVEC_COMPRESSED_BYTES = $pvcb:literal,
        INDCPA_PK_BYTES = $ipkb:literal,
        INDCPA_SK_BYTES = $iskb:literal,
        INDCPA_BYTES = $ib:literal,
        PK_BYTES = $pkb:literal,
        SK_BYTES = $skb:literal,
        CT_BYTES = $ctb:literal
    ) => {
        impl sealed::Sealed for $name {}

        impl ParameterSet for $name {
            const K: usize = $K;
            type Eta1 = $Eta1;
            type Eta2 = $Eta2;
            type Du = $Du;
            type Dv = $Dv;

            const POLYVEC_BYTES: usize = $pvb;
            const POLY_COMPRESSED_BYTES: usize = $pcb;
            const POLYVEC_COMPRESSED_BYTES: usize = $pvcb;
            const INDCPA_PK_BYTES: usize = $ipkb;
            const INDCPA_SK_BYTES: usize = $iskb;
            const INDCPA_BYTES: usize = $ib;
            const PK_BYTES: usize = $pkb;
            const SK_BYTES: usize = $skb;
            const CT_BYTES: usize = $ctb;

            type PkArray = $PkArray;
            type SkArray = $SkArray;
            type CtArray = $CtArray;

            type NttVec = NttVector<$K>;
            type Vec = Vector<$K>;
            type Matrix = NttMatrix<$K>;

            fn gen_matrix(seed: &[u8; SYMBYTES], transposed: bool) -> Self::Matrix {
                gen_matrix_inner::<$K>(seed, transposed)
            }

            fn sample_noise_eta1(seed: &[u8; SYMBYTES], nonce: &mut u8) -> Self::NttVec {
                sample_noise_ntt::<$Eta1, $K>(seed, nonce)
            }

            fn sample_noise_eta1_std(seed: &[u8; SYMBYTES], nonce: &mut u8) -> Self::Vec {
                sample_noise_std::<$Eta1, $K>(seed, nonce)
            }

            fn sample_noise_eta2(seed: &[u8; SYMBYTES], nonce: &mut u8) -> Self::Vec {
                sample_noise_std::<$Eta2, $K>(seed, nonce)
            }

            fn ntt_vec_from_bytes(bytes: &[u8]) -> Self::NttVec {
                NttVector::<$K>::from_bytes(bytes)
            }

            fn ntt_vec_to_bytes(v: &Self::NttVec, out: &mut [u8]) {
                v.to_bytes(out);
            }

            fn vec_compress(v: &Self::Vec, out: &mut [u8]) {
                v.compress::<$Du>(out);
            }

            fn vec_decompress(bytes: &[u8]) -> Self::Vec {
                Vector::<$K>::decompress::<$Du>(bytes)
            }

            fn mat_mul_vec_tomont(a: &Self::Matrix, v: &Self::NttVec) -> Self::NttVec {
                a.mul_vec_tomont(v)
            }

            fn mat_mul_vec(a: &Self::Matrix, v: &Self::NttVec) -> Self::NttVec {
                a.mul_vec(v)
            }

            fn inner_product(a: &Self::NttVec, b: &Self::NttVec) -> NttPolynomial {
                a.inner_product(b)
            }

            fn ntt_vec(v: Self::Vec) -> Self::NttVec {
                v.ntt()
            }

            fn inv_ntt_vec(v: Self::NttVec) -> Self::Vec {
                v.ntt_inverse()
            }

            fn add_ntt_vecs(a: &Self::NttVec, b: &Self::NttVec) -> Self::NttVec {
                let mut r = a.clone();
                r += b;
                r
            }

            fn add_vecs(a: &Self::Vec, b: &Self::Vec) -> Self::Vec {
                let mut r = a.clone();
                r += b;
                r
            }

            fn reduce_ntt_vec(v: &mut Self::NttVec) {
                v.reduce();
            }

            fn reduce_vec(v: &mut Self::Vec) {
                v.reduce();
            }
        }
    };
}

// -- Helper functions used by the macro impls --------------------------------

fn gen_matrix_inner<const K: usize>(seed: &[u8; SYMBYTES], transposed: bool) -> NttMatrix<K> {
    use sha3::digest::XofReader;
    let mut a = NttMatrix::<K>::zero();
    for (i, a_row) in a.rows_mut().iter_mut().enumerate() {
        for (j, poly) in a_row.polys_mut().iter_mut().enumerate() {
            let (x, y) = if transposed {
                (i as u8, j as u8)
            } else {
                (j as u8, i as u8)
            };
            let mut xof = crate::hash::xof_absorb(seed, x, y);
            sample::rej_uniform(poly.coeffs_mut(), |buf| xof.read(buf));
        }
    }
    a
}

fn sample_noise_ntt<Eta: CbdWidth, const K: usize>(
    seed: &[u8; SYMBYTES], nonce: &mut u8,
) -> NttVector<K> {
    let mut v = NttVector::<K>::zero();
    let mut buf = [0u8; 192]; // max CBD buffer: eta=3 -> 192 bytes
    for p in v.polys_mut() {
        crate::hash::prf(seed, *nonce, &mut buf[..Eta::BUF_BYTES]);
        let poly = Polynomial::sample_cbd::<Eta>(&buf[..Eta::BUF_BYTES]);
        *p = poly.ntt();
        *nonce += 1;
    }
    v
}

fn sample_noise_std<Eta: CbdWidth, const K: usize>(
    seed: &[u8; SYMBYTES], nonce: &mut u8,
) -> Vector<K> {
    let mut v = Vector::<K>::zero();
    let mut buf = [0u8; 192];
    for p in v.polys_mut() {
        crate::hash::prf(seed, *nonce, &mut buf[..Eta::BUF_BYTES]);
        *p = Polynomial::sample_cbd::<Eta>(&buf[..Eta::BUF_BYTES]);
        *nonce += 1;
    }
    v
}

// -- Parameter set marker types ----------------------------------------------

/// ML-KEM-512 (k = 2, NIST security level 1).
#[derive(Debug, Clone, Copy)]
pub struct MlKem512;

/// ML-KEM-768 (k = 3, NIST security level 3).
#[derive(Debug, Clone, Copy)]
pub struct MlKem768;

/// ML-KEM-1024 (k = 4, NIST security level 5).
#[derive(Debug, Clone, Copy)]
pub struct MlKem1024;

impl_parameter_set!(
    MlKem512,
    K = 2,
    Eta1 = sample::Eta3,
    Eta2 = sample::Eta2,
    Du = compress::D10,
    Dv = compress::D4,
    PkArray = [u8; 800],
    SkArray = [u8; 1632],
    CtArray = [u8; 768],
    POLYVEC_BYTES = 768,
    POLY_COMPRESSED_BYTES = 128,
    POLYVEC_COMPRESSED_BYTES = 640,
    INDCPA_PK_BYTES = 800,
    INDCPA_SK_BYTES = 768,
    INDCPA_BYTES = 768,
    PK_BYTES = 800,
    SK_BYTES = 1632,
    CT_BYTES = 768
);

impl_parameter_set!(
    MlKem768,
    K = 3,
    Eta1 = sample::Eta2,
    Eta2 = sample::Eta2,
    Du = compress::D10,
    Dv = compress::D4,
    PkArray = [u8; 1184],
    SkArray = [u8; 2400],
    CtArray = [u8; 1088],
    POLYVEC_BYTES = 1152,
    POLY_COMPRESSED_BYTES = 128,
    POLYVEC_COMPRESSED_BYTES = 960,
    INDCPA_PK_BYTES = 1184,
    INDCPA_SK_BYTES = 1152,
    INDCPA_BYTES = 1088,
    PK_BYTES = 1184,
    SK_BYTES = 2400,
    CT_BYTES = 1088
);

impl_parameter_set!(
    MlKem1024,
    K = 4,
    Eta1 = sample::Eta2,
    Eta2 = sample::Eta2,
    Du = compress::D11,
    Dv = compress::D5,
    PkArray = [u8; 1568],
    SkArray = [u8; 3168],
    CtArray = [u8; 1568],
    POLYVEC_BYTES = 1536,
    POLY_COMPRESSED_BYTES = 160,
    POLYVEC_COMPRESSED_BYTES = 1408,
    INDCPA_PK_BYTES = 1568,
    INDCPA_SK_BYTES = 1536,
    INDCPA_BYTES = 1568,
    PK_BYTES = 1568,
    SK_BYTES = 3168,
    CT_BYTES = 1568
);

const _: () = {
    macro_rules! check_params {
        ($t:ty) => {
            assert!(<$t>::POLYVEC_BYTES == <$t>::K * POLYBYTES);
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
