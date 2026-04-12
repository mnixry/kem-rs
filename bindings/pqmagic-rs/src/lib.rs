#[cfg(not(unix))]
compile_error!("pqmagic-rs only supports Unix-like platforms");

use core::ffi::c_int;

pub const MLKEM_BYTES: usize = 32;

pub const MLKEM512_PK_BYTES: usize = 800;
pub const MLKEM512_SK_BYTES: usize = 1632;
pub const MLKEM512_CT_BYTES: usize = 768;

pub const MLKEM768_PK_BYTES: usize = 1184;
pub const MLKEM768_SK_BYTES: usize = 2400;
pub const MLKEM768_CT_BYTES: usize = 1088;

pub const MLKEM1024_PK_BYTES: usize = 1568;
pub const MLKEM1024_SK_BYTES: usize = 3168;
pub const MLKEM1024_CT_BYTES: usize = 1568;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Fail,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("pqmagic operation failed")
    }
}

impl std::error::Error for Error {}

const fn check(rc: c_int) -> Result<(), Error> {
    match rc {
        0 => Ok(()),
        _ => Err(Error::Fail),
    }
}

pub(crate) mod ffi {
    use core::ffi::c_int;

    unsafe extern "C" {
        pub fn pqmagic_ml_kem_512_std_keypair_internal(
            pk: *mut u8, sk: *mut u8, coins: *const u8,
        ) -> c_int;
        pub fn pqmagic_ml_kem_512_std_enc_internal(
            ct: *mut u8, ss: *mut u8, pk: *const u8, coins: *const u8,
        ) -> c_int;
        pub fn pqmagic_ml_kem_512_std_dec(ss: *mut u8, ct: *const u8, sk: *const u8) -> c_int;

        pub fn pqmagic_ml_kem_768_std_keypair_internal(
            pk: *mut u8, sk: *mut u8, coins: *const u8,
        ) -> c_int;
        pub fn pqmagic_ml_kem_768_std_enc_internal(
            ct: *mut u8, ss: *mut u8, pk: *const u8, coins: *const u8,
        ) -> c_int;
        pub fn pqmagic_ml_kem_768_std_dec(ss: *mut u8, ct: *const u8, sk: *const u8) -> c_int;

        pub fn pqmagic_ml_kem_1024_std_keypair_internal(
            pk: *mut u8, sk: *mut u8, coins: *const u8,
        ) -> c_int;
        pub fn pqmagic_ml_kem_1024_std_enc_internal(
            ct: *mut u8, ss: *mut u8, pk: *const u8, coins: *const u8,
        ) -> c_int;
        pub fn pqmagic_ml_kem_1024_std_dec(ss: *mut u8, ct: *const u8, sk: *const u8) -> c_int;
    }
}

macro_rules! impl_mlkem {
    (
        $mod:ident,
        pk = $pk:literal, sk = $sk:literal, ct = $ct:literal,
        keypair_derand = $ffi_kg:path,
        enc_derand = $ffi_enc:path,
        dec = $ffi_dec:path $(,)?
    ) => {
        #[allow(clippy::missing_errors_doc)]
        pub mod $mod {
            use super::{Error, MLKEM_BYTES, check};

            pub fn keypair_derand(coins: &[u8; 64]) -> Result<([u8; $pk], [u8; $sk]), Error> {
                let mut pk = [0u8; $pk];
                let mut sk = [0u8; $sk];
                check(unsafe { $ffi_kg(pk.as_mut_ptr(), sk.as_mut_ptr(), coins.as_ptr()) })?;
                Ok((pk, sk))
            }

            pub fn enc_derand(
                pk: &[u8; $pk], coins: &[u8; 32],
            ) -> Result<([u8; $ct], [u8; MLKEM_BYTES]), Error> {
                let mut ct = [0u8; $ct];
                let mut ss = [0u8; MLKEM_BYTES];
                check(unsafe {
                    $ffi_enc(
                        ct.as_mut_ptr(),
                        ss.as_mut_ptr(),
                        pk.as_ptr(),
                        coins.as_ptr(),
                    )
                })?;
                Ok((ct, ss))
            }

            pub fn dec(ct: &[u8; $ct], sk: &[u8; $sk]) -> Result<[u8; MLKEM_BYTES], Error> {
                let mut ss = [0u8; MLKEM_BYTES];
                check(unsafe { $ffi_dec(ss.as_mut_ptr(), ct.as_ptr(), sk.as_ptr()) })?;
                Ok(ss)
            }
        }
    };
}

impl_mlkem!(
    mlkem512,
    pk = 800,
    sk = 1632,
    ct = 768,
    keypair_derand = crate::ffi::pqmagic_ml_kem_512_std_keypair_internal,
    enc_derand = crate::ffi::pqmagic_ml_kem_512_std_enc_internal,
    dec = crate::ffi::pqmagic_ml_kem_512_std_dec,
);

impl_mlkem!(
    mlkem768,
    pk = 1184,
    sk = 2400,
    ct = 1088,
    keypair_derand = crate::ffi::pqmagic_ml_kem_768_std_keypair_internal,
    enc_derand = crate::ffi::pqmagic_ml_kem_768_std_enc_internal,
    dec = crate::ffi::pqmagic_ml_kem_768_std_dec,
);

impl_mlkem!(
    mlkem1024,
    pk = 1568,
    sk = 3168,
    ct = 1568,
    keypair_derand = crate::ffi::pqmagic_ml_kem_1024_std_keypair_internal,
    enc_derand = crate::ffi::pqmagic_ml_kem_1024_std_enc_internal,
    dec = crate::ffi::pqmagic_ml_kem_1024_std_dec,
);
